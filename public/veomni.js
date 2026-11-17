import { prepareConfigLog, dictionaryBlocks, readConfigFields } from './config.js';
import { BATCH, LAYERS, EXPERTS, COMPLETENESS, modelDefaults, formatBatch } from './record-fields.js';

const positive = value => Number.isInteger(value) && value > 0;
const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const leaf = path => typeof path === 'string' ? path.replace(/\/$/, '').split('/').at(-1) : '';

export function isVeOmniLog(text) {
  if (/['"]actor_rollout_ref['"]\s*:/.test(text)) return false;
  return /\bveomni\b/i.test(text) || (/['"]model_path['"]\s*:/.test(text) && /['"]train['"]\s*:/.test(text));
}

export function extractVeOmniConfig(text) {
  const { clean, ignored } = prepareConfigLog(text);
  const configs = [], modelConfigs = [], notes = [], fields = {}, evidence = {}, issues = [];
  const lineAt = pos => clean.slice(0, pos).split('\n').length;
  for (const block of dictionaryBlocks(clean)) {
    // The resolved dataclass dump and HF model dump are separate objects.
    const prefix = clean.slice(clean.lastIndexOf('\n', block.start - 1) + 1, block.start);
    if (!/['"](?:model|train|data)['"]\s*:/.test(block.text) && !/Model config \S+\s*$/.test(prefix)) continue;
    const parsed = readConfigFields(block.text);
    const entry = { ...parsed, configClass: prefix.match(/Model config (\S+)\s*$/)?.[1], line: lineAt(block.start), endLine: lineAt(block.start + block.text.length) };
    entry.diagnostics = parsed.issues.map(item => `第 ${lineAt(block.start + item.position)} 行 ${item.path}：${item.reason}`);
    if (parsed.rootKeys.has('train') && (parsed.rootKeys.has('model') || parsed.rootKeys.has('data'))) configs.push(entry);
    else if (/Model config \S+\s*$/.test(prefix)) modelConfigs.push(entry);
  }
  // Resolve each requested path independently, including duplicate/conflicting dumps.
  function pick(entries, path) {
    const parts = path.split('.'), values = [];
    let invalid = false;
    for (const entry of entries) {
      if (entry.issues.some(item => item.path === path || path.startsWith(item.path + '.'))) {
        // A parent may be incomplete while its child leaves are fully read.
        invalid ||= entry.issues.some(item => (item.path === path || path.startsWith(item.path + '.')) && /重复字段/.test(item.reason));
      }
      let value = entry.config;
      for (const part of parts) value = object(value)[part];
      if (value !== undefined && value !== null) values.push(value);
      else if (entry.issues.some(item => item.path === path)) invalid = true;
    }
    if (invalid || values.some(value => JSON.stringify(value) !== JSON.stringify(values[0]))) {
      issues.push(`${path} 无法确认或存在冲突，仅相关字段留空`);
      return undefined;
    }
    return values[0];
  }
  const set = (key, value, source) => {
    if (value !== undefined && value !== null && value !== '') { fields[key] = value; evidence[key] = source; }
  };
  function runtime(pattern, label) {
    const values = [...clean.matchAll(pattern)].map(match => Number(match[1]));
    if (values.some(value => value !== values[0])) { issues.push(`${label} 存在冲突，未自动填写`); return undefined; }
    return positive(values[0]) ? values[0] : undefined;
  }
  const get = path => pick(configs, path);
  // VeOmni releases place accelerator/checkpoint settings under different
  // owners. Resolve aliases per leaf, never switch the entire schema based
  // on one key or let a conflicting/broken alias hide behind a valid one.
  function compatible(paths, sources) {
    const found = []; let invalid = false;
    for (const path of paths) {
      const before = issues.length, value = get(path);
      invalid ||= issues.length > before;
      if (value !== undefined) found.push({ path, value });
    }
    if (invalid || found.some(item => JSON.stringify(item.value) !== JSON.stringify(found[0].value))) {
      issues.push(`${paths.join(' / ')} 无法确认或存在冲突，仅相关字段留空`);
      return undefined;
    }
    for (const item of found) sources.add(item.path);
    return found[0]?.value;
  }
  const modelPath = get('model.model_path');
  const detectedTypes = [...new Set([...clean.matchAll(/^[ \t]*Detected DiT model type:\s*(\w+)\./gm)].map(match => match[1]))];
  const dit = get('data.data_type') === 'diffusion' || detectedTypes.length > 0;
  const mainDumps = dit ? modelConfigs.filter(entry => detectedTypes.length === 1
    ? entry.config.model_type === detectedTypes[0] || entry.configClass === detectedTypes[0] + 'Config'
    : /DiTModelConfig$/.test(entry.configClass || '')) : modelConfigs;
  if (detectedTypes.length > 1) issues.push('DiT 主干模型类型存在冲突，层数需核对');
  let model = leaf(modelPath), modelSource = 'model.model_path';
  if (/^(?:transformer(?:_\d+)?|unet|dit|model)$/i.test(model)) {
    const names = typeof modelPath === 'string' ? modelPath.split('/').filter(name =>
      /^(?:Qwen|DeepSeek|Llama|Gemma|GLM|Mistral|MiniCPM|MiniMax|Wan|LTX)[-_.]?[a-z0-9]/i.test(name)) : [];
    if (names.length) { model = names.at(-1); modelSource += '（模型目录，忽略组件子目录）'; }
    else { model = ''; issues.push('model.model_path 指向组件子目录，无法确认模型名称，请手动填写'); }
  }
  set('模型', model, modelSource);
  set('参数', model.match(/(?:^|-)(\d+(?:\.\d+)?[BM])(?:-|$)/i)?.[1], '模型名称中的参数标识');
  set('系列', model.match(/^(Qwen|DeepSeek|Llama|Gemma|GLM|Mistral|MiniCPM|MiniMax|Wan|LTX)/i)?.[1], '模型名称中的系列标识');
  const seq = get('data.max_seq_len');
  if (dit) notes.push('DiT 序列取决于实际 latent / patch 形状；data.max_seq_len 不作为视频/音频序列长度，未记录实际长度时留空');
  if (!dit && positive(seq)) set('序列', seq % 1024 === 0 ? `${seq / 1024}k` : String(seq), 'data.max_seq_len');
  const world = runtime(/^[ \t]*Process rank:\s*\d+,\s*world size:\s*(\d+)/gm, 'world size');
  if (world) set('集群规格', `${world}P`, 'Process rank / world size 运行日志');
  const accSources = new Set();
  const accGet = key => compatible(['model.accelerator.' + key, 'train.accelerator.' + key], accSources);
  const fsdp = accGet('fsdp_config.fsdp_mode');
  set('训练后端', typeof fsdp === 'string' ? `VeOmni (${fsdp.toUpperCase()})` : 'VeOmni', [...accSources].join(' / ') || 'VeOmni 日志，未确认具体后端');
  const dpRep = accGet('dp_replicate_size'), dpShard = accGet('dp_shard_size');
  const dp = positive(dpRep) && positive(dpShard) ? dpRep * dpShard : undefined;
  const parallel = [];
  if (dp) parallel.push(`DP${dp}`);
  if (positive(dpRep)) parallel.push(`DP-Replica${dpRep}`);
  if (positive(dpShard)) parallel.push(`DP-Shard${dpShard}`);
  for (const [label, key] of [['TP', 'tp_size'], ['PP', 'pp_size'], ['CP', 'cp_size'], ['Ulysses SP', 'ulysses_size'], ['EP', 'ep_size']]) {
    const value = accGet(key);
    if (positive(value)) parallel.push(label + value);
  }
  set('并行策略', parallel.join(' '), [...accSources].join(' / ') + '；DP = dp_replicate_size × dp_shard_size；保留复制与分片维度');
  if (!parallel.length) issues.push('未找到有效的 model.accelerator 或 train.accelerator 并行配置，不能仅凭总卡数判断并行策略');
  const mbs = get('train.micro_batch_size'), gbs = get('train.global_batch_size');
  const gas = runtime(/^[ \t]*Set gradient accumulation to (\d+)\./gm, '梯度累积步数');
  set(BATCH, formatBatch([positive(mbs) ? mbs : '未记录', positive(gbs) ? gbs : '未记录', gas || '未记录']),
    'train.micro_batch_size / train.global_batch_size / Set gradient accumulation 运行日志；不从动态 batch 推算');
  const checkpointSources = new Set();
  const checkpointGet = key => compatible(['model.accelerator.gradient_checkpointing.', 'train.gradient_checkpointing.', 'train.accelerator.gradient_checkpointing.'].map(root => root + key), checkpointSources);
  const enabled = checkpointGet('enable'), layers = checkpointGet('recompute_layers'), selective = checkpointGet('selective_recompute_layers');
  const checkpointSource = [...checkpointSources].join(' / ');
  if (enabled === false || (enabled === true && layers === 0)) set('重计算', '未开重计算', checkpointSource);
  else if (enabled === true) {
    set('重计算', positive(selective) ? '选择性重计算' : layers === -1 && selective === 0 ? '全重计算' : positive(layers) ? `部分重计算（${layers}层）` : '已开重计算（范围待核实）',
      checkpointSource);
    if (layers === undefined || selective === undefined) notes.push('重计算配置未完整记录层范围；不采用其他版本的默认值补全，也不把开启开关直接判为全重计算');
    if (positive(selective)) notes.push(`VeOmni 重计算范围：${layers === -1 ? '全部层' : positive(layers) ? `末尾 ${layers} 层（不超过模型层数）` : '层范围未记录或无效'}；其中前 ${selective} 层使用选择性重计算，范围内其余层完整重算，数量受实际层数限制`);
    else if (positive(layers)) notes.push(`VeOmni 重计算配置为末尾 ${layers} 层；实际数量不超过模型层数，未据此推断覆盖全部层`);
  }
  const known = modelDefaults(model).source;
  if (dit && !mainDumps.length) issues.push('未找到可确认的 DiT 主干 Model config，层数不从条件模型推断');
  for (const [column, key, totalKey] of [[LAYERS, 'num_hidden_layers', 'layers'], [EXPERTS, 'num_experts', 'experts']]) {
    // DiT and condition encoders are separate models. Never borrow the
    // condition encoder's depth, token-refiner depth or vision tower depth.
    const textDumps = mainDumps.filter(entry => Object.hasOwn(entry.config, 'text_config'));
    const path = dit && column === LAYERS ? 'num_layers' : textDumps.length ? `text_config.${key}` : key;
    const count = dit && detectedTypes.length > 1 ? undefined : pick(textDumps.length ? textDumps : mainDumps, path);
    if (positive(count)) {
      const total = known?.[totalKey];
      set(column, total && total !== count ? `${count}/${total}` : count, `Model config ${path}`);
      if (total && total !== count) set(COMPLETENESS, '减层/减配', '日志模型配置与官方总数不同');
    } else if (issues.some(issue => issue.startsWith(path + ' '))) fields[column] = '';
  }
  notes.push(...configs.flatMap(entry => entry.diagnostics), ...modelConfigs.flatMap(entry => entry.diagnostics));
  const ignoredConfig = ignored.filter(line => configs.some(c => line >= c.line && line <= c.endLine));
  if (ignoredConfig.length) notes.push(`已忽略配置间交错的日志输出（原日志行 ${ignoredConfig.slice(0, 10).join('、')}）`);
  if (!configs.length) issues.push('未找到 VeOmni 的 model / data / train 运行配置，未记录的字段留空');
  if (!model) issues.push('model.model_path 缺失或无效，模型名称需核对');
  if (get('train.dyn_bsz') === true) notes.push('动态 batch；MBS / GBS 为日志配置值，GAS 仅采用明确打印的值');
  notes.push(...issues);
  return { fields, evidence, notes, warning: notes.filter(note => /第 \d+ 行|冲突|无法确认|未找到|缺失或无效/.test(note)).slice(0, 3).join('；') };
}

// Native field names from VeOmni EnvironMeter. Only explicitly step-labelled
// metric records are handled here; progress time estimates are a separate fallback.
export function veomniMetricRecords(text) {
  const clean = prepareConfigLog(text).clean, records = [], notes = [];
  for (const block of dictionaryBlocks(clean)) {
    if (!/['"](?:tokens_per_second\(M\)|mfu|step_time_s|training\/(?:total_loss|loss)|consume_tokens\(M\))['"]\s*:/.test(block.text)) continue;
    const parsed = readConfigFields(block.text), c = parsed.config;
    const stepKeys = ['global_step', 'step', '_step'].filter(key => Object.hasOwn(c, key));
    const line = clean.slice(0, block.start).split('\n').length;
    if (!stepKeys.length) { notes.push(`第 ${line} 行指标没有明确步号，未计入均值`); continue; }
    const step = c[stepKeys[0]];
    if (!Number.isSafeInteger(step) || step < 0 || stepKeys.some(key => c[key] !== step)) throw new Error(`第 ${line} 行 VeOmni 指标步号无效或冲突`);
    const values = {};
    for (const [input, output, scale] of [['tokens_per_second(M)', 'tps', 1e6], ['mfu', 'mfu', 1], ['step_time_s', 'step_time_s', 1]]) {
      if (!parsed.rootKeys.has(input)) continue;
      values[output] = c[input] === null ? null : typeof c[input] === 'number' ? c[input] * scale : NaN;
    }
    if (c['flops_promised(T)'] === 0 && 'mfu' in values) {
      values.mfu = NaN; notes.push(`第 ${line} 行 flops_promised(T) 为 0，MFU 不可用`);
    }
    records.push({ step, values, line });
    notes.push(...parsed.issues.map(item => `第 ${line} 行指标 ${item.path}：${item.reason}`));
  }
  return { records, notes };
}
