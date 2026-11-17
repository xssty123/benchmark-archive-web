import { cleanConfigLog, dictionaryBlocks, readConfigFields, readLiteral } from './config.js';
import { BATCH, LAYERS, EXPERTS, COMPLETENESS, formatBatch, modelDefaults } from './record-fields.js';

export const SLIME_FIELDS = {
  '训练MFU': 'unsupported_mfu', '吞吐TPS': 'perf/actor_train_tok_per_s',
  '训练耗时 actor_update': 'perf/actor_train_time', '推理耗时': 'perf/rollout_time',
  'resharding耗时': 'perf/update_weights_time', '单步耗时s': 'perf/step_time',
};
const numericKeys = new Set(('actor_num_nodes actor_num_gpus_per_node rollout_num_gpus rollout_num_gpus_per_engine ' +
  'critic_num_nodes critic_num_gpus_per_node rollout_max_prompt_len rollout_max_response_len rollout_max_context_len ' +
  'num_layers num_experts micro_batch_size global_batch_size gradient_accumulation_steps rollout_batch_size n_samples_per_prompt ' +
  'tensor_model_parallel_size pipeline_model_parallel_size context_parallel_size expert_model_parallel_size ' +
  'expert_tensor_parallel_size virtual_pipeline_model_parallel_size data_parallel_size recompute_num_layers ' +
  'sglang_tp_size sglang_pp_size sglang_dp_size sglang_ep_size sglang_tensor_parallel_size sglang_pipeline_parallel_size ' +
  'sglang_data_parallel_size sglang_expert_parallel_size').split(' '));
const boolKeys = new Set(['colocate', 'use_critic', 'use_dynamic_batch_size']);
const stringKeys = new Set(['hf_checkpoint', 'train_backend', 'rollout_function_path', 'recompute_granularity', 'recompute_method']);
const configKeys = new Set([...numericKeys, ...boolKeys, ...stringKeys]);
const positive = value => Number.isSafeInteger(value) && value > 0;
const clean = text => cleanConfigLog(text).replace(/^(?:[\w./-]+\.py:\d+)[ \t]*-[ \t]*/gm, '');

export function isSlimeLog(text) {
  return (/\bperf\s+\d+\s*:\s*\{/.test(text) && /['"]perf\/(?:actor_train_|rollout_time|train_wait_time|step_time|update_weights_time)/.test(text)) ||
    (/\b(?:actor_num_nodes|actor_num_gpus_per_node)\b/.test(text) && /\b(?:hf_checkpoint|rollout_batch_size)\b/.test(text)) ||
    /(?:slime[./](?:rollout|backends|ray|observability)|\bslime RL\b)/i.test(text);
}

// Convert Namespace keys only outside strings. Literal values still go through
// the same safe, field-independent reader as dictionaries; nothing is evaluated.
function namespaceDict(body) {
  let out = '{', quote = '', escaped = false, depth = 0;
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (quote) {
      out += c;
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === quote || c === '\n') quote = '';
      continue;
    }
    if (c === '"' || c === "'") { quote = c; out += c; continue; }
    if (!depth && (i === 0 || /[\s,]/.test(body[i - 1]))) {
      const key = body.slice(i).match(/^([A-Za-z_]\w*)\s*=/);
      if (key) { out += JSON.stringify(key[1]) + ':'; i += key[0].length - 1; continue; }
    }
    if ('([{'.includes(c)) depth++;
    if (')]}'.includes(c)) depth--;
    out += c;
  }
  return out + '}';
}

export function extractSlimeConfig(text) {
  const input = clean(text), entries = new Map(), issues = [], fields = {}, evidence = {}, notes = [];
  const add = (key, value, line, invalid = false) => {
    if (!configKeys.has(key)) return;
    const valid = value === null || (numericKeys.has(key) ? Number.isSafeInteger(value) && value >= 0 :
      boolKeys.has(key) ? typeof value === 'boolean' : typeof value === 'string');
    const list = entries.get(key) || [];
    list.push({ value, line, invalid: invalid || !valid }); entries.set(key, list);
  };
  const read = (body, line) => {
    const result = readConfigFields(body);
    for (const key of result.rootKeys) add(key, result.config[key], line, result.issues.some(issue => issue.path === key));
    for (const issue of result.issues) if (configKeys.has(issue.path)) issues.push(`第 ${line} 行 ${issue.path}：${issue.reason}`);
  };
  input.split('\n').forEach((line, i) => {
    const dotted = line.match(/^\s*([A-Za-z_]\w*)\s+\.{2,}\s+(.*?)\s*$/);
    if (dotted && configKeys.has(dotted[1])) {
      const [ , key, raw ] = dotted;
      try {
        // Megatron prints plain string paths (not repr); numbers/bools remain strict.
        const parsed = stringKeys.has(key) && !/^(?:None|null|['"])/.test(raw)
          ? { value: raw, end: raw.length } : readLiteral(raw);
        add(key, parsed.value, i + 1, parsed.end !== raw.length);
      } catch { add(key, undefined, i + 1, true); }
    }
    const ns = line.match(/(?:^|\bargs\s*[:=]\s*)Namespace\((.*)\)\s*$/);
    if (ns) read(namespaceDict(ns[1]), i + 1);
  });
  for (const block of dictionaryBlocks(input)) {
    const result = readConfigFields(block.text);
    if (!['hf_checkpoint', 'rollout_batch_size', 'actor_num_nodes'].some(key => result.rootKeys.has(key))) continue;
    read(block.text, input.slice(0, block.start).split('\n').length);
  }
  const invalidKeys = new Set();
  const get = key => {
    const list = entries.get(key) || [];
    if (list.some(entry => entry.invalid) || new Set(list.map(entry => JSON.stringify(entry.value))).size > 1) {
      if (!invalidKeys.has(key)) issues.push(`${key} 无效或存在冲突（原日志行 ${list.map(entry => entry.line).join('、')}），仅该字段留空`);
      invalidKeys.add(key); return undefined;
    }
    return list[0]?.value;
  };
  // Validate even fields whose dependent result cannot be computed.
  for (const key of entries.keys()) get(key);
  const set = (key, value, source) => { if (value !== undefined && value !== null && value !== '') { fields[key] = value; evidence[key] = source; } };
  const modelPath = get('hf_checkpoint');
  const model = typeof modelPath === 'string' ? modelPath.replace(/\/+$/, '').split('/').at(-1) : '';
  set('模型', model, 'slime hf_checkpoint');
  set('系列', model.match(/^(Qwen|DeepSeek|Llama|GLM|MiniMax|MiniCPM|Gemma|Mistral)/i)?.[1], '模型名称');
  set('参数', model.match(/(?:^|-)(\d+(?:\.\d+)?[BM])(?:-|$)/i)?.[1], '模型名称中的参数标识');
  const known = modelDefaults(model).source;
  for (const [column, key, totalKey] of [[LAYERS, 'num_layers', 'layers'], [EXPERTS, 'num_experts', 'experts']]) {
    const count = get(key), total = known?.[totalKey];
    if (positive(count)) {
      set(column, total && count !== total ? `${count}/${total}` : count, `slime ${key}`);
      if (total && count !== total) set(COMPLETENESS, '减层/减配', '运行配置与已知模型总数不同');
    } else if (invalidKeys.has(key)) fields[column] = '';
  }
  const rollout = get('rollout_function_path');
  if (typeof rollout === 'string') {
    if (/fully_async/.test(rollout)) set('同步/异步', '异步', 'rollout_function_path fully_async');
    else if (rollout === 'slime.rollout.sglang_rollout.generate_rollout') notes.push('标准 SGLang 生成函数本身不证明训练调度为同步；未记录明确调度方式时同步/异步留空');
  }
  const train = get('train_backend');
  set('训练后端', train === 'megatron' ? 'Megatron Core' : train, 'slime train_backend');
  if (/sglang/.test(rollout || '') || [...entries.keys()].some(key => key.startsWith('sglang_'))) set('推理后端', 'SGLang', 'slime rollout_function_path / sglang 参数');
  const k = n => n % 1024 === 0 ? `${n / 1024}k` : String(n);
  const prompt = get('rollout_max_prompt_len'), response = get('rollout_max_response_len');
  if (positive(prompt) && positive(response)) set('序列', `${k(prompt)}→${k(response)}`, 'slime rollout_max_prompt_len → rollout_max_response_len（上限）');
  const actorNodes = get('actor_num_nodes'), actorGpus = get('actor_num_gpus_per_node');
  const actor = positive(actorNodes) && positive(actorGpus) ? actorNodes * actorGpus : null;
  const rolloutGpus = get('rollout_num_gpus'), colocate = get('colocate'), critic = get('use_critic');
  if (actor) notes.push(`actor 训练设备数：${actor}；集群规格仅在设备共享及分离信息明确时填写`);
  if (actor && colocate === true && critic !== true) set('集群规格', `${actor}P`, 'actor_num_nodes × actor_num_gpus_per_node；colocate=true，共享设备不重复计数');
  if (actor && colocate === false && Number.isSafeInteger(rolloutGpus)) {
    const cn = get('critic_num_nodes'), cg = get('critic_num_gpus_per_node');
    if (critic === false || (critic === true && positive(cn) && positive(cg))) set('集群规格', `${actor + rolloutGpus + (critic ? cn * cg : 0)}P`, '分离部署：actor + rollout + 明确启用的 critic 设备数');
  }
  const parallel = [['TP','tensor_model_parallel_size'],['PP','pipeline_model_parallel_size'],['CP','context_parallel_size'],['EP','expert_model_parallel_size'],['ETP','expert_tensor_parallel_size'],['VPP','virtual_pipeline_model_parallel_size'],['DP','data_parallel_size']];
  set('训练并行策略', parallel.filter(([, key]) => positive(get(key))).map(([name, key]) => name + get(key)).join(' '), 'slime / Megatron 并行参数');
  const aliases = [['TP','sglang_tp_size','sglang_tensor_parallel_size'],['PP','sglang_pp_size','sglang_pipeline_parallel_size'],['DP','sglang_dp_size','sglang_data_parallel_size'],['EP','sglang_ep_size','sglang_expert_parallel_size']];
  set('推理并行策略', aliases.map(([label, current, legacy]) => {
    const a = get(current), b = get(legacy);
    if (invalidKeys.has(current) || invalidKeys.has(legacy)) return '';
    if (a != null && b != null && a !== b) { issues.push(`${current} 与 ${legacy} 冲突，${label} 留空`); return ''; }
    return positive(a ?? b) ? label + (a ?? b) : '';
  }).filter(Boolean).join(' '), 'slime sglang 并行参数（不将每引擎设备数直接视为 TP）');
  const mbs = get('micro_batch_size'), gbs = get('global_batch_size'), gas = get('gradient_accumulation_steps');
  set(BATCH, formatBatch([positive(mbs) ? mbs : '未记录', positive(gbs) ? gbs : '未记录', positive(gas) ? gas : get('use_dynamic_batch_size') === true ? '动态（未推算）' : '未记录']), 'slime micro_batch_size / global_batch_size；GAS 仅采用显式值');
  const prompts = get('rollout_batch_size'), n = get('n_samples_per_prompt');
  if (positive(prompts) && positive(n)) set('train_batch_size × rollout_n', `${prompts} × ${n} = ${prompts * n}`, 'slime rollout_batch_size × n_samples_per_prompt（生成样本数，不等同训练 GBS）');
  const granularity = get('recompute_granularity'), method = get('recompute_method'), layers = get('recompute_num_layers');
  if (granularity === 'selective') set('重计算', '选择性重计算', 'recompute_granularity');
  else if (granularity === null) set('重计算', '未开重计算', '显式 recompute_granularity=None');
  else if (granularity === 'full') set('重计算', method === 'uniform' ? '全重计算' : method === 'block' && positive(layers) ? `部分重计算（每流水线阶段 ${layers} 层）` : '已开重计算（范围待核实）', 'recompute_granularity / recompute_method / recompute_num_layers');
  if (!model) issues.push('未找到有效 hf_checkpoint，模型需手动核对');
  if (!entries.size) issues.push('未找到受支持的 slime 参数表、Namespace 或配置字典；配置字段留空');
  notes.push('框架：slime；配置按独立字段识别；没有明确值的配置不采用框架默认值', ...issues);
  return { fields, evidence, notes, warning: issues.slice(0, 3).join('；') };
}

export function slimeRecords(text) {
  const input = clean(text), records = [], notes = [];
  const markers = [...input.matchAll(/^\s*perf\s+(\d+)\s*:\s*(?=\{)/gm)];
  for (let i = 0; i < markers.length; i++) {
    const marker = markers[i], start = marker.index + marker[0].length;
    const segment = input.slice(start, markers[i + 1]?.index);
    const block = dictionaryBlocks(segment).next().value;
    if (!block || block.start !== 0) continue;
    const parsed = readConfigFields(block.text), line = input.slice(0, marker.index).split('\n').length;
    const values = {};
    for (const key of Object.values(SLIME_FIELDS).filter(key => key !== 'unsupported_mfu')) {
      if (parsed.rootKeys.has(key)) values[key] = parsed.config[key] === null ? null : typeof parsed.config[key] === 'number' ? parsed.config[key] : NaN;
    }
    if (!Object.hasOwn(values, 'perf/step_time') && ['perf/train_wait_time', 'perf/train_time'].every(key => typeof parsed.config[key] === 'number' && Number.isFinite(parsed.config[key]) && parsed.config[key] >= 0)) {
      values['perf/step_time'] = parsed.config['perf/train_wait_time'] + parsed.config['perf/train_time'];
      notes.push('单步耗时按 slime 定义取 perf/train_wait_time + perf/train_time');
    }
    if (Object.keys(values).length) records.push({ step: Number(marker[1]), values, line });
    for (const issue of parsed.issues) notes.push(`第 ${line} 行指标 ${issue.path}：${issue.reason}`);
  }
  notes.push('统计单位为日志 perf 后的 rollout_id（保留从 0 开始的编号）；同一编号的训练、rollout 指标合并，默认包含全部已记录编号',
    'TPS 采用 perf/actor_train_tok_per_s：整个 rollout batch 的 token 数 / actor_train_time，属于训练阶段全局 tokens/s，不是 verl 的端到端单设备吞吐；不混用生成阶段 tokens_per_gpu_per_sec',
    '单步耗时采用 perf/step_time（train_wait_time + train_time），按 slime 计时范围记录，不把可重叠的阶段耗时相加',
    'resharding 采用 perf/update_weights_time 的已记录计时值；Timer 在 perf 输出时重置，权重同步可能属于前一轮更新，不自动移动步号',
    '当前适配的 slime 日志没有直接记录训练 MFU，留空；actor_train_tflops 不是 MFU，不据此换算');
  return { records, notes: [...new Set(notes)] };
}
