import { BATCH, COMPLETENESS, LAYERS, EXPERTS, modelDefaults, formatBatch } from './record-fields.js';
import { normalizeRecompute } from './recompute.js';
// Read Python pprint data literals only. Never evaluate executable log content.
export function readLiteral(text) {
  let i = 0;
  const fail = message => { const error = new Error(message); error.position = i; throw error; };
  const ws = () => { while (/\s/.test(text[i] || '') && i < text.length) i++; };
  function string() {
    const quote = text[i++]; let out = '';
    while (i < text.length) {
      let c = text[i++];
      if (c === quote) return out;
      if (c === '\\') {
        c = text[i++];
        if (['u', 'x'].includes(c)) {
          const n = c === 'u' ? 4 : 2, hex = text.slice(i, i + n);
          if (!new RegExp(`^[0-9a-f]{${n}}$`, 'i').test(hex)) fail('Invalid escape');
          out += String.fromCharCode(parseInt(hex, 16)); i += n;
        } else out += ({ n: '\n', r: '\r', t: '\t', '\\': '\\', "'": "'", '"': '"' })[c] ?? ('\\' + c);
      } else out += c;
    }
    fail('Unclosed string');
  }
  function value(depth = 0) {
    if (depth > 100) fail('Config too deep');
    ws();
    if (i >= text.length) fail('Incomplete literal');
    const c = text[i];
    if (c === "'" || c === '"') {
      let out = string(); ws();
      while (text[i] === "'" || text[i] === '"') { out += string(); ws(); }
      return out;
    }
    if (c === '{' || c === '[' || c === '(') {
      i++; const dict = c === '{', close = dict ? '}' : c === '[' ? ']' : ')';
      const out = dict ? Object.create(null) : [];
      try {
        ws();
        while (text[i] !== close) {
          const item = value(depth + 1); ws();
          let next;
          if (dict) {
            if (typeof item !== 'string' || text[i++] !== ':') fail('Invalid dict');
            if (Object.hasOwn(out, item)) fail('Duplicate config key');
            try { next = value(depth + 1); }
            catch (error) {
              if (error.partial && !error.fatal) out[item] = error.partial;
              throw error;
            }
          }
          ws();
          if (i >= text.length) fail('Incomplete literal');
          if (text[i] !== ',' && text[i] !== close) fail('Invalid separator');
          // Commit a field only after its value and delimiter have been validated.
          if (dict) out[item] = next; else out.push(item);
          if (text[i] === ',') { i++; ws(); }
        }
        i++; return out;
      } catch (error) {
        error.fatal = error.fatal || error.message === 'Duplicate config key';
        if (dict && !error.fatal) error.partial = out;
        else delete error.partial;
        throw error;
      }
    }
    const token = text.slice(i).match(/^(?:True|False|None|true|false|null|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)/i)?.[0];
    if (!token) fail('Unsupported literal');
    i += token.length;
    const literals = { True: true, False: false, None: null, true: true, false: false, null: null };
    return Object.hasOwn(literals, token) ? literals[token] : Number(token);
  }
  const result = value();
  return { value: result, end: i };
}

// Strip only recognized logger prefixes at the start of each physical line.
export function cleanConfigLog(text) {
  return text.replace(/(?:\x1b|\\x1b|\\u001b)\[[0-?]*[ -/]*[@-~]/gi, '').split(/\r\n|\n|\r/).map(line => {
    line = line.replace(/^\uFEFF/, '');
    const prefixes = [
      /^[ \t]*\[(?:\d{1,3};)*\d{1,3}m/, // A pasted ANSI prefix may lose its ESC byte.
      /^[ \t]*\([^\n)]*\bpid=\d+[^)]*\)[ \t]?/,
      /^[ \t]*\[rank[ \t]*\d+\][ \t]*:?[ \t]*/i,
      /^[ \t]*(?:\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)[ \t]*(?:[-|:][ \t]*)?/,
      /^[ \t]*(?:\[(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\]|(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b)[ \t]*(?:[-:|][ \t]*)?/,
    ];
    for (;;) {
      const prefix = prefixes.find(pattern => pattern.test(line));
      if (!prefix) return line;
      line = line.replace(prefix, '');
    }
  }).join('\n');
}

// Ray stdout/stderr records may share a physical line. Split only outside strings.
function raySegments(line) {
  const starts = [0]; let quote = '', escaped = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === quote) quote = '';
    } else if (c === '"' || c === "'") quote = c;
    else if (c === '(') {
      const marker = line.slice(i).match(/^\([^\n)]*\bpid=\d+[^)]*\)/);
      if (marker) { if (i) starts.push(i); i += marker[0].length - 1; }
    }
  }
  return starts.map((start, index) => line.slice(start, starts[index + 1]));
}

function prepareConfigLog(text) {
  const ignored = new Set(); let depth = 0, quote = '', escaped = false, owner = '', warningSource = false;
  const lines = text.replace(/(?:\x1b|\\x1b|\\u001b)\[[0-?]*[ -/]*[@-~]/gi, '').split(/\r\n|\n|\r/);
  const clean = lines.map((line, lineIndex) => raySegments(line).map(segment => {
    const body = cleanConfigLog(segment), trimmed = body.trim();
    const origin = segment.match(/\([^\n)]*\bpid=\d+[^)]*\)/)?.[0] || segment.match(/\[rank\s*\d+\]/i)?.[0] || '';
    const dataLine = /^(?:[{'"\[\]})]|[-+]?\d|True\b|False\b|None\b|true\b|false\b|null\b)/.test(trimmed);
    const logLevel = /(?:^|\s|\))(?:\[?(?:INFO|DEBUG|WARNING|WARN|ERROR|CRITICAL)\]?)(?=[:\s])/.test(segment);
    const warning = /^(?:.*\.py:\d+:\s*(?:\w*Warning|\w*Error):)/.test(trimmed);
    const source = warningSource && /^(?:self\.\w+\s*=|if\s+.+:\s*$|warnings\.warn\()/.test(trimmed);
    if ((!dataLine && warning) || source || (logLevel && !dataLine)) {
      if (depth) ignored.add(lineIndex + 1);
      warningSource = warning || source;
      return '';
    }
    if (trimmed) warningSource = false;
    // Never join a second worker's literal-looking output into the active config.
    if (depth && owner && origin && origin !== owner && dataLine) return '__OTHER_WORKER_OUTPUT__';
    if (!depth && trimmed.startsWith('{')) owner = origin;
    for (const c of body) {
      if (quote) {
        if (escaped) escaped = false;
        else if (c === '\\') escaped = true;
        else if (c === quote) quote = '';
      } else if (depth && (c === '"' || c === "'")) quote = c;
      else if (c === '{') depth++;
      else if (c === '}' && depth) depth--;
    }
    if (!depth) owner = '';
    // pprint/JSON strings cannot contain an unescaped physical newline.
    quote = ''; escaped = false;
    return body;
  }).join(' ')).join('\n');
  return { clean, ignored: [...ignored] };
}

// Balanced outer dictionaries, respecting strings, in a single pass over the log.
function* dictionaryBlocks(text) {
  let start = -1, depth = 0, quote = '', escaped = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (start < 0) {
      if (c === '{') {
        let next = i + 1;
        while (/\s/.test(text[next] || '') && next < text.length) next++;
        if (text[next] === '"' || text[next] === "'" || text.slice(i + 1, next).includes('\n')) { start = i; depth = 1; }
      }
      continue;
    }
    if (quote) {
      if (c === '\n' || c === '\r') { quote = ''; escaped = false; }
      else if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === quote) quote = '';
    } else if (c === '"' || c === "'") quote = c;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) {
      yield { start, text: text.slice(start, i + 1), complete: true };
      start = -1;
    }
  }
  if (start >= 0) yield { start, text: text.slice(start), complete: false };
}

function sameConfig(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && sameConfig(a[key], b[key]));
}

// Walk each dictionary entry independently. A bad value is never a reason to
// abandon later entries; only structural boundaries, not keyword order, matter.
function readConfigFields(text) {
  let i = 0;
  const issues = [], rootKeys = new Set();
  const ws = () => { while (i < text.length && /\s/.test(text[i])) i++; };
  const issue = (path, position, reason) => issues.push({ path: path.join('.') || '配置', position, reason });
  function quoted(start) {
    const quote = text[start]; let escaped = false;
    for (let p = start + 1; p < text.length; p++) {
      const c = text[p];
      if (c === '\n' || c === '\r') return { end: p, closed: false };
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === quote) return { end: p + 1, closed: true };
    }
    return { end: text.length, closed: false };
  }
  function keyAt(start) {
    if (text[start] !== '"' && text[start] !== "'") return null;
    const q = quoted(start); if (!q.closed) return null;
    let p = q.end;
    while (p < text.length && /\s/.test(text[p])) p++;
    if (text[p] !== ':') return null;
    try { return { key: readLiteral(text.slice(start, q.end)).value, end: p + 1 }; }
    catch { return null; }
  }
  const columnAt = p => p - text.lastIndexOf('\n', p - 1) - 1;
  // Locate this value's delimiter without interpreting Python expressions.
  // Quoted strings and nested lists/calls hide their internal commas and keys.
  function valueBoundary(start, column, recover) {
    const stack = []; let bad = '';
    for (let p = start; p < text.length; p++) {
      const c = text[p];
      if (c === '\n' || c === '\r') {
        let next = p + 1; while (next < text.length && /\s/.test(text[next])) next++;
        if (recover && (stack.length || bad) && keyAt(next) && columnAt(next) <= column) {
          return { end: next, reason: bad || '值的括号未闭合' };
        }
      }
      if (c === '"' || c === "'") {
        if (!stack.length && p > start && keyAt(p)) return { end: p, reason: bad, missingComma: true };
        const q = quoted(p);
        if (!q.closed) { bad = '字符串未闭合'; p = q.end - 1; }
        else p = q.end - 1;
      } else if ('[{('.includes(c)) stack.push(({ '[': ']', '{': '}', '(': ')' })[c]);
      else if (']})'.includes(c)) {
        if (!stack.length || stack.at(-1) !== c) return { end: p, reason: bad || (stack.length ? '值的括号不匹配' : '') };
        stack.pop();
      } else if (c === ',' && !stack.length) return { end: p, reason: bad };
    }
    return { end: text.length, reason: bad || (stack.length ? '值的括号未闭合' : '') };
  }
  function boundary(start, column, allowRecovery = true) {
    const stop = valueBoundary(start, column, false);
    // Indentation is only a recovery hint for a broken value, never a rule for
    // valid nested values whose formatting may use any amount of whitespace.
    return stop.reason && allowRecovery ? valueBoundary(start, column, true) : stop;
  }
  function object(path, depth) {
    const out = Object.create(null), seen = new Set();
    i++; // opening brace
    while (i < text.length) {
      ws(); if (i >= text.length) break;
      if (text[i] === '}') { i++; return out; }
      if (text[i] === ',') { i++; continue; }
      const start = i, entry = keyAt(i);
      if (!entry) {
        issue(path, i, '包含暂不支持的数据类型或日志前缀，无法确认字段归属');
        // Skip the whole unknown fragment, including any nested data. Never
        // promote keys from an unrecognized wrapper into the current object.
        const end = boundary(i, columnAt(i), false).end;
        i = Math.max(i + 1, end);
        continue;
      }
      if (!path.length) rootKeys.add(entry.key);
      i = entry.end; ws();
      const valueStart = i, fieldPath = [...path, entry.key];
      let value, valid = false;
      if (text[i] === '{' && depth < 100) {
        value = object(fieldPath, depth + 1); valid = true;
        ws();
        if (i < text.length && ![',', '}'].includes(text[i]) && !keyAt(i)) {
          const stop = boundary(i, columnAt(start));
          issue(fieldPath, valueStart, '字典后包含不支持的表达式');
          i = stop.end; valid = false;
        }
      } else {
        const stop = boundary(i, columnAt(start));
        const raw = text.slice(i, stop.end).trim();
        i = stop.end;
        try {
          if (stop.reason) throw new Error(stop.reason);
          const parsed = readLiteral(raw);
          if (parsed.end !== raw.length) throw new Error('值包含不支持的表达式');
          value = parsed.value; valid = true;
        } catch (error) {
          const reasons = { 'Unsupported literal': '包含暂不支持的数据类型或表达式', 'Invalid escape': '字符串转义无效',
            'Unclosed string': '字符串未闭合', 'Incomplete literal': '值缺失或未闭合', 'Config too deep': '嵌套层级超过支持范围',
            'Invalid separator': '值包含不支持的表达式', 'Invalid dict': '值的字典格式无效' };
          issue(fieldPath, valueStart, reasons[error.message] || error.message);
        }
      }
      if (seen.has(entry.key)) {
        delete out[entry.key];
        issue(fieldPath, start, '包含重复字段，该字段留空');
      } else if (valid) out[entry.key] = value;
      seen.add(entry.key);
      ws();
      if (keyAt(i)) issue(fieldPath, i, '字段间缺少逗号，已按明确的键值边界继续识别');
    }
    issue(path, i, '配置未闭合或输出中断');
    return out;
  }
  return { config: object([], 0), issues, rootKeys };
}

export function extractConfig(text) {
  const { clean, ignored } = prepareConfigLog(text), configs = [], problems = [];
  const lineAt = position => clean.slice(0, position).split('\n').length;
  const empty = warning => ({ fields: {}, evidence: {}, notes: [], warning });
  for (const block of dictionaryBlocks(clean)) {
    if (!/['"]actor_rollout_ref['"]\s*:/.test(block.text)) continue;
    const where = `第 ${lineAt(block.start)} 行开始的配置`;
    const parsed = readConfigFields(block.text), c = parsed.config;
    const diagnostics = parsed.issues.map(item => `第 ${lineAt(block.start + item.position)} 行 ${item.path}：${item.reason}`);
    const isObject = value => value && typeof value === 'object' && !Array.isArray(value);
    if (!parsed.rootKeys.has('actor_rollout_ref')) {
      problems.push(`${where}的 actor_rollout_ref 不在配置顶层或无法确认所属层级${diagnostics.length ? '；' + diagnostics.slice(0, 3).join('；') : ''}`);
      continue;
    }
    for (const key of ['actor_rollout_ref', 'trainer']) {
      if (!isObject(c[key])) {
        diagnostics.push(`${where}缺少有效的 ${key} 配置对象，仅相关字段留空`);
        delete c[key];
      }
    }
    configs.push({ config: c, line: lineAt(block.start), endLine: lineAt(block.start + block.text.length), diagnostics });
  }
  if (problems.length) return empty(problems.slice(0, 3).join('；') + '。配置字段未自动填入，请核对或手动填写');
  if (!configs.length) {
    if (/\bactor_rollout_ref(?:['"]?\s*:|\.)/.test(clean)) return empty('发现 actor_rollout_ref 配置线索，但未找到完整的字典配置块；YAML、启动参数或被截取的配置片段暂不支持，请提供完整配置打印或手动填写');
    return empty('未找到运行配置：日志中没有识别到包含 actor_rollout_ref 和 trainer 的配置块，请确认日志包含启动时的完整配置打印，或手动填写');
  }
  if ((configs.length > 1 && configs.some(c => c.diagnostics.length)) || configs.some(c => !sameConfig(c.config, configs[0].config))) {
    return empty(`发现多份不同运行配置（起始行 ${configs.map(c => c.line).join('、')}），配置字段未自动填入，请先拆分日志`);
  }
  const result = mapConfig(configs[0].config, text);
  const diagnostics = configs[0].diagnostics;
  if (diagnostics.length) {
    result.warning = diagnostics.slice(0, 3).join('；') + (diagnostics.length > 3 ? `；另有 ${diagnostics.length - 3} 项，见解析说明` : '') + '。已独立提取其余字段，请核对提示项';
    result.notes.push(...diagnostics);
  }
  const configIgnored = ignored.filter(line => configs.some(c => line >= c.line && line <= c.endLine));
  if (configIgnored.length) result.notes.push(`已排除配置中交错的警告或 INFO 输出（原日志行 ${configIgnored.slice(0, 12).join('、')}${configIgnored.length > 12 ? '等' : ''}）`);
  if (!result.fields['模型']) result.notes.push('已解析配置，但 actor_rollout_ref.model.path 缺失或不是有效字符串，模型名称需手动填写');
  return result;
}

export function mapConfig(c, text = '') {
  const ar = c.actor_rollout_ref || {}, a = ar.actor || {}, m = ar.model || {}, r = ar.rollout || {};
  const t = c.trainer || {}, d = c.data || {}, fields = {}, evidence = {}, notes = [];
  const set = (key, value, source) => {
    if (value !== undefined && value !== null && value !== '') { fields[key] = value; evidence[key] = source; }
  };
  const positive = n => Number.isInteger(n) && n > 0;
  const size = n => n % 1024 === 0 ? `${n / 1024}k` : String(n);
  const model = typeof m.path === 'string' ? m.path.replace(/\/$/, '').split('/').at(-1) : '';
  set('模型', model, 'actor_rollout_ref.model.path');
  const known = modelDefaults(model, m.model_stage).source;
  const override = m.override_config || {};
  const textOverride = override.thinker_config?.text_config || override.text_config || override;
  for (const [column, key, totalKey] of [[LAYERS, 'num_hidden_layers', 'layers'], [EXPERTS, 'num_experts', 'experts']]) {
    if (!positive(textOverride[key])) continue;
    const total = known?.[totalKey], actual = textOverride[key];
    set(column, total && actual !== total ? `${actual}/${total}` : actual, 'actor_rollout_ref.model.override_config 文本模型配置');
    if (total && actual !== total) set(COMPLETENESS, '减层/减配', '日志模型配置与官方总数不同');
  }
  set('参数', model.match(/(?:^|-)(\d+(?:\.\d+)?[BM])(?:-|$)/i)?.[1], '模型名称中的参数标识');
  set('系列', model.match(/^(Qwen|DeepSeek|Llama|Gemma|GLM|Mistral)/i)?.[1], '模型名称中的系列标识');
  const mode = t.v1?.trainer_mode;
  if (typeof mode === 'string') set('同步/异步', /(^|_)async($|_)/.test(mode) ? '异步' : /(^|_)sync($|_)/.test(mode) ? '同步' : '', 'trainer.v1.trainer_mode（不采用 rollout.mode）');
  if (positive(d.max_prompt_length) && positive(d.max_response_length)) set('序列', `${size(d.max_prompt_length)}→${size(d.max_response_length)}`, 'data.max_prompt_length / max_response_length');
  const devices = positive(t.nnodes) && positive(t.n_gpus_per_node) ? t.nnodes * t.n_gpus_per_node : null;
  if (devices) set('集群规格', `${devices}P`, 'trainer.nnodes × n_gpus_per_node');
  if (positive(d.train_batch_size) && positive(r.n)) set('train_batch_size × rollout_n', `${d.train_batch_size} × ${r.n} = ${d.train_batch_size * r.n}`, 'data.train_batch_size × actor_rollout_ref.rollout.n');
  const mbs = a.ppo_micro_batch_size_per_gpu;
  // This mapping follows the verified verl PPO sequence minibatch contract, not arbitrary SFT/OPD adapters.
  const ppo = /(?:ActorConfig)$/.test(a._target_ || '') && !c.algorithm?.paired_preference;
  const gbs = ppo && positive(a.ppo_mini_batch_size) && positive(r.n) ? a.ppo_mini_batch_size * r.n : null;
  const gas = a.gradient_accumulation_steps;
  set(BATCH, formatBatch([positive(mbs) ? mbs : Object.hasOwn(a, 'ppo_micro_batch_size_per_gpu') ? '未配置' : '未记录',
    gbs || '未记录', positive(gas) ? gas : a.use_dynamic_bsz === true ? '动态（未推算）' : '未记录']),
  'MBS: ppo_micro_batch_size_per_gpu；GBS: ppo_mini_batch_size × rollout.n；GAS: 显式配置或注明未推算');
  if (a.use_dynamic_bsz === true) notes.push('动态 batch，MBS 为配置值，未设置固定 GAS 时不推算');
  if (gbs) notes.push(`GBS 按 verl 配置 ${a.ppo_mini_batch_size} × ${r.n} = ${gbs} 推算`);
  const strategy = a.strategy;
  if (strategy === 'veomni') {
    set('训练后端', 'VeOmni', 'actor_rollout_ref.actor.strategy');
  } else if (strategy === 'fsdp2' || strategy === 'fsdp') {
    set('训练后端', `PyTorch ${strategy.toUpperCase()}`, 'actor.strategy');
    const sp = a.ulysses_sequence_parallel_size;
    set('训练并行策略', `${strategy.toUpperCase()}${devices ? ` (${devices}P)` : ''}${positive(sp) ? `, Ulysses SP${sp}` : ''}`, 'actor.strategy / ulysses_sequence_parallel_size；trainer 设备数');
  } else if (['megatron', 'mcore', 'mindspeed'].includes(strategy) || c.model_engine === 'megatron') {
    set('训练后端', /MindSpeed|mindspeed/.test(text) ? 'Megatron Core + MindSpeed' : 'Megatron Core', 'actor.strategy / model_engine；日志 MindSpeed 标识');
    const mg = a.megatron || {};
    const parallel = [['TP','tensor_model_parallel_size'],['EP','expert_model_parallel_size'],['ETP','expert_tensor_parallel_size'],['PP','pipeline_model_parallel_size'],['CP','context_parallel_size'],['VPP','virtual_pipeline_model_parallel_size']];
    set('训练并行策略', parallel.filter(([, key]) => positive(mg[key])).map(([label, key]) => label + mg[key]).join(' '), 'actor.megatron 并行配置');
    const rec = mg.override_transformer_config || {};
    if (rec.recompute_granularity) set('重计算', `${rec.recompute_granularity === 'full' ? '全重计算' : rec.recompute_granularity}${rec.recompute_method ? '，' + rec.recompute_method : ''}${positive(rec.recompute_num_layers) ? `，每组${rec.recompute_num_layers}层` : ''}`, 'actor.megatron.override_transformer_config（分组层数不是模型层数）');
  }
  const engineNames = { vllm: 'vLLM', vllm_omni: 'vLLM-Omni', sglang: 'SGLang' };
  if (typeof r.name === 'string') set('推理后端', (engineNames[r.name] || r.name) + (/^vllm/.test(r.name) && /vllm_ascend:register|Platform plugin ascend is activated/.test(text) ? ' + vLLM-Ascend' : ''), 'actor_rollout_ref.rollout.name；运行插件记录');
  set('推理并行策略', [['TP','tensor_model_parallel_size'],['PP','pipeline_model_parallel_size']].filter(([, k]) => positive(r[k])).map(([label, k]) => label + r[k]).join(' '), 'actor_rollout_ref.rollout 并行配置');
  if (!fields['重计算'] && typeof m.enable_gradient_checkpointing === 'boolean') set('重计算', m.enable_gradient_checkpointing ? '开启梯度检查点' : '关闭梯度检查点', 'actor_rollout_ref.model.enable_gradient_checkpointing');
  if (fields['重计算']) {
    const normalized = normalizeRecompute(fields['重计算'], { model, stage: m.model_stage, backend: fields['训练后端'] });
    fields['重计算'] = normalized.value;
    if (normalized.note) notes.push(normalized.note);
  }
  if (m.model_stage) notes.push(`训练子模型：${m.model_stage}`);
  if (r.dtype) notes.push(`生成精度：${r.dtype}`);
  if (a.fsdp_config?.model_dtype) notes.push(`训练精度：${a.fsdp_config.model_dtype}`);
  return { fields, evidence, notes, warning: '', modelStage: m.model_stage };
}
