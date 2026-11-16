import { BATCH, COMPLETENESS, LAYERS, EXPERTS, modelDefaults, formatBatch } from './record-fields.js';
import { normalizeRecompute } from './recompute.js';
// Read Python pprint data literals only. Never evaluate executable log content.
export function readLiteral(text) {
  let i = 0;
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
          if (!new RegExp(`^[0-9a-f]{${n}}$`, 'i').test(hex)) throw Error('Invalid escape');
          out += String.fromCharCode(parseInt(hex, 16)); i += n;
        } else out += ({ n: '\n', r: '\r', t: '\t', '\\': '\\', "'": "'", '"': '"' })[c] ?? ('\\' + c);
      } else out += c;
    }
    throw Error('Unclosed string');
  }
  function value(depth = 0) {
    if (depth > 100) throw Error('Config too deep');
    ws(); const c = text[i];
    if (c === "'" || c === '"') {
      let out = string(); ws();
      while (text[i] === "'" || text[i] === '"') { out += string(); ws(); }
      return out;
    }
    if (c === '{' || c === '[' || c === '(') {
      i++; const dict = c === '{', close = dict ? '}' : c === '[' ? ']' : ')';
      const out = dict ? Object.create(null) : [];
      ws();
      while (text[i] !== close) {
        const item = value(depth + 1); ws();
        if (dict) {
          if (typeof item !== 'string' || text[i++] !== ':') throw Error('Invalid dict');
          const next = value(depth + 1);
          if (Object.hasOwn(out, item)) throw Error('Duplicate config key');
          out[item] = next;
        } else out.push(item);
        ws(); if (text[i] === ',') { i++; ws(); } else if (text[i] !== close) throw Error('Invalid separator');
      }
      i++; return out;
    }
    const token = text.slice(i).match(/^(?:True|False|None|true|false|null|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)/i)?.[0];
    if (!token) throw Error('Unsupported literal');
    i += token.length;
    const literals = { True: true, False: false, None: null, true: true, false: false, null: null };
    return Object.hasOwn(literals, token) ? literals[token] : Number(token);
  }
  const result = value();
  return { value: result, end: i };
}

export function extractConfig(text) {
  const clean = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').split(/\r?\n/)
    .map(line => line.replace(/^\([^\n)]*pid=\d+[^)]*\)\s?/, '')).join('\n');
  const candidates = [...clean.matchAll(/\{\s*['"]actor_rollout_ref['"]\s*:/g)];
  const configs = [];
  for (const match of candidates) {
    try {
      const c = readLiteral(clean.slice(match.index)).value;
      if (c?.actor_rollout_ref && c?.trainer) configs.push(c);
    } catch { /* Unsupported/incomplete dumps remain visibly unparsed. */ }
  }
  if (!configs.length) return { fields: {}, evidence: {}, notes: [], warning: candidates.length ? '运行配置块不完整或格式暂不支持，请手动核对配置' : '' };
  if (configs.some(c => JSON.stringify(c) !== JSON.stringify(configs[0]))) {
    return { fields: {}, evidence: {}, notes: [], warning: '发现多份不同运行配置，配置字段未自动填入，请先拆分日志' };
  }
  return mapConfig(configs[0], clean);
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
  if (strategy === 'fsdp2' || strategy === 'fsdp') {
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
