import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRL, parseSFT, selectRange } from '../public/parser.js';
import { extractSlimeConfig } from '../public/slime.js';
import { BATCH, LAYERS, EXPERTS, displayRow } from '../public/record-fields.js';

// Synthetic config and perf records following slime's native logger and Megatron dump.
const args = {
  hf_checkpoint: '/models/Qwen-Fixture-30B-A3B', actor_num_nodes: 2, actor_num_gpus_per_node: 8,
  rollout_num_gpus: 16, colocate: true, use_critic: false, train_backend: 'megatron',
  rollout_function_path: 'slime.rollout.sglang_rollout.generate_rollout',
  rollout_max_prompt_len: 1024, rollout_max_response_len: 8192, num_layers: 48, num_experts: 128,
  micro_batch_size: 2, global_batch_size: 64, use_dynamic_batch_size: true,
  rollout_batch_size: 16, n_samples_per_prompt: 4,
  tensor_model_parallel_size: 4, pipeline_model_parallel_size: 2, context_parallel_size: 1,
  expert_model_parallel_size: 4, sglang_tp_size: 8, sglang_pp_size: 1,
  recompute_granularity: 'full', recompute_method: 'uniform', recompute_num_layers: 1,
};
const literal = value => value === null ? 'None' : value === true ? 'True' : value === false ? 'False' : String(value);
const dotted = (config = args) => '------------------------ arguments ------------------------\n' +
  Object.entries(config).map(([key, value]) => `  ${key} .... ${literal(value)}`).join('\n') + '\n-------------------- end of arguments ---------------------\n';
const prefix = '(MegatronTrainRayActor pid=123) [2026-11-16 10:00:00] train_metric_utils.py:404 - ';
const perf = (step, values) => prefix + `perf ${step}: ${JSON.stringify(values)}\n`;
const train = { 'perf/actor_train_tok_per_s': 1000, 'perf/actor_train_time': 20, 'perf/update_weights_time': 3,
  'perf/train_wait_time': 40, 'perf/train_time': 25, 'perf/step_time': 65, 'perf/actor_train_tflops': 100 };
const rollout = { 'perf/rollout_time': 35, 'perf/tokens_per_gpu_per_sec': 111, 'perf/effective_tokens_per_gpu_per_sec': 99 };

test('native slime logs map independent config fields and merge training/rollout perf by rollout ID including zero', () => {
  const p = parseRL(dotted() + perf(0, rollout) + perf(0, train) + perf(1, { ...train, 'perf/actor_train_tok_per_s': 2000 }) + perf(1, rollout));
  assert.equal(p.adapter, 'slime-text-v1'); assert.deepEqual(p.steps, [0, 1]);
  assert.equal(p.metrics['吞吐TPS'].mean, 1500); assert.equal(p.metrics['训练MFU'].mean, null);
  assert.equal(p.metrics['训练耗时 actor_update'].mean, 20); assert.equal(p.metrics['推理耗时'].mean, 35);
  assert.equal(p.metrics['resharding耗时'].mean, 3); assert.equal(p.metrics['单步耗时s'].mean, 65);
  assert.match(p.unit, /训练阶段.*全局/);
  const f = p.configuration.fields;
  assert.equal(f['模型'], 'Qwen-Fixture-30B-A3B'); assert.equal(f['参数'], '30B'); assert.equal(f['系列'], 'Qwen');
  assert.equal(f['集群规格'], '16P'); assert.equal(f['序列'], '1k→8k'); assert.equal(f['同步/异步'], undefined);
  assert.equal(f[LAYERS], 48); assert.equal(f[EXPERTS], 128); assert.equal(f['重计算'], '全重计算');
  assert.equal(f['训练后端'], 'Megatron Core'); assert.equal(f['推理后端'], 'SGLang');
  assert.equal(f['训练并行策略'], 'TP4 PP2 CP1 EP4'); assert.equal(f['推理并行策略'], 'TP8 PP1');
  assert.equal(f['train_batch_size × rollout_n'], '16 × 4 = 64');
  assert.equal(displayRow({ row: f })[BATCH], '2/64/dyn'); assert.equal(f['备注'], undefined);
  assert.equal(selectRange(p, 1, 1).metrics['吞吐TPS'].mean, 2000);
  assert.equal(selectRange(selectRange(p, 1, 1)).metrics['吞吐TPS'].mean, 1500);
  const saved = selectRange(JSON.parse(JSON.stringify(p)), 0, 0);
  assert.equal(saved.adapter, 'slime-text-v1'); assert.equal(saved.metrics['吞吐TPS'].mean, 1000);
  assert.equal(saved.configuration.fields['备注'], undefined);
});

test('dotted arguments are order independent with Ray ANSI rank prefixes and invalid fields isolated', () => {
  const broken = dotted(Object.fromEntries(Object.entries(args).reverse())).replace('num_layers .... 48', 'num_layers .... unknown()');
  const text = broken.split('\n').map(line => '\x1b[36m[rank0]: ' + line + '\x1b[0m').join('\r\n');
  const p = parseRL(text + '\n' + perf(0, train));
  assert.equal(p.configuration.fields[LAYERS], ''); assert.equal(p.configuration.fields[EXPERTS], 128);
  assert.equal(p.configuration.fields['模型'], 'Qwen-Fixture-30B-A3B'); assert.equal(p.metrics['吞吐TPS'].mean, 1000);
  assert.match(p.configuration.warning, /num_layers.*原日志行/);
});

test('JSON and Python Namespace configs recover valid fields after an invalid middle field without evaluation', () => {
  for (const text of [JSON.stringify(args).replace('"num_layers":48', '"num_layers":unknown()'),
    'Namespace(' + Object.entries(args).map(([key, value]) => `${key}=${key === 'num_layers' ? 'unknown()' : JSON.stringify(value)}`).join(', ') + ')']) {
    const p = parseRL(text);
    assert.equal(p.count, 0); assert.equal(p.configuration.fields['模型'], 'Qwen-Fixture-30B-A3B');
    assert.equal(p.configuration.fields[LAYERS], ''); assert.equal(p.configuration.fields[EXPERTS], 128);
    assert.match(p.configuration.notes.join('\n'), /num_layers/);
    assert.equal(p.configuration.fields['备注'], undefined);
  }
});

test('Namespace string contents cannot inject config keys and incomplete metric blocks recover sibling fields', () => {
  const p = parseRL("Namespace(hf_checkpoint='/models/Fixture', actor_num_nodes=1, num_layers=48, unused='num_layers=999, num_experts=999', num_experts=128)");
  assert.equal(p.configuration.fields[LAYERS], 48); assert.equal(p.configuration.fields[EXPERTS], 128);
  const recovered = parseRL(dotted() + "perf 0: {'perf/actor_train_time': unsupported(), 'perf/actor_train_tok_per_s': 123, 'perf/step_time': 65");
  assert.equal(recovered.metrics['吞吐TPS'].mean, 123); assert.equal(recovered.metrics['单步耗时s'].mean, 65);
  assert.equal(recovered.metrics['训练耗时 actor_update'].mean, null);
});

test('conflicting config keys invalidate only that field and same-value repeats remain valid', () => {
  const p = parseRL(dotted() + dotted({ ...args, num_experts: 64 }));
  assert.equal(p.configuration.fields[EXPERTS], ''); assert.equal(p.configuration.fields[LAYERS], 48);
  assert.equal(p.configuration.fields['模型'], 'Qwen-Fixture-30B-A3B');
  assert.equal(parseRL(dotted() + dotted()).configuration.fields[EXPERTS], 128);
  const a = extractSlimeConfig(JSON.stringify(args).replace('"num_layers":48', '"num_layers":48,"num_layers":24'));
  assert.equal(a.fields[LAYERS], ''); assert.equal(a.fields[EXPERTS], 128);
});

test('metric dictionary failures do not hide later performance keys or config', () => {
  const line = "perf 0: {'perf/actor_train_tok_per_s': 1000, 'perf/actor_train_time': unknown(), 'perf/rollout_time': 30, 'perf/update_weights_time': 0, 'perf/step_time': 65}";
  const p = parseRL(dotted() + prefix + line);
  assert.equal(p.metrics['吞吐TPS'].mean, 1000); assert.equal(p.metrics['训练耗时 actor_update'].mean, null);
  assert.deepEqual(p.metrics['训练耗时 actor_update'].invalid, [0]);
  assert.equal(p.metrics['推理耗时'].mean, 30); assert.equal(p.metrics['resharding耗时'].mean, 0);
  assert.match(p.configuration.notes.join('\n'), /actor_train_time.*不支持/);
});

test('perf dictionaries can be multiline and same records deduplicate; conflicting steps fail clearly', () => {
  const line = prefix + 'perf 0: ' + JSON.stringify(train, null, 2) + '\n';
  const p = parseRL(line + line); assert.equal(p.count, 1); assert.equal(p.metrics['吞吐TPS'].mean, 1000);
  assert.throws(() => parseRL(line + perf(0, { ...train, 'perf/actor_train_time': 30 })), /冲突/);
  const bad = parseRL(dotted() + perf(0, train) + perf(1, { 'perf/step_time': 60 }));
  assert.equal(bad.metrics['吞吐TPS'].mean, null); assert.deepEqual(bad.metrics['吞吐TPS'].missing, [1]);
});

test('step time uses explicit step_time or the framework wait+train definition, never rollout sums', () => {
  const p = parseRL(dotted() + perf(0, { 'perf/train_wait_time': 30, 'perf/train_time': 20, 'perf/rollout_time': 40 }));
  assert.equal(p.metrics['单步耗时s'].mean, 50);
  const noWait = parseRL(dotted() + perf(0, { 'perf/train_time': 20, 'perf/actor_train_time': 15, 'perf/rollout_time': 40 }));
  assert.equal(noWait.metrics['单步耗时s'].mean, null);
});

test('rollout-only throughput, TFLOPS and unnumbered timers never become main TPS, MFU or step time', () => {
  const p = parseRL(dotted() + perf(0, { 'perf/tokens_per_gpu_per_sec': 123, 'perf/actor_train_tflops': 456 }) + 'Timer update_weights end (elapsed: 1.2s)');
  assert.equal(p.count, 0);
  for (const metric of Object.values(p.metrics)) assert.equal(metric.mean, null);
  assert.match(p.configuration.notes.join('\n'), /TFLOPS|tflops/);
  assert.throws(() => selectRange(p), /没有已记录性能步骤/);
});

test('disaggregated GPUs include explicitly allocated rollout and critic; unknown topology is not guessed', () => {
  assert.equal(parseRL(dotted({ ...args, colocate: false })).configuration.fields['集群规格'], '32P');
  assert.equal(parseRL(dotted({ ...args, colocate: false, use_critic: true, critic_num_nodes: 1, critic_num_gpus_per_node: 8 })).configuration.fields['集群规格'], '40P');
  for (const extra of [{ colocate: null }, { colocate: false, use_critic: null }, { use_critic: true }]) {
    assert.equal(parseRL(dotted({ ...args, ...extra })).configuration.fields['集群规格'], undefined);
  }
});

test('async, batch, recompute and SGLang parallel aliases reflect explicit values rather than defaults', () => {
  const p = parseRL(dotted({ ...args, rollout_function_path: 'slime.rollout.fully_async_rollout.generate_rollout_fully_async',
    use_dynamic_batch_size: false, recompute_granularity: null, sglang_tensor_parallel_size: 4 }));
  const f = p.configuration.fields;
  assert.equal(f['同步/异步'], '异步'); assert.equal(f['重计算'], '未开重计算');
  assert.equal(displayRow({ row: f })[BATCH], '2/64/Nan');
  assert.equal(f['推理并行策略'], 'PP1'); assert.match(p.configuration.warning, /冲突/);
  const minimal = parseRL(dotted({ hf_checkpoint: '/models/Unknown', actor_num_nodes: 1 }));
  assert.equal(minimal.configuration.fields['训练后端'], undefined);
  assert.equal(minimal.configuration.fields['同步/异步'], undefined);
  assert.equal(minimal.configuration.fields['重计算'], undefined);
});

test('verl routing stays separate and mixed framework logs fail rather than silently dropping results', () => {
  assert.equal(parseRL('step:1 - perf/throughput:123').adapter, 'verl-text-v1');
  assert.throws(() => parseRL(dotted() + 'step:1 - perf/throughput:123'), /slime 和 verl/);
  assert.throws(() => parseSFT(dotted()), /JSONL/);
});
