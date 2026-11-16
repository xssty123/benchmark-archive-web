import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSFT, selectRange, parseRL } from '../public/parser.js';
import { BATCH, LAYERS, EXPERTS } from '../public/record-fields.js';

// Synthetic values; same logger and dataclass layout as the local native SFT log.
const prefix = '[INFO|/fixture/VeOmni/veomni/trainer/base.py:344] 10/16/2026 02:00:16 >> ';
const config = {
  model: { model_path: '/models/Qwen3.5-35B-A3B', accelerator: {
    dp_replicate_size: 1, dp_shard_size: 2, tp_size: 1, pp_size: 1, cp_size: 1, ulysses_size: 4, ep_size: 1,
    fsdp_config: { fsdp_mode: 'fsdp2' }, gradient_checkpointing: { enable: true, recompute_layers: -1, selective_recompute_layers: 0 } } },
  data: { max_seq_len: 16384 }, train: { micro_batch_size: 1, global_batch_size: 16, dyn_bsz: true },
};
const runtime = '\nProcess rank: 0, world size: 8\nProcess rank: 1, world size: 8\nSet gradient accumulation to 8.\n';
const modelDump = '\n[INFO|configuration_utils.py:921] 2026-10-16 02:00:18,533 >> Model config Qwen3_5MoeConfig ' + JSON.stringify({
  text_config: { num_hidden_layers: 40, num_experts: 256, num_experts_per_tok: 8 }, vision_config: { num_hidden_layers: 27 } });
const log = prefix + JSON.stringify(config, null, 2) + runtime + modelDump;
const metric = (step, values) => '\n' + prefix + JSON.stringify({ global_step: step, ...values });

test('native VeOmni failure log returns configuration with empty performance and no fake steps', () => {
  const p = parseSFT(log + '\nEpoch 1/2: 0%| | 1/306 [00:51<4:24:02, 51.94s/it, total_loss: 1.39]\n[rank0]: RuntimeError: fixture failure');
  assert.equal(p.adapter, 'veomni-text-v1');
  assert.equal(p.count, 0); assert.deepEqual(p.records, []);
  assert.deepEqual(p.range, { start: null, end: null });
  for (const m of Object.values(p.metrics)) { assert.equal(m.mean, null); assert.match(m.reason, /未记录/); }
  const f = p.configuration.fields;
  assert.equal(f['模型'], 'Qwen3.5-35B-A3B'); assert.equal(f['序列'], '16k');
  assert.equal(f['集群规格'], '8P'); assert.equal(f['训练后端'], 'VeOmni (FSDP2)');
  assert.match(f['并行策略'], /DP2.*Ulysses SP4/);
  assert.equal(f[BATCH], 'MBS=1；GBS=16；GAS=8'); assert.equal(f['重计算'], '全重计算');
  assert.equal(f[LAYERS], 40); assert.equal(f[EXPERTS], 256);
  assert.equal(f['备注'], undefined); assert.equal(p.configuration.warning, '');
  assert.throws(() => selectRange(p), /没有已记录性能步骤/);
});

test('each native config field survives failures before and after it', () => {
  for (const broken of ["unknown()", "Path('/fixture')", "[1,unknown()]", "'bad\\xZZ'"]) {
    const text = prefix + JSON.stringify(config).replace('"dp_shard_size":2', '"dp_shard_size":' + broken) + runtime;
    const p = parseSFT(text), f = p.configuration.fields;
    assert.equal(f['模型'], 'Qwen3.5-35B-A3B'); assert.equal(f['序列'], '16k');
    assert.equal(f['集群规格'], '8P'); assert.equal(f['训练后端'], 'VeOmni (FSDP2)');
    assert.match(f['并行策略'], /Ulysses SP4/); assert.doesNotMatch(f['并行策略'], /DP2/);
    assert.equal(f['重计算'], '全重计算'); assert.match(p.configuration.warning, /model.accelerator.dp_shard_size/);
  }
});

test('reordered fields, ANSI, rank prefixes and interleaved warnings remain parseable', () => {
  const reversed = { train: config.train, data: config.data, model: config.model };
  const text = prefix + JSON.stringify(reversed, null, 2).replace('"data": {', '/fixture.py:12: UserWarning: irrelevant\n"data": {');
  const prefixed = text.split('\n').map(line => '\x1b[36m[rank0]:\x1b[0m ' + line).join('\r\n');
  const p = parseSFT(prefixed + runtime);
  assert.equal(p.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
  assert.equal(p.configuration.fields['序列'], '16k');
  assert.equal(p.configuration.fields[BATCH], 'MBS=1；GBS=16；GAS=8');
});

test('duplicate or conflicting config fields are local failures, not whole-record failures', () => {
  const duplicate = prefix + JSON.stringify(config).replace('"micro_batch_size":1', '"micro_batch_size":1,"micro_batch_size":2') + runtime;
  const p = parseSFT(duplicate);
  assert.equal(p.configuration.fields[BATCH], 'MBS=未记录；GBS=16；GAS=8');
  assert.equal(p.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
  assert.match(p.configuration.warning, /重复字段/);
  const differing = prefix + JSON.stringify({ ...config, data: { max_seq_len: 2048 } });
  const conflict = parseSFT(log + '\n' + differing);
  assert.equal(conflict.configuration.fields['序列'], undefined);
  assert.equal(conflict.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
});

test('checkpointing scope and missing runtime sizes are not guessed', () => {
  for (const [recompute, expected] of [[{ enable: false }, '未开重计算'], [{ enable: true }, '已开重计算（范围待核实）'],
    [{ enable: true, recompute_layers: 2, selective_recompute_layers: 0 }, '部分重计算（2层）'],
    [{ enable: true, recompute_layers: -1, selective_recompute_layers: 2 }, '选择性重计算']]) {
    const c = structuredClone(config); c.model.accelerator.gradient_checkpointing = recompute;
    const p = parseSFT(prefix + JSON.stringify(c));
    assert.equal(p.configuration.fields['重计算'], expected);
    assert.equal(p.configuration.fields['集群规格'], undefined);
    assert.match(p.configuration.fields[BATCH], /GAS=未记录/);
  }
});

test('explicit native metrics use global tokens/s and independent per-step arithmetic averages', () => {
  const text = log + metric(1, { 'tokens_per_second(M)': 0.01, mfu: 0.2, step_time_s: 20 }) +
    metric(2, { 'tokens_per_second(M)': 0.03, mfu: 0.4, step_time_s: 10 });
  const p = parseSFT(text);
  assert.equal(p.unit, 'tokens/s'); assert.equal(p.count, 2);
  assert.equal(p.metrics['吞吐TPS'].mean, 20000); assert.equal(p.metrics['单步耗时s'].mean, 15);
  assert.ok(Math.abs(p.metrics.MFU.mean - 0.3) < 1e-10);
  assert.equal(selectRange(p, 2, 2).metrics['吞吐TPS'].mean, 30000);
});

test('invalid or missing metric affects only itself and all explicitly logged steps count', () => {
  const text = log + metric(1, { 'tokens_per_second(M)': 0.01, mfu: 0.2, step_time_s: 20 }) +
    '\n' + prefix + "{'global_step':2,'mfu':unknown(),'tokens_per_second(M)':0.03,'step_time_s':10}";
  const p = parseSFT(text);
  assert.equal(p.metrics.MFU.mean, null); assert.deepEqual(p.metrics.MFU.invalid, [2]);
  assert.equal(p.metrics['吞吐TPS'].mean, 20000); assert.equal(p.metrics['单步耗时s'].mean, 15);
  assert.equal(p.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
  const lossOnly = parseSFT(text + metric(3, { 'training/total_loss': 1 }));
  assert.equal(lossOnly.count, 3); assert.equal(lossOnly.metrics['吞吐TPS'].mean, null);
  assert.deepEqual(lossOnly.metrics['吞吐TPS'].missing, [3]);
});

test('unlabelled, cumulative, tqdm and per-module throughput are not substitutes for global metrics', () => {
  const p = parseSFT(log + '\nEpoch 1/2: 10%|xx| 10/100 [10:00<11:00, 8.22s/it, loss: 1.1]\n' +
    JSON.stringify({ 'tokens_per_second(M)': 100, mfu: 0.5 }) + metric(1, { 'consume_tokens(M)': 2, 'trace/vision/tokens_per_second(M)': 0.5 }));
  assert.equal(p.count, 1);
  assert.equal(p.metrics['吞吐TPS'].mean, null); assert.equal(p.metrics['单步耗时s'].mean, null);
  assert.match(p.configuration.notes.join('\n'), /没有明确步号/);
});

test('duplicate metric records deduplicate; conflicting restarted steps fail visibly', () => {
  const line = metric(1, { 'tokens_per_second(M)': 0.01, mfu: 0.2 });
  assert.equal(parseSFT(log + line + line).count, 1);
  assert.throws(() => parseSFT(log + line + metric(1, { mfu: 0.9 })), /冲突/);
  assert.throws(() => parseSFT(log + metric('1', { mfu: 0.2 })), /步号/);
  const zeroFlops = parseSFT(log + metric(1, { mfu: 0, 'flops_promised(T)': 0 }));
  assert.equal(zeroFlops.metrics.MFU.mean, null);
});

test('normalized JSONL and verl RL stay separate from the native VeOmni adapter', () => {
  assert.equal(parseSFT(JSON.stringify({ task: 'SFT', step: 1, tps: 3, tps_unit: 'tokens/s', framework: 'VeOmni' })).adapter, 'sft-normalized-jsonl-v1');
  const rl = "{'actor_rollout_ref': {'actor': {'strategy': 'veomni'}}, 'trainer': {}}\nstep:1 - perf/throughput:3";
  assert.throws(() => parseSFT(rl), /JSONL/);
  assert.equal(parseRL(rl).metrics['吞吐TPS'].mean, 3);
});
