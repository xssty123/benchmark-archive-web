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
  const p = parseSFT(log + '\n[rank0]: RuntimeError: fixture failure');
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
  assert.equal(lossOnly.count, 3); assert.equal(lossOnly.metrics['吞吐TPS'].mean, 20000);
  assert.deepEqual(lossOnly.metrics['吞吐TPS'].missing, [3]);
});

test('unlabelled, cumulative and per-module throughput are not substitutes for global metrics', () => {
  const p = parseSFT(log + '\nEpoch 1/2: 10%|xx| 10/100 [10:00<11:00, 8.22s/it, loss: 1.1]\n' +
    JSON.stringify({ 'tokens_per_second(M)': 100, mfu: 0.5 }) + metric(1, { 'consume_tokens(M)': 2, 'trace/vision/tokens_per_second(M)': 0.5 }));
  assert.equal(p.count, 2);
  assert.equal(p.metrics['吞吐TPS'].mean, null); assert.equal(p.metrics.MFU.mean, null);
  assert.equal(p.metrics['单步耗时s'].mean, 8.22);
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

test('MiniMax component paths and DiT num_layers identify the trained backbone', () => {
  const c = structuredClone(config);
  c.model.model_path = 'pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer';
  c.data.data_type = 'diffusion'; c.data.max_seq_len = 2048;
  c.model.accelerator.dp_shard_size = 16;
  const main = '\nModel config MiniMaxH3DiTModelConfig {"model_type":"MiniMaxH3DiTModel","num_layers":50,"token_refiner_num_layers":2}';
  const condition = '\nModel config MiniMaxH3ConditionModelConfig {"model_type":"MiniMaxH3ConditionModel","num_layers":7,"text_config":{"num_hidden_layers":7,"num_experts":99},"text_encoder_num_retained_layers":7}';
  const text = prefix + JSON.stringify(c) + '\nProcess rank: 0, world size: 16\nSet gradient accumulation to 1.\nDetected DiT model type: MiniMaxH3DiTModel.' + main + condition + main;
  const p = parseSFT(text), f = p.configuration.fields;
  assert.equal(f['模型'], 'MiniMax-H3'); assert.equal(f['系列'], 'MiniMax');
  assert.equal(f[LAYERS], 50); assert.equal(f[EXPERTS], undefined);
  assert.equal(f['序列'], undefined); assert.equal(f['参数'], undefined);
  assert.equal(f['集群规格'], '16P'); assert.equal(f[BATCH], 'MBS=1；GBS=16；GAS=1');
  assert.equal(f['重计算'], '全重计算'); assert.equal(p.count, 0);
  assert.match(p.configuration.notes.join('\n'), /latent/);
  const broken = parseSFT(text.replaceAll('"num_layers":50', '"num_layers":unknown()'));
  assert.equal(broken.configuration.fields[LAYERS], '');
  assert.equal(broken.configuration.fields['模型'], 'MiniMax-H3');
  assert.equal(broken.configuration.fields[BATCH], 'MBS=1；GBS=16；GAS=1');
});

test('DiT model selection never uses condition-only or conflicting backbone depths', () => {
  const c = structuredClone(config); c.data.data_type = 'diffusion'; c.model.model_path = '/models/MiniMax-H3/transformer';
  const base = prefix + JSON.stringify(c) + '\nDetected DiT model type: MiniMaxH3DiTModel.';
  const condition = '\nModel config MiniMaxH3ConditionModelConfig {"num_layers":7}';
  assert.equal(parseSFT(base + condition).configuration.fields[LAYERS], undefined);
  const first = '\nModel config MiniMaxH3DiTModelConfig {"num_layers":50}';
  const second = '\nModel config MiniMaxH3DiTModelConfig {"num_layers":48}';
  const p = parseSFT(base + first + second);
  assert.equal(p.configuration.fields[LAYERS], '');
  assert.equal(p.configuration.fields['模型'], 'MiniMax-H3');
  assert.match(p.configuration.warning, /num_layers.*冲突/);
});

test('generic unknown component paths stay blank while ordinary model names are unchanged', () => {
  for (const [path, expected] of [['/models/unlabelled/transformer', undefined], ['/models/Qwen-Fixture-27B', 'Qwen-Fixture-27B'],
    ['/models/MiniMax-H3', 'MiniMax-H3']]) {
    const c = structuredClone(config); c.model.model_path = path;
    assert.equal(parseSFT(prefix + JSON.stringify(c)).configuration.fields['模型'], expected);
  }
});

test('tqdm s/it averages provide estimated time while TPS, MFU and remarks remain empty', () => {
  const p = parseSFT(log + '\nEpoch 1/1: 3%| | 304/8800 [24:20<10:37:03, 4.50s/it, total_loss: 0.46, mse_video: 0.08, mse_audio: 0.38]\nEpoch 1/1: 3%| | 305/8800 [24:24<10:23:42, 4.41s/it, total_loss: 0.40, mse_video: 0.08, mse_audio: 0.32]');
  assert.equal(p.count, 2);
  assert.equal(p.metrics['单步耗时s'].mean, 4.455);
  assert.equal(p.metrics.MFU.mean, null); assert.equal(p.metrics['吞吐TPS'].mean, null);
  assert.equal(p.configuration.fields['备注'], undefined);
  assert.equal(p.estimatedStepTime, true);
  assert.equal(selectRange(p, 305, 305).metrics['单步耗时s'].mean, 4.41);
  assert.equal(selectRange(selectRange(p, 305, 305)).metrics['单步耗时s'].mean, 4.455);
  assert.match(p.configuration.notes.join('\n'), /单步耗时为估算/);
});

const progress = (step, speed, epoch = 1, totalEpochs = 1, size = 100) =>
  `Epoch ${epoch}/${totalEpochs}: 3%|▎ | ${step}/${size} [00:10<01:00, ${speed}, total_loss: 0.4]`;

test('progress redraws, ANSI, CR-only output, gaps and multi-epoch counters keep one sample per step', () => {
  const text = log + '\n' + [progress(1, '9.00s/it', 1, 2), progress(1, '8.00s/it', 1, 2),
    progress(2, '4.00s/it', 1, 2), progress(1, '6.00s/it', 2, 2), progress(1, '6.00s/it', 2, 2)]
    .map(line => '\x1b[36m[rank0]: ' + line + '\x1b[0m').join('\r');
  const p = parseSFT(text);
  assert.deepEqual(p.steps, [1, 2, 101]);
  assert.equal(p.metrics['单步耗时s'].mean, 6);
  assert.equal(selectRange(p, 1, 2).metrics['单步耗时s'].mean, 6);
  assert.equal(selectRange(p, 2, 101).metrics['单步耗时s'].mean, 5);
  assert.equal(p.metrics['单步耗时s'].validCount, 3);
});

test('invalid speeds, initial bars, it/s and validation/loading bars are not estimated', () => {
  const bars = ['?s/it', 'NaNs/it', 'infs/it', '0.00s/it', '-2.00s/it', '2.00it/s'];
  const text = log + '\n' + [...bars.map(speed => progress(1, speed)), progress(0, '5.00s/it'),
    'Validation: 3%| | 1/100 [00:10<01:00, 5.00s/it]', 'Loading: 1/100 5.00s/it'].join('\n');
  const p = parseSFT(text);
  assert.equal(p.count, 0); assert.equal(p.metrics['单步耗时s'].mean, null);
  const valid = parseSFT(text + '\n' + progress(2, '4.00s/it'));
  assert.equal(valid.count, 1); assert.equal(valid.metrics['单步耗时s'].mean, 4);
});

test('explicit step time takes precedence across the log without mixing estimates into missing steps', () => {
  const p = parseSFT(log + metric(1, { step_time_s: 20 }) + metric(2, { step_time_s: 10 }) +
    '\n' + progress(1, '4.00s/it') + '\n' + progress(2, '6.00s/it'));
  assert.equal(p.estimatedStepTime, false); assert.equal(p.metrics['单步耗时s'].mean, 15);
  assert.doesNotMatch(p.configuration.notes.join('\n'), /单步耗时为估算/);
  const partial = parseSFT(log + metric(1, { step_time_s: 20 }) + metric(2, { mfu: 0.2 }) + '\n' + progress(2, '6.00s/it'));
  assert.equal(partial.metrics['单步耗时s'].mean, 20);
  assert.deepEqual(partial.metrics['单步耗时s'].missing, [2]);
});

test('estimated time coexists with recorded TPS and MFU and survives saved JSON analysis', () => {
  const p = parseSFT(log + metric(1, { 'tokens_per_second(M)': 0.01, mfu: 0.2 }) +
    metric(2, { 'tokens_per_second(M)': 0.03, mfu: 0.4 }) + '\n' + progress(1, '4.00s/it') + '\n' + progress(2, '6.00s/it'));
  assert.equal(p.metrics['吞吐TPS'].mean, 20000);
  assert.ok(Math.abs(p.metrics.MFU.mean - 0.3) < 1e-10);
  const restored = selectRange(JSON.parse(JSON.stringify(p)), 2, 2);
  assert.equal(restored.metrics['单步耗时s'].mean, 6);
  assert.equal(restored.estimatedStepTime, true);
  assert.match(restored.configuration.notes.join('\n'), /估算/);
  assert.equal(restored.configuration.fields['备注'], undefined);
});

test('inconsistent epoch totals disable only the time estimate and explain why', () => {
  const p = parseSFT(log + metric(1, { mfu: 0.2 }) + '\n' + progress(1, '4.00s/it') + '\n' + progress(2, '6.00s/it', 1, 1, 200));
  assert.equal(p.metrics.MFU.mean, 0.2); assert.equal(p.metrics['单步耗时s'].mean, null);
  assert.equal(p.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
  assert.match(p.configuration.notes.join('\n'), /不一致.*未估算/);
});

function trainLayout() {
  const c = structuredClone(config);
  const { gradient_checkpointing, ...accelerator } = c.model.accelerator;
  delete c.model.accelerator;
  c.train.accelerator = accelerator;
  c.train.gradient_checkpointing = gradient_checkpointing;
  return c;
}

test('train-owned VeOmni accelerator and checkpoint settings work independently of model-owned layout', () => {
  const c = trainLayout();
  c.train.gradient_checkpointing = { enable: true, debug: false, enable_reentrant: false };
  const p = parseSFT(prefix + JSON.stringify(c) + runtime + metric(1, { step_time_s: 5 }));
  assert.equal(p.configuration.fields['训练后端'], 'VeOmni (FSDP2)');
  assert.equal(p.configuration.fields['并行策略'], 'DP2 DP-Replica1 DP-Shard2 TP1 PP1 CP1 Ulysses SP4 EP1');
  assert.equal(p.configuration.fields['重计算'], '已开重计算（范围待核实）');
  assert.match(p.configuration.evidence['并行策略'], /train.accelerator.dp_shard_size/);
  assert.match(p.configuration.evidence['重计算'], /train.gradient_checkpointing.enable/);
  assert.equal(p.configuration.fields['备注'], undefined);
  assert.equal(p.metrics['单步耗时s'].mean, 5);
});

test('checkpoint layout is resolved independently of accelerator layout', () => {
  for (const location of ['model-accelerator', 'train', 'train-accelerator']) {
    const c = trainLayout();
    const checkpoint = c.train.gradient_checkpointing;
    delete c.train.gradient_checkpointing;
    if (location === 'model-accelerator') c.model.accelerator = { gradient_checkpointing: checkpoint };
    else if (location === 'train') c.train.gradient_checkpointing = checkpoint;
    else c.train.accelerator.gradient_checkpointing = checkpoint;
    const p = parseSFT(prefix + JSON.stringify(c));
    assert.equal(p.configuration.fields['重计算'], '全重计算');
    assert.match(p.configuration.fields['并行策略'], /DP2.*Ulysses SP4/);
  }
});

test('same-valued aliases deduplicate while conflicts invalidate only dependent fields', () => {
  const c = trainLayout();
  c.model.accelerator = structuredClone(config.model.accelerator);
  let p = parseSFT(prefix + JSON.stringify(c));
  assert.equal(p.configuration.fields['重计算'], '全重计算');
  assert.equal(p.configuration.warning, '');
  c.train.accelerator.dp_shard_size = 4;
  c.train.gradient_checkpointing.enable = false;
  p = parseSFT(prefix + JSON.stringify(c));
  assert.doesNotMatch(p.configuration.fields['并行策略'], /DP2|DP-Shard/);
  assert.match(p.configuration.fields['并行策略'], /DP-Replica1.*Ulysses SP4/);
  assert.equal(p.configuration.fields['重计算'], undefined);
  assert.equal(p.configuration.fields['训练后端'], 'VeOmni (FSDP2)');
  assert.match(p.configuration.warning, /冲突/);
});

test('a malformed alternate leaf cannot silently fall back to another layout', () => {
  const c = trainLayout();
  c.model.accelerator = structuredClone(config.model.accelerator);
  c.train.accelerator.dp_shard_size = 'BROKEN';
  c.train.gradient_checkpointing.enable = 'BROKEN';
  const text = JSON.stringify(c).replaceAll('"BROKEN"', 'unsupported()');
  const p = parseSFT(prefix + text);
  assert.doesNotMatch(p.configuration.fields['并行策略'], /DP-Shard/);
  assert.match(p.configuration.fields['并行策略'], /TP1.*Ulysses SP4/);
  assert.equal(p.configuration.fields['重计算'], undefined);
  assert.equal(p.configuration.fields['模型'], 'Qwen3.5-35B-A3B');
  assert.match(p.configuration.warning, /无法确认|冲突/);
});

test('unknown parallel layout gives a specific explanation without inferring from device count', () => {
  const c = structuredClone(config);
  delete c.model.accelerator;
  const p = parseSFT(prefix + JSON.stringify(c) + runtime);
  assert.equal(p.configuration.fields['并行策略'], undefined);
  assert.equal(p.configuration.fields['集群规格'], '8P');
  assert.match(p.configuration.notes.join(' '), /model.accelerator 或 train.accelerator.*不能仅凭总卡数/);
});
