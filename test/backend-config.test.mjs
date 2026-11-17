import test from 'node:test';
import assert from 'node:assert/strict';
import { extractConfig, mapConfig } from '../public/config.js';
import { extractSlimeConfig } from '../public/slime.js';
import { parseSFT } from '../public/parser.js';
import { displayRow } from '../public/record-fields.js';

const config = (strategy, backend = {}, hf = false) => ({
  trainer: { nnodes: 2, n_gpus_per_node: 8 },
  actor_rollout_ref: { actor: { strategy, ...backend }, model: { enable_gradient_checkpointing: hf } },
});
const mcore = rec => config('megatron', { megatron: { override_transformer_config: rec } }, true);

test('verl and slime share explicit Megatron checkpoint coverage, independent of HF flags', () => {
  for (const [rec, expected] of [
    [{ recompute_granularity: null }, '未开重计算'],
    [{ recompute_granularity: 'selective' }, '选择性重计算'],
    [{ recompute_granularity: 'full', recompute_method: 'uniform', recompute_num_layers: 2 }, '全重计算'],
    [{ recompute_granularity: 'full', recompute_method: 'block', recompute_num_layers: 2 }, '按层重计算（每流水线阶段 2 层）'],
    [{ recompute_granularity: 'full', recompute_method: 'uniform' }, '已开重计算（范围待核实）'],
    [{ recompute_granularity: 'full', recompute_method: 'block', recompute_num_layers: 0 }, '已开重计算（范围待核实）'],
    [{}, undefined], [{ recompute_granularity: 'custom' }, undefined],
  ]) {
    assert.equal(mapConfig(mcore(rec)).fields['重计算'], expected);
    const slime = extractSlimeConfig(JSON.stringify({ hf_checkpoint: '/fixture', train_backend: 'megatron', ...rec }));
    assert.equal(slime.fields['重计算'], expected);
    assert.equal(slime.fields['备注'], undefined);
  }
});

test('backend identity does not come from import warnings and unsupported backends do not inherit checkpoint flags', () => {
  assert.equal(mapConfig(mcore({}), 'MindSpeed engine unavailable: ImportError').fields['训练后端'], 'Megatron Core');
  assert.equal(mapConfig(config('mindspeed')).fields['训练后端'], 'Megatron Core + MindSpeed');
  for (const name of ['custom', 'automodel', undefined]) {
    const r = mapConfig(config(name, {}, false));
    assert.equal(r.fields['重计算'], undefined);
    assert.equal(r.fields['训练并行策略'], undefined);
    assert.match(r.notes.join(' '), /尚未适配/);
  }
  const conflicting = { ...config('custom'), model_engine: 'megatron' };
  assert.equal(mapConfig(conflicting).fields['训练后端'], undefined);
});

test('Megatron conflicting process-group and model sizes stay blank while sequence parallel follows its override', () => {
  const c = config('megatron', { megatron: { tensor_model_parallel_size: 2, pipeline_model_parallel_size: 4,
    sequence_parallel: false, override_transformer_config: { tensor_model_parallel_size: 8, sequence_parallel: true } } });
  assert.equal(mapConfig(c).fields['训练并行策略'], 'PP4 Sequence Parallel 开启');
  assert.match(mapConfig(c).notes.join(' '), /进程组配置与模型.*不一致/);
  c.actor_rollout_ref.actor.megatron.override_transformer_config.tensor_model_parallel_size = 2;
  assert.equal(mapConfig(c).fields['训练并行策略'], 'TP2 PP4 Sequence Parallel 开启');
  c.actor_rollout_ref.actor.megatron.override_transformer_config.tensor_model_parallel_size = null;
  assert.equal(mapConfig(c).fields['训练并行策略'], 'PP4 Sequence Parallel 开启');
  const broken = JSON.stringify(c).replace('"tensor_model_parallel_size":null', '"tensor_model_parallel_size":unknown()');
  const r = extractConfig(broken);
  assert.equal(r.fields['训练并行策略'], 'PP4 Sequence Parallel 开启');
  assert.match(r.warning, /tensor_model_parallel_size/);
  const slime = extractSlimeConfig(JSON.stringify({ hf_checkpoint: '/fixture', train_backend: 'megatron', tensor_model_parallel_size: 2, sequence_parallel: true }));
  assert.equal(slime.fields['训练并行策略'], 'TP2 Sequence Parallel 开启');
});

test('FSDP1 and FSDP2 retain shard/replicate sizes and nested Ulysses, not VeOmni topology', () => {
  for (const strategy of ['fsdp', 'fsdp2']) {
    const c = config(strategy, { ulysses_sequence_parallel_size: 1,
      fsdp_config: { fsdp_size: -1, ulysses_sequence_parallel_size: 4 } });
    assert.equal(mapConfig(c).fields['训练并行策略'], 'FSDP16 Ulysses SP4');
    c.actor_rollout_ref.actor.fsdp_config.fsdp_size = 4;
    assert.equal(mapConfig(c).fields['训练并行策略'], 'DP-Replica4 FSDP4 Ulysses SP4');
    c.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 2;
    assert.equal(mapConfig(c).fields['训练并行策略'], 'DP-Replica4 FSDP4 Ulysses SP2');
    c.actor_rollout_ref.actor.fsdp_config.fsdp_size = 3;
    assert.equal(mapConfig(c).fields['训练并行策略'], 'Ulysses SP2');
    delete c.actor_rollout_ref.actor.fsdp_config.fsdp_size;
    assert.equal(mapConfig(c).fields['训练并行策略'], 'Ulysses SP2');
    c.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 1;
    c.actor_rollout_ref.actor.fsdp_config.fsdp_size = 4;
    const broken = JSON.stringify(c).replace('"ulysses_sequence_parallel_size":4', '"ulysses_sequence_parallel_size":broken()');
    assert.equal(extractConfig(broken).fields['训练并行策略'], 'DP-Replica4 FSDP4');
  }
});

test('Turbo disabled status needs both switches, while explicit enabled status remains independent', () => {
  const c = config('fsdp_turbo', { fsdp_config: { turbo_config: { memory: { recompute: false } } } });
  assert.equal(mapConfig(c).fields['重计算'], '未开重计算');
  delete c.actor_rollout_ref.model.enable_gradient_checkpointing;
  assert.equal(mapConfig(c).fields['重计算'], undefined);
  c.actor_rollout_ref.actor.fsdp_config.turbo_config.memory = { recompute: true, recompute_plan: ['model.layers.{*}', 7] };
  assert.equal(mapConfig(c).fields['重计算'], '已开重计算（范围待核实）');
});

test('native VeOmni preserves independent replicate/shard dimensions and explains mixed SAC coverage', () => {
  const c = { model: { model_path: '/fixture', accelerator: { dp_replicate_size: 2, dp_shard_size: 4,
    fsdp_config: { fsdp_mode: 'fsdp2' }, gradient_checkpointing: { enable: true, recompute_layers: 8, selective_recompute_layers: 2 } } }, train: {} };
  let r = parseSFT('VeOmni\n' + JSON.stringify(c)).configuration;
  assert.equal(r.fields['并行策略'], 'DP8 DP-Replica2 DP-Shard4');
  assert.equal(r.fields['重计算'], '选择性重计算');
  assert.match(r.notes.join(' '), /末尾 8 层.*前 2 层.*其余层完整重算/);
  delete c.model.accelerator.dp_replicate_size;
  r = parseSFT('VeOmni\n' + JSON.stringify(c)).configuration;
  assert.equal(r.fields['并行策略'], 'DP-Shard4');
  assert.equal(r.fields['备注'], undefined);
  c.model.accelerator.gradient_checkpointing.enable = false;
  assert.equal(parseSFT('VeOmni\n' + JSON.stringify(c)).configuration.fields['重计算'], '未开重计算');
});

test('slime custom backends cannot reuse Megatron settings and historical block labels are not presented as full', () => {
  const r = extractSlimeConfig(JSON.stringify({ hf_checkpoint: '/fixture', train_backend: 'custom', tensor_model_parallel_size: 4,
    recompute_granularity: 'full', recompute_method: 'uniform', recompute_num_layers: 1, sglang_tp_size: 2 }));
  assert.equal(r.fields['训练并行策略'], undefined);
  assert.equal(r.fields['重计算'], undefined);
  assert.equal(r.fields['推理并行策略'], 'TP2');
  assert.match(r.warning, /尚未适配/);
  const row = { 重计算: '全重计算，block，每组2层', 备注: '保留我的备注' };
  assert.equal(displayRow({ row })['重计算'], '按层重计算（范围待核实）');
  assert.equal(row['重计算'], '全重计算，block，每组2层');
  assert.equal(displayRow({ row })['备注'], row['备注']);
});
