import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { parseRL, parseSFT, selectRange } from '../public/parser.js';
import { extractConfig, readLiteral, mapConfig } from '../public/config.js';
import { makeCsv, buildZip, zipRecord } from '../public/archive.js';
import { BATCH, COMPLETENESS, LAYERS, EXPERTS, modelDefaults, parseBatch, formatBatch, displayRow } from '../public/record-fields.js';
import { clipboardRow } from '../public/copy-row.js';

test('RL keeps first/slow steps and averages rates rather than pooling tokens', () => {
  const p = parseRL('\x1b[32mstep:1 - perf/throughput:10 - perf/time_per_step:100\x1b[0m\nstep:2 - perf/throughput:np.float64(30) - perf/time_per_step:10');
  assert.equal(p.count, 2);
  assert.equal(p.metrics['吞吐TPS'].mean, 20);
  assert.equal(p.metrics['单步耗时s'].mean, 55);
  assert.equal(p.metrics['训练MFU'].mean, null);
});
test('incomplete and malformed metrics do not silently become partial averages', () => {
  const p = parseRL('step:1 - perf/throughput:10 - perf/time_per_step:1\nstep:2 - perf/throughput:nan\nstep:3 - perf/time_per_step:2');
  assert.equal(p.count, 3);
  assert.equal(p.metrics['吞吐TPS'].mean, null);
  assert.deepEqual(p.metrics['吞吐TPS'].invalid, [2]);
  assert.deepEqual(p.metrics['吞吐TPS'].missing, [3]);
});
test('duplicate fragments deduplicate; conflicting restarted steps fail', () => {
  const p = parseRL('step:1 - perf/throughput:10\nstep:1 - perf/throughput:10\nstep:1 - perf/time_per_step:5');
  assert.equal(p.count, 1);
  assert.equal(p.metrics['单步耗时s'].mean, 5);
  assert.throws(() => parseRL('step:1 - perf/throughput:10\nstep:1 - perf/throughput:20'), /冲突/);
});
test('MFU unsupported warning leaves values blank; update_weights uses whole stage', () => {
  const p = parseRL('MFU will always be zero\nstep:1 - perf/mfu/actor:0 - timing_s/update_weights:8');
  assert.equal(p.metrics['训练MFU'].mean, null);
  assert.equal(p.metrics['resharding耗时'].mean, 8);
});
test('no matching records fail; gaps and source lines are retained', () => {
  assert.throws(() => parseRL('step:1 - val/score:0.5'), /未找到/);
  const p = parseRL('step:2 - perf/throughput:10\nstep:4 - perf/throughput:30');
  assert.equal(p.gaps, 1);
  assert.deepEqual(p.sources, { 2: [1], 4: [2] });
});
test('SFT supports explicit normalized records only, with consistent units', () => {
  const row = { task: 'SFT', step: 1, tps_unit: 'tokens/s', tps: 100, step_time_s: 5, mfu: 0.2 };
  const p = parseSFT([row, { ...row, step: 2, tps: 200, mfu: null }].map(JSON.stringify).join('\n'));
  assert.equal(p.metrics['吞吐TPS'].mean, 150);
  assert.equal(p.metrics.MFU.mean, null);
  assert.throws(() => parseSFT('step:1 - perf/throughput:10'), /JSONL/);
  assert.throws(() => parseSFT(JSON.stringify({ ...row, task: 'RL' })), /task/);
  assert.throws(() => parseSFT([row, { ...row, step: 2, tps_unit: 'tokens/s/device' }].map(JSON.stringify).join('\n')), /单位/);
});
test('SFT rejects invalid steps and does not coerce string metrics', () => {
  const row = { task: 'SFT', step: 1, tps_unit: 'tokens/s', tps: '100' };
  assert.equal(parseSFT(JSON.stringify(row)).metrics['吞吐TPS'].mean, null);
  assert.throws(() => parseSFT(JSON.stringify({ ...row, step: '1' })), /步号/);
});
test('approved schemas keep exact counts and final material columns', async () => {
  const columns = JSON.parse(await fs.readFile(new URL('../public/columns.json', import.meta.url)));
  assert.equal(columns.RL.length, 30); assert.equal(columns.SFT.length, 23);
  for (const keys of Object.values(columns)) assert.deepEqual(keys.slice(-4), ['备注', '环境信息', '日志存档', 'profiling采集']);
});
test('CSV escapes Excel formulas, quotes and embedded line breaks', () => {
  const csv = makeCsv(['a', 'b'], { a: ' =HYPERLINK("bad")', b: 'x\ny' });
  assert.ok(csv.startsWith('\ufeff'));
  assert.ok(csv.includes('"\' =HYPERLINK(""bad"")"'));
  assert.ok(csv.includes('"x\ny"'));
});
test('ZIP stores exact binary bytes, UTF-8 paths, rejects traversal', async () => {
  const bytes = new Uint8Array([0, 255, 1, 2]);
  const zip = new Uint8Array(await buildZip([{ name: '运行/run.log', bytes }]).arrayBuffer());
  const d = new DataView(zip.buffer);
  assert.equal(d.getUint32(0, true), 0x04034b50);
  assert.equal(d.getUint16(6, true), 0x800);
  const start = 30 + d.getUint16(26, true);
  assert.deepEqual(zip.slice(start, start + 4), bytes);
  assert.equal(d.getUint32(zip.length - 22, true), 0x06054b50);
  assert.throws(() => buildZip([{ name: '../run.log', bytes }]), /非法/);
});
test('log-only archives keep optional cloud links and retain legacy attachment bytes', async () => {
  async function unpack(blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer()), view = new DataView(bytes.buffer), files = {};
    let offset = 0;
    while (view.getUint32(offset, true) === 0x04034b50) {
      const size = view.getUint32(offset + 18, true), nameSize = view.getUint16(offset + 26, true);
      const name = new TextDecoder().decode(bytes.slice(offset + 30, offset + 30 + nameSize));
      const start = offset + 30 + nameSize + view.getUint16(offset + 28, true);
      files[name] = bytes.slice(start, start + size); offset = start + size;
    }
    return files;
  }
  const logBytes = new Uint8Array([0, 255, 13, 10]);
  const record = { id: 'RL-test', log: new Blob([logBytes]), row: {} };
  const onlyLog = await unpack(await zipRecord(record));
  assert.deepEqual(Object.keys(onlyLog), ['RL-test/run.log']);
  assert.deepEqual(onlyLog['RL-test/run.log'], logBytes);
  record.row = { 环境信息: 'https://example.org/env', 日志存档: '', profiling采集: 'https://example.org/profile' };
  const linked = await unpack(await zipRecord(record));
  assert.deepEqual(JSON.parse(new TextDecoder().decode(linked['RL-test/material-links.json'])), {
    环境信息: record.row['环境信息'], profiling采集: record.row['profiling采集'],
  });
  assert.equal(linked['RL-test/environment.txt'], undefined);
  const old = await unpack(await zipRecord({ ...record, environment: new Blob(['old env']), profiles: [new File([logBytes], 'trace.zip')] }));
  assert.equal(new TextDecoder().decode(old['RL-test/environment.txt']), 'old env');
  assert.deepEqual(old['RL-test/profiling/1-trace.zip'], logBytes);
});

test('range is inclusive, defaults to all and can restore all from a subset', () => {
  const p = parseRL('step:1 - perf/throughput:10\nstep:2 - perf/throughput:20\nstep:3 - perf/throughput:90');
  const selected = selectRange(p, 2, 3);
  assert.equal(selected.count, 2);
  assert.equal(selected.metrics['吞吐TPS'].mean, 55);
  assert.deepEqual(selected.sources, { 2: [2], 3: [3] });
  assert.equal(selectRange(selected).count, 3);
  assert.equal(selectRange(selected).metrics['吞吐TPS'].mean, 40);
  assert.equal(selectRange(p, 2, 2).metrics['吞吐TPS'].mean, 20);
});
test('range validates bounds, gaps and coverage within the selected window', () => {
  const p = parseRL('step:1 - perf/throughput:10\nstep:3 - perf/time_per_step:2\nstep:5 - perf/throughput:30');
  assert.throws(() => selectRange(p, 4, 2), /统计范围/);
  assert.throws(() => selectRange(p, 0, 5), /统计范围/);
  assert.throws(() => selectRange(p, 1.5, 5), /统计范围/);
  assert.throws(() => selectRange(p, 2, 2), /没有/);
  assert.equal(selectRange(p, 2, 5).gaps, 2);
  assert.equal(selectRange(p, 3, 5).metrics['吞吐TPS'].mean, null);
  assert.equal(selectRange(p, 5, 5).metrics['吞吐TPS'].mean, 30);
});
test('SFT uses the same selection without changing units or unsupported metric rules', () => {
  const p = parseSFT([1, 2, 3].map(step => JSON.stringify({ task: 'SFT', step, tps_unit: 'tokens/s', tps: step * 10 })).join('\n'));
  const selected = selectRange(p, 2, 3);
  assert.equal(selected.unit, 'tokens/s');
  assert.equal(selected.metrics['吞吐TPS'].mean, 25);
  assert.equal(selected.metrics.MFU.mean, null);
  const rl = parseRL('MFU will always be zero\nstep:1 - perf/mfu/actor:0\nstep:2 - perf/mfu/actor:0');
  assert.equal(selectRange(rl, 2, 2).metrics['训练MFU'].mean, null);
});
test('literal parser handles pprint strings/null/bools without executing expressions', () => {
  const p = readLiteral("{'path': '/models/' 'Qwen-test', 'none': None, 'on': True, 'off': False, 'list': [1, 2,], 'escaped': 'it\\'s'}").value;
  assert.equal(p.path, '/models/Qwen-test'); assert.equal(p.none, null);
  assert.equal(p.on, true); assert.equal(p.off, false); assert.equal(p.escaped, "it's");
  assert.throws(() => readLiteral("{'x': __import__('os')}"));
  assert.throws(() => readLiteral("{'x':1,'x':2}"));
});
test('resolved actor config maps model and batch without taking critic or rollout mode', () => {
  const c = { actor_rollout_ref: {
    actor: { _target_: 'example.OmniActorConfig', strategy: 'fsdp2', ppo_mini_batch_size: 16, ppo_micro_batch_size_per_gpu: null, use_dynamic_bsz: true, ulysses_sequence_parallel_size: 1 },
    model: { path: '/models/Qwen3-Omni-30B-A3B-Instruct', enable_gradient_checkpointing: true },
    rollout: { name: 'vllm_omni', n: 16, mode: 'async', tensor_model_parallel_size: 2, pipeline_model_parallel_size: 1 },
  }, trainer: { nnodes: 1, n_gpus_per_node: 8, v1: { trainer_mode: 'omni_sync' } },
  critic: { ppo_mini_batch_size: 999 }, data: { train_batch_size: 128, max_prompt_length: 4096, max_response_length: 12288 } };
  const p = extractConfig('vllm_ascend:register\n' + JSON.stringify(c));
  assert.equal(p.fields['同步/异步'], '同步'); assert.equal(p.fields['序列'], '4k→12k');
  assert.equal(p.fields['集群规格'], '8P'); assert.equal(p.fields[BATCH], 'MBS=未配置；GBS=256；GAS=动态（未推算）');
  assert.equal(p.fields['推理后端'], 'vLLM-Omni + vLLM-Ascend');
  assert.equal(p.fields['训练并行策略'], 'FSDP2 (8P), Ulysses SP1');
  assert.equal(p.fields['满血/减层 关键配置'], undefined);
  assert.equal(p.fields['层数/总层数（满血填总）'], undefined);
  assert.ok(p.evidence['模型']);
  const conflicting = extractConfig(JSON.stringify(c) + '\n' + JSON.stringify({ ...c, trainer: { nnodes: 2 } }));
  assert.deepEqual(conflicting.fields, {}); assert.match(conflicting.warning, /不同运行配置/);
});

test('model completeness defaults to full with traceable text-model totals', () => {
  const omni = modelDefaults('Qwen3-Omni-30B-A3B-Instruct', 'thinker');
  assert.equal(omni.fields[COMPLETENESS], '满血');
  assert.equal(omni.fields[LAYERS], 48); assert.equal(omni.fields[EXPERTS], 128);
  assert.ok(omni.source.url.startsWith('https://huggingface.co/Qwen/'));
  assert.equal(modelDefaults('Qwen3.5-35B-A3B').fields[LAYERS], 40);
  assert.equal(modelDefaults('Qwen3.5-35B-A3B').fields[EXPERTS], 256);
  assert.equal(modelDefaults('unknown').fields[LAYERS], undefined);
  assert.equal(modelDefaults('unknown').fields[COMPLETENESS], '满血');
  assert.equal(modelDefaults('Qwen3-Omni-30B-A3B-Instruct', 'talker').fields[LAYERS], undefined);
});
test('recomputation distinguishes verified Omni layer checkpointing from unknown scopes', () => {
  const config = (model, strategy = 'fsdp2') => ({ actor_rollout_ref: { actor: { strategy }, model } });
  const model = { path: '/models/Qwen3-Omni-30B-A3B-Instruct', model_stage: 'thinker', enable_gradient_checkpointing: true };
  const omni = mapConfig(config(model));
  assert.equal(omni.fields['重计算'], '全重计算');
  assert.ok(omni.notes.some(note => note.includes('Thinker 文本主干全部 Transformer 层')));
  for (const variant of [{ ...model, path: '/models/Unknown' }, { ...model, model_stage: 'talker' }, { ...model, model_stage: undefined }]) {
    assert.equal(mapConfig(config(variant)).fields['重计算'], '已开重计算（范围待核实）');
  }
  assert.equal(mapConfig(config(model, 'veomni')).fields['重计算'], '已开重计算（范围待核实）');
  assert.equal(mapConfig(config({ ...model, enable_gradient_checkpointing: false })).fields['重计算'], '未开重计算');
  assert.equal(mapConfig(config({ ...model, enable_gradient_checkpointing: undefined })).fields['重计算'], undefined);
  for (const [granularity, expected] of [['full', '全重计算'], ['selective', '选择性重计算']]) {
    const p = mapConfig({ actor_rollout_ref: { actor: { strategy: 'megatron', megatron: {
      override_transformer_config: { recompute_granularity: granularity, recompute_method: 'uniform', recompute_num_layers: 1 },
    } }, model: { enable_gradient_checkpointing: false } } });
    assert.equal(p.fields['重计算'], expected);
    assert.ok(p.notes.some(note => note.includes('uniform')));
  }
});
test('legacy recomputation labels are normalized for tables and exports without changing stored records', () => {
  const old = { row: { 模型: 'Qwen3-Omni-30B-A3B-Instruct', 训练后端: 'PyTorch FSDP2', 重计算: '开启梯度检查点' }, analysis: { configuration: { modelStage: 'thinker' } } };
  const row = displayRow(old);
  assert.equal(row['重计算'], '全重计算');
  assert.equal(row['备注'], undefined);
  assert.equal(old.row['重计算'], '开启梯度检查点');
  assert.equal(displayRow({ row: { ...old.row, 模型: 'Unknown' } })['重计算'], '已开重计算（范围待核实）');
  assert.equal(displayRow({ row: { 重计算: '关闭梯度检查点' } })['重计算'], '未开重计算');
  assert.equal(displayRow({ row: { 重计算: '全重计算，uniform，每组1层', 备注: '' } })['备注'], '');
  assert.equal(displayRow({ row: { 重计算: '全重计算，uniform，每组1层', 备注: '  用户备注  ' } })['备注'], '  用户备注  ');
  assert.equal(displayRow({ row: { 重计算: '自定义策略' } })['重计算'], '自定义策略');
});
test('batch fields render missing values explicitly and preserve zero or fixed values', () => {
  assert.equal(formatBatch(parseBatch('/256/', true)), 'MBS=未记录；GBS=256；GAS=动态（未推算）');
  assert.equal(formatBatch(parseBatch('4/32/1')), 'MBS=4；GBS=32；GAS=1');
  assert.equal(formatBatch(parseBatch('')), 'MBS=未记录；GBS=未记录；GAS=未记录');
  assert.deepEqual(parseBatch('MBS=未配置；GBS=256；GAS=动态（未推算）'), ['未配置', '256', '动态（未推算）']);
  const record = { row: { [BATCH]: '/256/', '备注': '动态batch' } };
  assert.equal(displayRow(record)[BATCH], 'Nan/256/dyn');
  assert.equal(record.row[BATCH], '/256/'); // Old stored records remain unchanged.
});
test('explicit reduced-model configuration overrides full-size defaults', () => {
  const p = mapConfig({ actor_rollout_ref: { model: {
    path: '/models/Qwen3-Omni-30B-A3B-Instruct', model_stage: 'thinker',
    override_config: { thinker_config: { text_config: { num_hidden_layers: 16, num_experts: 64 } } },
  } } });
  assert.equal(p.fields[COMPLETENESS], '减层/减配');
  assert.equal(p.fields[LAYERS], '16/48'); assert.equal(p.fields[EXPERTS], '64/128');
  const fixed = mapConfig({ actor_rollout_ref: { actor: { ppo_micro_batch_size_per_gpu: 4, gradient_accumulation_steps: 2 } } });
  assert.equal(fixed.fields[BATCH], 'MBS=4；GBS=未记录；GAS=2');
});

test('clipboard rows preserve RL/SFT column ordering, empty trailing cells and omit headers', async () => {
  const schemas = JSON.parse(await fs.readFile(new URL('../public/columns.json', import.meta.url)));
  for (const columns of Object.values(schemas)) {
    const row = { 模型: 'Example-30B', 吞吐TPS: 0, 备注: '范围 2–10' };
    const copied = clipboardRow(columns, row);
    const cells = copied.text.split('\t');
    assert.equal(cells.length, columns.length);
    assert.equal(cells[columns.indexOf('模型')], 'Example-30B');
    assert.equal(cells[columns.indexOf('吞吐TPS')], '0');
    assert.deepEqual(cells.slice(-3), ['', '', '']);
    assert.equal((copied.html.match(/<tr>/g) || []).length, 1);
    assert.equal((copied.html.match(/<td>/g) || []).length, columns.length);
    assert.ok(!copied.html.includes('<th>'));
  }
});
test('clipboard values cannot add rows, columns, formulas or HTML markup', () => {
  const copied = clipboardRow(['a', 'b'], { a: '=1+1\t\nextra', b: '<img src=x onerror=alert(1)>"text"' });
  assert.equal(copied.text.split('\t').length, 2);
  assert.equal(copied.text.split('\n').length, 1);
  assert.ok(copied.text.startsWith("'=1+1 extra"));
  assert.ok(copied.html.includes('&lt;img'));
  assert.ok(!copied.html.includes('<img'));
  assert.ok(copied.text.includes('""text""'));
});
