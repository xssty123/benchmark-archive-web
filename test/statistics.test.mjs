import test from 'node:test';
import assert from 'node:assert/strict';
import { parseLog, selectRange } from '../public/parser.js';

const adapters = [
  { name: 'verl', task: 'RL', mfu: '训练MFU', build: rows => rows.map(({ step, ...v }) =>
    `step:${step} - timing_s/update_weights:0` + Object.entries(v).map(([key, value]) =>
      ` - ${{ tps: 'perf/throughput', time: 'perf/time_per_step', mfu: 'perf/mfu/actor' }[key]}:${value === null ? '' : value}`).join('')).join('\n') },
  { name: 'slime', task: 'RL', mfu: '训练MFU', unsupportedMfu: true, build: rows => rows.map(({ step, ...v }) =>
    `perf ${step}: ` + JSON.stringify({ 'perf/update_weights_time': 0, ...Object.fromEntries(Object.entries(v).filter(([key]) => key !== 'mfu').map(([key, value]) =>
      [{ tps: 'perf/actor_train_tok_per_s', time: 'perf/step_time' }[key], value])) })).join('\n') },
  { name: 'VeOmni', task: 'SFT', mfu: 'MFU', build: rows => 'VeOmni\n' + rows.map(({ step, ...v }) =>
    JSON.stringify({ global_step: step, 'training/total_loss': 1, ...Object.fromEntries(Object.entries(v).map(([key, value]) =>
      [{ tps: 'tokens_per_second(M)', time: 'step_time_s', mfu: 'mfu' }[key], key === 'tps' && typeof value === 'number' ? value / 1e6 : value])) })).join('\n') },
  { name: 'SFT JSONL', task: 'SFT', mfu: 'MFU', build: rows => rows.map(({ step, time, ...v }) =>
    JSON.stringify({ task: 'SFT', step, tps_unit: 'tokens/s', ...v, ...(time !== undefined ? { step_time_s: time } : {}) })).join('\n') },
];

for (const adapter of adapters) {
  test(`${adapter.name}: partial metric coverage retains means, real zero values and separate sample counts`, () => {
    const p = parseLog(adapter.task, adapter.build([
      { step: 0, tps: 10, time: 100, mfu: 0 }, { step: 1, time: 10 },
      { step: 2, tps: 30 }, { step: 4, mfu: 0.4 },
    ]));
    assert.deepEqual(p.steps, [0, 1, 2, 4]);
    assert.equal(p.metrics['吞吐TPS'].mean, 20);
    assert.equal(p.metrics['单步耗时s'].mean, 55);
    assert.equal(p.metrics['吞吐TPS'].validCount, 2);
    assert.deepEqual(p.metrics['吞吐TPS'].missing, [1, 4]);
    assert.deepEqual(p.metrics['单步耗时s'].missing, [2, 4]);
    assert.match(p.metrics['吞吐TPS'].reason, /未记录 2 步.*实际记录的 2 步/);
    assert.equal(p.metrics[adapter.mfu].mean, adapter.unsupportedMfu ? null : 0.2);
    assert.equal(p.independentMetrics, true);
    assert.match(p.statisticsNote, /个别步缺字段.*不清空/);
    assert.equal(p.configuration?.fields?.['备注'], undefined);
    const one = selectRange(p, 1, 1);
    assert.equal(one.metrics['吞吐TPS'].mean, null);
    assert.equal(one.metrics['单步耗时s'].mean, 10);
    assert.equal(selectRange(one).metrics['吞吐TPS'].mean, 20);
    assert.equal(selectRange(p, 4, 4).metrics['单步耗时s'].mean, null);
    if (!adapter.unsupportedMfu) assert.equal(selectRange(p, 0, 0).metrics[adapter.mfu].mean, 0);
  });

  test(`${adapter.name}: range recalculation of old saved analyses uses the unified policy`, () => {
    const original = parseLog(adapter.task, adapter.build([{ step: 1, tps: 10 }, { step: 2, time: 3 }]));
    const saved = JSON.parse(JSON.stringify(original));
    saved.independentMetrics = false; delete saved.statisticsNote;
    saved.metrics['吞吐TPS'].mean = null;
    const recalculated = selectRange(saved);
    assert.equal(recalculated.metrics['吞吐TPS'].mean, 10);
    assert.equal(recalculated.metrics['单步耗时s'].mean, 3);
    assert.equal(recalculated.independentMetrics, true);
    assert.match(recalculated.statisticsNote, /各指标分别/);
    assert.equal(saved.metrics['吞吐TPS'].mean, null); // no silent rewrite of saved/user-edited rows
  });

  test(`${adapter.name}: invalid values are reported independently rather than treated as missing`, () => {
    const p = parseLog(adapter.task, adapter.build([{ step: 1, tps: 100, time: 10 }, { step: 2, tps: -1, time: 20 }, { step: 3, time: 30 }]));
    assert.equal(p.metrics['吞吐TPS'].mean, null);
    assert.deepEqual(p.metrics['吞吐TPS'].invalid, [2]);
    assert.deepEqual(p.metrics['吞吐TPS'].missing, [3]);
    assert.equal(p.metrics['单步耗时s'].mean, 20);
  });
}

test('explicit null in SFT JSONL and VeOmni is missing, while an unsupported MFU remains blank', () => {
  for (const adapter of adapters.filter(a => a.task === 'SFT')) {
    const p = parseLog(adapter.task, adapter.build([{ step: 1, tps: 100, mfu: 0.3 }, { step: 2, tps: null, mfu: null }]));
    assert.equal(p.metrics['吞吐TPS'].mean, 100); assert.equal(p.metrics.MFU.mean, 0.3);
    assert.deepEqual(p.metrics.MFU.missing, [2]); assert.deepEqual(p.metrics.MFU.invalid, []);
  }
  const p = parseLog('RL', 'MFU will always be zero\nstep:1 - perf/mfu/actor:0 - perf/throughput:100\nstep:2 - timing_s/update_weights:0');
  assert.equal(p.metrics['训练MFU'].mean, null);
  assert.match(p.metrics['训练MFU'].reason, /不受支持/);
  assert.equal(p.metrics['吞吐TPS'].mean, 100);
  const nullable = parseLog('RL', 'step:1 - perf/throughput:100 - perf/mfu/actor:0.3\nstep:2 - perf/throughput:None - perf/mfu/actor:null');
  assert.equal(nullable.metrics['吞吐TPS'].mean, 100);
  assert.equal(nullable.metrics['训练MFU'].mean, 0.3);
  assert.deepEqual(nullable.metrics['吞吐TPS'].missing, [2]);
  assert.deepEqual(nullable.metrics['训练MFU'].invalid, []);
});
