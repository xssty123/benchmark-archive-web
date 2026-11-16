import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import { parseLog } from '../public/parser.js';

// Exercise the real browser handlers with controlled file reads and minimal DOM adapters.
const source = await fs.readFile(new URL('../public/app.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function fixture() {
  const elements = new Map();
  const $ = key => {
    if (!elements.has(key)) elements.set(key, { value: '', hidden: false, textContent: '',
      replaceChildren() { this.cleared = true; }, addEventListener(event, fn) { this[event] = fn; } });
    return elements.get(key);
  };
  const state = { revision: 0, task: 'RL', log: null, parsed: null, analyzedLog: null };
  const context = vm.createContext({ state, $, parseLog, safely: (_, fn) => fn(), message() {},
    renderFiles() {}, renderReview() { context.renderCount++; }, stage(n) { context.currentStage = n; }, renderCount: 0 });
  vm.runInContext(section('function invalidateAnalysis()', 'function reset()') + '\n' +
    section("$('#analyze').addEventListener", "$('#back-upload').addEventListener"), context);
  return { state, $, context, choose: log => context.chooseLog(log), analyze: () => $('#analyze').click({ currentTarget: $('#analyze') }) };
}
const file = text => ({ size: text.length, text: async () => text });
test('replacing a log invalidates previous results, draft and saved-record association', async () => {
  const f = fixture();
  f.choose(file('step:1 - perf/throughput:131'));
  await f.analyze();
  f.state.savedRecord = { id: 'old' };
  f.choose(file('step:1 - perf/throughput:222'));
  assert.equal(f.state.parsed, null);
  assert.equal(f.state.savedRecord, null);
  assert.equal(f.$('#resume-review').hidden, true);
  assert.equal(f.$('#record-form').cleared, true);
  await f.analyze();
  assert.equal(f.state.parsed.metrics['吞吐TPS'].mean, 222);
  assert.equal(f.state.analyzedLog, f.state.log);
});
test('continuing the same log retains the current form rather than rendering defaults again', async () => {
  const f = fixture(); f.choose(file('step:1 - perf/throughput:131'));
  await f.analyze();
  const parsed = f.state.parsed;
  await f.analyze();
  assert.equal(f.context.renderCount, 1);
  assert.equal(f.state.parsed, parsed);
  assert.equal(f.context.currentStage, 2);
});
test('a late file read cannot restore results after changing the log or task', async () => {
  const f = fixture(); let finish;
  f.choose({ size: 1, text: () => new Promise(resolve => { finish = resolve; }) });
  const pending = f.analyze();
  f.choose(file('step:1 - perf/throughput:222'));
  finish('step:1 - perf/throughput:131'); await pending;
  assert.equal(f.state.parsed, null);
  assert.equal(f.context.renderCount, 0);
  await f.analyze(); assert.equal(f.state.parsed.metrics['吞吐TPS'].mean, 222);
});
test('metric cards follow edited, zero and blank values without falling back to the log mean', () => {
  const f = fixture(); f.state.parsed = { unit: 'tokens/s/device', metrics: {
    '吞吐TPS': { mean: 131 }, '单步耗时s': { mean: 179 }, '训练MFU': { mean: null, reason: '不支持' },
  } };
  f.context.escape = String; f.context.format = String;
  vm.runInContext(section('function renderMetricCards(', 'function currentPreviewRow()'), f.context);
  f.context.renderMetricCards({ '吞吐TPS': '999', '单步耗时s': '', '训练MFU': '0' });
  const html = f.$('#metric-cards').innerHTML;
  assert.match(html, /<strong>999<\/strong>/); assert.match(html, /<strong>0<\/strong>/);
  assert.match(html, /<strong>—<\/strong>/); assert.doesNotMatch(html, /131|179|不支持/);
});
