import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import { formatMfu } from '../public/record-fields.js';
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
  const links = ['环境信息', '日志存档', 'profiling采集'].map(key => ({ dataset: { material: key }, value: '',
    setCustomValidity(text) { this.error = text; }, reportValidity() { this.reported = true; return !this.error; } }));
  const state = { revision: 0, task: 'RL', log: null, parsed: null, analyzedLog: null };
  const context = vm.createContext({ state, $, URL, window: { confirm: () => true, addEventListener() {} }, currentFingerprint: () => 'draft', $$: selector => selector.includes('data-material') ? links : [], parseLog, safely: (_, fn) => fn(), message() {},
    renderFiles() {}, renderReview() { context.renderCount++; }, stage(n) { context.currentStage = n; }, renderCount: 0 });
  vm.runInContext(section('function invalidateAnalysis()', 'function reset()') + '\n' +
    section("$('#analyze').addEventListener", "$('#back-upload').addEventListener"), context);
  context.confirmLeave = async () => !context.hasUnsavedChanges() || context.allowLeave !== false;
  return { state, $, links, context, choose: log => context.chooseLog(log), analyze: () => $('#analyze').click({ currentTarget: $('#analyze') }) };
}
const file = text => ({ size: text.length, text: async () => text });
test('replacing a log invalidates previous results, draft and saved-record association', async () => {
  const f = fixture();
  await f.choose(file('step:1 - perf/throughput:131'));
  await f.analyze();
  f.state.savedRecord = { id: 'old' };
  await f.choose(file('step:1 - perf/throughput:222'));
  assert.equal(f.state.parsed, null);
  assert.equal(f.state.savedRecord, null);
  assert.equal(f.$('#resume-review').hidden, true);
  assert.equal(f.$('#record-form').cleared, true);
  await f.analyze();
  assert.equal(f.state.parsed.metrics['吞吐TPS'].mean, 222);
  assert.equal(f.state.analyzedLog, f.state.log);
});
test('continuing the same log retains the current form rather than rendering defaults again', async () => {
  const f = fixture(); await f.choose(file('step:1 - perf/throughput:131'));
  await f.analyze();
  const parsed = f.state.parsed;
  await f.analyze();
  assert.equal(f.context.renderCount, 1);
  assert.equal(f.state.parsed, parsed);
  assert.equal(f.context.currentStage, 2);
});
test('a late file read cannot restore results after changing the log or task', async () => {
  const f = fixture(); let finish;
  await f.choose({ size: 1, text: () => new Promise(resolve => { finish = resolve; }) });
  const pending = f.analyze();
  await f.choose(file('step:1 - perf/throughput:222'));
  finish('step:1 - perf/throughput:131'); await pending;
  assert.equal(f.state.parsed, null);
  assert.equal(f.context.renderCount, 0);
  await f.analyze(); assert.equal(f.state.parsed.metrics['吞吐TPS'].mean, 222);
});
test('metric cards follow edited, zero and blank values without falling back to the log mean', () => {
  const f = fixture(); f.state.parsed = { unit: 'tokens/s/device', metrics: {
    '吞吐TPS': { mean: 131 }, '单步耗时s': { mean: 179 }, '训练MFU': { mean: null, reason: '不支持' },
  } };
  f.context.escape = String; f.context.format = String; f.context.formatMfu = formatMfu;
  vm.runInContext(section('function renderMetricCards(', 'function currentPreviewRow()'), f.context);
  f.context.renderMetricCards({ '吞吐TPS': '999', '单步耗时s': '', '训练MFU': '0' });
  const html = f.$('#metric-cards').innerHTML;
  assert.match(html, /<strong>999<\/strong>/); assert.match(html, /<strong>0%<\/strong>/);
  assert.match(html, /<strong>—<\/strong>/); assert.doesNotMatch(html, /131|179|不支持/);
});

test('MFU percent inputs normalize to ratios and reject values above 100 for both tasks', () => {
  for (const key of ['训练MFU', 'MFU']) {
    const f = fixture();
    const inputs = [{ dataset: { column: '模型' }, value: 'Demo' }, { dataset: { column: key }, value: '13' }];
    f.context.$$ = selector => selector.includes('data-material') ? f.links : inputs;
    f.context.schema = { RL: ['环境信息', '日志存档', 'profiling采集'] };
    f.state.parsed = { metrics: { [key]: { mean: 0.13 } } };
    vm.runInContext(section('function currentRow()', 'function currentPreviewRow()') + '\n' +
      section('function validateCurrentRow()', 'function prepareCurrentRow()'), f.context);
    assert.equal(f.context.currentRow()[key], 0.13);
    assert.doesNotThrow(() => f.context.validateCurrentRow());
    inputs[1].value = '0.5'; assert.equal(f.context.currentRow()[key], 0.005);
    inputs[1].value = '100'; assert.doesNotThrow(() => f.context.validateCurrentRow());
    assert.equal(f.context.currentRow()[key], 1);
    inputs[1].value = '101'; assert.throws(() => f.context.validateCurrentRow(), /0–100/);
    inputs[1].value = ''; assert.equal(f.context.currentRow()[key], '');
  }
});

test('material links entered before the first log survive returning to review, but clear on replacement', async () => {
  const f = fixture();
  f.links[0].value = 'https://example.com/environment.txt';
  f.links[1].value = 'https://example.com/run.log';
  await f.choose(file('step:1 - perf/throughput:131'));
  await f.analyze();
  vm.runInContext(section('function currentRow()', 'function currentPreviewRow()'), f.context);
  assert.equal(f.context.currentRow()['环境信息'], f.links[0].value);
  assert.equal(f.context.currentRow()['日志存档'], f.links[1].value);
  f.state.parsed = { ...f.state.parsed, range: { start: 1, end: 1 } };
  await f.analyze();
  assert.equal(f.context.currentRow()['环境信息'], 'https://example.com/environment.txt');
  f.links[0].value = 'https://example.com/new.txt';
  assert.equal(f.context.currentRow()['环境信息'], 'https://example.com/new.txt');
  await f.choose(file('step:1 - perf/throughput:222'));
  assert.ok(f.links.every(input => input.value === ''));
});

test('invalid material links return to their first-page field and prevent analysis', async () => {
  const f = fixture(); await f.choose(file('step:1 - perf/throughput:131'));
  for (const value of ['http://example.com', 'not a url', 'javascript:alert(1)', 'https:example.com']) {
    f.links[0].value = value;
    await f.analyze();
    assert.equal(f.state.parsed, null);
    assert.equal(f.context.currentStage, 1);
    assert.equal(f.links[0].reported, true);
    assert.match(f.links[0].error, /HTTPS/);
  }
  f.links[0].value = ''; await f.analyze();
  assert.equal(f.state.parsed.count, 1);
});

test('cancelling replacement retains the parsed draft, links and saved identity', async () => {
  const f = fixture(); const first = file('step:1 - perf/throughput:131');
  await f.choose(first); await f.analyze();
  f.links[0].value = 'https://example.com/env';
  const parsed = f.state.parsed;
  f.context.allowLeave = false;
  await f.choose(file('step:1 - perf/throughput:222'));
  assert.equal(f.state.log, first); assert.equal(f.state.parsed, parsed);
  assert.equal(f.links[0].value, 'https://example.com/env');
});

test('dirty protection covers draft-only links, pending ranges and clean history', () => {
  const f = fixture();
  assert.equal(f.context.hasUnsavedChanges(), false);
  f.links[0].value = 'https://example.com/env'; assert.equal(f.context.hasUnsavedChanges(), true);
  f.links[0].value = '';
  f.state.parsed = { range: { start: 1, end: 10 } };
  f.$('#range-start').value = '1'; f.$('#range-end').value = '10';
  f.state.savedRecord = { fingerprint: 'draft' };
  assert.equal(f.context.hasUnsavedChanges(), false);
  f.$('#range-start').value = '2'; assert.equal(f.context.hasUnsavedChanges(), true);
  f.$('#range-start').value = '1'; f.context.currentFingerprint = () => 'modified';
  assert.equal(f.context.hasUnsavedChanges(), true);
});

test('copy/save validation never applies a pending range or overwrites edited metrics', () => {
  const f = fixture(); const log = file('step:1 - perf/throughput:131');
  Object.assign(f.state, { log, analyzedLog: log, parsed: { range: { start: 1, end: 10 } } });
  let reprocessed = false;
  f.context.applyRange = () => { reprocessed = true; };
  f.context.validateCurrentRow = () => {};
  f.$('#record-form').reportValidity = () => true;
  vm.runInContext(section('function prepareCurrentRow()', 'function currentFingerprint()'), f.context);
  f.$('#range-start').value = '3'; f.$('#range-end').value = '10';
  assert.throws(() => f.context.prepareCurrentRow(), /范围尚未应用/);
  assert.equal(reprocessed, false);
  f.$('#range-start').value = '1'; assert.equal(f.context.prepareCurrentRow(), true);
});

test('history editing restores original identity, precise MFU, links and custom fields', async () => {
  for (const task of ['RL', 'SFT']) {
    const f = fixture(); let preserved;
    f.context.reset = () => Object.assign(f.state, { parsed: null, savedRecord: null });
    f.context.selectTask = task => { f.state.task = task; };
    f.context.renderReview = row => { preserved = row; };
    f.context.updateSaveStatus = () => {};
    vm.runInContext(section('function metricInputValues(', 'function renderReview('), f.context);
    const key = task === 'RL' ? '训练MFU' : 'MFU';
    const record = { id: 'existing-id', task, createdAt: 'original-date', log: file('original'),
      analysis: { records: [], range: { start: 2, end: 8 } }, row: {
        模型: 'Custom', 备注: 'user note', [key]: 0.12345678, 环境信息: 'https://example.com/env',
      } };
    await f.context.editRecord(record);
    assert.equal(f.state.savedRecord.id, 'existing-id');
    assert.equal(f.state.savedRecord.createdAt, 'original-date');
    assert.equal(f.state.editingRecord, record); assert.equal(f.state.log, record.log);
    assert.equal(preserved[key], 12.345678); assert.equal(preserved['备注'], 'user note');
    assert.equal(f.links[0].value, record.row['环境信息']);
    assert.equal(f.context.historyFormValues({ [key]: '13.75%' })[key], '13.75');
    assert.equal(f.context.historyFormValues({ [key]: '' })[key], '');
  }
});

test('saving an edited history row updates its ID and preserves original attachments; failure keeps the draft', async () => {
  for (const fails of [false, true]) {
    const f = fixture();
    const original = { id: 'same-id', createdAt: 'first-date', log: file('original'), environment: 'old-env', profiles: ['old-profile'] };
    Object.assign(f.state, { log: original.log, editingRecord: original, savedRecord: { id: original.id, createdAt: original.createdAt },
      parsed: { metrics: { '训练MFU': {}, '吞吐TPS': {} } }, initial: { '训练MFU': 13, '吞吐TPS': 131 } });
    let captured, reset = false, historyShown = false;
    Object.assign(f.context, {
      prepareCurrentRow: () => true, currentRow: () => ({ '训练MFU': 0.255, '吞吐TPS': 131, 备注: 'edited' }),
      date: () => '2026-11-16', updateSaveStatus() {}, refreshRecords: async () => {},
      saveRecord: async record => { if (fails) throw new Error('storage unavailable'); captured = record; },
      reset: () => { reset = true; }, showRecords: async () => { historyShown = true; },
    });
    vm.runInContext(section("$('#save-history').addEventListener", 'function download('), f.context);
    const save = () => f.$('#save-history').click({ currentTarget: f.$('#save-history') });
    if (fails) {
      await assert.rejects(save, /storage unavailable/);
      assert.equal(reset, false); assert.equal(f.state.log, original.log);
    } else {
      await save();
      assert.equal(captured.id, original.id); assert.equal(captured.createdAt, original.createdAt);
      assert.equal(captured.environment, original.environment); assert.equal(captured.profiles, original.profiles);
      assert.equal(captured.row['训练MFU'], 0.255); assert.equal(captured.row['备注'], 'edited');
      assert.deepEqual([...captured.editedMetrics], ['训练MFU']);
      assert.equal(reset, true); assert.equal(historyShown, true);
    }
  }
});


test('draft dialog only permits leaving after explicit discard', async () => {
  const f = fixture(); f.links[0].value = 'https://example.com/env';
  vm.runInContext(section('async function confirmLeave()', "window.addEventListener('beforeunload'"), f.context);
  const dialog = f.$('#discard-dialog'); dialog.showModal = () => {};
  const keep = f.context.confirmLeave(); dialog.close();
  assert.equal(await keep, false);
  const discard = f.context.confirmLeave(); dialog.returnValue = 'discard'; dialog.close();
  assert.equal(await discard, true);
});

for (const [task, configLog] of [
  ['SFT', 'VeOmni\n{"model":{"model_path":"/models/Fixture-7B"},"data":{"max_seq_len":2048},"train":{}}'],
  ['RL', 'actor_num_nodes .... 1\nhf_checkpoint .... /models/Fixture-7B'],
]) test(`config-only ${task} results can be reviewed, copied and saved without a step range`, async () => {
  const f = fixture(); f.state.task = task;
  await f.choose(file(configLog));
  await f.analyze();
  assert.equal(f.context.currentStage, 2); assert.equal(f.state.parsed.totalCount, 0);
  vm.runInContext(section('function renderRangeStatus(', 'function renderReview('), f.context);
  f.context.renderRangeStatus(f.state.parsed);
  assert.equal(f.$('.range-panel').hidden, true); assert.equal(f.$('#range-start').disabled, true);
  assert.equal(f.$('#range-start').value, ''); assert.match(f.$('#coverage').textContent, /性能字段留空/);
  assert.equal(f.context.rangePending(), false);
  f.$('#record-form').reportValidity = () => true;
  f.context.validateCurrentRow = () => {};
  vm.runInContext(section('function prepareCurrentRow()', 'function currentFingerprint()'), f.context);
  assert.equal(f.context.prepareCurrentRow(), true);
  const row = { 模型: 'Fixture-7B', MFU: '', 吞吐TPS: '', 单步耗时s: '', 备注: '' };
  let captured, reset = false;
  Object.assign(f.context, { currentRow: () => row, date: () => '2026-11-16', crypto: { randomUUID: () => 'fixture' },
    updateSaveStatus() {}, refreshRecords: async () => {}, saveRecord: async record => { captured = record; },
    reset: () => { reset = true; }, showRecords: async () => {} });
  f.state.initial = { MFU: '', 吞吐TPS: '', 单步耗时s: '' };
  vm.runInContext(section("$('#save-history').addEventListener", 'function download('), f.context);
  await f.$('#save-history').click({ currentTarget: f.$('#save-history') });
  assert.equal(captured.task, task); assert.equal(captured.analysis.count, 0);
  assert.equal(captured.row.MFU, ''); assert.equal(captured.row['吞吐TPS'], ''); assert.equal(reset, true);
  const normal = parseLog('SFT', '{"task":"SFT","step":1,"tps_unit":"tokens/s","tps":100}');
  f.context.renderRangeStatus(normal);
  assert.equal(f.$('.range-panel').hidden, false); assert.equal(f.$('#range-start').disabled, false);
});
