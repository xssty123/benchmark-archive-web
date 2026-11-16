import { parseLog, selectRange } from './parser.js';
import { makeCsv, zipRecord } from './archive.js';
import { saveRecord, listRecords, deleteRecord } from './storage.js';
import { clipboardRow } from './copy-row.js';
import { BATCH, COMPLETENESS, LAYERS, EXPERTS, modelDefaults, parseBatch, formatBatch, displayRow, formatMfu } from './record-fields.js';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
// Keep error feedback visible from both the wizard and the history view.
$('.topbar').after($('#alert'));
const schema = await fetch('./columns.json').then(response => {
  if (!response.ok) throw new Error('表头配置无法加载');
  return response.json();
});
const state = { task: 'RL', log: null, demo: false,
  parsed: null, analyzedLog: null, revision: 0, savedRecord: null, filter: 'RL', records: [] };
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const format = value => value == null ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 });
const date = () => new Date().toLocaleDateString('sv-SE');
function message(text) {
  $('#alert').textContent = text;
  $('#alert').classList.toggle('hidden', !text);
}
async function safely(button, action) {
  if (button.disabled) return;
  button.disabled = true;
  try { await action(); } catch (error) {
    message(error.message);
    $('#alert').scrollIntoView({ block: 'center', behavior: 'smooth' });
  } finally { button.disabled = false; }
}
function stage(n) {
  $('#new-view').classList.remove('hidden');
  $('#records-view').classList.add('hidden');
  ['upload', 'review'].forEach((id, i) => $('#stage-' + id).classList.toggle('hidden', i + 1 !== n));
  $$('.steps li').forEach(el => el.classList.toggle('current', Number(el.dataset.stage) === n));
  $('#demo').classList.toggle('hidden', n !== 1);
  $('#back-upload').classList.toggle('hidden', n !== 2);
  $('#nav-new').classList.add('active'); $('#nav-records').classList.remove('active');
  message('');
  window.scrollTo({ top: 0 });
}
function selectTask(task) {
  state.task = task;
  $$('.task').forEach(el => {
    el.classList.toggle('selected', el.dataset.task === task);
    el.setAttribute('aria-pressed', String(el.dataset.task === task));
  });
  $('#parser-hint').textContent = task === 'RL' ?
    'RL 支持 verl 的逐步文本日志，保留日志原有吞吐口径' :
    'SFT 原型仅支持标准化 JSONL（step / task / tps / tps_unit / step_time_s / mfu），真实框架日志适配待补充';
}
function renderFiles() {
  $('#analyze').textContent = state.parsed && state.analyzedLog === state.log ? '继续核对 →' : '解析并核对 →';
  $('#log-name').textContent = state.log?.name || '拖放运行日志到这里，或点击选择';
  $('#log-hint').textContent = state.log ? `${(state.log.size / 1024).toFixed(1)} KB · ${state.demo ? '合成演示数据，非实测结果' : '原始内容仅在本地读取'}` : '支持 .log / .txt / .jsonl，单文件不超过 50 MB';
}
function invalidateAnalysis() {
  state.revision++;
  Object.assign(state, { parsed: null, analyzedLog: null, savedRecord: null, initial: null, defaultModel: null });
  $('#record-form').replaceChildren();
  $('#resume-review').hidden = true;
}
function chooseLog(log) {
  if (log === state.log) return;
  invalidateAnalysis();
  state.log = log; state.demo = false;
  renderFiles(); message('');
}
function reset() {
  invalidateAnalysis();
  Object.assign(state, { log: null, demo: false });
  $$('.file-input').forEach(input => { input.value = ''; });
  state.initial = null; state.defaultModel = null;
  $('#record-form').replaceChildren();
  renderFiles(); stage(1);
}
$$('.task').forEach(el => el.addEventListener('click', () => { if (state.task !== el.dataset.task) { reset(); selectTask(el.dataset.task); } }));
$('#log-input').addEventListener('change', event => { if (event.target.files[0]) chooseLog(event.target.files[0]); });
$('#log-drop').addEventListener('dragover', event => { event.preventDefault(); });
$('#log-drop').addEventListener('drop', event => {
  event.preventDefault();
  if (event.dataTransfer.files.length !== 1) { message('每次请选择一份单次运行日志'); return; }
  chooseLog(event.dataTransfer.files[0]);
});
$('#demo').addEventListener('click', () => {
  reset(); state.demo = true;
  const lines = state.task === 'RL' ? Array.from({ length: 12 }, (_, i) =>
    `step:${i + 1} - perf/throughput:${120 + i * 2} - perf/time_per_step:${190 - i * 2} - timing_s/update_actor:${65 - i} - timing_s/gen:${100 - i} - timing_s/update_weights:4 - perf/mfu/actor:0.13`) :
    Array.from({ length: 10 }, (_, i) => JSON.stringify({ task: 'SFT', step: i + 1, tps: 21000 + i * 100,
      tps_unit: 'tokens/s', step_time_s: 9 - i * 0.1, mfu: 0.32 }));
  state.log = new File([lines.join('\n') + '\n'], `demo-${state.task.toLowerCase()}.${state.task === 'RL' ? 'log' : 'jsonl'}`, { type: 'text/plain' });
  renderFiles(); message('已加载合成演示数据，点击“解析并核对”继续');
});

function renderReview(preserved = {}) {
  const p = state.parsed;
  $('#review-title').textContent = `${state.task} · 核对性能结果${state.demo ? '（演示）' : ''}`;
  $('#coverage').textContent = `日志共 ${p.totalCount} 步（${p.totalFirst}–${p.totalLast}） · 当前统计 ${p.count} 步（${p.range.start}–${p.range.end}），逐步算术平均` +
    (p.gaps ? ` · 所选范围有 ${p.gaps} 个步号未记录` : '');
  for (const [id, value] of [['range-start', p.range.start], ['range-end', p.range.end]]) {
    const input = $('#' + id); input.min = p.totalFirst; input.max = p.totalLast; input.value = value;
  }
  const initial = Object.fromEntries(schema[state.task].map(key => [key, '']));
  const defaultName = preserved['模型'] ?? p.configuration?.fields?.['模型'];
  const defaults = modelDefaults(defaultName, defaultName === p.configuration?.fields?.['模型'] ? p.configuration?.modelStage : undefined);
  state.defaultModel = defaults.source;
  Object.assign(initial, defaults.fields);
  Object.assign(initial, p.configuration?.fields || {});
  initial[BATCH] = formatBatch(parseBatch(initial[BATCH]));
  Object.entries(p.metrics).forEach(([key, item]) => { initial[key] = item.mean == null ? '' : Number((item.mean * (key.includes('MFU') ? 100 : 1)).toFixed(6)); });
  initial[state.task === 'RL' ? 'A5场景' : '场景'] = state.task === 'RL' ? 'RL' : '';
  initial[state.task === 'RL' ? '更新时间' : '数据更新时间'] = date();
  initial['备注'] = '';
  if (state.demo) Object.assign(initial, { 系列: 'Demo', 模型: 'Demo-Model-30B', 参数: '30B', 集群规格: '8P',
    '满血/减层 关键配置': '满血', 序列: state.task === 'RL' ? '1k→8k' : '16k',
    '训练后端': state.task === 'RL' ? 'Megatron Core（演示）' : 'FSDP2（演示）' });
  // Recalculating a range updates metrics only; retain all user-edited configuration and URLs.
  Object.assign(initial, preserved);
  state.initial = initial;
  $('#parse-details').hidden = !p.configuration?.notes?.length;
  $('#parse-notes').textContent = (p.configuration?.notes || []).join('\n');
  $('#config-status').textContent = p.configuration?.warning || (p.configuration && Object.keys(p.configuration.fields).length ?
    `已从运行配置提取 ${Object.keys(p.configuration.fields).length} 项；默认满血，已知模型填入官方总层数和总专家数，请按实际运行核对` : '默认满血；未识别到的模型层数和专家数需填写，减层时改为实际数/总数');
  const metricKeys = Object.keys(p.metrics);
  const otherKeys = schema[state.task].filter(key => !metricKeys.includes(key) && !schema[state.task].slice(-3).includes(key));
  const groups = [['性能指标', metricKeys], ['模型与运行配置', otherKeys], ['在线材料链接', schema[state.task].slice(-3)]];
  let html = '<div class="form-grid">';
  for (const [title, keys] of groups) {
    if (title === '在线材料链接') html += `<section class="row-preview" aria-label="单行结果预览"><div class="row-preview-heading"><div><h3>单行结果</h3><p>与当前参数同步 · 按 ${state.task} 表头顺序 · 复制不含表头，可直接粘贴到 Excel / 在线表格</p></div><button type="button" class="button secondary" id="copy-row">复制整行</button></div><div id="row-preview-table" class="row-preview-scroll"></div><p id="copy-status" role="status" class="footnote"></p><textarea id="copy-fallback" class="hidden" readonly aria-label="手动复制整行内容"></textarea></section>`;
    html += `<h3 class="form-heading">${title}${title === '在线材料链接' ? '（可选）' : ''}</h3>`;
    if (title === '在线材料链接') html += '<p class="material-link-hint">材料请自行归档到团队云空间，链接会随整行复制；也可留空，直接在在线表格补充</p>';
    for (const key of keys) {
      const index = schema[state.task].indexOf(key), metric = p.metrics[key], link = index >= schema[state.task].length - 3;
      const source = p.configuration?.evidence?.[key];
      if (key === BATCH) {
        const parts = parseBatch(initial[key]);
        html += `<fieldset class="batch-fields"><legend>MBS / GBS / GAS</legend><input type="hidden" data-column="${BATCH}" value="${escape(formatBatch(parts))}"><div class="batch-grid">${['MBS · 每设备微批量', 'GBS · 全局批量', 'GAS · 梯度累积'].map((label, i) => `<div class="form-field"><label for="batch-${i}">${label}</label><input id="batch-${i}" data-batch="${i}" value="${escape(parts[i])}" placeholder="未记录"></div>`).join('')}</div><small>${escape(source || '未知项明确标为未记录，不用空斜杠或 0 代替')}${/动态/.test(parts[2]) ? '；动态 batch 无法据此得到固定梯度累积步数' : ''}</small></fieldset>`;
        continue;
      }
      const dimensions = [LAYERS, EXPERTS].includes(key);
      const fieldHint = key === COMPLETENESS ? '默认满血，可改为减层；减层时填写实际数/总数' : dimensions ?
        defaults.source ? `满血默认总数 · ${defaults.source.scope} · 专家数指每层路由专家总数` : '待填写总数；减层时填写实际数/总数' : '';
      html += `<div class="form-field"><label for="field-${index}">${escape(key)}${key.includes('MFU') ? '（%）' : ''}${key === '模型' ? ' *' : ''}</label><input id="field-${index}" data-column="${escape(key)}" value="${escape(initial[key])}" ${link ? 'type="url" placeholder="可留空，或粘贴云空间 HTTPS 链接"' : metric ? `type="number" min="0" step="any"${key.includes('MFU') ? ' max="100"' : ''}` : 'type="text" placeholder="未提供，留空"'} ${key === '模型' ? 'required' : ''}>${metric ? `<small>${escape(metric.key)} · ${metric.validCount}/${p.count} 步${metric.reason ? ' · ' + escape(metric.reason) : ''}</small>` : source ? `<small>来源：${escape(source)}</small>` : ''}</div>`;
      if (fieldHint) html = html.slice(0, -6) + `<small>${escape(fieldHint)}${dimensions && defaults.source ? ` · <a href="${defaults.source.url}" target="_blank" rel="noopener noreferrer">模型配置 ↗</a>` : ''}</small></div>`;
    }
  }
  $('#record-form').innerHTML = html + '</div><p class="footnote">A5/A3 填可比运行的吞吐比，无基线留空' +
    (state.task === 'RL' ? ' · resharding 当前采用 update_weights 阶段总耗时' : '') +
    ' · 备注仅保留你填写的内容</p>';
  renderRowPreview();
  $('#copy-row').addEventListener('click', event => safely(event.currentTarget, copyCurrentRow));
}
async function copyTableRow(task, row, statusSelector, fallbackSelector) {
  const content = clipboardRow(schema[task], row);
  const status = $(statusSelector), fallback = $(fallbackSelector);
  fallback.classList.add('hidden');
  try {
    if (navigator.clipboard?.write && typeof ClipboardItem !== 'undefined') {
      await navigator.clipboard.write([new ClipboardItem({
        'text/plain': new Blob([content.text], { type: 'text/plain' }),
        'text/html': new Blob([content.html], { type: 'text/html' }),
      })]);
    } else if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(content.text);
    else throw new Error('Clipboard unavailable');
    status.textContent = `已复制 ${schema[task].length} 列，请粘贴到在线表格`;
  } catch {
    fallback.value = content.text; fallback.classList.remove('hidden'); fallback.focus(); fallback.select();
    status.textContent = '浏览器未允许自动复制，内容已选中，请按 Ctrl+C / ⌘C';
  }
}
async function copyCurrentRow() {
  if (!prepareCurrentRow()) return;
  renderRowPreview();
  await copyTableRow(state.task, currentPreviewRow(), '#copy-status', '#copy-fallback');
  $('#copy-footer-status').textContent = $('#copy-status').textContent;
}
function renderMetricCards(row) {
  const p = state.parsed, mfu = state.task === 'RL' ? '训练MFU' : 'MFU';
  $('#metric-cards').innerHTML = ['吞吐TPS', '单步耗时s', mfu].map(key => {
    const value = row[key], metric = p.metrics[key];
    const empty = value === '' || value == null;
    const valid = !empty && Number.isFinite(Number(value)) && Number(value) >= 0 &&
      (key !== mfu || Number(value) <= 1) && (key !== '单步耗时s' || Number(value) > 0);
    const unit = key === '吞吐TPS' ? p.unit : key === mfu ? '百分比' : '秒 / step';
    const hint = empty ? metric.reason || '未填写' : valid ? unit : '请检查输入值';
    return `<div class="metric"><small>${escape(key)}</small><strong>${valid ? (key === mfu ? formatMfu(value) : format(value)) : '—'}</strong><small>${escape(hint)}</small></div>`;
  }).join('');
}
function currentRow() {
  const row = Object.fromEntries($$('#record-form [data-column]').map(input => [input.dataset.column, input.dataset.column === '备注' ? input.value : input.value.trim()]));
  // Inputs use percent units; stored records and parsed metrics retain ratios.
  for (const key of ['训练MFU', 'MFU']) {
    if (row[key] !== undefined && row[key] !== '') row[key] = Number(row[key]) / 100;
  }
  return row;
}
function currentPreviewRow() {
  return displayRow({ row: currentRow() });
}
function renderRowPreview() {
  const row = currentPreviewRow(), columns = schema[state.task];
  renderMetricCards(currentRow());
  $('#row-preview-table').innerHTML = `<table><thead><tr>${columns.map(key => `<th scope="col">${escape(key)}</th>`).join('')}</tr></thead><tbody><tr>${columns.map(key => `<td>${escape(row[key])}</td>`).join('')}</tr></tbody></table>`;
  $('#copy-status').textContent = '';
  $('#copy-footer-status').textContent = '可在上方复制整行；保存到历史记录后开始下一份';
  updateSaveStatus();
  $('#copy-fallback').classList.add('hidden');
}
$('#record-form').addEventListener('input', event => {
  if (event.target.dataset.batch !== undefined) {
    $('#record-form input[type="hidden"]').value = formatBatch($$('[data-batch]').map(input => input.value));
  }
  if (event.target.dataset.column !== undefined || event.target.dataset.batch !== undefined) renderRowPreview();
});
$('#record-form').addEventListener('change', event => {
  if (event.target.dataset.column !== '模型') { renderRowPreview(); return; }
  const previous = state.defaultModel;
  const next = modelDefaults(event.target.value.trim(), state.parsed.configuration?.modelStage).source;
  for (const [key, count] of [[LAYERS, 'layers'], [EXPERTS, 'experts']]) {
    const input = $(`#record-form [data-column="${key}"]`);
    if (!input.value || (previous && input.value === String(previous[count]))) input.value = next?.[count] ?? '';
    const help = input.parentElement.querySelector('small');
    if (help) help.textContent = next ? `满血默认总数 · ${next.scope}；减层时填写实际数/总数` : '待填写总数；减层时填写实际数/总数';
  }
  state.defaultModel = next;
  renderRowPreview();
});
function applyRange(start = Number($('#range-start').value), end = Number($('#range-end').value)) {
  if (!$('#range-start').value || !$('#range-end').value) throw new Error('请填写统计开始步数和结束步数');
  const selected = selectRange(state.parsed, start, end);
  const preserved = {};
  $$('#record-form [data-column]').forEach(input => {
    if (!(input.dataset.column in state.parsed.metrics)) preserved[input.dataset.column] = input.value;
  });
  state.parsed = selected;
  renderReview(preserved); message('');
}
$('#range-apply').addEventListener('click', event => safely(event.currentTarget, async () => applyRange()));
$('#range-reset').addEventListener('click', event => safely(event.currentTarget, async () => {
  $('#range-start').value = state.parsed.totalFirst;
  $('#range-end').value = state.parsed.totalLast;
  applyRange();
}));
$('#analyze').addEventListener('click', event => safely(event.currentTarget, async () => {
  if (!state.log) throw new Error('请先添加运行日志，或试用演示数据');
  if (state.log.size > 50 * 1024 ** 2) throw new Error('日志超过 50 MB，请使用完整的单次运行日志片段或后续服务端版本');
  if (state.parsed && state.analyzedLog === state.log) { stage(2); return; }
  const log = state.log, task = state.task, revision = state.revision;
  const text = await log.text();
  // A file/task change or reset while reading invalidates this pending parse.
  if (revision !== state.revision || log !== state.log || task !== state.task) return;
  const parsed = parseLog(task, text);
  state.parsed = parsed; state.analyzedLog = log; state.savedRecord = null;
  renderReview(); renderFiles(); stage(2);
}));
$('#back-upload').addEventListener('click', () => stage(1));
$('#record-form').addEventListener('submit', event => event.preventDefault());
function validateCurrentRow() {
  const row = {};
  $$('#record-form [data-column]').forEach(input => { row[input.dataset.column] = input.value.trim(); });
  if (!row['模型']) throw new Error('请补充模型名称');
  for (const key of Object.keys(state.parsed.metrics)) {
    if (row[key] === '') continue;
    const value = Number(row[key]);
    if (!Number.isFinite(value) || value < 0 || (key.includes('MFU') && value > 100) || (key === '单步耗时s' && value === 0)) {
      throw new Error(`${key}无效：耗时应大于零，MFU 使用 0–100 百分比数值（例如 13 表示 13%）`);
    }
  }
  for (const key of schema[state.task].slice(-3)) {
    if (row[key] && new URL(row[key]).protocol !== 'https:') throw new Error(`${key}须使用 HTTPS 链接；也可留空后在在线表格补充`);
  }
}

function prepareCurrentRow() {
  if (!state.parsed || state.analyzedLog !== state.log) throw new Error('日志已更换，请先解析当前日志');
  if (!$('#range-start').value || !$('#range-end').value || Number($('#range-start').value) !== state.parsed.range.start || Number($('#range-end').value) !== state.parsed.range.end) applyRange();
  if (!$('#record-form').reportValidity()) return false;
  validateCurrentRow();
  return true;
}
function currentFingerprint() {
  return JSON.stringify({ row: currentRow(), range: state.parsed.range });
}
function updateSaveStatus() {
  const saved = state.savedRecord;
  $('#save-history').textContent = '保存并解析下一份日志 →';
  $('#save-status').textContent = !saved ? '尚未保存到历史记录' :
    saved.fingerprint === currentFingerprint() ? '已保存到历史记录' : '有修改尚未保存，点击“保存并解析下一份日志”继续';
}
$('#save-history').addEventListener('click', event => safely(event.currentTarget, async () => {
  if (!prepareCurrentRow()) return;
  const previous = state.savedRecord;
  const row = currentRow();
  const now = new Date().toISOString();
  const record = {
    id: previous?.id || `${state.task}-${date().replaceAll('-', '')}-${crypto.randomUUID()}`,
    task: state.task, demo: state.demo, row, analysis: state.parsed,
    editedMetrics: Object.keys(state.parsed.metrics).filter(key => String(key.includes('MFU') && state.initial[key] !== '' ? Number(state.initial[key]) / 100 : state.initial[key]) !== String(row[key])),
    createdAt: previous?.createdAt || now, updatedAt: now, log: state.log,
  };
  const fingerprint = currentFingerprint();
  await saveRecord(record);
  if (state.log !== record.log) { await refreshRecords(); return; }
  state.savedRecord = { id: record.id, createdAt: record.createdAt, fingerprint };
  updateSaveStatus();
  message('');
  await refreshRecords();
  if (state.log !== record.log) return;
  if (currentFingerprint() !== fingerprint) {
    message('保存期间内容有修改，已保留当前页面，请再次保存后继续');
    return;
  }
  state.filter = record.task;
  $$('[data-filter]').forEach(el => el.classList.toggle('selected', el.dataset.filter === state.filter));
  reset();
  message('已保存到历史记录，请选择下一份日志');
}));

function download(blob, name) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
const exportZip = async record => download(await zipRecord(record), record.id + '.zip');
const exportCsv = record => download(new Blob([makeCsv(schema[record.task], displayRow(record))], { type: 'text/csv;charset=utf-8' }), record.id + '.csv');
async function refreshRecords() {
  state.records = (await listRecords()).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const count = state.records.filter(record => !record.deletedAt).length;
  $('#nav-count').textContent = count;

  renderHistory();
}
function renderHistory() {
  $('#history-copy-status').textContent = '';
  $('#history-copy-fallback').classList.add('hidden');
  const records = state.records.filter(record => record.task === state.filter && !record.deletedAt);
  if (!records.length) { $('#history').innerHTML = '<div class="empty">暂无该类型的历史记录，可在核对页点击“保存并解析下一份日志”</div>'; return; }
  // Full approved schema, including the three links at the end; horizontal scrolling keeps every field accessible.
  $('#history').innerHTML = `<table><thead><tr><th>操作</th>${schema[state.filter].map(key => `<th>${escape(key)}</th>`).join('')}</tr></thead><tbody>${records.map(record => {
    const row = displayRow(record);
    return `<tr><td><button data-copy-record="${record.id}">复制整行</button><button data-zip="${record.id}">本地备份</button><button data-csv="${record.id}">CSV</button><button class="delete-record" data-delete="${record.id}">删除</button>${record.demo ? '<span class="badge">演示</span>' : ''}</td>${schema[state.filter].map(key => `<td>${schema[state.filter].slice(-3).includes(key) && row[key] ? `<a href="${escape(row[key])}" target="_blank" rel="noopener noreferrer">打开 ↗</a>` : escape(row[key] === '' || row[key] == null ? '—' : row[key])}</td>`).join('')}</tr>`;
  }).join('')}</tbody></table>`;
  $$('[data-copy-record]').forEach(button => button.addEventListener('click', () => safely(button, async () => {
    const record = records.find(record => record.id === button.dataset.copyRecord);
    await copyTableRow(record.task, displayRow(record), '#history-copy-status', '#history-copy-fallback');
  })));
  $$('[data-zip]').forEach(button => button.addEventListener('click', () => safely(button, () => exportZip(records.find(record => record.id === button.dataset.zip)))));
  $$('[data-csv]').forEach(button => button.addEventListener('click', () => exportCsv(records.find(record => record.id === button.dataset.csv))));
  $$('[data-delete]').forEach(button => button.addEventListener('click', () => safely(button, async () => {
    const id = button.dataset.delete;
    await deleteRecord(id);
    if (state.savedRecord?.id === id) { state.savedRecord = null; updateSaveStatus(); }
    await showRecords();
    message('记录及本地附件已删除；已下载文件和在线材料不受影响');
  })));
}
async function showRecords() {
  await refreshRecords();
  $('#resume-review').hidden = !state.log;
  $('#resume-review').textContent = state.parsed ? '返回当前核对' : '返回日志选择';
  $('#new-view').classList.add('hidden'); $('#records-view').classList.remove('hidden');
  $('#nav-new').classList.remove('active'); $('#nav-records').classList.add('active');
}
$('#nav-new').addEventListener('click', reset);
$('#resume-review').addEventListener('click', () => {
  if (state.log) stage(state.parsed && state.analyzedLog === state.log ? 2 : 1);
});
$('#nav-records').addEventListener('click', event => safely(event.currentTarget, showRecords));
$$('[data-filter]').forEach(button => button.addEventListener('click', () => {
  state.filter = button.dataset.filter;
  $$('[data-filter]').forEach(el => el.classList.toggle('selected', el === button)); renderHistory();
}));
// History storage is optional for parsing and copying.
refreshRecords().catch(() => message('历史记录暂时无法读取，日志解析和复制仍可使用'));

$$('[data-copy-command]').forEach(button => button.addEventListener('click', event => safely(event.currentTarget, async () => {
  const command = $('#' + button.dataset.copyCommand).textContent;
  const fallback = $('#env-copy-fallback');
  fallback.classList.add('hidden');
  try {
    await navigator.clipboard.writeText(command);
    $('#env-copy-status').textContent = '命令已复制，请在实际训练环境的终端执行';
  } catch {
    fallback.value = command;
    fallback.classList.remove('hidden'); fallback.focus(); fallback.select();
    $('#env-copy-status').textContent = '内容已选中，请按 Ctrl+C / ⌘C 复制';
  }
})));
