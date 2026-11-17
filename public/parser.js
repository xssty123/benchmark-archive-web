import { extractConfig } from './config.js';
import { isVeOmniLog, extractVeOmniConfig, veomniMetricRecords } from './veomni.js';
import { isSlimeLog, extractSlimeConfig, slimeRecords, SLIME_FIELDS } from './slime.js';
// Explicit adapters only: do not infer task type, units or defaults from model names.
export const RL_FIELDS = {
  '训练MFU': 'perf/mfu/actor', '吞吐TPS': 'perf/throughput',
  '训练耗时 actor_update': 'timing_s/update_actor', '推理耗时': 'timing_s/gen',
  'resharding耗时': 'timing_s/update_weights', '单步耗时s': 'perf/time_per_step',
};
const SFT_FIELDS = { MFU: 'mfu', '吞吐TPS': 'tps', '单步耗时s': 'step_time_s' };
const ANSI = /\x1b\[[0-?]*[ -/]*[@-~]/g;
const STATISTICS_NOTE = '各指标分别对所选范围内实际记录的步数取算术平均，默认包含全部已记录步。个别步缺字段或为 null 时不补 0，也不清空已有均值；每项显示有效、缺失及无效步数。完全未记录、包含无效数值或日志明确声明不受支持的指标留空并提示';

function addStep(rows, step, values, line) {
  if (!Number.isSafeInteger(step) || step < 0) throw new Error(`第 ${line} 行步号无效`);
  const previous = rows.get(step) || { values: {}, lines: [] };
  for (const [key, value] of Object.entries(values)) {
    if (key in previous.values && !Object.is(previous.values[key], value)) {
      throw new Error(`第 ${line} 行的 step ${step} 存在冲突值（${key}），请先拆分重启或混合运行日志`);
    }
    previous.values[key] = value;
  }
  previous.lines.push(line);
  rows.set(step, previous);
}

function summarize(rows, fields, unsupportedMFU = false) {
  if (!rows.size) throw new Error('未找到支持的逐步训练指标，请检查任务类型与日志格式');
  const steps = [...rows.keys()].sort((a, b) => a - b);
  const metrics = {};
  for (const [label, key] of Object.entries(fields)) {
    const missing = [], invalid = [], values = [];
    for (const step of steps) {
      const value = rows.get(step).values[key];
      if (value === undefined || value === null) missing.push(step);
      else if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 ||
        (label === '单步耗时s' && value === 0) || (label.includes('MFU') && value > 1)) invalid.push(step);
      else values.push(value);
    }
    // One policy for every adapter and range, including saved analyses from older versions.
    const unsupported = label.includes('MFU') && unsupportedMFU;
    const canAverage = values.length > 0 && !invalid.length && !unsupported;
    const reason = unsupported ? '日志声明 MFU 不受支持' :
      canAverage && missing.length ? `该指标未记录 ${missing.length} 步，仅平均实际记录的 ${values.length} 步` :
      missing.length || invalid.length ? `缺失 ${missing.length} 步 / 无效 ${invalid.length} 步` : '';
    metrics[label] = { key, mean: canAverage ? values.reduce((a, b) => a + b / values.length, 0) : null,
      validCount: values.length, missing, invalid, reason };
  }
  return { steps, count: steps.length, first: steps[0], last: steps.at(-1),
    gaps: steps.at(-1) - steps[0] + 1 - steps.length,
    sources: Object.fromEntries([...rows].map(([step, entry]) => [step, entry.lines])), metrics };
}

export function parseRL(text) {
  if (isSlimeLog(text)) {
    if (/\bstep\s*:\s*\d+\s+-\s+perf\/throughput/.test(text)) throw new Error('同时发现 slime 和 verl 日志，请先拆分不同运行记录');
    return parseSlimeRL(text);
  }
  const rows = new Map();
  const lines = text.replace(ANSI, '').split(/\r?\n/);
  lines.forEach((line, index) => {
    const starts = [...line.matchAll(/(?<![\w/])step\s*:\s*(\d+)\s+-\s+/g)];
    starts.forEach((start, i) => {
      const body = line.slice(start.index + start[0].length, starts[i + 1]?.index);
      const values = {};
      for (const part of body.split(/\s+-\s+/)) {
        const match = part.match(/^([\w./@-]+):\s*(.*)$/);
        if (!match || !Object.values(RL_FIELDS).includes(match[1])) continue;
        const raw = match[2].trim().replace(/^(?:np|numpy)\.(?:float\d*|u?int\d*|longdouble)\(\s*(.*?)\s*\)$/, '$1');
        const value = /^(?:None|null)$/.test(raw) ? null : /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?$/i.test(raw) ? Number(raw) : NaN;
        if (match[1] in values && !Object.is(values[match[1]], value)) throw new Error(`第 ${index + 1} 行指标重复且冲突`);
        values[match[1]] = value;
      }
      if (Object.keys(values).length) addStep(rows, Number(start[1]), values, index + 1);
    });
  });
  return withRecords(rows, RL_FIELDS, /MFU will always be zero/i.test(text), {
    unit: 'tokens/s/device', adapter: 'verl-text-v1', configuration: extractConfig(text),
  });
}

function parseSlimeRL(text) {
  const configuration = extractSlimeConfig(text), parsed = slimeRecords(text), rows = new Map();
  for (const record of parsed.records) addStep(rows, record.step, record.values, record.line);
  configuration.notes.push(...parsed.notes);
  const metadata = { adapter: 'slime-text-v1', unit: 'tokens/s（训练阶段，全局）', configuration, independentMetrics: true };
  if (rows.size) return withRecords(rows, SLIME_FIELDS, false, metadata);
  const reason = '未找到受支持的 slime perf 指标，缺失字段留空';
  configuration.notes.push('未识别到 perf <rollout_id>: {...} 中可填写的性能字段；仅有 TFLOPS、生成吞吐或无步号 Timer 输出时不推算主表指标');
  return { ...metadata, count: 0, steps: [], first: null, last: null, gaps: 0, sources: {},
    records: [], fields: SLIME_FIELDS, unsupportedMFU: false, totalCount: 0, totalFirst: null, totalLast: null,
    range: { start: null, end: null }, metrics: Object.fromEntries(Object.entries(SLIME_FIELDS).map(([label, key]) =>
      [label, { key, mean: null, validCount: 0, missing: [], invalid: [], reason }])) };
}

export function parseSFT(text) {
  const first = text.replace(/^\uFEFF/, '').split(/\r?\n/).find(line => line.trim()) || '';
  if (/"task"\s*:/.test(first)) return parseNormalizedSFT(text.replace(/^\uFEFF/, ''));
  if (isVeOmniLog(text)) return parseVeOmniSFT(text);
  return parseNormalizedSFT(text);
}

function parseVeOmniSFT(text) {
  const configuration = extractVeOmniConfig(text), rows = new Map();
  const parsed = veomniMetricRecords(text);
  for (const record of parsed.records) addStep(rows, record.step, record.values, record.line);
  configuration.notes.push(...parsed.notes,
    'VeOmni 吞吐使用 tokens_per_second(M) × 1,000,000，单位为全局 tokens/s；MFU 原始值为比例，页面显示百分比',
    'TPS、MFU 仅采用带明确步号的指标字典，缺失时留空，不从进度条推算；单步耗时优先采用显式 step_time_s');
  let estimatedStepTime = false;
  if (![...rows.values()].some(row => Object.hasOwn(row.values, 'step_time_s'))) {
    const progress = veomniProgressTimes(text);
    configuration.notes.push(...progress.notes);
    for (const record of progress.records) addStep(rows, record.step, { step_time_s: record.seconds }, record.line);
    estimatedStepTime = progress.records.length > 0;
    if (estimatedStepTime) configuration.notes.push('单步耗时为估算：取所选范围内进度条已打印的有效 s/it 的算术平均；该速率默认经过平滑，并非逐步实测耗时。相同步数重复刷新只取最后一个有效值；没有 s/it 的步骤不参与耗时均值，不补 0。多 epoch 步号按 (epoch−1)×每 epoch 总步数+当前步展开');
  }
  const metadata = { unit: 'tokens/s', adapter: 'veomni-text-v1', configuration, estimatedStepTime };
  if (rows.size) return withRecords(rows, SFT_FIELDS, /Returning zero FLOPs|MFU will always be zero/i.test(text), metadata);
  const reason = '日志未记录可用的逐步性能数据';
  configuration.notes.push('本地核对的 VeOmni 默认终端日志不打印完整效率指标，W&B 中的指标不会自动出现在终端日志里');
  return { ...metadata, count: 0, steps: [], first: null, last: null, gaps: 0, sources: {},
    records: [], fields: SFT_FIELDS, unsupportedMFU: false, totalCount: 0, totalFirst: null, totalLast: null,
    range: { start: null, end: null },
    metrics: Object.fromEntries(Object.entries(SFT_FIELDS).map(([key, source]) => [key,
      { key: source, mean: null, validCount: 0, missing: [], invalid: [], reason }])) };
}

function veomniProgressTimes(text) {
  const samples = new Map(), notes = [];
  let total, epochs;
  // Match training Epoch bars only, not validation/loading bars or loss postfix numbers.
  const pattern = /\bEpoch\s+(\d+)\s*\/\s*(\d+)\s*:[^\r\n]*?\|\s*(\d+)\s*\/\s*(\d+)\s*\[[^\]\r\n]*?,\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*s\/it(?=\s*[,\]])/gi;
  const lines = text.replace(ANSI, '').split(/\r\n|\r|\n/);
  for (const [index, line] of lines.entries()) {
    for (const match of line.matchAll(pattern)) {
      const [epoch, epochCount, current, size, seconds] = match.slice(1).map(Number);
      const step = (epoch - 1) * size + current;
      if (![epoch, epochCount, current, size, step].every(Number.isSafeInteger) || epoch < 1 || epoch > epochCount ||
        current < 1 || current > size || !Number.isFinite(seconds) || seconds <= 0) continue;
      if ((total !== undefined && total !== size) || (epochs !== undefined && epochs !== epochCount)) {
        return { records: [], notes: ['进度条的 epoch 总数或每 epoch 总步数不一致，无法统一步号，未估算单步耗时'] };
      }
      total = size; epochs = epochCount;
      samples.set(step, { step, seconds, line: index + 1 });
    }
  }
  return { records: [...samples.values()], notes };
}

function parseNormalizedSFT(text) {
  const rows = new Map();
  let unit;
  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    let row;
    try { row = JSON.parse(line); } catch { throw new Error(`第 ${index + 1} 行不是标准化 JSONL；支持 VeOmni 原始日志或标准化 SFT JSONL，未识别当前格式`); }
    if (!row || row.task !== 'SFT' || !['tokens/s', 'tokens/s/device'].includes(row.tps_unit)) {
      throw new Error(`第 ${index + 1} 行须明确 task: SFT 和 tps_unit: tokens/s 或 tokens/s/device`);
    }
    if (unit && unit !== row.tps_unit) throw new Error('同一日志的吞吐单位不一致，请先核对');
    unit = row.tps_unit;
    const values = Object.fromEntries(Object.values(SFT_FIELDS).filter(key => key in row).map(key => [key, row[key]]));
    addStep(rows, row.step, values, index + 1);
  });
  return withRecords(rows, SFT_FIELDS, false, { unit, adapter: 'sft-normalized-jsonl-v1' });
}

function withRecords(rows, fields, unsupportedMFU, metadata) {
  const all = summarize(rows, fields, unsupportedMFU);
  return { ...all, ...metadata, independentMetrics: true, statisticsNote: STATISTICS_NOTE, records: [...rows], fields, unsupportedMFU,
    totalCount: all.count, totalFirst: all.first, totalLast: all.last,
    range: { start: all.first, end: all.last } };
}

export function selectRange(parsed, start = parsed.totalFirst, end = parsed.totalLast) {
  if (parsed.totalCount === 0) throw new Error('没有已记录性能步骤，无需选择统计范围');
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start < parsed.totalFirst || end > parsed.totalLast) {
    throw new Error(`统计范围须为 ${parsed.totalFirst}–${parsed.totalLast} 内的整数，开始步不能大于结束步`);
  }
  const rows = new Map(parsed.records.filter(([step]) => step >= start && step <= end));
  if (!rows.size) throw new Error('所选范围没有已记录训练步');
  const selected = summarize(rows, parsed.fields, parsed.unsupportedMFU);
  return { ...parsed, ...selected, independentMetrics: true, statisticsNote: STATISTICS_NOTE,
    gaps: end - start + 1 - selected.count, range: { start, end } };
}

export function parseLog(task, text) {
  if (task === 'RL') return parseRL(text);
  if (task === 'SFT') return parseSFT(text);
  throw new Error('未知任务类型');
}
