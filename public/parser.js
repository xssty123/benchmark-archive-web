import { extractConfig } from './config.js';
// Explicit adapters only: do not infer task type, units or defaults from model names.
export const RL_FIELDS = {
  '训练MFU': 'perf/mfu/actor', '吞吐TPS': 'perf/throughput',
  '训练耗时 actor_update': 'timing_s/update_actor', '推理耗时': 'timing_s/gen',
  'resharding耗时': 'timing_s/update_weights', '单步耗时s': 'perf/time_per_step',
};
const SFT_FIELDS = { MFU: 'mfu', '吞吐TPS': 'tps', '单步耗时s': 'step_time_s' };
const ANSI = /\x1b\[[0-?]*[ -/]*[@-~]/g;

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
    const reason = label.includes('MFU') && unsupportedMFU ? '日志声明 MFU 不受支持' :
      missing.length || invalid.length ? `缺失 ${missing.length} 步 / 无效 ${invalid.length} 步` : '';
    metrics[label] = { key, mean: reason ? null : values.reduce((a, b) => a + b / values.length, 0),
      validCount: values.length, missing, invalid, reason };
  }
  return { steps, count: steps.length, first: steps[0], last: steps.at(-1),
    gaps: steps.at(-1) - steps[0] + 1 - steps.length,
    sources: Object.fromEntries([...rows].map(([step, entry]) => [step, entry.lines])), metrics };
}

export function parseRL(text) {
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
        const value = /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?$/i.test(raw) ? Number(raw) : NaN;
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

export function parseSFT(text) {
  const rows = new Map();
  let unit;
  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    let row;
    try { row = JSON.parse(line); } catch { throw new Error(`第 ${index + 1} 行不是标准化 JSONL；暂不支持直接解析任意 SFT 原始日志`); }
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
  return { ...all, ...metadata, records: [...rows], fields, unsupportedMFU,
    totalCount: all.count, totalFirst: all.first, totalLast: all.last,
    range: { start: all.first, end: all.last } };
}

export function selectRange(parsed, start = parsed.totalFirst, end = parsed.totalLast) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start < parsed.totalFirst || end > parsed.totalLast) {
    throw new Error(`统计范围须为 ${parsed.totalFirst}–${parsed.totalLast} 内的整数，开始步不能大于结束步`);
  }
  const rows = new Map(parsed.records.filter(([step]) => step >= start && step <= end));
  if (!rows.size) throw new Error('所选范围没有已记录训练步');
  const selected = summarize(rows, parsed.fields, parsed.unsupportedMFU);
  return { ...parsed, ...selected, gaps: end - start + 1 - selected.count, range: { start, end } };
}

export function parseLog(task, text) {
  if (task === 'RL') return parseRL(text);
  if (task === 'SFT') return parseSFT(text);
  throw new Error('未知任务类型');
}
