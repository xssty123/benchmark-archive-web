import { normalizeRecompute } from './recompute.js';
export const BATCH = 'mbs/gbs/gas（gas=梯度累积）';
export const COMPLETENESS = '满血/减层 关键配置';
export const LAYERS = '层数/总层数（满血填总）';
export const EXPERTS = '专家/总专家（满血填总）';
// Official model defaults checked 2026-11-16. These are defaults, not measured run evidence.
const models = {
  'Qwen3-Omni-30B-A3B-Instruct': { layers: 48, experts: 128, scope: 'Thinker 文本主干',
    url: 'https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json' },
  'Qwen3.5-35B-A3B': { layers: 40, experts: 256, scope: '文本主干',
    url: 'https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json' },
};
export function modelDefaults(name, modelStage) {
  const model = models[name];
  // Omni's Talker has different dimensions; never reuse Thinker defaults for it.
  const compatible = model && !(name.startsWith('Qwen3-Omni') && modelStage && modelStage !== 'thinker');
  return { fields: { [COMPLETENESS]: '满血', ...(compatible ? { [LAYERS]: model.layers, [EXPERTS]: model.experts } : {}) },
    source: compatible ? model : null };
}
export function parseBatch(value, dynamic = false) {
  const text = String(value ?? '');
  const labels = ['MBS', 'GBS', 'GAS'];
  const parts = /MBS\s*=/.test(text) ? labels.map(label => text.match(new RegExp(`${label}\\s*=([^；;]*)`))?.[1]?.trim() || '') : text.split('/');
  return parts.slice(0, 3).concat(Array(3).fill('')).slice(0, 3)
    .map((part, i) => part.trim() || (i === 2 && dynamic ? '动态（未推算）' : '未记录'));
}
export function formatBatch(parts) {
  return ['MBS', 'GBS', 'GAS'].map((label, i) => `${label}=${String(parts[i] ?? '').trim() || '未记录'}`).join('；');
}
// Old records keep ratios; percent strings are also accepted without scaling twice.
export function formatMfu(value) {
  if (value == null || String(value).trim() === '') return '';
  const text = String(value).trim();
  const percent = text.endsWith('%') ? Number(text.slice(0, -1)) : Number(text) * 100;
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) return '';
  return `${Number(percent.toFixed(2))}%`;
}
export function displayRow(record) {
  const parts = parseBatch(record.row[BATCH], /动态\s*batch/i.test(record.row['备注'] || ''));
  const batch = parts.map(value => {
    if (/^(未配置|未记录|nan)$/i.test(value)) return 'Nan';
    if (/^动态|^dyn$/i.test(value)) return 'dyn';
    return value;
  }).join('/');
  const row = { ...record.row, [BATCH]: batch };
  for (const key of ['训练MFU', 'MFU']) {
    if (key in row) row[key] = formatMfu(row[key]);
  }
  if (row['重计算']) {
    const normalized = normalizeRecompute(row['重计算'], {
      model: row['模型'], backend: row['训练后端'], stage: record.analysis?.configuration?.modelStage,
    });
    row['重计算'] = normalized.value;
  }
  return row;
}
