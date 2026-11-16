// Verified standard HF implementation; this is a configuration-based classification,
// not a claim that an unversioned historical runtime was profiled.
export const OMNI_RECOMPUTE_NOTE = '重计算范围：Thinker 文本主干全部 Transformer 层（含 Attention、MoE），按标准 Transformers 实现判定；历史运行精确版本及自定义补丁未核验';
export function isOmniFullCheckpoint(model, stage, backend) {
  return model === 'Qwen3-Omni-30B-A3B-Instruct' && stage === 'thinker' && /^(?:PyTorch )?FSDP2?(?:\b|$)/i.test(backend || '');
}
export function normalizeRecompute(value, { model, stage, backend } = {}) {
  const text = String(value ?? '').trim();
  if (text === '开启梯度检查点') {
    return isOmniFullCheckpoint(model, stage, backend)
      ? { value: '全重计算', note: OMNI_RECOMPUTE_NOTE }
      : { value: '已开重计算（范围待核实）', note: '' };
  }
  if (text === '关闭梯度检查点') return { value: '未开重计算', note: '' };
  const match = text.match(/^(全重计算|selective|选择性重计算)(?:[，,](.*))?$/);
  if (match) return { value: match[1] === 'selective' ? '选择性重计算' : match[1], note: match[2] ? `重计算细节：${match[2]}` : '' };
  return { value: text, note: '' };
}
