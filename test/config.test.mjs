import test from 'node:test';
import assert from 'node:assert/strict';
import { extractConfig, readLiteral } from '../public/config.js';
import { parseRL } from '../public/parser.js';

const actor = { actor: { _target_: 'example.VeOmniActorConfig', strategy: 'veomni', ppo_micro_batch_size_per_gpu: 1,
  ppo_mini_batch_size: 8, use_dynamic_bsz: true }, model: { path: '/models/Qwen-Fixture-27B', enable_gradient_checkpointing: true },
  rollout: { n: 2, name: 'vllm' } };
const trainer = { nnodes: 2, n_gpus_per_node: 8 };
const config = { trainer, data: { train_batch_size: 8, max_prompt_length: 1024, max_response_length: 8192 }, actor_rollout_ref: actor };
const dump = JSON.stringify(config, null, 2);
const metrics = 'step:1 - perf/throughput:100 - perf/time_per_step:10';

test('top-level and nested key order do not affect extraction or duplicate comparison', () => {
  const reordered = { actor_rollout_ref: { rollout: actor.rollout, model: actor.model, actor: actor.actor },
    data: config.data, trainer: { n_gpus_per_node: 8, nnodes: 2 } };
  const result = extractConfig(dump + '\n' + JSON.stringify(reordered));
  assert.equal(result.warning, '');
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.fields['训练后端'], 'VeOmni');
  assert.equal(result.fields['集群规格'], '16P');
  assert.equal(result.fields['序列'], '1k→8k');
});

test('Ray, rank, timestamp, level and ANSI prefixes are accepted with CRLF and BOM', () => {
  const prefixes = [
    '\x1b[36m(TaskRunnerV1 pid=123)\x1b[0m ',
    '\\u001b[36m(TaskRunnerV1 pid=123)\\u001b[0m ',
    '\\x1b[36m(TaskRunnerV1 pid=123)\\x1b[0m ',
    '[36m(TaskRunnerV1 pid=123) ', '[rank0]: ', '[rank 1] ',
    '2026-01-02 03:04:05,123 INFO ', '[2026-01-02T03:04:05Z] [INFO] [rank0]: ',
    '  [rank0]: \x1b[36m(TaskRunnerV1 pid=123, ip=127.0.0.1)\x1b[0m ',
  ];
  for (const prefix of prefixes) {
    const result = extractConfig('\uFEFF' + dump.split('\n').map(line => prefix + line).join('\r\n'));
    assert.equal(result.warning, '', prefix);
    assert.equal(result.fields['模型'], 'Qwen-Fixture-27B', prefix);
  }
});

test('Python pprint literals and string contents are preserved without executing objects', () => {
  const text = "{'data': {'description': '[rank0] {INFO}: not a prefix'}, 'trainer': {'nnodes': 1, 'n_gpus_per_node': 8},\n" +
    "'actor_rollout_ref': {'model': {'path': '/models/' 'Qwen-Fixture-27B'}, 'actor': {'strategy': 'veomni', 'enabled': True, 'unset': None}}}";
  assert.equal(extractConfig(text).fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(readLiteral("{'text':'[rank0] {INFO}: not a prefix'}").value.text, '[rank0] {INFO}: not a prefix');
  const unsupported = text.replace("'unset': None", "'unset': dangerous_function()");
  assert.match(extractConfig(unsupported).warning, /第 2 行.*暂不支持/);
});

test('missing, YAML, truncated, malformed and missing-trainer configs have distinct diagnostics', () => {
  const cases = [
    ['step:1 - perf/throughput:1', /未找到运行配置/],
    ['actor_rollout_ref:\n  model:\n    path: example', /YAML、启动参数/],
    ['+ python train.py actor_rollout_ref.model.path=example', /YAML、启动参数/],
    ["{'actor_rollout_ref': {'model': {'path': 'unfinished", /第 1 行.*未闭合/],
    ["{'actor_rollout_ref': {'model': Path('example')}, 'trainer': {}}", /第 1 行.*暂不支持的数据类型/],
    [JSON.stringify({ actor_rollout_ref: actor }), /缺少有效的.*trainer/],
    ["{'actor_rollout_ref': {}, 'trainer': {}, 'trainer': {}}", /重复字段/],
  ];
  for (const [text, warning] of cases) {
    const result = extractConfig(text);
    assert.deepEqual(result.fields, {});
    assert.match(result.warning, warning);
  }
});

test('different configs and incomplete additional dumps never silently choose one run', () => {
  const result = extractConfig(dump + '\n' + JSON.stringify({ ...config, trainer: { nnodes: 3 } }));
  assert.match(result.warning, /多份不同运行配置.*起始行/);
  assert.deepEqual(result.fields, {});
  const partial = extractConfig(dump + "\n{'actor_rollout_ref': {");
  assert.match(partial.warning, /未闭合/); assert.deepEqual(partial.fields, {});
});

test('config errors do not prevent throughput parsing and do not enter remarks', () => {
  for (const configText of [dump, "{'actor_rollout_ref': {'model': ", 'nothing here']) {
    const result = parseRL(configText + '\n' + metrics);
    assert.equal(result.metrics['吞吐TPS'].mean, 100);
    assert.equal(result.configuration.fields['备注'], undefined);
  }
});

test('unrecognized per-line prefixes and missing model paths are explained', () => {
  const prefixed = dump.split('\n').map(line => 'CUSTOM_LOG | ' + line).join('\n');
  assert.match(extractConfig(prefixed).warning, /第 \d+ 行.*前缀/);
  const noModel = extractConfig(JSON.stringify({ trainer, actor_rollout_ref: { actor: actor.actor } }));
  assert.match(noModel.notes.join('\n'), /model.path 缺失/);
});


test('unrelated log braces do not hide a later complete config', () => {
  const result = extractConfig('progress { pending\n' + dump + '\n' + metrics);
  assert.equal(result.warning, '');
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
});

const ray = '\x1b[36m(TaskRunnerV1 pid=123)\x1b[0m ';
const worker = '\x1b[36m(WorkerDict pid=456)\x1b[0m ';
const warningNoise = [
  ray + '/fixture/trainer.py:129: UserWarning: Disabled critic as algorithm.adv_estimator != gae',
  ray + '  self.use_critic = need_critic(self.config)',
  ray + 'INFO:2026-01-02 03:04:05,690:train and validate dataloader initialized',
  ray + "INFO:2026-01-02 03:04:05,693:worker group kwargs: {'device_name': 'npu'}",
  worker + "/fixture/engine.py:32: UserWarning: engine unavailable: ModuleNotFoundError(\"No module named 'fixture'\")",
  worker + '  warnings.warn(f"engine unavailable: {e!r}", stacklevel=1)',
];
const actorFirst = JSON.stringify({ actor_rollout_ref: actor, data: config.data });
const startTrainer = actorFirst.slice(0, -1) + ", 'trainer': {'del_local_ckpt_after_load': False,";

test('same-line Ray warnings, source echoes and INFO output do not break a complete config', () => {
  const text = ray + startTrainer + warningNoise[0] + '\n' + warningNoise.slice(1).join('\n') + '\n' +
    ray + "'nnodes': 2, 'n_gpus_per_node': 8}}\n" + metrics;
  const result = parseRL(text);
  assert.equal(result.configuration.warning, '');
  assert.equal(result.configuration.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.configuration.fields['集群规格'], '16P');
  assert.equal(result.configuration.fields['训练后端'], 'VeOmni');
  assert.match(result.configuration.notes.join('\n'), /交错.*原日志行 1/);
  assert.equal(result.metrics['吞吐TPS'].mean, 100);
});

test('interrupted config retains only complete actor fields without inventing trainer values', () => {
  const text = ray + startTrainer + warningNoise[0] + '\n' + warningNoise.slice(1).join('\n');
  const result = extractConfig(text);
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.fields['训练后端'], 'VeOmni');
  assert.equal(result.fields['集群规格'], undefined);
  assert.equal(result.fields['同步/异步'], undefined);
  assert.match(result.warning, /未闭合.*仅恢复已完整解析/);
});

test('unknown values preserve earlier verified fields but not partial expressions or lists', () => {
  const text = startTrainer + "'nnodes': 2 + unknown(), 'n_gpus_per_node': 8}}";
  const result = extractConfig(text);
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.fields['集群规格'], undefined);
  assert.match(result.warning, /仅恢复已完整解析/);
  try {
    readLiteral("{'safe': 7, 'bad': [1, unsupported()], 'later': 9}");
    assert.fail('must reject unsupported value');
  } catch (error) {
    assert.equal(error.partial.safe, 7);
    assert.equal(error.partial.bad, undefined);
    assert.equal(error.partial.later, undefined);
  }
});

test('duplicate keys and multiple runs still fail instead of selecting a partial result', () => {
  const duplicate = startTrainer + "'nnodes': 2, 'nnodes': 3, 'n_gpus_per_node': 8}}";
  assert.deepEqual(extractConfig(duplicate).fields, {});
  assert.match(extractConfig(duplicate).warning, /重复字段/);
  const mixed = dump + '\n' + startTrainer;
  assert.deepEqual(extractConfig(mixed).fields, {});
  assert.match(extractConfig(mixed).warning, /多份不同运行配置/);
});

test('literal-looking output from another Ray worker cannot supply trainer fields', () => {
  const text = ray + startTrainer + '\n' + worker + "'nnodes': 99, 'n_gpus_per_node': 99," + '\n' + ray + '}}';
  const result = extractConfig(text);
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.fields['集群规格'], undefined);
  assert.match(result.warning, /仅恢复已完整解析/);
});

test('logger-like strings inside quoted config values are not removed', () => {
  const model = '/models/(TaskRunnerV1 pid=123) INFO:fixture';
  const text = JSON.stringify({ actor_rollout_ref: { model: { path: model } }, trainer: {} });
  assert.equal(extractConfig(ray + text).fields['模型'], '(TaskRunnerV1 pid=123) INFO:fixture');
});

test('warning-like text inside a config string remains data', () => {
  const withText = { ...config, actor_rollout_ref: { ...actor, model: { path: '/models/Qwen-Fixture-27B', description: '/fixture.py:12: UserWarning: example' } } };
  const result = extractConfig(JSON.stringify(withText, null, 2));
  assert.equal(result.warning, '');
  assert.equal(result.fields['模型'], 'Qwen-Fixture-27B');
  assert.equal(result.notes.some(note => note.includes('交错')), false);
});
