/**
 * smoke.mjs —— 契约自检：不需要 GPT-SoVITS 环境，只验证插件注册形状。
 *
 * 验的是「插件能不能被 DSH 正确加载」，不是「训练能不能跑」。
 * 后者要 node test/live.mjs。
 */
import { strict as assert } from 'node:assert';

const EXPECTED = [
  'tts_doctor', 'tts_plan', 'tts_script', 'tts_scan', 'tts_build', 'tts_asr',
  'tts_patch_text', 'tts_preprocess', 'tts_run', 'tts_status', 'tts_stop',
  'tts_infer', 'tts_register', 'tts_overview',
];

const mod = await import('../src/index.js');
assert.equal(mod.name, 'tts-train-pipeline', 'name 必须是 tts-train-pipeline');
assert.deepEqual(mod.inject, ['tools'], 'inject 必须是 ["tools"]');
assert.equal(typeof mod.apply, 'function', 'apply 必须是函数');

const registered = new Map();
const ctx = {
  logger: { info() {}, warn() {}, error() {} },
  get(service) {
    if (service !== 'tools') return undefined;
    return {
      register(def) {
        assert.ok(def?.name, '工具必须有 name');
        assert.ok(def?.description, `${def?.name} 必须有 description`);
        assert.ok(def?.parameters, `${def?.name} 必须有 parameters`);
        assert.ok(def?.output?.schema, `${def?.name} 必须有 output.schema`);
        assert.equal(typeof def?.output?.render, 'function', `${def?.name} 必须有 output.render`);
        assert.equal(typeof def?.execute, 'function', `${def?.name} 必须有 execute`);
        assert.ok(!registered.has(def.name), `工具重名: ${def.name}`);
        registered.set(def.name, def);
      },
    };
  },
};

await mod.apply(ctx, {});

assert.equal(registered.size, EXPECTED.length,
  `工具数应为 ${EXPECTED.length}，实际 ${registered.size}: ${[...registered.keys()]}`);
for (const n of EXPECTED) assert.ok(registered.has(n), `缺少工具: ${n}`);

// 参数 schema 必须是合法 JSON Schema 的壳，且 additionalProperties 显式声明
for (const [n, def] of registered) {
  assert.equal(def.parameters.type, 'object', `${n}.parameters.type`);
  assert.ok(def.parameters.properties, `${n}.parameters.properties 缺失`);
  assert.equal(def.parameters.additionalProperties, false, `${n} 应显式 additionalProperties:false`);
}

// 带必填参数的工具，必填项必须真的在 properties 里
const REQUIRED = {
  tts_scan: ['source'],
  tts_infer: ['text'],
};
for (const [n, reqs] of Object.entries(REQUIRED)) {
  const def = registered.get(n);
  const listed = def.parameters.required || [];
  for (const r of reqs) {
    assert.ok(listed.includes(r), `${n} 的 required 应含 ${r}`);
    assert.ok(def.parameters.properties[r], `${n} 的 properties 应含 ${r}`);
  }
}

// 无参数工具（tts_doctor 现在有可选的 deep，不在此列）
for (const n of ['tts_stop', 'tts_overview']) {
  const p = registered.get(n).parameters.properties;
  assert.equal(Object.keys(p).length, 0, `${n} 不应有参数（实际: ${Object.keys(p)}）`);
}

// 所有工具的参数都必须是可选的（工具要能在无参调用下走默认路径）
for (const [n, def] of registered) {
  const req = def.parameters.required || [];
  for (const r of req) {
    assert.ok(def.parameters.properties[r], `${n} 的 required 里有未声明的参数 ${r}`);
  }
}
const doctorReq = registered.get('tts_doctor').parameters.required || [];
assert.equal(doctorReq.length, 0, 'tts_doctor 的 deep 必须是可选参数');

// render 必须能吃空值而不炸
for (const [n, def] of registered) {
  const out = def.output.render({}, {});
  assert.ok(Array.isArray(out) && out[0]?.type === 'text', `${n}.render 应返回 [{type:'text'}]`);
}

console.log(`✅ smoke: ${EXPECTED.length} 个工具契约通过`);
console.log('   ' + [...registered.keys()].join(', '));
