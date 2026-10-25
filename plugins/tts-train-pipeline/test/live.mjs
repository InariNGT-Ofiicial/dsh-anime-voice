/**
 * live.mjs —— 真机只读验证：doctor / plan / overview。
 *
 * 不训练、不写文件，只验证「插件能不能在这台机器上认出环境」。
 * 用法: node test/live.mjs [doctor|plan|overview|all]
 */
const mod = await import('../src/index.js');

const tools = new Map();
const ctx = {
  logger: { info() {}, warn: (m) => console.error(m), error: (m) => console.error(m) },
  get: (s) => (s === 'tools' ? {
    register(def) { tools.set(def.name, def); },
  } : undefined),
};

await mod.apply(ctx, {});

const which = process.argv[2] || 'all';
const run = async (name, args = {}) => {
  const def = tools.get(name);
  if (!def) throw new Error(`工具未注册: ${name}`);
  const t0 = Date.now();
  const v = await def.execute(args);
  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  const text = def.output.render(args, v).map((x) => x.text).join('\n');
  console.log(`\n${'='.repeat(72)}\n[${name}] ok=${v.ok} ${dt}s\n${'='.repeat(72)}`);
  console.log(text);
  return v;
};

const out = {};
if (which === 'all' || which === 'doctor') out.doctor = await run('tts_doctor');
if (which === 'all' || which === 'plan') out.plan = await run('tts_plan');
if (which === 'all' || which === 'overview') out.overview = await run('tts_overview');

if (which === 'all') {
  console.log('\n【结论】');
  console.log(`  体检      : ${out.doctor.ok ? '✅ 通过' : '❌ ' + out.doctor.errorCount + ' 个阻断项'}`);
  console.log(`  工程发现  : ${out.doctor.info?.gsvDir || '(未发现)'}`);
  console.log(`  GPU       : ${out.doctor.info?.gpu?.name || '(未探测到)'}`);
  process.exitCode = out.doctor.ok ? 0 : 1;
}
