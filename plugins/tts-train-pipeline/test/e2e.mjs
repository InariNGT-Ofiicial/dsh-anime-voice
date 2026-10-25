/**
 * e2e.mjs —— 真实数据链验证：剧本原文 → 盘点 → 出料。
 *
 * 不训练、不碰 GPU，只验证「台词能不能从剧本里被正确抽出来，并变成合法的 list.txt」。
 * 这是整条桥接最核心的一步：如果这一步的命中率是 100%，ASR 就完全不需要。
 *
 * 用法: node test/e2e.mjs [scnDir] [voiceDir]
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';

const SCN = process.argv[2] || 'D:/AI-Song/04_训练工作区/corpus/senren-unpacked/data/scn';
const VOICE = process.argv[3] || 'D:/AI-Song/04_训练工作区/corpus/senren-unpacked/data/voice';
const EXP = 'e2e-yoshino';
const SPEAKER = '芳乃';
const LIMIT = 30;

const mod = await import('../src/index.js');
const tools = new Map();
await mod.apply({
  logger: { info() {}, warn: (m) => console.error(m), error: (m) => console.error(m) },
  get: (s) => (s === 'tools' ? { register: (d) => tools.set(d.name, d) } : undefined),
}, { expName: EXP });

const run = async (name, args = {}) => {
  const t0 = Date.now();
  const v = await tools.get(name).execute(args);
  const text = tools.get(name).output.render(args, v).map((x) => x.text).join('\n');
  console.log(`\n${'='.repeat(72)}\n[${name}] ok=${v.ok} ${((Date.now() - t0) / 1000).toFixed(1)}s\n${'='.repeat(72)}`);
  console.log(text);
  if (!v.ok) { console.error(`\n❌ ${name} 失败，中止`); process.exit(1); }
  return v;
};

if (!existsSync(SCN)) { console.error(`剧本目录不存在: ${SCN}`); process.exit(2); }
if (!existsSync(VOICE)) { console.error(`语音目录不存在: ${VOICE}`); process.exit(2); }

// ── 1. 找回原文 ────────────────────────────────────────────────────────────
const s1 = await run('tts_script', { scnDir: SCN, voiceDir: VOICE, expName: EXP, lang: 'jp' });
const lines = s1.script;
if (!lines.with_both) { console.error('❌ 没有一条同时具备台词与语音'); process.exit(1); }
if (lines.voice_unmatched > 0) {
  console.error(`⚠️  ${lines.voice_unmatched} 条 voice ID 未命中文件，检查命名规则`);
}
const hitRate = lines.voice_matched / (lines.voice_matched + lines.voice_unmatched);
console.log(`\n语音命中率: ${(hitRate * 100).toFixed(1)}%`);

// 交叉验：抽 3 条，确认 text 是日文、voice_path 真存在
const jsonl = readFileSync(s1.out, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const sample = jsonl.filter((r) => r.voice_path).slice(0, 3);
console.log('\n抽样（台词 ↔ 语音 对应）:');
for (const r of sample) {
  const okFile = existsSync(r.voice_path);
  console.log(`  ${okFile ? '✓' : '✗'} [${r.speaker}] ${r.text.slice(0, 40)}  ← ${r.voice}`);
}
if (sample.length === 0 || sample.some((r) => !existsSync(r.voice_path))) {
  console.error('❌ 抽样里的 voice_path 不存在'); process.exit(1);
}

// ── 2. 盘点 ────────────────────────────────────────────────────────────────
const s2 = await run('tts_scan', { source: s1.out, voiceDir: VOICE, minSec: 3, maxSec: 10 });

// ── 3. 出料 ────────────────────────────────────────────────────────────────
const s3 = await run('tts_build', {
  lines: s1.out, voiceDir: VOICE, expName: EXP,
  speaker: SPEAKER, lang: 'ja', limit: LIMIT,
});

// ── 4. 校验 list.txt 契约 ─────────────────────────────────────────────────
const listPath = s3.build.list;
const rows = readFileSync(listPath, 'utf8').trim().split('\n').filter(Boolean);
console.log(`\nlist.txt: ${rows.length} 行`);
let bad = 0;
for (const [i, line] of rows.entries()) {
  const parts = line.split('|');
  if (parts.length !== 4) { console.error(`  ❌ 第 ${i + 1} 行不是 4 段: ${line.slice(0, 80)}`); bad += 1; continue; }
  const [wav, spk, lang, text] = parts;
  if (!wav.endsWith('.wav')) { console.error(`  ❌ 第 ${i + 1} 行 wav 名异常: ${wav}`); bad += 1; }
  if (!existsSync(`${s3.build.wav_dir}/${wav}`)) { console.error(`  ❌ 第 ${i + 1} 行音频不存在: ${wav}`); bad += 1; }
  if (!['zh', 'ja', 'en', 'ko', 'yue'].includes(lang)) { console.error(`  ❌ 第 ${i + 1} 行语言非法: ${lang}`); bad += 1; }
  if (!text.trim()) { console.error(`  ❌ 第 ${i + 1} 行文本为空`); bad += 1; }
}
const wavCount = readdirSync(s3.build.wav_dir).filter((f) => f.endsWith('.wav')).length;
console.log(`wav 文件: ${wavCount} 个`);

console.log('\n【结论】');
console.log(`  剧本台词 ${lines.with_text} 条 / 带语音 ${lines.with_voice} 条 / 命中 ${lines.voice_matched}`);
console.log(`  出料 ${rows.length} 条，list.txt 契约 ${bad === 0 ? '✅ 全部合规' : `❌ ${bad} 处违规`}`);
console.log(`  可训时长 ${s3.build.minutes} 分钟`);
process.exit(bad === 0 ? 0 : 1);
