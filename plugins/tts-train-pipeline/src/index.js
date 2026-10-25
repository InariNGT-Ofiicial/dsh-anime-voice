/**
 * dsh-tts-train-pipeline —— 把角色语料变成 GPT-SoVITS TTS 音色的 DSH 插件。
 *
 * 与 dsh-train-pipeline 的分工（这是整个设计的前提）：
 *   dsh-train-pipeline → DDSP/SVC 模型，用途是**翻唱**（给已有歌声换音色）
 *   本插件            → GPT-SoVITS 模型，用途是**说话**（文本转语音，念任意台词）
 *
 *   两者**不通用也不互转**：SVC 模型不能拿来 TTS，TTS 模型也不能拿来 SVC。
 *   所以本插件不重复实现取材/解包（那是 dsh-train-pipeline 的强项），
 *   而是**吃它的产物**：解包出的 `lines.jsonl`（带 speaker/text）与语音目录。
 *   桥接的实质是把「解包阶段本来就有、却在 SVC 出料时被丢弃的台词」接回来。
 *
 * Cordis 插件契约：
 *   export const name / inject / async function apply(ctx, config)
 *   tools.register({ name, description, parameters, output, execute })
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';

import {
  resolveConfig, join, PY_DIR, scanWeights, findVoice, listVoices,
  LANGUAGES, needsSv, datasetDirOf, resolveSpeed, LANG_SPEED, BASELINE, pickGptMode,
} from './config.js';
import { asrGate } from './asr-gate.js';
import {
  runOnce, spawnJob, pythonEnv, tailFile, parseProgress, detectFinish,
  readJobs, upsertJob, jobAlive, killJob,
} from './execution.js';
import { doctor, configSnippet, gitInfo } from './checks.js';
import {
  runPrepStage, resetPrepArtifacts, preflight, preflightPrep, s2Overrides, s1Overrides,
  makeTrainConfig, startS2, startS1, countListLines, PREP_ARTIFACTS, ensureTrainDirs,
} from './gsv.js';
import { FLOW, DECISIONS, RULES, VERSION_NOTES, SCN_STRUCTURE } from './stages.js';

export const name = 'tts-train-pipeline';
export const inject = ['tools'];

/**
 * 读数据集 list.txt 统计**训练语料的主语言**（第 3 列）。
 *
 * 用途只有一个：给 gptMode='auto' 判断「这是不是跨语言合成」。
 * 为什么必须从数据里读、不能靠配置猜：同一个插件下可以并存日文语料训的
 * 芳乃和中文语料训的别的角色，写死一个 cfg.lang 一定会在某个角色上判断错。
 */
function trainLangOf(cfg, name) {
  const p = join(datasetDirOf(cfg, name), 'list.txt');
  if (!existsSync(p)) return '';
  const cnt = new Map();
  let n = 0;
  for (const line of readFileSync(p, 'utf8').split(/\r?\n/)) {
    if (n >= 800) break;                 // 抽样即可，不必读全量
    const col = line.split('|');
    if (col.length < 3) continue;
    const lang = (col[2] || '').trim().toLowerCase();
    if (!lang) continue;
    cnt.set(lang, (cnt.get(lang) || 0) + 1);
    n += 1;
  }
  let best = '';
  let bestN = 0;
  for (const [k, v] of cnt) if (v > bestN) { best = k; bestN = v; }
  return best;
}

/**
 * 解析 s1 用哪一档，返回 { mode, note }。
 *
 * auto 的判据是「目标语言 vs 训练语言」：
 *   一致   → trained（自训 s1 在该语言上有语料，语气最贴）
 *   不一致 → base（跨语言必须换底模：自训 s1 把音素→语义的映射训窄到训练语言了）
 * 没有自训 s1 时一律回落 base。
 */
function resolveGptMode(cfg, req, targetLang, hasTrained) {
  const trainLang = trainLangOf(cfg, cfg.expName);
  if (req !== 'auto') return { mode: req, note: '', trainLang };
  const p = pickGptMode(trainLang, targetLang, hasTrained);
  return { mode: p.mode, note: p.note, trainLang };
}

const OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    rendered: { type: 'string' },
    stage: { type: 'string' },
  },
  required: ['ok', 'rendered'],
};

/** Cordis 的 output 形状是 { schema, render }，不是裸 schema */
const wrap = (run) => ({
  output: {
    schema: OUTPUT_SCHEMA,
    render: (_args, v) => [{ type: 'text', text: String(v?.rendered ?? '(无输出)') }],
  },
  execute: run,
});

/** 从混合输出里取最后一段合法 JSON（脚本常先打日志） */
function lastJson(text) {
  if (!text) return null;
  const t = String(text).trim();
  try { return JSON.parse(t); } catch { /* 继续 */ }
  const lines = t.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (!lines[i].startsWith('{')) continue;
    try { return JSON.parse(lines[i]); } catch { /* 继续 */ }
  }
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a >= 0 && b > a) {
    try { return JSON.parse(t.slice(a, b + 1)); } catch { /* ignore */ }
  }
  return null;
}

const fmtMin = (sec) => {
  const s = Number(sec) || 0;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  if (m < 60) return `${m}m${r}s`;
  return `${Math.floor(m / 60)}h${m % 60}m`;
};

const fmtBytes = (n) => {
  const b = Number(n) || 0;
  if (b < 1024) return `${b}B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)}KB`;
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)}MB`;
  return `${(b / 1024 ** 3).toFixed(2)}GB`;
};

function renderDoctor(res, cfg) {
  const L = [];
  L.push(`【GPT-SoVITS 训练环境体检】${res.ok ? '✅ 可以开训' : '❌ 有阻断项'}`);
  L.push('');
  const i = res.info || {};
  L.push(`工程:   ${i.gsvDir || '(未发现)'}`);
  L.push(`Python: ${i.python?.exe || '(未发现)'}  ${i.python?.version ? `(${i.python.version}${i.python.prefixVenv ? ' · venv' : ' · 非 venv'})` : ''}`);
  if (i.gpu) {
    L.push(`GPU:    ${i.gpu.name} · ${i.gpu.capability} · ${i.gpu.vramGB}GB · torch ${i.gpu.torch} (cu${i.gpu.cudaBuild})`);
  }
  L.push(`版本:   ${cfg.version} · ${cfg.profile.sovitsWeights} / ${cfg.profile.gptWeights}`);
  if (i.disk) L.push(`磁盘:   ${i.disk.freeGB}GB 可用 / ${i.disk.totalGB}GB`);
  L.push(`ffmpeg: ${i.ffmpeg || '(未发现)'}`);
  L.push(`ASR:    ${i.asr?.cached ? `本地已缓存 ${i.asr.model}（${i.asr.modelBinMB}MB）` : `${i.asr?.model} 未缓存（首次需联网）`}`);
  L.push(`实验:   ${cfg.expDir}`);
  L.push(`语料区: ${cfg.workspace.datasets}`);
  if (i.modulesSkipped?.length) {
    L.push(`未探模块: ${i.modulesSkipped.join(', ')}  （deep=true 可查）`);
  }
  if (i.probeNote) L.push(i.probeNote);
  L.push('');

  if (res.errors.length) {
    L.push(`── 阻断项 (${res.errors.length}) ──`);
    for (const e of res.errors) {
      L.push(`  ❌ ${e.msg}`);
      if (e.fix) L.push(`     → ${e.fix}`);
      if (e.detail) L.push(`     ${String(e.detail).split('\n').slice(-3).join('\n     ')}`);
    }
    L.push('');
  }
  if (res.warnings.length) {
    L.push(`── 注意 (${res.warnings.length}) ──`);
    for (const w of res.warnings) L.push(`  ⚠️  ${w}`);
    L.push('');
  }
  if (i.versions) {
    L.push('── 版本说明 ──');
    for (const [k, v] of Object.entries(VERSION_NOTES)) L.push(`  ${k.padEnd(10)} ${v}`);
    L.push('');
  }
  L.push('已注册音色:');
  const vs = i.voices || [];
  L.push(vs.length ? vs.map((v) => `  ${v.ok ? '✓' : '✗'} ${v.name}`).join('\n') : '  (空)');

  if (!res.ok) {
    L.push('', '可直接粘贴的配置（写进 tts.local.json 或 cordis.patch.yml）:');
    L.push(configSnippet(cfg));
  }
  return L.join('\n');
}

// ───────────────────────────────────────────────────────────────────────────

export async function apply(ctx, config) {
  const logger = ctx.logger ?? console;
  const rawConfig = config ?? {};
  const cfg = resolveConfig(rawConfig);
  const tools = ctx.get('tools');

  if (!tools || typeof tools.register !== 'function') {
    logger.warn('[tts-train-pipeline] tools 服务不可用，插件未注册任何工具');
    return;
  }
  const register = (def) => tools.register(def);

  /** 用一组覆盖值重新解析配置（版本/实验名这类会影响派生路径的键必须走这里） */
  const withCfg = (over = {}) => resolveConfig({ ...rawConfig, ...over });

  /** 数据集目录 */
  const dsDir = (c, expName) => join(c.workspace.datasets, expName || c.expName);

  /** 跑一个 py 脚本并取 JSON */
  const runPy = async (c, script, args, timeoutMs) => {
    const r = await runOnce({
      cmd: c.gsvEnv || 'python',
      args: [join(PY_DIR, script), ...args],
      cwd: c.gsvDir || PY_DIR,
      env: pythonEnv(c.gsvEnv, {}),
      timeoutMs: timeoutMs ?? c.timeoutMs,
    });
    return { ...r, json: lastJson(r.stdout), tail: ((r.stderr || '') + (r.stdout || '')).slice(-1500) };
  };

  // ── 1 · 环境体检 ───────────────────────────────────────────────────────
  register({
    name: 'tts_doctor',
    description:
      'GPT-SoVITS 训练环境体检：工程与 7 个关键文件、版本档案要求的全部底模（s2G/s2D/s1/BERT/hubert/SV）、'
      + 'Python 与 torch 的 CUDA 版本是否匹配 GPU 算力（Blackwell + cu126 是个死局）、ffmpeg、'
      + 'faster-whisper 与本地缓存、磁盘、已注册音色。★ 任何训练前先跑，每个阻断项都带修复办法。'
      + '默认只探参与判定的模块；deep=true 追加 librosa/transformers/gradio/peft/flash_attn（约 +15s）。',
    parameters: {
      type: 'object',
      properties: {
        deep: { type: 'boolean', description: '连重模块（peft/transformers/gradio 等）一起探，慢约 15s' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const res = await doctor(cfg, { deep: !!args.deep });
      res.info.versions = true;
      const rendered = renderDoctor(res, cfg);
      logger.info(`[tts-train-pipeline] doctor ok=${res.ok} errors=${res.errors.length}`);
      return {
        ok: res.ok, stage: 'doctor',
        errorCount: res.errors.length, warningCount: res.warnings.length,
        info: res.info, errors: res.errors, warnings: res.warnings, rendered,
      };
    }),
  });

  // ── 2 · 流程说明 ───────────────────────────────────────────────────────
  register({
    name: 'tts_plan',
    description:
      'TTS 训练六段流程 + 4 个决策点 + 5 条红线。开始前先读：尤其是「TTS 不能切分音频」'
      + '与「s2 必须成功 s1 才有意义」这两条，它们在 SVC 链路里不成立，容易照搬出错。',
    parameters: {
      type: 'object',
      properties: { version: { type: 'string', description: '想确认的版本（v1/v2/v2Pro/v2ProPlus/v3/v4）' } },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg(args.version ? { version: args.version } : {});
      const L = ['【TTS 音色训练流程】'];
      L.push('★ 原则：能用原文就绝不用 ASR。剧本/字幕里的台词是引擎级无损数据；ASR 是有损重听写。', '');
      for (const s of FLOW) {
        const tag = s.first ? '  ⟵ 首选入口' : (s.fallback ? '  ⟵ 兜底，能避免就避免' : '');
        L.push(`${s.id}  →  ${s.tool}${tag}`);
        L.push(`   做什么: ${s.what}`);
        L.push(`   会卡在: ${s.fail}`);
      }
      L.push('', '【KrKrZ 剧本结构（为什么必须这么抽）】');
      L.push(`  引擎: ${SCN_STRUCTURE.engine}`);
      L.push(`  台词: ${SCN_STRUCTURE.where_text_lives}`);
      L.push(`  说话人: ${SCN_STRUCTURE.where_speaker}`);
      L.push(`  语音: ${SCN_STRUCTURE.where_voice}`);
      L.push(`  语言列: ${SCN_STRUCTURE.lang_columns}`);
      L.push(`  ★ ${SCN_STRUCTURE.why_lines_fails}`);
      L.push('', '【决策点】');
      for (const d of DECISIONS) {
        L.push(`\n${d.id}`);
        for (const o of d.options) L.push(`   · ${o}`);
        L.push(`   ★ ${d.rule}`);
      }
      L.push('', '【红线】');
      for (const r of RULES) L.push(`  ${r}`);
      L.push('', `本次实际会用: version=${c.version}（${VERSION_NOTES[c.version] || '自定义'}）`);
      L.push(`  s2 底模: ${c.profile.s2G}`);
      L.push(`  s1 底模: ${c.profile.s1}`);
      L.push(`  权重落点: ${c.sovitsWeightDir}`);
      L.push(`            ${c.gptWeightDir}`);
      return { ok: true, stage: 'plan', version: c.version, flow: FLOW, decisions: DECISIONS, rendered: L.join('\n') };
    }),
  });

  // ── 3 · 找回原文（首选路径，先于任何 ASR）────────────────────────────
  register({
    name: 'tts_script',
    description:
      '★ 首选入口：从剧本反编译产物里抽「说话人 + 台词原文 + 语音文件名」。'
      + 'KrKrZ 引擎的台词存放在 scene.texts[i] 里（不在 lines 里，lines 只有指令流）——'
      + '这就是按 lines 找文本只能捞到选项菜单的原因。'
      + '抽出结果与 tts_build 的 lines.jsonl 同构，可直接喂下去。'
      + '★ 有原文就别跑 ASR：ASR 是有损重听写，会丢标点、认错人名、写错汉字。',
    parameters: {
      type: 'object',
      properties: {
        scnDir: { type: 'string', description: '含 *.ks.json 的剧本目录（省略则从 voiceDir 的兄弟目录推断）' },
        voiceDir: { type: 'string', description: '语音目录，用于校验 voice ID 能否命中文件并报命中率' },
        out: { type: 'string', description: '输出 lines.jsonl（默认 <语料区>/<expName>.lines.jsonl）' },
        lang: {
          type: 'string', enum: ['jp', 'cn', 'tw', 'en'],
          description: 'jp=日文原文（与日文语音匹配，默认）；cn/tw/en 是官方翻译列',
        },
        expName: { type: 'string' },
        limit: { type: 'number' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg(args.expName ? { expName: args.expName } : {});

      // 自动推断：voiceDir 的兄弟目录里找 scn
      let scnDir = args.scnDir || '';
      if (!scnDir && args.voiceDir) {
        const parent = args.voiceDir.replace(/[\\/][^\\/]+[\\/]?$/, '');
        for (const cand of ['scn', 'script', 'scenario']) {
          const p = join(parent, cand);
          if (existsSync(p)) { scnDir = p; break; }
        }
      }
      if (!scnDir) {
        const msg = '需要 scnDir：指向反编译出的 *.ks.json 所在目录。\n'
          + '若只有 .scn 二进制，先用 dsh-train-pipeline 的 train_unpack 反编译。';
        return { ok: false, stage: 'script', error: msg, rendered: msg };
      }
      if (!existsSync(scnDir)) {
        const msg = `剧本目录不存在: ${scnDir}`;
        return { ok: false, stage: 'script', error: msg, rendered: msg };
      }

      const out = args.out || join(c.workspace.corpus, `${c.expName}.lines.jsonl`);
      mkdirSync(c.workspace.corpus, { recursive: true });
      const argv = ['--scn-dir', scnDir, '--out', out, '--lang', args.lang || 'jp'];
      if (args.voiceDir) argv.push('--voice-dir', args.voiceDir);
      if (args.limit) argv.push('--limit', String(args.limit));

      const r = await runPy(c, 'extract_scn_text.py', argv, 900_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'script', error: '提取脚本无输出', detail: r.tail, rendered: `提取失败：\n${r.tail}` };
      }
      if (!j.ok) {
        return { ok: false, stage: 'script', error: j.error, detail: j, rendered: `❌ ${j.error}` };
      }

      const L = [`【剧本原文提取】${j.files_story} 个剧情文件 · 语言列 ${j.lang}`];
      L.push(`台词总数 ${j.with_text} · 带语音 ${j.with_voice} · 两者都有 ${j.with_both}`);
      if (args.voiceDir) {
        const hit = j.voice_matched || 0;
        const miss = j.voice_unmatched || 0;
        const rate = hit + miss > 0 ? ((hit / (hit + miss)) * 100).toFixed(1) : '—';
        L.push(`语音命中 ${hit} / 未命中 ${miss}（${rate}%）`);
        if (miss > 0) {
          L.push(`  未命中样本（voice ID 与文件命名对不上）: ${(j.unmatched_samples || []).slice(0, 5).join(', ')}`);
        }
      } else {
        L.push('⚠️  没给 voiceDir，未校验 voice ID 能否落到文件');
      }
      if (j.fallback_to_jp) L.push(`⚠️  ${j.fallback_to_jp} 条缺所选语言，已回落日文原文`);
      if (j.multi_voice) L.push(`（${j.multi_voice} 条有多个语音指令，取第一条）`);

      L.push('', '说话人分布（仅有语音的台词）:');
      for (const [name, n] of (j.speakers || []).slice(0, 15)) {
        L.push(`  ${String(name).padEnd(12)} ${n}`);
      }
      L.push('', `输出: ${j.out}`);
      L.push(`结构依据: 台词在 ${SCN_STRUCTURE.where_text_lives} · 说话人在 ${SCN_STRUCTURE.where_speaker}`
        + ` · 语音在 ${SCN_STRUCTURE.where_voice}`);
      L.push('', '下一步: tts_build（lines=上面这个 jsonl，voiceDir=语音目录）');
      L.push('★ 这些是引擎级原文，不是听写结果 —— 不要再去跑 tts_asr。');
      return { ok: true, stage: 'script', script: j, out, rendered: L.join('\n') };
    }),
  });

  // ── 4 · 盘语料 ─────────────────────────────────────────────────────────
  register({
    name: 'tts_scan',
    description:
      '盘点语料：每个角色「有多少条语音 / 多少带台词 / 多少条两者都有 / 总时长 / 落在 3~10s 区间的条数」，'
      + '并给出能不能训的判定。★ 关键区别：「900 个语音文件」不等于「900 条可训数据」——'
      + '后者才决定能不能训。支持 lines.jsonl（台词驱动）与按角色分目录（无台词）两种源。',
    parameters: {
      type: 'object',
      properties: {
        source: { type: 'string', description: 'lines.jsonl 路径，或语音目录' },
        voiceDir: { type: 'string', description: 'lines 模式下语音文件所在目录（省略则用 source 所在目录）' },
        minSec: { type: 'number', description: '可训时长下限（默认 2；galgame 单句多在 1.5~4s）' },
        maxSec: { type: 'number', description: '可训时长上限（默认 10）' },
      },
      required: ['source'],
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      let isLines = false;
      try { isLines = statSync(args.source).isFile(); } catch { isLines = false; }

      const argv = [];
      if (isLines) {
        argv.push('--lines', args.source);
        if (args.voiceDir) argv.push('--voice-dir', args.voiceDir);
      } else {
        argv.push('--voice-dir', args.source);
      }
      if (args.minSec) argv.push('--min-sec', String(args.minSec));
      if (args.maxSec) argv.push('--max-sec', String(args.maxSec));

      const r = await runPy(cfg, 'scan_corpus.py', argv, 300_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'scan', error: '扫描失败', detail: r.tail, rendered: `扫描失败：\n${r.tail}` };
      }
      if (!j.ok) {
        return { ok: false, stage: 'scan', error: j.error, rendered: `扫描失败：${j.error}` };
      }

      const L = [`【语料盘点】${j.mode === 'lines' ? '台词驱动' : '目录驱动'}${j.truncated ? ' · 已截断（超时）' : ''}`];
      L.push(`时长探测: ${j.has_soundfile ? 'soundfile（快）' : 'ffprobe（慢）'}  耗时 ${j.elapsedSec}s`, '');
      L.push('角色                          语音  带台词  两者都有   时长     可训判定');
      let anyUsable = false;
      for (const s of j.speakers) {
        const name = String(s.speaker).padEnd(22).slice(0, 22);
        const usable = j.mode === 'lines' ? (s.both ?? 0) : (s.inRange ?? 0);
        if (usable > 0) anyUsable = true;
        L.push(`${name} ${String(s.withVoice).padStart(6)} ${String(s.withText).padStart(7)} `
          + `${String(usable).padStart(8)}  ${String(s.minutes).padStart(6)}m  ${s.verdict}`);
      }
      L.push('');
      if (!anyUsable) {
        L.push('⚠️  没有任何角色有可训数据。');
        if (j.mode === 'lines') {
          L.push('   台词与语音对不上：要么剧本没标说话人，要么 voice 字段匹配不到文件。');
          L.push('   先跑 tts_scan 看对方目录结构，改用「目录驱动」路线再走 ASR 回标。');
        } else {
          L.push('   这是无台词源，必须走 tts_asr 回标才有监督信号。');
        }
      } else {
        L.push('下一步: tts_build（有台词）或 tts_asr → tts_build（无台词）');
      }
      return { ok: anyUsable, stage: 'scan', scan: j, rendered: L.join('\n') };
    }),
  });

  // ── 4 · 出料 ───────────────────────────────────────────────────────────
  register({
    name: 'tts_build',
    description:
      '出料：把「台词清单 + 语音目录」变成 GPT-SoVITS 可直接预处理的 32kHz/单声道 wav 目录 + '
      + 'list.txt（wav名|说话人|语言|文本）。优先用 lines 里的 voice_path 精确路径，回落到按名字模糊匹配。'
      + '★ 台词来自 tts_script 抽出的原文（引擎级无损数据），不是 ASR 结果。'
      + '★ 超长音频**不切分**而是整条丢弃——切了音频和文本就对不上，那是 TTS 最致命的错。'
      + '无台词语料加 allowEmptyText=true 只出 wav（给 tts_asr 兜底用）。',
    parameters: {
      type: 'object',
      properties: {
        lines: { type: 'string', description: 'lines.jsonl（台词驱动）' },
        voiceDir: { type: 'string', description: '语音文件目录；目录驱动时一级子目录=角色' },
        speaker: { type: 'string', description: '角色名，逗号分隔；all=全部' },
        voiceOnly: { type: 'boolean', description: '目录驱动模式：只规格化，忽略台词' },
        expName: { type: 'string', description: '实验名（决定输出目录与训练目录，默认取配置）' },
        out: { type: 'string', description: '输出目录（默认 <语料区>/<expName>）' },
        lang: { type: 'string', description: `语言 ${LANGUAGES.join('/')}（默认取配置）` },
        langAuto: { type: 'boolean', description: '按文本字符自动判语言（假名→ja，汉字→zh）' },
        minSec: { type: 'number' },
        maxSec: { type: 'number' },
        limit: { type: 'number', description: '每角色最多出多少条（调试用）' },
        allowEmptyText: { type: 'boolean', description: '无台词也出 wav（为 ASR 回标做准备）' },
        noTrim: { type: 'boolean', description: '不裁首尾静音' },
        tailPad: { type: 'number', description: '句尾兜底静音秒数，默认 0.16；设 0 关闭' },
        trimTail: { type: 'boolean', description: '★仅复现旧数据集：裁掉句尾（训出来的模型会吞末字，别在生产上用）' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg(args.expName ? { expName: args.expName } : {});
      if (!args.voiceDir && !args.lines) {
        return { ok: false, stage: 'build', error: '需要 lines 或 voiceDir', rendered: '需要 lines 或 voiceDir' };
      }
      const out = args.out || dsDir(c, args.expName);
      const argv = ['--out', out];
      if (args.lines) argv.push('--lines', args.lines);
      if (args.voiceDir) argv.push('--voice-dir', args.voiceDir);
      if (args.voiceOnly) argv.push('--voice-only');
      if (args.speaker) argv.push('--speaker', args.speaker);
      if (args.lang) argv.push('--lang', args.lang);
      if (args.langAuto) argv.push('--lang-auto');
      if (args.minSec) argv.push('--min-sec', String(args.minSec));
      if (args.maxSec) argv.push('--max-sec', String(args.maxSec));
      if (args.limit) argv.push('--limit', String(args.limit));
      if (args.allowEmptyText) argv.push('--allow-empty-text');
      if (args.noTrim) argv.push('--no-trim');
      if (args.tailPad !== undefined) argv.push('--tail-pad', String(args.tailPad));
      if (args.trimTail === true) argv.push('--trim-tail');
      argv.push('--sample-rate', String(c.sampleRate));
      argv.push('--ffmpeg', c.ffmpeg || 'ffmpeg');

      const r = await runPy(c, 'build_tts_dataset.py', argv, 1_800_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'build', error: '出料脚本无输出', detail: r.tail, rendered: `出料失败：\n${r.tail}` };
      }
      if (!j.ok) {
        const L = [`❌ 出料失败：${j.error || '未知'}`];
        if (j.speakers) for (const s of j.speakers) L.push(`  ${s.speaker}: 候选 ${s.candidates} 出 ${s.built}`);
        return { ok: false, stage: 'build', detail: j, rendered: L.join('\n') };
      }

      const L = [`【出料完成】${j.out}`];
      L.push(`wav: ${j.wav_dir}   list: ${j.list}`);
      L.push(`规格: ${j.sample_rate}Hz / 单声道 / PCM16  语言: ${j.lang_auto ? '自动' : j.lang}`);
      L.push(`时长窗: ${j.min_sec}~${j.max_sec}s${j.trim ? '（已裁头尾静音）' : '（未裁静音）'}`);
      L.push('');
      for (const s of j.speakers) {
        L.push(`  ${s.speaker || '(旁白)'}: 出 ${s.built} 条 / ${s.minutes}m  `
          + `(短 ${s.skipped_short} · 长 ${s.skipped_long} · 无文本 ${s.skipped_notext} `
          + `· 无语音 ${s.skipped_missing ?? 0} · 错 ${s.skipped_error})`);
      }
      L.push('', `合计 ${j.lines} 条 · ${j.minutes} 分钟`);
      if (j.duration_stats) {
        const d = j.duration_stats;
        L.push(`时长分布: 最短 ${d.min}s · 中位 ${d.median}s · P90 ${d.p90}s · 最长 ${d.max}s`);
      }
      // 尾音体检：这是「切句尾」唯一可判读的信号，别只看句末有没有标点
      if (j.tail_stats) {
        const t = j.tail_stats;
        L.push(`尾音体检: ${t.n} 条 · 收尾电平中位 ${t.median_db}dB · `
          + `${t.cut_ratio > 0.3 ? '❌' : '✅'} ${t.verdict}（切断 ${(t.cut_ratio * 100).toFixed(0)}%）`);
        if (t.cut_ratio > 0.3) {
          if (j.cut_tail?.length) L.push('  样本: ' + j.cut_tail.slice(0, 3).join(' | '));
          L.push('  ⚠️  源音频本身停在满电平，补静音底救不了；要从原始 ogg 重建，'
            + '不要拿切过的 wav 再切一次');
        }
      }
      if (args.trimTail === true) {
        L.push('', '⚠️  trimTail=true：这是旧数据集行为（句尾停在满电平），'
          + '训出来的模型会吞末字，只用于对齐历史数据');
      }
      if (j.duration_histogram && Object.keys(j.duration_histogram).length) {
        const hist = Object.entries(j.duration_histogram).filter(([, v]) => v > 0);
        L.push('  ' + hist.map(([k, v]) => `${k}:${v}`).join('  '));
      }
      if (j.skipped_missing) {
        L.push(`ℹ️  ${j.skipped_missing} 条台词没有对应语音文件（旁白或无配音句，正常）`);
      }
      if (j.advice) L.push('', `⚠️  ${j.advice}`);
      if (j.skipped_long) {
        L.push(`\n⚠️  ${j.skipped_long} 条超长被丢弃（TTS 不切分，切了文不对音）：`);
        L.push('   ' + (j.rejected_long || []).slice(0, 8).join(', '));
      }
      if (j.errors?.length) {
        L.push('\n出错样本:');
        for (const e of j.errors.slice(0, 5)) L.push(`  ${e}`);
      }
      L.push('', '下一步: tts_preprocess（1-get-text → 2-get-hubert → 2-get-sv → 3-get-semantic）');
      return { ok: true, stage: 'build', build: j, rendered: L.join('\n') };
    }),
  });

  // ── 5 · ASR 回标 ───────────────────────────────────────────────────────
  register({
    name: 'tts_asr',
    description:
      '**兜底手段，不是首选**：只在语料真的没有文本时用（如语音按角色分目录、剧本里没有对白、'
      + '或语音来自录屏/番剧）。用 faster-whisper 逐条转写，按 avg_logprob / no_speech 置信度过滤。'
      + '★ 有原文却跑 ASR 是返工：日文 ASR 会把「橋」听成「はし」写错汉字、丢标点、认错人名，'
      + '模型学到的是错误发音。先试 tts_script / 找旁挂字幕，都不行再来这里。',
    parameters: {
      type: 'object',
      properties: {
        wavDir: { type: 'string', description: 'wav 目录（默认 <语料区>/<expName>/wav）' },
        expName: { type: 'string' },
        speaker: { type: 'string', description: '写进 list.txt 的说话人名（默认 expName）' },
        lang: { type: 'string', description: 'auto/zh/ja/en/ko/yue（默认 auto）' },
        model: { type: 'string', description: 'whisper 模型（默认取配置，large-v3）' },
        limit: { type: 'number' },
        minLogprob: { type: 'number', description: '平均对数概率下限（默认 -1.0，越低越宽松）' },
        allowDownload: { type: 'boolean', description: '本地无缓存时允许联网下载' },
        mergeExisting: { type: 'string', description: '已有 list.txt：只补缺失的行（断点续跑）' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg(args.expName ? { expName: args.expName } : {});
      const wavDir = args.wavDir || join(dsDir(c, args.expName), 'wav');
      if (!existsSync(wavDir)) {
        const msg = `wav 目录不存在: ${wavDir}\n先用 tts_build（voiceOnly + allowEmptyText）出 wav。`;
        return { ok: false, stage: 'asr', error: msg, rendered: msg };
      }
      const argv = ['--wav-dir', wavDir, '--out', dsDir(c, args.expName),
        '--speaker', args.speaker || c.expName,
        '--lang', args.lang || 'auto',
        '--model', args.model || c.asrModel,
        '--compute-type', c.asrComputeType,
        '--beam-size', String(c.asrBeamSize),
        '--min-logprob', String(args.minLogprob ?? -1.0)];
      if (c.asrCacheDir) argv.push('--cache-dir', c.asrCacheDir);
      if (args.limit) argv.push('--limit', String(args.limit));
      if (args.allowDownload) argv.push('--allow-download');
      if (args.mergeExisting) argv.push('--merge-existing', args.mergeExisting);

      const r = await runPy(c, 'asr_label.py', argv, 7_200_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'asr', error: 'ASR 无输出', detail: r.tail, rendered: `ASR 失败：\n${r.tail}` };
      }

      const L = [`【ASR 回标】${j.ok ? '✅' : '❌'} ${j.model}`];
      L.push(`设备 ${j.device} · ${j.compute_type} · 语言 ${j.lang}`);
      L.push(`wav ${j.wav_total} → 标出 ${j.labelled} · 剔除 ${j.rejected} · 失败 ${j.failed}`);
      L.push(`list: ${j.list}`);
      if (j.rejected_samples?.length) {
        L.push('', '被剔除的样本（前 10）：');
        for (const s of j.rejected_samples.slice(0, 10)) {
          L.push(`  ${s.wav}  ${s.why}${s.avg_logprob !== undefined ? ` lp=${s.avg_logprob}` : ''}${s.text ? ` "${s.text}"` : ''}`);
        }
      }
      if (j.failed_samples?.length) {
        L.push('', '失败：');
        for (const s of j.failed_samples.slice(0, 5)) L.push(`  ${s}`);
      }
      if (!j.ok) L.push('', `❌ ${j.error || '一条都没标出来'}`);
      else L.push('', '下一步: tts_preprocess');
      return { ok: !!j.ok, stage: 'asr', asr: j, rendered: L.join('\n') };
    }),
  });

  // ── 5.5 · 文本条件：消 UNK ─────────────────────────────────────────────
  register({
    name: 'tts_patch_text',
    description:
      '给 GPT-SoVITS 工程打「消 UNK」文本补丁（日语线）：japanese.py 剥掉无音值符号'
      + '（「」『』～―（）() ），cleaner.py 过滤声调短语边界 "#"。这两类字符都不在 symbols2 的 732 项里，'
      + '会被 cleaner 降级成 UNK —— 代价是所有词边界共享同一个 embedding，模型分不清短停与连读。'
      + '★ 幂等（打过就跳）、自动备份 .bak_dsh_unk、可 revert 回滚、打前核对符号表避免误删有效符号。'
      + '★ 默认只报计划，apply=true 才写盘。'
      + '★ 打了必须重训：实测只打补丁不重训是负收益（PER 0.197→0.236），因为推理条件变了、权重还是旧条件训的。'
      + '给旧权重做对照推理时设 GSV_LEGACY_UNK=1 还原上游行为。',
    parameters: {
      type: 'object',
      properties: {
        apply: { type: 'boolean', description: '真的写盘（默认只报计划）' },
        revert: { type: 'boolean', description: '从 .bak_dsh_unk 回滚' },
        gsvRoot: {
          type: 'string',
          description: '显式指定 GPT-SoVITS 根目录。不传则用配置/自动发现的结果——'
            + '注意自动发现会在路径「不像 GSV」时换成另一个工程，改文件前请核对渲染里的实际路径',
        },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg({});
      const root = args.gsvRoot || c.gsvDir;
      if (!root) {
        return { ok: false, stage: 'patch-text', rendered: '❌ 没有找到 GPT-SoVITS 工程目录；先跑 tts_doctor 或配 gsvDir' };
      }
      const argv = ['--gsv-root', root];
      if (args.apply) argv.push('--apply');
      if (args.revert) argv.push('--revert');

      const r = await runPy(c, 'patch_unk.py', argv, 180_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'patch-text', rendered: `❌ 补丁脚本无输出：\n${r.tail}` };
      }
      if (!j.ok) {
        const L = [`❌ 补丁失败：${j.error || '未知'}`];
        for (const f of j.files || []) L.push(`  ${f.file}: ${f.action || ''} ${f.note || f.error || ''}`);
        for (const e of j.errors || []) L.push(`  ${e.file}: ${e.error}`);
        for (const m of j.missing || []) L.push(`  缺文件: ${m}`);
        return { ok: false, stage: 'patch-text', patch: j, rendered: L.join('\n') };
      }

      if (j.reverted) {
        const L = ['【消 UNK 补丁 · 回滚】'];
        for (const f of j.files || []) L.push(`  ${f.file}: ${f.action}${f.note ? ` （${f.note}）` : ''}`);
        L.push('', '已恢复上游行为。用旧权重推理时本来就是这个状态，训练侧需重新打补丁才能吃到修复。');
        return { ok: true, stage: 'patch-text', patch: j, rendered: L.join('\n') };
      }

      const L = [`【消 UNK 文本补丁｜${j.applied ? '已写盘' : '计划（未写盘）'}】${j.gsv_root}`];
      for (const f of j.files || []) L.push(`  ${f.file}: ${f.action}  ${f.note || ''}`);
      const s = j.symbols || {};
      if (s.checked) {
        L.push(`  符号表: ${s.total} 项${s.conflicts?.length ? ` ❌ 冲突 ${s.conflicts.join(' ')}` : ' · 待剥离字符均不在表内'}`);
      } else {
        L.push(`  ⚠️  符号表未校验：${s.reason || '未知'}（补丁仍会打，但少了这层保护）`);
      }
      const ext = (j.files || []).filter((f) => f.action === 'already-external');
      if (ext.length) {
        L.push('', `ℹ️  ${ext.length} 个文件已有等价的 UNK 修复（不是本工具打的），未改动。`
          + '要换成这套补丁，先手工还原上游原版——回滚只认本工具自己的 .bak_dsh_unk。');
      }
      if (!j.applied) {
        L.push('', '下一步: 确认无误后 apply=true 写盘 → tts_preprocess force=true（重算文本特征）→ tts_run');
      } else {
        L.push('', '下一步: tts_preprocess force=true（★ 旧特征会被跳过，必须重置）→ tts_run。'
          + '★ 必须从 s2 重训到 s1：只打补丁不重训是负收益（推理条件变了、权重没变）。');
      }
      return { ok: true, stage: 'patch-text', patch: j, rendered: L.join('\n') };
    }),
  });

  // ── 6 · 预处理 ─────────────────────────────────────────────────────────
  register({
    name: 'tts_preprocess',
    description:
      '预处理：1-get-text（音素+BERT）→ 2-get-hubert（SSL 特征+32k）→ '
      + '[v2Pro/v2ProPlus 专属] 2-get-sv（7-sv_cn 说话人向量）→ 3-get-semantic（语义 token）。'
      + '★ 这四个脚本**只读环境变量**，不吃 argv。★ 它们有「产物已存在就跳过」逻辑，'
      + '换数据集必须 force=true 重置，否则会拿旧台词去训新音频（最难查的一类错）。'
      + '★ 7-sv_cn 不是可选步骤：缺了上游会把 spec/ssl/wav 一起换成全零继续训，不报错但训出来是废的。',
    parameters: {
      type: 'object',
      properties: {
        stage: { type: 'string', enum: ['all', 'text', 'hubert', 'sv', 'semantic'], description: '默认 all' },
        expName: { type: 'string' },
        listFile: { type: 'string', description: '默认 <语料区>/<expName>/list.txt' },
        wavDir: { type: 'string', description: '默认 <语料区>/<expName>/wav' },
        force: { type: 'boolean', description: '重置旧产物（改名而非删除）后重跑' },
        gpu: { type: 'string', description: 'GPU 序号，默认 0' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg(args.expName ? { expName: args.expName } : {});
      const listFile = args.listFile || join(dsDir(c, args.expName), 'list.txt');
      const wavDir = args.wavDir || join(dsDir(c, args.expName), 'wav');

      const pre = preflightPrep(c, { listFile, wavDir });
      if (!pre.ok) {
        return {
          ok: false, stage: 'preprocess', problems: pre.problems,
          rendered: ['❌ 预处理前置检查未通过：', ...pre.problems.map((p) => `  · ${p}`)].join('\n'),
        };
      }

      let reset = { moved: [], staleDir: '' };
      if (args.force) reset = resetPrepArtifacts(c.expDir);

      const stages = args.stage && args.stage !== 'all'
        ? [args.stage]
        : ['text', 'hubert', ...(needsSv(c.version) ? ['sv'] : []), 'semantic'];
      const results = [];
      const L = [`【预处理】exp=${c.expName} → ${c.expDir}`];
      if (reset.moved.length) L.push(`♻️  已重置旧产物 ${reset.moved.length} 项 → ${reset.staleDir}`);
      L.push(`list.txt: ${countListLines(listFile)} 行`, '');
      if (!needsSv(c.version) && args.stage === 'sv') {
        const msg = `${c.version} 不需要 7-sv_cn（只有 v2Pro/v2ProPlus 要），stage=sv 无意义`;
        return { ok: false, stage: 'preprocess', error: msg, rendered: msg };
      }

      for (const st of stages) {
        const r = await runPrepStage(c, { stage: st, listFile, wavDir, gpu: args.gpu || '0' });
        results.push(r);
        const mark = r.ok ? '✅' : '❌';
        L.push(`${mark} ${st}  (${(r.ms / 1000).toFixed(1)}s) exit=${r.exitCode}`);
        if (st === 'text' && r.textLines !== null) {
          L.push(`   文本特征 ${r.textLines} 行 / list ${countListLines(listFile)} 行`
            + (r.textLines < countListLines(listFile) ? '  ⚠️ 有句子被跳过（语言不在白名单或文本清洗后为空）' : ''));
        }
        if (!r.ok) {
          L.push('   ' + String(r.logTail || '').split('\n').slice(-8).join('\n   '));
          break;   // 前一步失败，后面没有意义
        }
      }

      const allOk = results.length === stages.length && results.every((x) => x.ok);
      if (allOk) L.push('', '下一步: tts_run（s2 → s1）');
      return { ok: allOk, stage: 'preprocess', results, reset, rendered: L.join('\n') };
    }),
  });

  // ── 7 · 训练 ───────────────────────────────────────────────────────────
  register({
    name: 'tts_run',
    description:
      '启动训练（后台，不阻塞）：phase=s2（SoVITS 声学）/ s1（GPT 语义）/ all（串联，s2 落权重才起 s1）。'
      + '★ 顺序不可倒、不可并行。门禁看的是**权重文件真的出现**，不是退出码——'
      + 'epoch 小于保存间隔时退出码 0 但一个权重都没有，那是失败。'
      + 'cwd 固定为工程根（权重目录是相对路径）。启动后用 tts_status 看进度。',
    parameters: {
      type: 'object',
      properties: {
        phase: { type: 'string', enum: ['all', 's2', 's1'], description: '默认 all' },
        expName: { type: 'string' },
        version: { type: 'string', description: '覆盖版本（决定底模与输出目录）' },
        s2Epochs: { type: 'number' },
        s2BatchSize: { type: 'number' },
        s2SaveEveryEpoch: { type: 'number' },
        s1Epochs: { type: 'number' },
        s1BatchSize: { type: 'number' },
        s1SaveEveryNEpoch: { type: 'number' },
        gpu: { type: 'string', description: 'GPU 序号，默认 0' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const over = {};
      if (args.expName) over.expName = args.expName;
      if (args.version) over.version = args.version;
      if (args.s2Epochs) over.s2Epochs = args.s2Epochs;
      if (args.s2BatchSize) over.s2BatchSize = args.s2BatchSize;
      if (args.s2SaveEveryEpoch) over.s2SaveEveryEpoch = args.s2SaveEveryEpoch;
      if (args.s1Epochs) over.s1Epochs = args.s1Epochs;
      if (args.s1BatchSize) over.s1BatchSize = args.s1BatchSize;
      if (args.s1SaveEveryNEpoch) over.s1SaveEveryNEpoch = args.s1SaveEveryNEpoch;
      const c = withCfg(over);

      const phase = args.phase || 'all';

      // 前置校验
      if (phase === 'all' || phase === 's2') {
        const pre = preflight(c, { phase: 's2' });
        if (!pre.ok) {
          return {
            ok: false, stage: 'run', problems: pre.problems,
            rendered: ['❌ 不能开训：', ...pre.problems.map((p) => `  · ${p}`)].join('\n'),
          };
        }
      }
      const w = scanWeights(c);
      if (phase === 's1' && w.sovits.length === 0) {
        const msg = `没有已落盘的 s2 权重可续训 s1（${w.sovitsDir} 为空）。\n先跑 phase=s2。`;
        return { ok: false, stage: 'run', error: msg, rendered: msg };
      }
      if (phase === 's1') {
        const pre = preflight(c, { phase: 's1' });
        if (!pre.ok) {
          return {
            ok: false, stage: 'run', problems: pre.problems,
            rendered: ['❌ 不能开训 s1：', ...pre.problems.map((p) => `  · ${p}`)].join('\n'),
          };
        }
      }

      // 生成配置
      mkdirSync(c.workspace.logs, { recursive: true });
      // ★ 权重/检查点目录必须先建：上游 s2 的 save_checkpoint 不建 logs_s2_<ver>，
      //   命令行直跑会在保存那一刻抛 FileNotFoundError，整轮训练白跑。
      const dirs = ensureTrainDirs(c);
      const tmpBase = join(c.expDir, '_train_tmp');
      const s2Path = join(tmpBase, 'tmp_s2.json');
      const s1Path = join(tmpBase, 'tmp_s1.yaml');

      let mk2 = null;
      if (phase === 'all' || phase === 's2') {
        mk2 = await makeTrainConfig(c, { kind: 's2', overrides: s2Overrides(c), tmpPath: s2Path });
        if (!mk2.ok) {
          return {
            ok: false, stage: 'run', error: 's2 配置生成失败',
            rendered: `❌ s2 配置生成失败：${mk2.error || mk2.stderr}`,
          };
        }
      }
      let mk1 = null;
      if (phase === 'all' || phase === 's1') {
        mk1 = await makeTrainConfig(c, { kind: 's1', overrides: s1Overrides(c), tmpPath: s1Path });
        if (!mk1.ok) {
          return {
            ok: false, stage: 'run', error: 's1 配置生成失败',
            rendered: `❌ s1 配置生成失败：${mk1.error || mk1.stderr}`,
          };
        }
      }

      const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
      const id = `${c.expName}-${phase}-${stamp}`;
      const logPath = join(c.workspace.logs, `${id}.log`);

      let job;
      let cmdDesc;
      if (phase === 'all') {
        // 一条自持进程串 s2 → s1：门禁、单一日志、失败即停
        job = spawnJob({
          cmd: c.gsvEnv,
          args: [join(PY_DIR, 'run_two_stage.py'),
            '--gsv-dir', c.gsvDir,
            '--python', c.gsvEnv,
            '--s2-script', c.profile.s2Script,
            '--s2-config', s2Path,
            '--s1-config', s1Path,
            '--weight-dir', c.sovitsWeightDir,
            '--exp-name', c.expName,
            '--gpu', args.gpu || '0'],
          cwd: c.gsvDir,
          env: pythonEnv(c.gsvEnv, {}),
          logPath,
        });
        cmdDesc = 'run_two_stage.py（s2 → s1）';
      } else if (phase === 's2') {
        job = startS2({ ...c, gsvEnv: c.gsvEnv }, { configPath: s2Path, logPath, gpu: args.gpu || '0' });
        cmdDesc = c.profile.s2Script;
      } else {
        job = startS1({ ...c, gsvEnv: c.gsvEnv }, { configPath: s1Path, logPath, gpu: args.gpu || '0' });
        cmdDesc = 'GPT_SoVITS/s1_train.py';
      }

      upsertJob(c.workspace.jobs, {
        id, phase, expName: c.expName, version: c.version,
        pid: job.pid, logPath, startedAt: new Date().toISOString(),
        s2Config: s2Path, s1Config: s1Path,
        weightDir: c.sovitsWeightDir, gptDir: c.gptWeightDir,
        epochs: { s2: c.s2Epochs, s1: c.s1Epochs },
        batch: { s2: c.s2BatchSize, s1: c.s1BatchSize },
      });

      const L = [`【已启动训练】${c.expName} · phase=${phase} · ${c.version}`];
      L.push(`进程 ${cmdDesc}  pid=${job.pid}`);
      L.push(`日志 ${logPath}`);
      L.push('');
      L.push(`s2: epochs=${c.s2Epochs} batch=${c.s2BatchSize} 每 ${c.s2SaveEveryEpoch} epoch 存权重`);
      if (phase !== 's2') L.push(`s1: epochs=${c.s1Epochs} batch=${c.s1BatchSize}`);
      L.push(`权重落点: ${c.sovitsWeightDir}`);
      if (phase !== 's2') L.push(`            ${c.gptWeightDir}`);
      if (dirs.made.length) L.push(`已建目录: ${dirs.made.join(' · ')}`);
      L.push('');
      L.push('★ 后台运行，不阻塞。看进度用 tts_status，要停用 tts_stop。');
      L.push(`  预计：8GB 显存下 batch=${c.s2BatchSize}，s2 每 epoch 约数分钟量级（随语料量线性）。`);
      return {
        ok: true, stage: 'run', jobId: id, pid: job.pid, logPath, phase,
        s2Config: s2Path, s1Config: s1Path, rendered: L.join('\n'),
      };
    }),
  });

  // ── 8 · 进度 ───────────────────────────────────────────────────────────
  register({
    name: 'tts_status',
    description:
      '训练进度：最近一次任务的进程存活、日志尾部、从日志解析的 epoch/loss、'
      + '以及**权重目录里真实落盘的文件**。★ 权重文件是最可靠的事实——'
      + '日志可能滚屏、进程可能已被回收，但落盘的 .pth/.ckpt 不会骗人。',
    parameters: {
      type: 'object',
      properties: { lines: { type: 'number', description: '日志尾部行数（默认 20）' } },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const jobs = readJobs(cfg.workspace.jobs);
      const job = jobs[jobs.length - 1];
      const w = scanWeights(cfg);
      const L = [`【训练状态】${cfg.expName} · ${cfg.version}`];

      if (!job) {
        L.push('（没有训练记录）');
        L.push(`权重目录: ${w.sovitsDir}`);
        L.push(`  s2 权重 ${w.sovits.length} 个 · s1 权重 ${w.gpt.length} 个`);
        return { ok: true, stage: 'status', weightCount: w.sovits.length + w.gpt.length, rendered: L.join('\n') };
      }

      const alive = await jobAlive(job);
      const text = tailFile(job.logPath, args.lines || 20);
      const fin = detectFinish(tailFile(job.logPath, 400));
      const prog = parseProgress(tailFile(job.logPath, 400));

      L.push(`任务 ${job.id}  phase=${job.phase}`);
      L.push(`进程 ${alive ? `运行中 (pid ${job.pid})` : '已结束 / 不在了'}`);
      if (prog.epoch !== null) L.push(`进度 epoch=${prog.epoch}${prog.step !== null ? ` step=${prog.step}/${prog.total ?? '?'}` : ''}${prog.pct !== null ? ` ${prog.pct}%` : ''}`);
      if (prog.lossG !== null || prog.lossD !== null) L.push(`loss G=${prog.lossG ?? '?'} D=${prog.lossD ?? '?'}`);
      if (fin.failed) L.push(`⛔ 失败迹象: ${fin.reason}`);
      else if (fin.done) L.push('✅ 日志出现完成标记');

      L.push('', `s2 权重 (${w.sovitsDir}):`);
      if (!w.sovits.length) L.push('  (无)');
      for (const x of w.sovits.slice(-6)) L.push(`  ${x.file}  ${fmtBytes(x.sizeBytes)}`);
      L.push(`s1 权重 (${w.gptDir}):`);
      if (!w.gpt.length) L.push('  (无)');
      for (const x of w.gpt.slice(-6)) L.push(`  ${x.file}  ${fmtBytes(x.sizeBytes)}`);

      L.push('', '── 日志尾部 ──');
      L.push(text || '(空)');

      const done = w.sovits.length > 0 && (job.phase === 'all' || job.phase === 's2');
      if (done) L.push('\n下一步: tts_infer（试听一句）→ tts_register');
      return {
        ok: alive || w.sovits.length > 0, stage: 'status',
        alive, jobId: job.id, progress: prog, finish: fin,
        weights: { sovits: w.sovits, gpt: w.gpt }, rendered: L.join('\n'),
      };
    }),
  });

  // ── 9 · 停止 ───────────────────────────────────────────────────────────
  register({
    name: 'tts_stop',
    description: '停止正在跑的训练（taskkill /T /F 连 dataloader 子进程一起收）。已落盘的权重保留。',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    ...wrap(async () => {
      const jobs = readJobs(cfg.workspace.jobs);
      const job = jobs[jobs.length - 1];
      if (!job) return { ok: false, stage: 'stop', rendered: '没有训练记录可停。' };
      const alive = await jobAlive(job);
      if (!alive) {
        return { ok: true, stage: 'stop', rendered: `任务 ${job.id} 的进程已不在（pid ${job.pid}），无需停止。` };
      }
      const r = await killJob(job);
      const w = scanWeights(cfg);
      const L = [`${r.ok ? '✅ 已停止' : '❌ 停止失败'} 任务 ${job.id} (pid ${job.pid})`];
      if (!r.ok) L.push(String(r.stderr || '').slice(-400));
      L.push(`已落版权重 ${w.sovits.length} 个（保留，可续训或直接推理）`);
      return { ok: r.ok, stage: 'stop', rendered: L.join('\n') };
    }),
  });

  // ── 10 · 试听 ──────────────────────────────────────────────────────────
  register({
    name: 'tts_infer',
    description:
      '用训好的权重合成一句话，产出 wav。默认自动取该实验最新落盘的 .pth + .ckpt。'
      + 'asrCheck=true 时用 faster-whisper 把合成结果再转写回来与原文比对——'
      + '★「能出声」不等于「念对了」，只听有声音是最容易放过的错。'
      + '参考音频决定音色底子与情感基调，3~10 秒、干净、无 BGM。',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: '要合成的台词' },
        lang: { type: 'string', description: 'zh/ja/en（默认取配置）' },
        expName: { type: 'string' },
        version: { type: 'string' },
        sovits: { type: 'string', description: '手动指定 .pth（默认取最新）' },
        gpt: { type: 'string', description: '手动指定 .ckpt（默认取最新）' },
        gptMode: {
          type: 'string',
          enum: ['auto', 'trained', 'base'],
          description: 's1 用哪个：auto=按「目标语言 vs 训练语料语言」自动选（默认）· '
            + 'trained=强制自训 · base=强制底模 s1v3。'
            + '★ 跨语言（如日文语料念中文）必须用 base：自训 s1 会把音素→语义的映射训窄到训练语言。'
            + '注意：s1 决定**断句、停顿、重音、语气结构**，s2 只决定音色 —— '
            + '跨语言时若 s1 用错，读音可能是对的，但整句的语气仍是外语的。',
        },
        ref: { type: 'string', description: '参考音频路径（默认取音色清单里该实验的记录）' },
        refText: { type: 'string', description: '参考音频的台词' },
        refLang: {
          type: 'string',
          description: '参考音频语言 zh/ja/en/ko/yue，默认 auto（按参考台词判）。'
            + '★ 必须与参考音频真实语言一致：报错语言会让输出塌成静音',
        },
        out: { type: 'string', description: '输出 wav 路径' },
        asrCheck: { type: 'boolean', description: '合成后 ASR 回读并比对' },
        speed: {
          type: 'number',
          description: '语速系数（1.0 = 与训练语料同速，越小越慢）。默认按目标语言自动取：'
            + `zh ${LANG_SPEED.zh} · en ${LANG_SPEED.en} · ja ${LANG_SPEED.ja}。`
            + '★ 实测这是纯时长缩放（改 s2 latent 长度），F0 偏移仅 -1~-3%，不变调。'
            + '中文散文想更舒缓可用 0.75；低于 0.6 会开始拖沓。',
        },
        split: {
          type: 'string',
          description: '分句策略：cut5（默认，按标点切小句）· cut0（整段不切，朗读韵律更连贯，'
            + '但超过约 15 秒的文本容易中途崩）',
        },
        temperature: {
          type: 'number',
          description: '采样温度（默认 1.0）。★ 欠拟合的权重调低到 0.6~0.7 能明显减少乱读与跳字，'
            + '但过低会念得平板',
        },
        topK: { type: 'number', description: 'top-k（默认 15）。欠拟合时降到 5 更稳' },
        topP: { type: 'number', description: 'top-p（默认 1.0）' },
        repetitionPenalty: {
          type: 'number',
          description: '重复惩罚（默认 1.35）。调高可抑制复读，过高会导致吞字',
        },
      },
      required: ['text'],
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg({
        ...(args.expName ? { expName: args.expName } : {}),
        ...(args.version ? { version: args.version } : {}),
      });

      const reg = findVoice(c, c.expName);
      const w = scanWeights(c);
      const sovits = args.sovits || (w.sovits.length ? w.sovits[w.sovits.length - 1].path : '');
      // s1：默认 auto —— 按「目标语言 vs 训练语料语言」自动选。
      // ★ 不再拿 reg.gptMode 当推理默认：那是注册那一刻的静态快照，
      //   会把 auto 锁死（实测注册时记成 'base'，之后每次中文都被强制 base，
      //   而 auto 本来能做同样判断，却因为被 reg 抢先而拿不到 note，也失去了
      //   将来 s1 训够之后自动切回 trained 的能力）。
      //   显式传参 > auto；reg 里的值只作为音色档案保留，不参与决策。
      // ★ 字段名是 c.language，不是 c.lang —— 写错的后果是静默退回 'zh'，
      //   中文场景看不出问题，日文/英文配置会被判成跨语言。
      const targetLang = args.lang || c.language || 'zh';
      const gm = resolveGptMode(c, args.gptMode || 'auto', targetLang, w.gpt.length > 0);
      const gptMode = gm.mode;
      const gptTrained = w.gpt.length ? w.gpt[w.gpt.length - 1].path : '';
      const gpt = args.gpt || (gptMode === 'base' ? (c.pretrainedS1 || '') : gptTrained);
      if (gptMode === 'base' && !gpt) {
        const msg = `找不到底模 s1（${c.profile?.s1 || 's1v3.ckpt'}），检查工程目录: ${c.gsvDir}`;
        return { ok: false, stage: 'infer', error: msg, rendered: msg };
      }
      if (!sovits || !gpt) {
        const msg = `没有可用权重，先跑 tts_run 训练。目录: ${w.sovitsDir} / ${w.gptDir}`;
        return { ok: false, stage: 'infer', error: msg, rendered: msg };
      }

      const ref = args.ref || reg?.ref || '';
      if (!ref || !existsSync(ref)) {
        const msg = `没有可用的参考音频。\n`
          + `参考音频是「音色底子」的来源，必须有：用 tts_register 记录一条，或直接传 ref。\n`
          + (ref ? `(当前路径不存在: ${ref})` : '(未指定)');
        return { ok: false, stage: 'infer', error: msg, rendered: msg };
      }

      const outWav = args.out || join(c.workspace.reports, `${c.expName}_infer.wav`);
      // ★ 语速：默认按目标语言取（zh 0.85 / ja 1.0 / en 0.9），不再任由模型用训练语料
      //   的语速念。s2 的训练语料是日文，直接念中文会 4.9 字/秒，音节被压缩。
      const speed = resolveSpeed(targetLang, args.speed);
      const argv = ['--gsv-dir', c.gsvDir, '--version', c.version,
        '--sovits', sovits, '--gpt', gpt, '--ref', ref,
        '--text', args.text, '--lang', targetLang,
        '--out', outWav, '--device', 'cuda',
        '--speed', String(speed)];
      if (args.refText || reg?.refText) argv.push('--ref-text', args.refText || reg.refText);
      // ★ 不给 ref-lang 时交给 py 端 auto 判（按参考台词）。历史上默认 "zh" 是错的：
      //   日文参考音频 + prompt_lang=zh 会合成出 0.68s 的静音。
      const refLang = args.refLang || reg?.lang || 'auto';
      argv.push('--ref-lang', refLang);
      if (args.split) argv.push('--split', String(args.split));
      // ★ 采样默认取**基线**（temperature 0.6 / top-k 5），不是 py 端的通用默认（1.0 / 15）。
      //   基线是用户逐条听完选定的那条；插件默认路径必须能复现它，
      //   否则「基线」就只是 reports 里躺着一个样本，谁也不知道它怎么来的。
      //   s2 只训 8 轮，温度 1.0 时韵律容易飘；0.6 是稳定性与自然度之间的实测落点。
      const baselineSampling = args.temperature == null && args.topK == null;
      const temp = args.temperature ?? BASELINE.temperature;
      const topK = args.topK ?? BASELINE.topK;
      const topP = args.topP ?? BASELINE.topP;
      argv.push('--temperature', String(temp), '--top-k', String(topK), '--top-p', String(topP));
      const sampling = { temperature: temp, topK };
      if (args.topP != null) sampling.topP = topP;
      if (args.repetitionPenalty != null) {
        argv.push('--repetition-penalty', String(args.repetitionPenalty));
        sampling.repetitionPenalty = args.repetitionPenalty;
      }

      const r = await runPy(c, 'gsv_infer.py', argv, 900_000);
      const j = r.json;
      if (!j) {
        return { ok: false, stage: 'infer', error: '推理无输出', detail: r.tail, rendered: `推理失败：\n${r.tail}` };
      }
      if (!j.ok) {
        const L = [`❌ 推理失败（${j.stage}）: ${j.error}`];
        if (j.trace) L.push('', j.trace);
        return { ok: false, stage: 'infer', detail: j, rendered: L.join('\n') };
      }

      const L = ['【试听合成】✅'];
      L.push(`wav: ${j.out}`);
      L.push(`时长 ${j.durationSec}s · ${j.sampleRate}Hz · peak=${j.peak} rms=${j.rms}`);
      // 语速自证：中文散文 3.8~4.2 字/秒，播报 ~5.0。数字不对就是又念快了/拖了。
      if (j.charsPerSec) {
        const cp = j.charsPerSec;
        const verdict = cp > 4.8 ? '偏快（播报速度）' : cp > 4.3 ? '略快' : cp >= 3.4 ? '正常朗读' : '偏慢';
        L.push(`语速 ${cp} 字/秒（${verdict}）`);
      }
      L.push(`加载 ${j.loadSec}s · 合成 ${j.synthSec}s · 显存 ${j.vramGB}GB`);
      L.push(`权重: ${sovits.split('/').pop()} + ${gpt.split('/').pop()}`
        + `${gptMode === 'base' ? '（s1 用底模，跨语言可用）' : ''}`);
      if (gm.note) L.push(`s1 选择: ${gm.note}`);
      const speedNote = args.speed != null
        ? `（手动指定）`
        : (LANG_SPEED[targetLang] != null ? `（${targetLang} 默认）` : '');
      L.push(`语速: ${speed}${speedNote}`);
      if (Object.keys(sampling).length) {
        L.push(`采样: ${Object.entries(sampling).map(([k, v]) => `${k}=${v}`).join(' · ')}`
          + `${baselineSampling ? `（基线 ${BASELINE.label}）` : '（手动指定）'}`);
      }
      L.push(`参考语言: ${j.refLang}${j.refLangAuto ? '（自动判定）' : ''}`
        + `${j.refTextUsed ? '' : ' · 未给参考台词（音色更不稳）'}`);
      if (j.refLang !== (args.lang || c.language)) {
        L.push(`ℹ️  参考语言(${j.refLang}) 与目标语言(${args.lang || c.language}) 不同 —— 跨语言合成可以，`
          + `但参考台词必须与参考音频同语言，否则输出会塌成静音`);
      }
      if (j.durationSec < 0.8) {
        L.push('⚠️  输出过短。常见原因：训练轮数太少（先跑够 epoch）、参考音频太短或与目标音色差太远、'
          + '或 prompt_lang 与参考音频不符');
      }

      let asr = null;
      if (args.asrCheck) {
        if (!c.asrCacheDir && !existsSync(join(c.gsvDir, '_hf_home'))) {
          L.push('\n⚠️  没有 ASR 缓存，跳过回读（先用 tts_asr 拉一次模型，或手动放缓存）');
        } else {
          const ra = await runPy(c, 'asr_label.py',
            ['--wav-file', outWav, '--lang', 'auto',
              '--model', c.asrModel, '--compute-type', c.asrComputeType,
              ...(c.asrCacheDir ? ['--cache-dir', c.asrCacheDir] : [])], 600_000);
          asr = ra.json;
          if (asr?.ok) {
            const gate = asrGate(asr.text, args.text, asr.avgLogprob);
            L.push('', `ASR 回读: 「${asr.text}」`);
            L.push(`原文:     「${args.text}」`);
            L.push(`判定: ${gate.ok ? '✅ 念对了' : '❌ 不可用'}`
              + ` · 字错率 ${(gate.cer * 100).toFixed(1)}%（${gate.dist}/${gate.len} 字）`
              + ` · 置信度 ${gate.lp === null ? 'n/a' : gate.lp}`);
            if (!gate.cerOk) {
              L.push(`  ↳ 字错率超过 15% 上限：念错/漏字/跳句，不能只看"有没有声音"`);
            }
            if (!gate.lpOk) {
              L.push('  ↳ 置信度低于 -0.35：ASR 自己都不确定听到的是这些音，'
                + '常见于跨语言音素映射错误（见 README 4.19）或轮数严重不足');
            }
          } else {
            L.push(`\nASR 回读失败: ${asr?.error || ra.tail.slice(-300)}`);
          }
        }
      }
      return { ok: true, stage: 'infer', infer: j, asr, sovits, gpt, rendered: L.join('\n') };
    }),
  });

  // ── 11 · 注册音色 ──────────────────────────────────────────────────────
  register({
    name: 'tts_register',
    description:
      '把训好的权重登记进音色清单（voices.json），让「用 XX 的声音念这句」有名字可寻。'
      + 'GPT-SoVITS 本身没有音色库概念（每次推理现给权重路径），这份清单是本插件补的索引。'
      + '同时记录参考音频与参考文本 —— 推理时必须带参考音频才有音色底子。',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '音色名（默认实验名）' },
        expName: { type: 'string' },
        version: { type: 'string' },
        sovits: { type: 'string', description: '默认取最新 .pth' },
        gpt: { type: 'string', description: '默认取最新 .ckpt' },
        gptMode: {
          type: 'string',
          enum: ['trained', 'base'],
          description: '记录 s1 用自训还是底模。★ 跨语言音色建议注册成 base',
        },
        ref: { type: 'string', description: '参考音频（推理必用）' },
        refText: { type: 'string', description: '参考音频台词' },
        lang: { type: 'string', description: '主要语言' },
        notes: { type: 'string', description: '备注：语料来源、时长等' },
      },
      additionalProperties: false,
    },
    ...wrap(async (args) => {
      const c = withCfg({
        ...(args.expName ? { expName: args.expName } : {}),
        ...(args.version ? { version: args.version } : {}),
      });
      const w = scanWeights(c);
      const sovits = args.sovits || (w.sovits.length ? w.sovits[w.sovits.length - 1].path : '');
      const gptMode = args.gptMode || 'trained';
      const gptTrained = w.gpt.length ? w.gpt[w.gpt.length - 1].path : '';
      const gpt = args.gpt || (gptMode === 'base' ? (c.pretrainedS1 || '') : gptTrained);
      if (!sovits || !gpt) {
        const msg = `没有可注册的权重（s2=${sovits || '无'} s1=${gpt || '无'}）。先跑 tts_run。`;
        return { ok: false, stage: 'register', error: msg, rendered: msg };
      }

      const file = c.workspace.voices;
      let reg = [];
      try { reg = JSON.parse(readFileSync(file, 'utf8')); } catch { reg = []; }
      if (!Array.isArray(reg)) reg = [];

      const vname = (args.name || c.expName).trim();
      const entry = {
        name: vname,
        expName: c.expName,
        version: c.version,
        sovits, gpt,
        gptMode,
        ref: args.ref || findVoice(c, c.expName)?.ref || '',
        refText: args.refText || '',
        lang: args.lang || c.language,
        notes: args.notes || '',
        registeredAt: new Date().toISOString(),
      };
      const i = reg.findIndex((x) => x.name === vname);
      if (i >= 0) reg[i] = { ...reg[i], ...entry };
      else reg.push(entry);

      mkdirSync(join(file, '..'), { recursive: true });
      writeFileSync(file, JSON.stringify(reg, null, 2), 'utf8');

      const L = [`✅ 已注册音色「${vname}」${i >= 0 ? '（覆盖同名项）' : ''}`];
      L.push(`  版本 ${c.version}`);
      L.push(`  s2 ${sovits}`);
      L.push(`  s1 ${gpt}${gptMode === 'base' ? '（底模 · 跨语言可用）' : '（自训 · 仅训练语言可靠）'}`);
      L.push(`  参考音频 ${entry.ref || '⚠️ 未设置（推理时需要补）'}`);
      L.push(`  清单 ${file}（共 ${reg.length} 项）`);
      if (!entry.ref) L.push('\n⚠️  没有参考音频的记录无法直接推理，用 tts_register 带 ref 重新登记。');
      return { ok: true, stage: 'register', entry, total: reg.length, rendered: L.join('\n') };
    }),
  });

  // ── 12 · 总览 ──────────────────────────────────────────────────────────
  register({
    name: 'tts_overview',
    description:
      '一屏自查：工程与版本、语料区各实验的条数与时长、预处理产物齐不齐、权重与音色清单、最近任务。收工前跑一次。',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    ...wrap(async () => {
      const c = cfg;
      const w = scanWeights(c);
      const L = [`【总览】GPT-SoVITS ${c.version} · ${c.gsvDir || '(未发现工程)'}`];

      // 数据集
      L.push('', `── 数据集 (${c.workspace.datasets}) ──`);
      let exps = [];
      try {
        exps = readdirSync(c.workspace.datasets, { withFileTypes: true })
          .filter((d) => d.isDirectory()).map((d) => d.name);
      } catch { exps = []; }
      if (!exps.length) L.push('  (空)');
      for (const e of exps) {
        const dir = join(c.workspace.datasets, e);
        const lf = join(dir, 'list.txt');
        const n = existsSync(lf) ? countListLines(lf) : 0;
        const expDir = join(c.gsvDir, 'logs', e);
        const prepped = existsSync(join(expDir, PREP_ARTIFACTS.text)) && existsSync(join(expDir, PREP_ARTIFACTS.semantic));
        L.push(`  ${e.padEnd(20)} list ${String(n).padStart(5)} 条  预处理 ${prepped ? '✅' : '—'}`);
      }

      // 权重
      L.push('', `── 权重 ──`);
      L.push(`  s2 ${w.sovits.length} 个 → ${w.sovitsDir}`);
      for (const x of w.sovits.slice(-4)) L.push(`     ${x.file} ${fmtBytes(x.sizeBytes)}`);
      L.push(`  s1 ${w.gpt.length} 个 → ${w.gptDir}`);
      for (const x of w.gpt.slice(-4)) L.push(`     ${x.file} ${fmtBytes(x.sizeBytes)}`);

      // 音色
      const vs = listVoices(c);
      L.push('', `── 音色清单 (${c.workspace.voices}) ──`);
      if (!vs.length) L.push('  (空)');
      for (const v of vs) L.push(`  ${v.exists ? '✓' : '✗'} ${v.name}  ${v.version}  ref=${v.ref ? '有' : '无'}`);

      // 最近任务
      const jobs = readJobs(c.workspace.jobs);
      const job = jobs[jobs.length - 1];
      L.push('', '── 最近任务 ──');
      if (!job) L.push('  (无)');
      else {
        const alive = await jobAlive(job);
        const fin = detectFinish(tailFile(job.logPath, 400));
        L.push(`  ${job.id}  ${alive ? '运行中' : '已结束'}${fin.failed ? ' ⛔失败' : fin.done ? ' ✅完成' : ''}`);
        L.push(`  日志 ${job.logPath}`);
      }

      const okAll = w.sovits.length > 0 && w.gpt.length > 0;
      L.push('', okAll ? '✅ 至少有一套完整权重（s2 + s1），可直接 tts_infer' : '⚠️  还没有完整权重对，继续 tts_run');
      return {
        ok: okAll, stage: 'overview',
        datasets: exps.length, sovits: w.sovits.length, gpt: w.gpt.length, voices: vs.length,
        rendered: L.join('\n'),
      };
    }),
  });

  logger.info(`[tts-train-pipeline] 已注册 12 个工具 · gsvDir=${cfg.gsvDir || '(未发现)'} · version=${cfg.version}`);
}
