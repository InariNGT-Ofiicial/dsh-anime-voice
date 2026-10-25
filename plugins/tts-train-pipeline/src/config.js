/**
 * 配置：内置默认值 + 自动发现 + 版本档案 + 合并/归一化。
 *
 * 设计原则（沿用 dsh-train-pipeline / dsh-cover-pipeline 的实证经验）：
 *   1. **默认值是空的**，不把某一台机器的 `D:/AI-Voice` 写死 —— 插件要能分发。
 *   2. 自动发现只在「配置值不成立」时补，绝不覆盖用户明确配置且有效的路径。
 *   3. 所有路径统一成**正斜杠绝对路径**，避免 Windows 反斜杠在 JSON/日志里的转义问题。
 *   4. 可选 `tts.local.json`（仓库根，git 忽略）作为**本机覆盖层**，优先级最高。
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoPath = (rel) => fileURLToPath(new URL('../' + rel, import.meta.url)).replace(/\\/g, '/').replace(/\/+$/, '');

export const repoRoot = repoPath('.');
export const PY_DIR = repoPath('py');
export const LOCAL_CONFIG = repoPath('tts.local.json');

/** 机器相关的默认值。空字符串 = 交给自动发现。 */
export const DEFAULTS = {
  gsvDir: '',
  gsvEnv: '',

  songRoot: '',
  workRoot: '',

  version: 'v2ProPlus',

  expName: 'tts-test',
  s2BatchSize: 4,
  s2Epochs: 8,
  s2SaveEveryEpoch: 4,
  s1BatchSize: 4,
  s1Epochs: 10,
  s1SaveEveryNEpoch: 5,
  textLowLrRate: 0.4,
  gradCkpt: true,

  sampleRate: 32000,
  // 2.0 而非 GPT-SoVITS 官方建议的 3.0：galgame / 番剧单句语音中位只有约 3s，
  // 按 3.0 卡会丢掉近一半素材（实测芳乃 630 条中 <3s 占 50%）。
  minSec: 2.0,
  maxSec: 10.0,
  targetHours: 0.5,
  language: 'zh',

  asrModel: 'large-v3',
  asrComputeType: 'float16',
  asrBeamSize: 5,
  asrCacheDir: '',

  ffmpeg: '',
  timeoutMs: 7_200_000,
  maxOutputChars: 6000,
};

const PATH_KEYS = [
  'gsvDir', 'gsvEnv', 'songRoot', 'workRoot', 'ffmpeg', 'asrCacheDir',
];

/**
 * 判定「这一层就是 GPT-SoVITS 工程」的标记。
 * 三个都必须存在，缺任何一个都说明目录指错（指向了父目录或整合包外层）。
 */
export const GSV_MARKERS = [
  'webui.py',
  'GPT_SoVITS/s2_train.py',
  'GPT_SoVITS/prepare_datasets/1-get-text.py',
  'GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
  'GPT_SoVITS/prepare_datasets/3-get-semantic.py',
  'GPT_SoVITS/s1_train.py',
];

/**
 * 版本档案 —— 全部抄自上游 `config.py`（pretrained_sovits_name / pretrained_gpt_name）
 * 与 `webui.py` 的两处推导，**不要凭感觉改**：
 *
 *   s2D = pretrained_sovits_name[version].replace('s2G', 's2D')      （v3/v4 为空）
 *   config_file = s2.json      当 version ∉ {v2Pro, v2ProPlus}
 *               = s2<version>.json  当 version ∈ {v2Pro, v2ProPlus}
 *   s1 = pretrained_gpt_name[version]
 *
 * ★ 最容易踩的一条：**v2Pro / v2ProPlus 的 s1 底模是 s1v3.ckpt**，
 *   不是 v2final 那一套。用错能训，但音色收敛会明显变差，且很难查。
 */
export const VERSION_PROFILES = {
  v1: {
    s2G: 'GPT_SoVITS/pretrained_models/s2G488k.pth',
    s2D: 'GPT_SoVITS/pretrained_models/s2D488k.pth',
    s1: 'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2.json',
    s2Script: 'GPT_SoVITS/s2_train.py',
    sovitsWeights: 'SoVITS_weights',
    gptWeights: 'GPT_weights',
  },
  v2: {
    s2G: 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth',
    s2D: 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth',
    s1: 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2.json',
    s2Script: 'GPT_SoVITS/s2_train.py',
    sovitsWeights: 'SoVITS_weights_v2',
    gptWeights: 'GPT_weights_v2',
  },
  v2Pro: {
    s2G: 'GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth',
    s2D: 'GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth',
    s1: 'GPT_SoVITS/pretrained_models/s1v3.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2v2Pro.json',
    s2Script: 'GPT_SoVITS/s2_train.py',
    sovitsWeights: 'SoVITS_weights_v2Pro',
    gptWeights: 'GPT_weights_v2Pro',
  },
  v2ProPlus: {
    s2G: 'GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth',
    s2D: 'GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth',
    s1: 'GPT_SoVITS/pretrained_models/s1v3.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2v2ProPlus.json',
    s2Script: 'GPT_SoVITS/s2_train.py',
    sovitsWeights: 'SoVITS_weights_v2ProPlus',
    gptWeights: 'GPT_weights_v2ProPlus',
  },
  v3: {
    s2G: 'GPT_SoVITS/pretrained_models/s2Gv3.pth',
    s2D: '',
    s1: 'GPT_SoVITS/pretrained_models/s1v3.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2.json',
    s2Script: 'GPT_SoVITS/s2_train_v3_lora.py',
    sovitsWeights: 'SoVITS_weights_v3',
    gptWeights: 'GPT_weights_v3',
  },
  v4: {
    s2G: 'GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth',
    s2D: '',
    s1: 'GPT_SoVITS/pretrained_models/s1v3.ckpt',
    s2Config: 'GPT_SoVITS/configs/s2.json',
    s2Script: 'GPT_SoVITS/s2_train_v3_lora.py',
    sovitsWeights: 'SoVITS_weights_v4',
    gptWeights: 'GPT_weights_v4',
  },
};

/** 预处理三件套的文件名（它们**只读环境变量**，不吃 argv —— 这是上游的既定行为） */
export const PREP_SCRIPTS = {
  text: 'GPT_SoVITS/prepare_datasets/1-get-text.py',
  hubert: 'GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
  // ★ v2Pro / v2ProPlus 专属第 4 步（webui 里是 `if "Pro" in version` 才跑）。
  //   产出 7-sv_cn/<wav名>.pt。缺了它 data_utils 会在 try 里抛异常，
  //   然后把 spec/ssl/wav **一起换成全零**继续训 —— 不报错，但训出来是废的。
  sv: 'GPT_SoVITS/prepare_datasets/2-get-sv.py',
  semantic: 'GPT_SoVITS/prepare_datasets/3-get-semantic.py',
};

/** 预处理要用的 BERT / SSL / SV 权重（相对 gsvDir） */
export const PREP_ASSETS = {
  bert: 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
  hubert: 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
  sv: 'GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt',
};

/** list.txt 里合法的语言标记（1-get-text.py 的白名单，非此列会静默跳过该句） */
export const LANGUAGES = ['zh', 'ja', 'en', 'ko', 'yue'];

/**
 * 该版本是否需要跑 2-get-sv.py 生成 7-sv_cn 特征。
 * ★ 判断方式照抄上游 webui.py 的 `if "Pro" in version` —— v2Pro / v2ProPlus 要，
 *   v1/v2/v3/v4 都不要。用 includes 而不是查档案表，是为了以后新增 *Pro* 版本时
 *   行为自动跟上，不会因为漏改档案表而静默少跑一步。
 */
export const needsSv = (version) => String(version || '').includes('Pro');

/**
 * 各语言的「默认语速系数」（传给 s2 的 speed_factor，1.0 = 与训练语料同速）。
 *
 * ★ 为什么需要这张表：speed_factor 只对 s2 的 latent 做长度插值，**不改音高**
 *   （实测 F0 中位偏移仅 -1~-3%，属测量噪声），是干净的时长缩放。
 *   s2 的语速来自训练语料 —— 本机芳乃语料是日文台词，语速偏快（约 7 字/秒），
 *   拿它念中文就会「音节被压缩」，即使加了停顿也还是快。
 *
 * 实测（46 字中文，含停顿总时长）：
 *   speed 1.00 → 4.92 字/秒（新闻播报语速，偏快）
 *   speed 0.85 → 4.21 字/秒（正常朗读）   ← 中文默认
 *   speed 0.75 → 3.77 字/秒（舒缓散文）
 *   speed 0.65 → 3.30 字/秒（偏慢，适合诗歌）
 *
 * 日文不调：训练语料就是日文，1.0 才是它的自然语速。
 */
export const LANG_SPEED = { zh: 0.85, ja: 1.0, en: 0.9, ko: 1.0, yue: 0.85 };

/** 显式传了就用显式的（含 0 以外的值）；否则按目标语言取默认。 */
export function resolveSpeed(lang, explicit) {
  if (typeof explicit === 'number' && explicit > 0) return explicit;
  return LANG_SPEED[lang] ?? 1.0;
}

/**
 * ★★ 基线（2026-10-19 用户选定）：**日文参考音频 + 自训 s2 + 底模 s1 + zh 0.85**。
 *
 * 为什么这不是「待优化的中间态」，而是基线本身：
 *
 * 用户的原话是「发音方式本来就是角色特质非常重要的一部分，拿 VC 转感觉丢很多信息，
 * 暂时不追求标准中文」。这句话把优化目标换了 —— 之前所有努力（换中文 ref、DDSP 换音色）
 * 都在追「像标准普通话」，而真正的目标是**保住角色的发音方式**。
 *
 * 于是「带日语腔的中文」不再是要修的缺陷，而是这个角色念中文时该有的样子。
 * 换中文 ref 虽然 CER 0.0%、字时长 CV 0.50，但那是**另一个人的发音习惯**，
 * 音色对、人不对 —— 和 VC 路线一样属于丢信息。
 *
 * 基线参数全部锁在下面，`tts_infer` 不传参数时即产出这条基线。
 * 任何一条被改动，`test/baseline-lock.mjs` 会失败 —— 这是故意的：
 * 改基线必须是一次有意识的决定，不能是某次调参的副作用。
 */
export const BASELINE = {
  label: '2026-10-19 芳乃·日文腔中文',
  // 参考音频是角色母语原声：它同时给音色和「怎么发音」的示范，后者才是角色特质
  refLang: 'ja',
  refText: 'ですが、絶対に何とかしますから。私に任せて下さい',
  // ja 语料 → zh 目标 = 跨语言 ⇒ 必须底模 s1（自训 s1 的音素→语义映射被训窄在日文上）
  gptMode: 'auto',
  speed: 0.85,        // 实测 4.21 字/秒，正常朗读；1.0 是 4.92 的播报速度
  split: 'cut5',
  temperature: 0.6,   // s2 只训 8 轮，低温度换稳定性；1.0 时韵律容易飘
  topK: 5,
  topP: 1.0,
  seed: 42,
  repetitionPenalty: 1.35,
  gap: 0.22,          // 分段间静音，叠加 cut5 自带的 0.9s 尾巴后约 0.24~0.36s
};

/**
 * s1 档位判定（纯函数，便于测试）：target ≠ trainLang 即跨语言，必须底模。
 * 没有自训 s1、或读不到训练语言时，保守用底模。
 */
export function pickGptMode(trainLang, targetLang, hasTrained) {
  if (!hasTrained) return { mode: 'base', note: '尚无自训 s1，用底模' };
  if (!trainLang) return { mode: 'base', note: '读不到训练语料语言，保守用底模 s1' };
  if (!targetLang) return { mode: 'trained', note: '' };
  const cross = String(trainLang).toLowerCase() !== String(targetLang).toLowerCase();
  return {
    mode: cross ? 'base' : 'trained',
    note: `训练语料 ${trainLang} → 目标 ${targetLang}，${cross ? '跨语言，用底模 s1' : '同语言，用自训 s1'}`,
  };
}

export function toPosixPath(p) {
  if (typeof p !== 'string' || p === '') return p;
  const s = p.replace(/\\/g, '/');
  // 相对路径转绝对；含 `..` 的也归一化 —— 自动发现拼出来的
  // `.../02_GPT-SoVITS/../hf_cache` 能跑，但日志里显示成那样没人看得懂
  if (!isAbsolute(s) || s.includes('..')) return resolve(s).replace(/\\/g, '/');
  return s;
}

export function join(...parts) {
  return parts
    .filter((p) => p !== undefined && p !== null && p !== '')
    .map((p, i) => (i === 0 ? String(p).replace(/\/+$/, '') : String(p).replace(/^\/+|\/+$/g, '')))
    .join('/');
}

export const isFile = (p) => { try { return !!p && existsSync(p) && statSync(p).isFile(); } catch { return false; } };
export const isDir = (p) => { try { return !!p && existsSync(p) && statSync(p).isDirectory(); } catch { return false; } };
const hasFile = (dir, name) => isFile(join(dir, name));

function children(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => join(dir, d.name));
  } catch { return []; }
}

function filesOf(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true }).filter((d) => d.isFile()).map((d) => d.name);
  } catch { return []; }
}

const childByPrefix = (dir, re) => children(dir).find((p) => re.test(basename(p))) || '';

/** 读本机覆盖层（不存在或格式坏掉都不报错） */
function readLocal() {
  try {
    if (!isFile(LOCAL_CONFIG)) return {};
    return JSON.parse(readFileSync(LOCAL_CONFIG, 'utf8'));
  } catch { return {}; }
}

/** 一个目录是不是 GPT-SoVITS 工程根（标记全在才算） */
export function looksLikeGsv(dir) {
  if (!dir || !isDir(dir)) return false;
  return GSV_MARKERS.every((m) => hasFile(dir, m));
}

// ───────────────────────────────────────────────────────────────────────────
// 自动发现
// ───────────────────────────────────────────────────────────────────────────

function pickSongRoot(cfg) {
  const candidates = [cfg.songRoot, process.env.DSH_COVER_ROOT, process.env.DSH_TTS_ROOT,
    'D:/AI-Song', 'C:/AudioWork'];
  for (const c of candidates) if (c && isDir(c)) return toPosixPath(c);
  for (const c of candidates) if (c) return toPosixPath(c);
  return toPosixPath(process.cwd());
}

/**
 * 找 GPT-SoVITS 工程根。
 * 搜索面刻意收窄到「AI-Voice 一类目录的下一层」，因为整盘递归扫 node_modules
 * 和系统目录会拖到几十秒，而 GPT-SoVITS 只会装在语音工程目录里。
 */
function findGsv(cfg) {
  if (looksLikeGsv(cfg.gsvDir)) return toPosixPath(cfg.gsvDir);

  const direct = [
    process.env.DSH_GSV_ROOT,
    'D:/AI-Voice/02_GPT-SoVITS',
    'D:/AI-Voice/GPT-SoVITS',
    'C:/AI-Voice/02_GPT-SoVITS',
  ].filter(Boolean);
  for (const c of direct) if (looksLikeGsv(c)) return toPosixPath(c);

  const roots = [...new Set([cfg.songRoot, pickSongRoot(cfg),
    'D:/AI-Voice', 'D:/AI-Song', 'C:/AudioWork', process.cwd()].filter(Boolean))];
  for (const r of roots) {
    if (looksLikeGsv(r)) return toPosixPath(r);
    for (const d of children(r)) {
      if (looksLikeGsv(d)) return toPosixPath(d);
      // 再下探一层：02_xxx/GPT-SoVITS 这种双层
      for (const dd of children(d)) {
        if (looksLikeGsv(dd)) return toPosixPath(dd);
      }
    }
  }
  return '';
}

function pickPython(gsvDir) {
  if (!gsvDir) return '';
  for (const rel of ['.venv/Scripts/python.exe', '.venv/bin/python',
    'runtime/python.exe', 'python.exe', 'env/Scripts/python.exe']) {
    const p = join(gsvDir, rel);
    if (isFile(p)) return p;
  }
  return '';
}

function pickFfmpeg(cfg, gsvDir) {
  if (isFile(cfg.ffmpeg)) return toPosixPath(cfg.ffmpeg);
  for (const rel of ['ffmpeg.exe', 'tools/ffmpeg.exe']) {
    const p = join(gsvDir, rel);
    if (isFile(p)) return p;
  }
  const roots = [cfg.songRoot, process.env.DSH_COVER_ROOT, 'D:/AI-Song'].filter(Boolean);
  for (const r of roots) {
    for (const d of children(r)) {
      if (!/01[_-]/.test(basename(d))) continue;
      for (const rel of ['env/ffmpeg.exe', 'ffmpeg/bin/ffmpeg.exe', 'env/Scripts/ffmpeg.exe']) {
        const p = join(d, rel);
        if (isFile(p)) return p;
      }
    }
  }
  return 'ffmpeg';
}

/** faster-whisper 模型缓存位置（优先工程的 _hf_home，其次上一级的 hf_cache） */
function pickAsrCache(cfg, gsvDir) {
  if (cfg.asrCacheDir && isDir(cfg.asrCacheDir)) return toPosixPath(cfg.asrCacheDir);
  const cands = [
    join(gsvDir, '_hf_home'),
    join(gsvDir, '..', 'hf_cache'),
    join(process.env.USERPROFILE || 'C:/Users/Default', '.cache/huggingface'),
  ];
  for (const c of cands) if (isDir(join(c, 'hub'))) return toPosixPath(c);
  return '';
}

function discover(cfg) {
  const out = {};
  const root = pickSongRoot(cfg);
  if (!isDir(cfg.songRoot) || cfg.songRoot !== root) out.songRoot = root;

  if (!isDir(cfg.workRoot)) out.workRoot = join(root, '04_训练工作区');

  const gsv = findGsv({ ...cfg, songRoot: root });
  if (gsv && gsv !== cfg.gsvDir) out.gsvDir = gsv;

  const python = isFile(cfg.gsvEnv) ? cfg.gsvEnv : pickPython(gsv || cfg.gsvDir);
  if (python && python !== cfg.gsvEnv) out.gsvEnv = python;

  if (!cfg.ffmpeg || cfg.ffmpeg === 'ffmpeg') out.ffmpeg = pickFfmpeg({ ...cfg, songRoot: root }, gsv);

  const cache = pickAsrCache(cfg, gsv);
  if (cache && cache !== cfg.asrCacheDir) out.asrCacheDir = cache;

  return out;
}

// ───────────────────────────────────────────────────────────────────────────
// 派生路径
// ───────────────────────────────────────────────────────────────────────────

/** 合并默认值 → 本机覆盖层 → 传入配置 → 自动发现，并归一化类型与派生路径。 */
export function resolveConfig(raw = {}) {
  const local = readLocal();
  const merged = { ...DEFAULTS, ...local, ...raw };
  for (const k of PATH_KEYS) merged[k] = toPosixPath(merged[k] ?? '');

  for (const k of ['minSec', 'maxSec', 'targetHours', 's2BatchSize', 's2Epochs', 's2SaveEveryEpoch',
    's1BatchSize', 's1Epochs', 's1SaveEveryNEpoch', 'textLowLrRate', 'asrBeamSize']) {
    merged[k] = Number(merged[k]) || DEFAULTS[k];
  }
  for (const k of ['sampleRate', 'timeoutMs', 'maxOutputChars']) merged[k] = Number(merged[k]) || DEFAULTS[k];
  merged.gradCkpt = merged.gradCkpt === true || merged.gradCkpt === 'true';

  const found = discover(merged);
  const cfg = { ...merged, ...found };
  for (const k of PATH_KEYS) cfg[k] = toPosixPath(cfg[k] ?? '');

  // 版本档案：不认识的版本回落 v2ProPlus，并把回落这件事记下来，不静默改
  const wantVersion = String(cfg.version || '').trim();
  cfg.version = VERSION_PROFILES[wantVersion] ? wantVersion : DEFAULTS.version;
  cfg.versionFallback = wantVersion && !VERSION_PROFILES[wantVersion] ? wantVersion : '';
  cfg.profile = VERSION_PROFILES[cfg.version];

  const g = cfg.gsvDir;
  // 上游 exp_root = "logs"（相对工程根），必须与 webui 一致，否则 WebUI 里看不见这个实验
  cfg.expDir = g ? join(g, 'logs', cfg.expName) : '';
  cfg.sovitsWeightDir = g ? join(g, cfg.profile.sovitsWeights) : '';
  cfg.gptWeightDir = g ? join(g, cfg.profile.gptWeights) : '';
  cfg.s2ConfigPath = g ? join(g, cfg.profile.s2Config) : '';
  cfg.s2ScriptPath = g ? join(g, cfg.profile.s2Script) : '';
  cfg.pretrainedS2G = g ? join(g, cfg.profile.s2G) : '';
  cfg.pretrainedS2D = g ? join(g, cfg.profile.s2D) : '';
  cfg.pretrainedS1 = g ? join(g, cfg.profile.s1) : '';
  cfg.bertDir = g ? join(g, PREP_ASSETS.bert) : '';
  cfg.hubertDir = g ? join(g, PREP_ASSETS.hubert) : '';
  cfg.svPath = g ? join(g, PREP_ASSETS.sv) : '';

  cfg.workspace = workspacePaths(cfg);
  cfg.voices = listVoices(cfg);
  return cfg;
}

/** 语料 / 数据集 / 日志 / 报告 的固定落点（都在 workRoot 下，不进 gsv 工程） */
export function workspacePaths(cfg) {
  const base = cfg.workRoot || join(cfg.songRoot, '04_训练工作区');
  const tts = join(base, 'tts');
  return {
    base,
    tts,
    corpus: join(tts, 'corpus'),        // 出料前的原始语料（可选）
    datasets: join(tts, 'datasets'),    // <角色>/wav/*.wav + list.txt
    logs: join(tts, 'logs'),            // 训练日志（后台任务的唯一进度来源）
    reports: join(tts, 'reports'),      // build / asr / qc 报告
    voices: join(tts, 'voices.json'),   // 音色注册清单
    jobs: join(tts, 'jobs.json'),       // 后台任务登记
  };
}

/** 一个数据集的名字（= 实验名，默认） */
export const datasetDirOf = (cfg, name) => join(cfg.workspace.datasets, name || cfg.expName);

// ───────────────────────────────────────────────────────────────────────────
// 权重扫描 / 音色注册
// ───────────────────────────────────────────────────────────────────────────

/**
 * 扫某个版本目录下的成品权重。
 * s2_train 落盘名形如 `<name>_e<epoch>_s<step>.pth`，s1_train 形如 `-e<epoch>.ckpt`。
 */
export function scanWeights(cfg, version = cfg.version) {
  const prof = VERSION_PROFILES[version] || cfg.profile;
  const sd = join(cfg.gsvDir, prof.sovitsWeights);
  const gd = join(cfg.gsvDir, prof.gptWeights);

  const sizeOf = (p) => { try { return statSync(p).size; } catch { return 0; } };

  // <name>_e<epoch>_s<step>.pth —— 名字里可能带下划线，用贪婪 + 后缀锚定，避免截错
  const sovits = filesOf(sd)
    .map((f) => {
      const m = /^(.+)_e(\d+)_s(\d+)\.pth$/i.exec(f);
      if (!m) return null;
      const p = join(sd, f);
      return { kind: 'sovits', file: f, path: p, name: m[1], epoch: Number(m[2]), step: Number(m[3]), sizeBytes: sizeOf(p) };
    })
    .filter(Boolean)
    .sort((a, b) => a.epoch - b.epoch || a.step - b.step);

  // <name>-e<epoch>.ckpt
  const gpt = filesOf(gd)
    .map((f) => {
      const m = /^(.+)-e(\d+)\.ckpt$/i.exec(f);
      if (!m) return null;
      const p = join(gd, f);
      return { kind: 'gpt', file: f, path: p, name: m[1], epoch: Number(m[2]), sizeBytes: sizeOf(p) };
    })
    .filter(Boolean)
    .sort((a, b) => a.epoch - b.epoch);

  return { version, sovitsDir: sd, gptDir: gd, sovits, gpt };
}

/**
 * 音色清单：GPT-SoVITS 本身没有"音色库"概念（每次推理现给权重路径），
 * 所以这里维护一份 `voices.json`，让「用芳乃的声音念这句话」有名字可寻。
 */
export function listVoices(cfg) {
  const file = cfg.workspace?.voices || join(cfg.workRoot || '', 'tts/voices.json');
  let reg = [];
  try { reg = JSON.parse(readFileSync(file, 'utf8')); } catch { reg = []; }
  if (!Array.isArray(reg)) reg = [];
  return reg.map((v) => ({
    ...v,
    exists: isFile(v.sovits) && isFile(v.gpt),
  }));
}

/** 按名字找音色（精确优先，再做包含匹配） */
export function findVoice(cfg, query) {
  const vs = Array.isArray(cfg.voices) ? cfg.voices : listVoices(cfg);
  const norm = (s) => String(s ?? '').replace(/[\s_·・.\-]/g, '').toLowerCase();
  const q = norm(query);
  if (!q) return null;
  // ★ name 和 expName 都要匹配：调用方（tts_infer / tts_register）传的是 expName，
  //   而用户给音色起的名字通常是中文（如「芳乃」）。只按 name 匹配会让清单形同虚设。
  const exact = vs.filter((v) => norm(v.name) === q || norm(v.expName) === q);
  if (exact.length) return exact[0];
  return vs.find((v) => norm(v.name).includes(q) || q.includes(norm(v.name))
    || norm(v.expName).includes(q) || q.includes(norm(v.expName))) ?? null;
}

export function readJson(p) {
  try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; }
}
