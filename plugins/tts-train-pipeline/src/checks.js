/**
 * 体检：把「能不能训」这件事拆成可判定的事实，而不是凭感觉说"环境应该没问题"。
 *
 * 三条原则：
 *   1. **error 就是跑不起来**，不写"可能"。
 *   2. 版本相关的缺失单独归类 —— v3/v4 本来就没有 s2D，不能算错。
 *   3. 把 GPU 算力与 torch 的 CUDA 版本一起报出来：Blackwell（sm_120）配 cu126 是个经典死局，
 *      报出来比事后 debug 省几个小时。
 */
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, PY_DIR, GSV_MARKERS, PREP_ASSETS, LANGUAGES } from './config.js';
import { pythonEnv, runOnce } from './execution.js';

/** 跑 env_probe.py，拿一份环境实况 */
export async function probeEnv(cfg, { timeoutMs = 180_000, deep = false } = {}) {
  const payload = JSON.stringify({
    gsvDir: cfg.gsvDir,
    asrModel: cfg.asrModel,
    asrCacheDir: cfg.asrCacheDir,
    ffmpeg: cfg.ffmpeg,
    deep,
  });
  const r = await runOnce({
    cmd: cfg.gsvEnv || 'python',
    args: [join(PY_DIR, 'env_probe.py'), payload],
    cwd: cfg.gsvDir || PY_DIR,
    env: pythonEnv(cfg.gsvEnv, {}),
    timeoutMs,
  });
  const line = (r.stdout || '').trim().split(/\r?\n/).filter(Boolean).pop() || '';
  try {
    return { ok: true, data: JSON.parse(line), elapsedMs: Math.round(Number(r.ms) || 0) };
  } catch {
    return { ok: false, data: null, stderr: (r.stderr || r.stdout || '').slice(-1500) };
  }
}

/** GPU 算力与 torch CUDA 版本是否匹配（Blackwell 陷阱） */
function checkCudaCompat(torch) {
  const issues = [];
  if (!torch?.installed) {
    issues.push({ level: 'error', msg: 'torch 未安装' });
    return issues;
  }
  if (!torch.cudaAvailable) {
    issues.push({ level: 'error', msg: `torch 装了但 CUDA 不可用（build=${torch.cudaBuild ?? '?'}）；训练跑不起来` });
    return issues;
  }
  const cap = String(torch.capability || '');
  const build = String(torch.cudaBuild || '');
  // sm_120 = Blackwell（50 系）。cu126 及以下没有对应 kernel，会报 no kernel image available
  if (cap === 'sm_120' && /^12\.[0-6]/.test(build)) {
    issues.push({
      level: 'error',
      msg: `GPU 是 ${cap}（Blackwell）但 torch 是 CUDA ${build}：会报 "no kernel image is available"`,
      fix: 'torch 必须换 cu128 或更新版本',
    });
  }
  return issues;
}

/**
 * 全量体检。返回 { ok, errors, warnings, info, probe }
 * errors 非空 = 别开训。
 */
export async function doctor(cfg, opts = {}) {
  const errors = [];
  const warnings = [];
  const info = {};

  // ── 工程目录 ───────────────────────────────────────────────────────────
  if (!cfg.gsvDir) {
    errors.push({
      msg: '没找到 GPT-SoVITS 工程（需要同时含 webui.py 与 GPT_SoVITS/s2_train.py 的那一层）',
      fix: '设 gsvDir，或用环境变量 DSH_GSV_ROOT 指向它',
    });
  } else {
    info.gsvDir = cfg.gsvDir;
    const missing = GSV_MARKERS.filter((m) => !existsSync(join(cfg.gsvDir, m)));
    if (missing.length) {
      errors.push({
        msg: `工程缺文件：${missing.join(', ')}`,
        fix: '确认 gsvDir 指向含 webui.py 的那一层，而不是它的父目录',
      });
    }
  }

  if (!cfg.gsvEnv) {
    errors.push({
      msg: '没找到 Python 解释器',
      fix: '工程里应存在 .venv/Scripts/python.exe；或显式配置 gsvEnv',
    });
  }

  // ── 版本档案的关键权重 ────────────────────────────────────────────────
  const weightChecks = [
    ['s2G 底模', cfg.pretrainedS2G],
    ['s1 底模', cfg.pretrainedS1],
    ['BERT 权重', cfg.bertDir],
    ['cnhubert 权重', cfg.hubertDir],
    ['SV 权重（v2ProPlus 的 3-get-semantic 要用）', cfg.svPath],
    ['s2 配置模板', cfg.s2ConfigPath],
    ['s2 训练脚本', cfg.s2ScriptPath],
  ];
  if (cfg.pretrainedS2D) weightChecks.push(['s2D 底模', cfg.pretrainedS2D]);

  for (const [label, p] of weightChecks) {
    if (!p) { warnings.push(`${label} 未配置路径`); continue; }
    if (!existsSync(p)) {
      const isOptional = !cfg.pretrainedS2D && label.startsWith('s2D');
      (isOptional ? warnings : errors).push({
        msg: `${label} 缺失: ${p}`,
        fix: `版本 ${cfg.version} 的档案要求这个文件`,
      });
    }
  }

  // ── ffmpeg ─────────────────────────────────────────────────────────────
  if (!cfg.ffmpeg) {
    warnings.push('ffmpeg 未找到：出料时会失败（可用 --ffmpeg 指定绝对路径）');
  } else if (cfg.ffmpeg !== 'ffmpeg' && !existsSync(cfg.ffmpeg)) {
    warnings.push(`ffmpeg 路径不存在: ${cfg.ffmpeg}`);
  } else {
    info.ffmpeg = cfg.ffmpeg;
  }

  // ── 语言白名单 ─────────────────────────────────────────────────────────
  if (!LANGUAGES.includes(cfg.language)) {
    errors.push({
      msg: `language=${cfg.language} 不在 1-get-text.py 的白名单里（${LANGUAGES.join('/')}）`,
      fix: '不在白名单的句子会被静默跳过，最终 0 条可用',
    });
  }

  // ── 环境实况 ───────────────────────────────────────────────────────────
  const probe = await probeEnv(cfg, { timeoutMs: opts.timeoutMs ?? 180_000, deep: !!opts.deep });
  if (!probe.ok) {
    errors.push({
      msg: 'Python 环境探测失败',
      fix: '单独跑一次 env_probe.py 看报错',
      detail: probe.stderr,
    });
  } else {
    const d = probe.data;
    info.python = d.python;
    info.platform = d.platform;

    for (const i of checkCudaCompat(d.torch)) errors.push(i);
    if (d.torch?.installed && d.torch.cudaAvailable) {
      info.gpu = {
        name: d.torch.deviceName,
        capability: d.torch.capability,
        vramGB: d.torch.totalVramGB,
        torch: d.torch.version,
        cudaBuild: d.torch.cudaBuild,
      };
      if (cfg.version === 'v2ProPlus' || cfg.version === 'v2Pro') {
        info.gpu.note = 'v2ProPlus 推理实测约 1.75 GB 显存；训练随 batch_size 线性增长';
      }
    }

    const needMods = ['faster_whisper', 'yaml'];    for (const m of needMods) {
      const st = d.modules?.[m];
      if (st?.error) {
        warnings.push(`${m} 不可用：${st.error}` + (m === 'faster_whisper'
          ? '（只影响 ASR 回标，不影响有台词的语料）' : '（s1 配置写入会失败，属 error）'));
        if (m === 'yaml') errors.push({ msg: 'pyyaml 缺失：s1 训练配置无法生成' });
      }
    }

    if (!d.modules?.['TTS_infer_pack.TTS']?.import) {
      errors.push({
        msg: `TTS_infer_pack 无法 import：${d.modules?.['TTS_infer_pack.TTS']?.error ?? '未知'}`,
        fix: '推理与版本一致性都依赖它',
      });
    }

    if (d.asrCache?.found) {
      info.asr = {
        model: cfg.asrModel, cached: true,
        path: d.asrCache.path, modelBinMB: d.asrCache.modelBinMB,
      };
    } else {
      const st = d.modules?.faster_whisper;
      if (!st?.error) {
        warnings.push(`faster-whisper 已装但本地没有 ${cfg.asrModel} 缓存：ASR 首次会联网下载`);
      }
      info.asr = { model: cfg.asrModel, cached: false, cacheRoot: cfg.asrCacheDir || '' };
    }

    if (d.disk?.gsv) info.disk = d.disk.gsv;

    // 未探的模块要如实报出来，不能让「没查」看起来像「查了没问题」
    info.deep = !!d.deep;
    info.probeMs = probe.elapsedMs;
    info.probeNote = probe.elapsedMs > 8000
      ? `环境探测 ${(probe.elapsedMs / 1000).toFixed(1)}s（TTS_infer_pack 首次 import 约 16s，属正常）`
      : '';
    if (d.modulesSkipped?.length) info.modulesSkipped = d.modulesSkipped;
    info.modules = Object.fromEntries(Object.entries(d.modules || {}).map(([k, v]) => {
      const ok = Boolean(v.version) || v.import === 'ok';
      return [k, ok ? 'ok' : 'missing'];
    }));
  }

  // ── 工作区与已有音色 ───────────────────────────────────────────────────
  info.workspace = cfg.workspace;
  info.version = cfg.version;
  info.expDir = cfg.expDir;
  if (cfg.versionFallback) {
    warnings.push(`version=${cfg.versionFallback} 不认识，已回落 ${cfg.version}`);
  }
  info.voices = (cfg.voices || []).map((v) => ({ name: v.name, sovits: v.sovits, gpt: v.gpt, ok: v.exists }));

  return { ok: errors.length === 0, errors, warnings, info, probe: probe.data };
}

/** 生成可直接粘贴的配置片段（发现失败时用） */
export function configSnippet(cfg) {
  const lines = [
    '{',
    `  "gsvDir": "${cfg.gsvDir || 'D:/AI-Voice/02_GPT-SoVITS'}",`,
    `  "gsvEnv": "${cfg.gsvEnv || 'D:/AI-Voice/02_GPT-SoVITS/.venv/Scripts/python.exe'}",`,
    `  "workRoot": "${cfg.workRoot || ''}",`,
    `  "version": "${cfg.version}"`,
    '}',
  ];
  return lines.join('\n');
}

/** 磁盘/Git 版本之类的轻量事实（失败就算了，不阻断） */
export function gitInfo(dir) {
  try {
    const out = execFileSync('git', ['-C', dir, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8', timeout: 8000 });
    return out.trim();
  } catch { return ''; }
}
