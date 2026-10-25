/**
 * GPT-SoVITS 训练链：预处理三件套 → s2（SoVITS 声学）→ s1（GPT 语义）→ 推理。
 *
 * ★ 全部字段都抄自上游 webui.py，不是自己设计的。三个必须记住的硬事实：
 *
 *   1. **预处理脚本不吃 argv，只读环境变量**（os.environ.get("inp_text") 之类）。
 *      所以这里不是拼命令行，而是给子进程塞 env。参数名错一个，脚本会拿 None 去拼路径。
 *
 *   2. **cwd 必须是工程根**。s2 的 `save_weight_dir`、s1 的 `half_weights_save_dir`、
 *      `train_semantic_path` 全是**相对路径**，换个 cwd 就写到别处去，而且不报错。
 *
 *   3. **预处理有「产物已存在就跳过」逻辑**（1-get-text.py 第 47 行）。
 *      换数据集不先重置 opt_dir，就会拿旧台词去训新音频 —— 这是最难查的一类错。
 *      本模块用**改名**而不是删除来重置：不触发沙箱的批量删除限制，也留了回滚余地。
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { join, PY_DIR, PREP_SCRIPTS, needsSv } from './config.js';
import { pythonEnv, runOnce, spawnJob, artifactsOf, artifactsOk } from './execution.js';

/** 预处理产物（相对 opt_dir），用于自证与重置 */
export const PREP_ARTIFACTS = {
  text: '2-name2text.txt',
  textPart: '2-name2text-0.txt',
  bert: '3-bert',
  hubert: '4-cnhubert',
  wav32k: '5-wav32k',
  semantic: '6-name2semantic.tsv',
  semanticPart: '6-name2semantic-0.tsv',
  sv: '7-sv_cn',
};

/** 上游 Python 的 is_half 走 strtobool，必须是 "True"/"False" 这两个字符串 */
const boolStr = (v) => (v ? 'True' : 'False');

/**
 * 重置 opt_dir 里的预处理产物。
 * ★ 改名而不是删除：3-bert 可能有上千个小文件，批量删除在受限沙箱里会被拦，
 *   而且删了就回不去了。改名既能强制重跑，又能事后核对上一版。
 */
export function resetPrepArtifacts(expDir) {
  const moved = [];
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
  const staleDir = join(expDir, `_stale_${stamp}`);
  const targets = [
    PREP_ARTIFACTS.textPart, PREP_ARTIFACTS.text,
    PREP_ARTIFACTS.bert, PREP_ARTIFACTS.hubert, PREP_ARTIFACTS.wav32k,
    PREP_ARTIFACTS.sv,
    PREP_ARTIFACTS.semanticPart, PREP_ARTIFACTS.semantic,
  ];
  for (const t of targets) {
    const p = join(expDir, t);
    if (!existsSync(p)) continue;
    try {
      mkdirSync(staleDir, { recursive: true });
      renameSync(p, join(staleDir, t));
      moved.push(t);
    } catch { /* 改名失败就跳过，由后续「产物已存在」的检查兜底报错 */ }
  }
  return { moved, staleDir: moved.length ? staleDir : '' };
}

/** 单步预处理的环境变量。参数名必须与脚本里的 os.environ.get 完全一致。 */
function prepEnv(cfg, { stage, listFile, wavDir, gpu = '0' }) {
  const base = {
    inp_text: listFile,
    exp_name: cfg.expName,
    opt_dir: cfg.expDir,
    i_part: '0',
    all_parts: '1',
    _CUDA_VISIBLE_DEVICES: String(gpu),
    is_half: boolStr(true),
    // ★ 1-get-text.py 用它决定 symbols_v1 还是 symbols_v2（clean_text 的第三个参数）。
    //   不传时 cleaner 自己回落 "v2"，对 v2/v2Pro/v2ProPlus 恰好等价，
    //   但 v1 会拿错符号表 —— 显式传，别依赖默认值。
    version: cfg.version,
  };
  if (stage === 'text') {
    return { ...base, inp_wav_dir: wavDir, bert_pretrained_dir: cfg.bertDir };
  }
  if (stage === 'hubert') {
    return { ...base, inp_wav_dir: wavDir, cnhubert_base_dir: cfg.hubertDir, sv_path: cfg.svPath };
  }
  if (stage === 'sv') {
    // 读 5-wav32k（上一步产出），不需要 inp_text
    return {
      ...base,
      inp_wav_dir: wavDir,
      cnhubert_base_dir: cfg.hubertDir,
      sv_path: cfg.svPath,
    };
  }
  // semantic：注意**没有** inp_wav_dir（脚本从 4-cnhubert 读）
  return { ...base, pretrained_s2G: cfg.pretrainedS2G, s2config_path: cfg.profile.s2Config };
}

/**
 * 跑一步预处理。
 * 产物自证 + 单 part 合并：上游 webui 是多进程按 part 分摊再合并，
 * 我们固定 all_parts=1，所以要把 `-0` 文件改名成合并后的正式名（s1 训练读正式名）。
 */
export async function runPrepStage(cfg, { stage, listFile, wavDir, gpu = '0', timeoutMs }) {
  const script = PREP_SCRIPTS[stage];
  const scriptPath = join(cfg.gsvDir, script);
  if (!existsSync(scriptPath)) {
    return { ok: false, stage, error: `预处理脚本缺失: ${scriptPath}` };
  }
  mkdirSync(cfg.expDir, { recursive: true });

  const env = pythonEnv(cfg.gsvEnv, prepEnv(cfg, { stage, listFile, wavDir, gpu }));
  const args = ['-s', script];

  const r = await runOnce({
    cmd: cfg.gsvEnv,
    args,
    cwd: cfg.gsvDir,                 // ★ 必须
    env,
    timeoutMs: timeoutMs ?? cfg.timeoutMs,
  });

  // 合并单 part 产物
  const mergeOne = (partName, finalName) => {
    const part = join(cfg.expDir, partName);
    const fin = join(cfg.expDir, finalName);
    if (existsSync(part)) {
      try { renameSync(part, fin); return true; } catch { return false; }
    }
    return existsSync(fin);
  };

  let produced = [];
  let merged = null;
  if (stage === 'text') {
    merged = mergeOne(PREP_ARTIFACTS.textPart, PREP_ARTIFACTS.text);
    produced = [join(cfg.expDir, PREP_ARTIFACTS.text)];
  } else if (stage === 'hubert') {
    produced = [join(cfg.expDir, PREP_ARTIFACTS.hubert), join(cfg.expDir, PREP_ARTIFACTS.wav32k)];
  } else if (stage === 'sv') {
    produced = [join(cfg.expDir, PREP_ARTIFACTS.sv)];
  } else {
    merged = mergeOne(PREP_ARTIFACTS.semanticPart, PREP_ARTIFACTS.semantic);
    produced = [join(cfg.expDir, PREP_ARTIFACTS.semantic)];
  }

  const arts = artifactsOf(produced);
  const hasArtifact = arts.every((a) => a.ok);
  const tail = (r.stderr || r.stdout || '').split(/\r?\n/).filter((l) => l.trim()).slice(-12).join('\n');

  // 1-get-text 会静默跳过语言不在白名单的句子，必须核对行数
  let textLines = null;
  if (stage === 'text' && hasArtifact) {
    try {
      textLines = readFileSync(join(cfg.expDir, PREP_ARTIFACTS.text), 'utf8')
        .split(/\r?\n/).filter((l) => l.trim() !== '').length;
    } catch { textLines = null; }
  }

  return {
    ok: r.exitCode === 0 && hasArtifact,
    stage,
    exitCode: r.exitCode,
    ms: r.ms,
    merged,
    artifacts: arts,
    textLines,
    logTail: tail,
    cmdline: r.cmdline,
  };
}

/** 读 list.txt 的行数（用于出料后核对） */
export function countListLines(listFile) {
  try {
    const txt = readFileSync(listFile, 'utf8');
    return txt.split(/\r?\n/).filter((l) => l.trim() !== '').length;
  } catch { return 0; }
}

/**
 * s2 配置覆盖项 —— 逐字段对应 webui.py 的 open1Ba。
 * 其中 `data.exp_dir` / `s2_ckpt_dir` / `save_weight_dir` 是**相对工程根**的路径。
 */
export function s2Overrides(cfg, opts = {}) {
  const expRel = `logs/${cfg.expName}`;
  const o = {
    train: {
      batch_size: Number(opts.batchSize ?? cfg.s2BatchSize),
      epochs: Number(opts.epochs ?? cfg.s2Epochs),
      text_low_lr_rate: Number(cfg.textLowLrRate),
      pretrained_s2G: opts.pretrainedS2G ?? cfg.pretrainedS2G,
      pretrained_s2D: opts.pretrainedS2D ?? cfg.pretrainedS2D,
      if_save_latest: true,
      if_save_every_weights: true,
      save_every_epoch: Number(opts.saveEveryEpoch ?? cfg.s2SaveEveryEpoch),
      gpu_numbers: String(opts.gpu ?? '0'),
      grad_ckpt: cfg.gradCkpt === true,
    },
    model: { version: cfg.version },
    data: { exp_dir: expRel },
    s2_ckpt_dir: expRel,
    save_weight_dir: cfg.profile.sovitsWeights,
    name: cfg.expName,
    version: cfg.version,
  };
  if (cfg.version === 'v3' || cfg.version === 'v4') {
    o.train.lora_rank = Number(opts.loraRank ?? 32);
  }
  return o;
}

/** s1 配置覆盖项 —— 对应 webui.py 的 open1B（s1 段） */
export function s1Overrides(cfg, opts = {}) {
  const expRel = `logs/${cfg.expName}`;
  const o = {
    train: {
      batch_size: Number(opts.batchSize ?? cfg.s1BatchSize),
      epochs: Number(opts.epochs ?? cfg.s1Epochs),
      save_every_n_epoch: Number(opts.saveEveryNEpoch ?? cfg.s1SaveEveryNEpoch),
      if_save_every_weights: true,
      if_save_latest: false,
      if_dpo: false,
      half_weights_save_dir: cfg.profile.gptWeights,
      exp_name: cfg.expName,
    },
    pretrained_s1: opts.pretrainedS1 ?? cfg.pretrainedS1,
    train_semantic_path: `${expRel}/6-name2semantic.tsv`,
    train_phoneme_path: `${expRel}/2-name2text.txt`,
    output_dir: `${expRel}/logs_s1_${cfg.version}`,
  };
  if (cfg.version === 'v1') {
    // v1 用另一份模板（s1longer.yaml），由调用方选择模板，此处不覆盖
  }
  return o;
}

/** 用 make_train_config.py 合成临时配置，返回产物路径与自证 */
export async function makeTrainConfig(cfg, { kind, overrides, tmpPath, timeoutMs = 120_000 }) {
  const template = kind === 's2'
    ? cfg.s2ConfigPath
    : join(cfg.gsvDir, cfg.version === 'v1'
      ? 'GPT_SoVITS/configs/s1longer.yaml'
      : 'GPT_SoVITS/configs/s1longer-v2.yaml');

  if (!existsSync(template)) {
    return { ok: false, error: `配置模板不存在: ${template}` };
  }

  const ovPath = `${tmpPath}.overrides.json`;
  mkdirSync(join(tmpPath, '..'), { recursive: true });
  writeFileSync(ovPath, JSON.stringify(overrides, null, 2), 'utf8');

  const r = await runOnce({
    cmd: cfg.gsvEnv,
    args: [join(PY_DIR, 'make_train_config.py'),
      '--kind', kind, '--template', template, '--overrides', ovPath, '--out', tmpPath],
    cwd: cfg.gsvDir,
    env: pythonEnv(cfg.gsvEnv, {}),
    timeoutMs,
  });

  let parsed = null;
  try { parsed = JSON.parse((r.stdout || '').trim().split(/\r?\n/).pop()); } catch { parsed = null; }

  let sizeBytes = 0;
  try { sizeBytes = statSync(tmpPath).size; } catch { sizeBytes = 0; }

  return {
    ok: r.exitCode === 0 && sizeBytes > 50,
    template,
    path: tmpPath,
    sizeBytes,
    overridesPath: ovPath,
    detail: parsed,
    stderr: r.exitCode === 0 ? '' : (r.stderr || r.stdout || '').slice(-1200),
  };
}

/** 后台启动 s2 训练 */
export function startS2(cfg, { configPath, logPath, gpu = '0' }) {
  return spawnJob({
    cmd: cfg.gsvEnv,
    args: ['-s', cfg.profile.s2Script, '--config', configPath],
    cwd: cfg.gsvDir,                                   // ★ 相对 save_weight_dir 的基准
    env: pythonEnv(cfg.gsvEnv, { _CUDA_VISIBLE_DEVICES: String(gpu) }),
    logPath,
  });
}

/** 后台启动 s1 训练 */
export function startS1(cfg, { configPath, logPath, gpu = '0' }) {
  return spawnJob({
    cmd: cfg.gsvEnv,
    args: ['-s', 'GPT_SoVITS/s1_train.py', '--config_file', configPath],
    cwd: cfg.gsvDir,
    env: pythonEnv(cfg.gsvEnv, {
      _CUDA_VISIBLE_DEVICES: String(gpu),
      hz: '25hz',                                      // 上游写死 25hz，语义帧率
    }),
    logPath,
  });
}

/**
 * 训练前必须先建好的目录。
 *
 * ★ 这四个目录不是"顺手建一下"，是**不建就一定崩**：
 *   上游 s2_train.py 的 save_checkpoint 会 shutil.move 到
 *   `<exp_dir>/logs_s2_<version>/G_<step>.pth`，而它自己**不建这个目录**
 *   （建目录的动作在 webui.py 第 517 行，s1 的在第 611 行）。
 *   直接用命令行跑就会在保存权重那一刻抛 FileNotFoundError —— 训练白跑一轮。
 *   另外两个是 cwd 相对的权重输出目录，同理。
 */
export function ensureTrainDirs(cfg) {
  const dirs = [
    join(cfg.expDir, `logs_s2_${cfg.version}`),
    join(cfg.expDir, 'logs_s1'),
    cfg.sovitsWeightDir,
    cfg.gptWeightDir,
  ].filter(Boolean);
  const made = [];
  for (const d of dirs) {
    try {
      if (!existsSync(d)) { mkdirSync(d, { recursive: true }); made.push(d); }
    } catch { /* 建不出来就让训练自己报错，别在这里吞掉 */ }
  }
  return { dirs, made };
}

/** 训练前置校验：数据齐不齐、权重在不在。返回问题清单，空数组 = 可以开训 */
export function preflight(cfg, { phase }) {
  const problems = [];
  const warn = [];

  if (!cfg.gsvDir) problems.push('gsvDir 未发现：找不到含 GPT_SoVITS/s2_train.py 的工程目录');
  if (!cfg.gsvEnv) problems.push('gsvEnv 未发现：工程里没有 .venv/Scripts/python.exe');

  if (phase === 's2' || phase === 's1') {
    const need = [
      [PREP_ARTIFACTS.text, '文本特征（1-get-text 产物）'],
      [PREP_ARTIFACTS.semantic, '语义 token（3-get-semantic 产物）'],
    ];
    for (const [f, label] of need) {
      const p = join(cfg.expDir, f);
      if (!existsSync(p)) problems.push(`${label} 缺失: ${p}`);
    }
    const needDirs = [PREP_ARTIFACTS.hubert, PREP_ARTIFACTS.wav32k];
    // ★ Pro 版本必须有 7-sv_cn：缺了 data_utils 会把 spec/ssl/wav 一起换成全零继续训，
    //   不报错但训出来是废的 —— 所以这里当硬阻断，不当 warning。
    if (needsSv(cfg.version)) needDirs.push(PREP_ARTIFACTS.sv);
    for (const dirName of needDirs) {
      const p = join(cfg.expDir, dirName);
      if (!existsSync(p)) {
        problems.push(`预处理目录缺失: ${p}` + (dirName === PREP_ARTIFACTS.sv
          ? `（${cfg.version} 需要 7-sv_cn，跑 tts_preprocess stage=sv）` : ''));
        continue;
      }
      try {
        if (readdirSync(p).length === 0) problems.push(`预处理目录是空的: ${p}`);
      } catch { /* 读不了就算了，后面训练会报 */ }
    }
  }

  if (phase === 's2') {
    if (!existsSync(cfg.pretrainedS2G)) problems.push(`s2G 底模缺失: ${cfg.pretrainedS2G}`);
    if (cfg.pretrainedS2D && !existsSync(cfg.pretrainedS2D)) problems.push(`s2D 底模缺失: ${cfg.pretrainedS2D}`);
    if (!cfg.pretrainedS2D) warn.push(`${cfg.version} 不需要 s2D 底模（v3/v4 路径），正常`);
  }

  if (phase === 's1') {
    if (!existsSync(cfg.pretrainedS1)) problems.push(`s1 底模缺失: ${cfg.pretrainedS1}`);
  }

  return { ok: problems.length === 0, problems, warn };
}

/** 预处理前置校验：list.txt 与 wav 目录 */
export function preflightPrep(cfg, { listFile, wavDir }) {
  const problems = [];
  if (!existsSync(listFile)) problems.push(`list.txt 不存在: ${listFile}`);
  if (!existsSync(wavDir)) problems.push(`wav 目录不存在: ${wavDir}`);
  if (!existsSync(cfg.bertDir)) problems.push(`BERT 权重缺失: ${cfg.bertDir}`);
  if (!existsSync(cfg.hubertDir)) problems.push(`cnhubert 权重缺失: ${cfg.hubertDir}`);
  if (!existsSync(cfg.pretrainedS2G)) problems.push(`s2G 底模缺失（3-get-semantic 要用）: ${cfg.pretrainedS2G}`);
  return { ok: problems.length === 0, problems };
}

export { artifactsOk };
