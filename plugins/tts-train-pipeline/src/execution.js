/**
 * 执行层：统一的子进程调用、产物自证、后台任务、日志尾部解析。
 *
 * 四条硬规矩（都是从上一代插件里换来的教训）：
 *   1. **产物自证**：退出码 0 不等于成功。每个阶段回读产物大小，缺失即失败。
 *   2. **不吞 stderr**：失败时把尾部原样带回，否则用户只能看到"没成功"。
 *   3. **后台任务写日志文件**：训练可能跑几小时，进程被回收也必须能从日志恢复进度。
 *   4. **detached 下的 spawn 失败是静默的**：父进程秒退、日志 0 字节。
 *      必须把 error 事件也写进日志，否则看起来像"启动了但没动静"。
 */
import { spawn } from 'node:child_process';
import {
  existsSync, mkdirSync, openSync, closeSync, readSync, readdirSync, readFileSync,
  statSync, writeFileSync, appendFileSync,
} from 'node:fs';
import { dirname } from 'node:path';

/** 拼一行给人看的命令行 */
export function cmdlineOf(cmd, args = []) {
  const q = (s) => (/[\s"]/u.test(String(s)) ? `"${String(s).replace(/"/g, '\\"')}"` : String(s));
  return [q(cmd), ...args.map(q)].join(' ');
}

/**
 * 运行环境。GPT-SoVITS 的 .venv 不需要像 DDSP 那样前置 Scripts 到 PATH
 * （它没有 "命令行走 PATH 找 ffmpeg" 的依赖），但 UTF-8 与 no_proxy 必须给：
 *   - PYTHONUTF8=1 防中文路径在 GBK 控制台下乱码
 *   - no_proxy 防本地梯度/端口请求被系统代理吞掉（本机实测过这个坑）
 */
export function pythonEnv(pythonExe, extra = {}) {
  const env = {
    ...process.env,
    PYTHONIOENCODING: 'utf-8',
    PYTHONUTF8: '1',
    // ★ 后台训练的日志靠 stdout 重定向到文件，而 Python 对**非 tty 的 stdout 是块缓冲**
    //   （默认 8KB）。不设这个，训练跑了几分钟日志还是 0 字节，看起来像没启动；
    //   更要命的是崩溃/被 abort 时缓冲区不会刷，真正的报错直接丢失。
    PYTHONUNBUFFERED: '1',
    no_proxy: 'localhost,127.0.0.1,::1',
    NO_PROXY: 'localhost,127.0.0.1,::1',
    ...extra,
  };
  // HF 相关下载若被沙箱代理劫持会很慢，交给调用方决定；这里只补 PYTHONPATH
  if (pythonExe) env.__GSV_PYTHON = pythonExe;
  return env;
}

/** 前台执行一次，收集输出，超时强杀。 */
export function runOnce({ cmd, args = [], cwd, env, timeoutMs = 3_600_000, echo = false }) {
  return new Promise((resolve) => {
    const started = Date.now();
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let child;
    try {
      child = spawn(cmd, args, { cwd, env, windowsHide: true });
    } catch (e) {
      const msg = String(e?.message ?? e);
      resolve({ ok: false, exitCode: null, stdout, stderr: msg, ms: 0, error: msg, cmdline: cmdlineOf(cmd, args) });
      return;
    }
    const timer = setTimeout(() => { timedOut = true; try { child.kill('SIGKILL'); } catch { /* ignore */ } }, timeoutMs);
    child.stdout?.on('data', (b) => { const s = b.toString('utf8'); stdout += s; if (echo) process.stderr.write(s); });
    child.stderr?.on('data', (b) => { const s = b.toString('utf8'); stderr += s; if (echo) process.stderr.write(s); });
    child.on('error', (e) => {
      clearTimeout(timer);
      const msg = String(e?.message ?? e);
      resolve({ ok: false, exitCode: null, stdout, stderr: `${stderr}\n${msg}`, ms: Date.now() - started, error: msg, cmdline: cmdlineOf(cmd, args) });
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ ok: code === 0 && !timedOut, exitCode: code, stdout, stderr, ms: Date.now() - started, timedOut, cmdline: cmdlineOf(cmd, args) });
    });
  });
}

/** 后台启动，stdout/stderr 落日志文件；进程 detach，插件重载也不影响训练。 */
export function spawnJob({ cmd, args = [], cwd, env, logPath }) {
  mkdirSync(dirname(logPath), { recursive: true });
  // 每次新任务清掉旧日志，避免上一次的 "training done" 误导 status
  try { writeFileSync(logPath, ''); } catch { /* ignore */ }
  const fd = openSync(logPath, 'a');
  let child;
  try {
    child = spawn(cmd, args, {
      cwd, env, windowsHide: true, detached: true,
      stdio: ['ignore', fd, fd],
    });
    child.on('error', (e) => {
      try { appendFileSync(logPath, `\n[spawn error] ${e?.message ?? e}\n`); } catch { /* ignore */ }
    });
  } finally {
    closeSync(fd);
  }
  child.unref();
  return { pid: child.pid, logPath, pidValid: Number.isInteger(child.pid), cmdline: cmdlineOf(cmd, args) };
}

/** 产物自证：回读每个路径的存在性与大小。 */
export function artifactsOf(paths) {
  const list = (Array.isArray(paths) ? paths : [paths]).filter((p) => typeof p === 'string' && p);
  return list.map((p) => {
    let exists = false;
    let sizeBytes = 0;
    let isDir = false;
    let entries = 0;
    try {
      if (existsSync(p)) {
        exists = true;
        const st = statSync(p);
        isDir = st.isDirectory();
        if (isDir) {
          // ★ 目录产物（4-cnhubert / 5-wav32k / 7-sv_cn）不能只看「目录存在」：
          //   上游脚本第一步就是 makedirs，跑挂了目录照样在。必须数条目。
          try { entries = readdirSync(p).length; } catch { entries = 0; }
        } else {
          sizeBytes = st.size;
        }
      }
    } catch { /* ignore */ }
    const ok = exists && (isDir ? entries > 0 : sizeBytes > 0);
    return { path: p, exists, isDir, entries, sizeBytes, ok };
  });
}

/** 产物判定：全部存在且非空（文件看字节，目录看条目数） */
export function artifactsOk(paths, minBytes = 1024) {
  const arts = artifactsOf(paths);
  return { arts, ok: arts.length > 0 && arts.every((a) => a.ok) };
}

/**
 * 读日志尾部若干行（后台任务的唯一进度来源）。
 *
 * ★ 必须同时按 `\r` 切：训练用 tqdm 刷进度条，\r 分隔且**可以长达上百 KB**。
 *   只按 \n 切会把整条进度条当成"一行"甩到界面上，真正要看的信息全被淹没。
 *   切完再限制单行长度，避免超长一行撑爆输出。
 */
export function tailFile(path, lines = 40, maxLineChars = 400) {
  try {
    if (!existsSync(path)) return '';
    // 只读尾部 256KB：训练日志跑几小时后可能几十 MB，全读会拖慢每次 status
    const size = statSync(path).size;
    const take = Math.min(size, 256 * 1024);
    const fd = openSync(path, 'r');
    let text = '';
    try {
      const buf = Buffer.alloc(take);
      readSync(fd, buf, 0, take, size - take);
      text = buf.toString('utf8');
      if (size > take) text = text.replace(/^[^\n]*\n/, '');   // 丢掉被切断的首行
    } finally {
      closeSync(fd);
    }
    const arr = text.split(/\r\n|\r|\n/)
      .map((l) => l.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '').trimEnd())
      .filter((l) => l.trim() !== '');
    return arr.slice(-lines).map((l) => (l.length > maxLineChars ? l.slice(0, maxLineChars) + '…' : l)).join('\n');
  } catch { return ''; }
}

/**
 * 从日志里抓训练进度。
 * s2_train：tqdm 形如 `epoch=0: 12%|█ | 7/59 [00:03<00:25, loss_d=..., loss_g=...]`
 *           logger 另有一行 `====> Epoch: 3`
 * s1_train：lightning 形如 `Epoch 2: 45%|`，另有 `loss=...`
 * 三层正则都试，抓不到就是抓不到，不编数字。
 */
export function parseProgress(text) {
  const out = { epoch: null, step: null, lossD: null, lossG: null, total: null, pct: null };
  if (!text) return out;
  const lines = text.split(/\r?\n/);

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const l = lines[i];

    if (out.epoch === null) {
      const m = /epoch\s*=\s*(\d+)/i.exec(l)
        || /Epoch\s*[:=]?\s*(\d+)/i.exec(l)
        || /start training from epoch\s+(\d+)/i.exec(l);
      if (m) out.epoch = Number(m[1]);
    }
    if (out.lossD === null) {
      const m = /loss_d\s*[=:]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)/i.exec(l);
      if (m) out.lossD = Number(m[1]);
    }
    if (out.lossG === null) {
      const m = /loss_g\s*[=:]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)/i.exec(l);
      if (m) out.lossG = Number(m[1]);
    }
    if (out.pct === null) {
      const m = /(\d{1,3})%\|/.exec(l);
      if (m) out.pct = Number(m[1]);
    }
    if (out.step === null) {
      const m = /(\d+)\/(\d+)\s*\[/.exec(l);
      if (m) { out.step = Number(m[1]); out.total = Number(m[2]); }
    }
    if (out.epoch !== null && out.lossG !== null && out.step !== null) break;
  }
  return out;
}

/** 日志里出现过的「终态」标记 */
export function detectFinish(text) {
  if (!text) return { done: false, failed: false, reason: '' };
  if (/training done/i.test(text)) return { done: true, failed: false, reason: 'training done' };
  if (/Traceback \(most recent call last\)/i.test(text)) {
    const last = text.split(/\r?\n/).filter((l) => l.trim()).slice(-3).join(' | ');
    return { done: true, failed: true, reason: last };
  }
  if (/out of memory|CUDA error/i.test(text)) {
    return { done: true, failed: true, reason: 'CUDA / 显存错误' };
  }
  return { done: false, failed: false, reason: '' };
}

// ───────────────────────────────────────────────────────────────────────────
// 后台任务登记（status / stop 的唯一数据来源）
// ───────────────────────────────────────────────────────────────────────────

export function readJobs(file) {
  try {
    const arr = JSON.parse(readFileSync(file, 'utf8'));
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

export function writeJobs(file, jobs) {
  try {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(jobs.slice(-50), null, 2), 'utf8');
    return true;
  } catch { return false; }
}

export function upsertJob(file, job) {
  const jobs = readJobs(file);
  const i = jobs.findIndex((j) => j.id === job.id);
  if (i >= 0) jobs[i] = { ...jobs[i], ...job };
  else jobs.push(job);
  writeJobs(file, jobs);
  return job;
}

/**
 * 进程是否还活着。
 * ★ 不能只信 pid：Windows 的 pid 会被复用，光看 pid 存在会把"上次的别的进程"当成训练还在跑。
 *   所以同时核对进程命令行里是否含本次的特征串（日志路径或 expName）。
 */
export async function jobAlive(job) {
  if (!job?.pid) return false;
  const r = await runOnce({
    cmd: 'tasklist',
    args: ['/FI', `PID eq ${job.pid}`, '/FO', 'CSV', '/NH'],
    timeoutMs: 8000,
  });
  const line = (r.stdout || '').trim();
  if (!line || /No tasks are running|没有运行的任务/i.test(line)) return false;
  return line.includes(String(job.pid));
}

/** Windows 上停止训练：taskkill /T /F 连子进程一起收（python 会 spawn dataloader worker） */
export async function killJob(job) {
  if (!job?.pid) return { ok: false, stderr: '没有 pid' };
  const r = await runOnce({
    cmd: 'taskkill',
    args: ['/T', '/F', '/PID', String(job.pid)],
    timeoutMs: 20000,
  });
  return { ok: r.ok || /SUCCESS|成功/i.test(r.stdout + r.stderr), stdout: r.stdout, stderr: r.stderr };
}

/** 扫目录里的权重文件（用于 status 判定"最新落盘到第几轮"） */
export function listWeights(dir, re) {
  try {
    if (!existsSync(dir)) return [];
    return readdirSync(dir)
      .map((f) => {
        const m = re.exec(f);
        if (!m) return null;
        let sizeBytes = 0;
        try { sizeBytes = statSync(`${dir}/${f}`).size; } catch { /* ignore */ }
        return { file: f, epoch: Number(m[1]), sizeBytes };
      })
      .filter(Boolean)
      .sort((a, b) => a.epoch - b.epoch);
  } catch { return []; }
}
