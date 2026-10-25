/**
 * asr-gate.js —— ASR 回读判定：这个合成结果到底念对了没有？
 *
 * ★ 为什么需要独立的判定模块
 *   早先的判定只看**开头 6 个字**是否包含：
 *     `norm(heard).includes(norm(want).slice(0, 6))`
 *   于是「我们即使没有足够的兵团也能成为赢下逃赛的成员」这种
 *   **开头对、后面全错**的输出照样被判「✅ 内容基本一致」。
 *   前缀判对对「系统性音素错误」完全失明 —— 而跨语言失败正是这一类。
 *
 * ★ 为什么单靠 ASR 置信度也不行
 *   whisper 有强语言模型先验：听不懂时会吐**高置信度的幻觉**
 *   （实测：一段音全错的中文，用日文 ASR 听得 -0.542，输出的却是
 *    「ご視聴ありがとうございました」；而念对了的中文反而 -0.818）。
 *   所以置信度必须和「同语言的字错率」一起看。
 */

/** 编辑距离（Levenshtein），用于算字错率 CER */
export function editDistance(a, b) {
  const m = a.length;
  const n = b.length;
  if (!m) return n;
  if (!n) return m;
  let prev = new Array(n + 1);
  let cur = new Array(n + 1);
  for (let j = 0; j <= n; j += 1) prev[j] = j;
  for (let i = 1; i <= m; i += 1) {
    cur[0] = i;
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev;
    prev = cur;
    cur = t;
  }
  return prev[n];
}

const normText = (s) => String(s || '')
  .replace(/[\s，。！？、,.!?「」『』"'…—·:：;；()（）]/g, '');

/**
 * 两个条件都过才算念对了：
 *   - 全文字错率 CER ≤ 0.15（日文 0.35，见下）
 *   - 声学置信度 avg_logprob ≥ -0.35
 *
 * 阈值依据（同一段中文散文，实测）：
 *   底模 s1（念对了）  lp -0.138 · CER ≈ 0.05
 *   自训 s1（音全错）  lp -0.419 · CER ≈ 0.38
 *
 * ★ 日文阈值必须单独放宽到 0.45：ASR 输出的是「私たち / 綺麗 / 透き通った / 食べ / 桃色」，
 *   而原文是「わたしたち / きれい / すきとおった / たべ / ももいろ」——
 *   同音不同表记在**字符级**比较里全算错。实测：同一段日文，念对了 CER 就有 30.5%，
 *   句尾少两个「ます」是 37.3%。两者只差 7 个百分点 —— 字符级 CER 在日文上信噪比极低，
 *   只能用来拦「大面积离谱」，细节得靠 avg_logprob 和人耳。
 *   中文没这个问题（表记统一），所以压到 0.15 仍然够用。
 */
export function asrGate(heard, want, avgLogprob, opts = {}) {
  const h = normText(heard);
  const w = normText(want);
  // 目标文本里有假名 ⇒ 按日文判，表记差异不算错
  const hasKana = /[\u3040-\u30ff]/.test(w);
  const CER_MAX = opts.cerMax ?? (hasKana ? 0.45 : 0.15);
  const LP_MIN = opts.lpMin ?? -0.35;
  const dist = editDistance(h, w);
  const cer = w.length ? dist / w.length : 1;
  const lp = typeof avgLogprob === 'number' ? avgLogprob : null;
  const cerOk = cer <= CER_MAX;
  const lpOk = lp === null ? true : lp >= LP_MIN;
  return { cer, cerMax: CER_MAX, dist, len: w.length, lp, cerOk, lpOk, ok: cerOk && lpOk };
}
