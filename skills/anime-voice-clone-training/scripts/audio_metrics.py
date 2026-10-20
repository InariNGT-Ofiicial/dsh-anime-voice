# -*- coding: utf-8 -*-
"""客观度量：F0 / 周期度 / 频谱包络（共振峰）/ 瞬态。

统一口径，所有评估脚本都从这取数，避免各写一份导致数字对不上。
"""
import numpy as np
import librosa
import soundfile as sf

SR = 32000
HOP = 160  # 5ms


def load(path, sr=SR):
    y, s0 = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if s0 != sr:
        y = librosa.resample(y, orig_sr=s0, target_sr=sr)
    return y, sr


def periodicity(y, sr, win=1024, hop=HOP):
    """帧级归一化自相关峰：1=纯周期音，<0.45 就是噪声/破音"""
    n = 1 + (len(y) - win) // hop
    out = np.zeros(max(n, 0), dtype=np.float32)
    for i in range(n):
        seg = y[i * hop:i * hop + win]
        seg = seg - seg.mean()
        if np.dot(seg, seg) < 1e-8:
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        lo, hi = int(sr / 1200), min(int(sr / 70), len(ac) - 1)
        if hi > lo:
            out[i] = float(np.max(ac[lo:hi]) / ac[0])
    return out


def envelope_peak(path):
    """频谱包络峰（cheaptrick，F0 无关）—— 判断「腔体有没有被改小」。
    只对 ≥300Hz 区域取峰：变调后低频空虚是正常的，不是共振峰移动。"""
    import pyworld as pw
    y, sr = load(path)
    x = y.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    v = f0 > 0
    if not v.any():
        return 0
    f = np.linspace(0, sr / 2, sp.shape[1])
    E = sp[v].mean(axis=0)
    m = (f > 300) & (f < 5000)
    from scipy.ndimage import uniform_filter1d
    Es = uniform_filter1d(E[m], size=max(3, int(m.sum() / 150)))
    return round(float(f[m][int(np.argmax(Es))]))


def formants(path):
    """LPC 的 F1/F2 —— 只在 F0 不高（<350Hz）时可信；F0 一旦接近 F1 会被谐波带偏，别用它判变调。"""
    y, sr = load(path)
    n_fft = 1024
    S = []
    for i in range(0, len(y) - n_fft, n_fft):
        seg = y[i:i + n_fft] * np.hamming(n_fft)
        if np.sum(seg ** 2) < 1e-6:
            continue
        try:
            a = librosa.lpc(seg, order=16)
        except Exception:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        fr = np.sort(np.angle(roots) * sr / (2 * np.pi))
        fr = fr[(fr > 200) & (fr < 5000)]
        if fr.size >= 2:
            S.append(fr[:2])
    if not S:
        return dict(f1=0, f2=0)
    S = np.array(S)
    return dict(f1=round(float(np.median(S[:, 0]))), f2=round(float(np.median(S[:, 1]))))


def measure(path):
    """一次给全：音高（前 30% / 全句 / 起落）、八度乱跳率、周期度、包络峰、削波"""
    y, sr = load(path)
    f0 = librosa.yin(y, fmin=90, fmax=1200, sr=sr, frame_length=2048, hop_length=HOP)
    per = periodicity(y, sr)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    n = min(len(f0), len(per), len(rms))
    f0, per, rms = f0[:n], per[:n], rms[:n]
    ok = np.isfinite(f0) & (rms > np.percentile(rms, 30))
    if not ok.any():
        return {}
    med = float(np.median(f0[ok]))
    k = int(n * 0.3)
    okh = ok[:k]
    head = float(np.median(f0[:k][okh])) if okh.any() else 0.0
    return dict(dur=round(len(y) / sr, 2), head=round(head), med=round(med),
                drop=round(12 * np.log2(head / max(med, 1)), 1),
                jump=round(float(np.mean(np.abs(12 * np.log2(f0[ok] / med)) > 6) * 100)),
                per=round(float(np.median(per[ok])), 2),
                peak_db=round(20 * np.log10(max(float(np.max(np.abs(y))), 1e-9)), 1),
                clip=round(float(np.mean(np.abs(y) > 0.999)) * 100, 3),
                env_peak=envelope_peak(path))
