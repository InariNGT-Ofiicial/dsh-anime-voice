#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_two_stage.py —— 一条后台进程串起 s2 → s1。

为什么需要它：
    s2（SoVITS 声学）必须成功，s1（GPT 语义）才有意义 —— 顺序不能倒、不能并行。
    如果由插件侧连续 spawn 两个后台进程，中间"等 s2 结束"这一步没有可靠的持有者：
    DSH 的插件生命周期与训练时长不是一个量级。

    所以把「串行」这件事变成一个自持的进程：它同时承担
      · 顺序与门禁（s2 没落权重就不启动 s1）
      · 单一日志（两个阶段的输出按序落到同一个文件，status 只看一个尾巴）
      · 失败即停（不继续烧 GPU 时间）

门禁的判定不靠退出码，而是**看权重文件真的出现了**：
    SoVITS_weights_<ver>/<exp>_e<N>_s<M>.pth
退出码 0 但没落权重（例如 epochs 小于 save_every_epoch）在这条链上是失败的。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def banner(msg):
    print(f"\n===== [two-stage] {msg} =====", flush=True)


def run_stage(title, cmd, cwd, env):
    banner(f"start {title}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env)
    dt = time.time() - t0
    banner(f"end {title} exit={p.returncode} elapsed={dt:.1f}s")
    return p.returncode, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="串行执行 s2 与 s1 训练")
    ap.add_argument("--gsv-dir", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--s2-script", default="GPT_SoVITS/s2_train.py")
    ap.add_argument("--s2-config", required=True)
    ap.add_argument("--s1-config", default="")
    ap.add_argument("--weight-dir", required=True, help="SoVITS 权重输出目录（门禁判定用）")
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-s1", action="store_true")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    env["_CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["hz"] = "25hz"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env["no_proxy"] = "localhost,127.0.0.1,::1"

    # ── s2 ────────────────────────────────────────────────────────────────
    rc, _ = run_stage("s2", [args.python, "-s", args.s2_script, "--config", args.s2_config], args.gsv_dir, env)
    if rc != 0:
        banner(f"FAILED: s2 退出码 {rc}，不启动 s1")
        return 10

    pattern = os.path.join(args.weight_dir, f"{glob.escape(args.exp_name)}_e*_s*.pth")
    got = sorted(glob.glob(pattern))
    if not got:
        banner("FAILED: s2 退出码为 0，但权重目录里没有落盘的 .pth")
        banner(f"  pattern: {pattern}")
        banner("  常见原因：epochs < save_every_epoch，导致一次都没保存")
        return 11
    banner(f"s2 落盘 {len(got)} 个权重，最新 {os.path.basename(got[-1])}")

    if args.skip_s1 or not args.s1_config:
        banner("跳过 s1（未给配置或显式跳过）")
        return 0

    # ── s1 ────────────────────────────────────────────────────────────────
    rc, _ = run_stage("s1", [args.python, "-s", "GPT_SoVITS/s1_train.py", "--config_file", args.s1_config], args.gsv_dir, env)
    if rc != 0:
        banner(f"FAILED: s1 退出码 {rc}")
        return 20

    banner("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
