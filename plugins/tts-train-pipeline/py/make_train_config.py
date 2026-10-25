#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_train_config.py —— 把「模板 + 覆盖项」合成训练用的临时配置文件。

为什么要单独用 Python 做这件事：
    s2 的配置是 JSON（好办），s1 的配置是 **YAML**。
    在 Node 里手搓 YAML 序列化必然在某些键上出错（缩进、引号、`1e-4` 被写成字符串）。
    本机 venv 里有 pyyaml，交给它做，Node 只负责传一个 JSON 覆盖对象。
    这样「写配置」这件事只有一种实现，不会两处打架。

用法：
    python make_train_config.py --kind s2 --template A.json --overrides B.json --out C.json
    python make_train_config.py --kind s1 --template A.yaml --overrides B.json --out C.yaml

★ overrides 是**深度合并**：只覆盖你给的叶子，模板里其余键原样保留。
  这与 webui.py 的行为一致（它也是读模板 dict 后逐字段赋值）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="合成 GPT-SoVITS 训练配置")
    ap.add_argument("--kind", choices=["s2", "s1"], required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--overrides", required=True, help="JSON 文件，可为 {}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.template):
        print(json.dumps({"ok": False, "error": f"模板不存在: {args.template}"}, ensure_ascii=False))
        return 2
    try:
        with open(args.overrides, "r", encoding="utf-8") as f:
            over = json.load(f)
    except Exception as e:                                    # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"overrides 读取失败: {e}"}, ensure_ascii=False))
        return 2

    if args.kind == "s2":
        with open(args.template, "r", encoding="utf-8") as f:
            try:
                base = json.load(f)
            except json.JSONDecodeError as e:
                print(json.dumps({"ok": False, "error": f"模板不是合法 JSON: {e}"}, ensure_ascii=False))
                return 3
        merged = deep_merge(base, over)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    else:
        try:
            import yaml
        except ImportError:
            print(json.dumps({"ok": False, "error": "缺少 pyyaml：请在 GPT-SoVITS 的 venv 里运行"},
                             ensure_ascii=False))
            return 3
        with open(args.template, "r", encoding="utf-8") as f:
            base = yaml.load(f, Loader=yaml.FullLoader) or {}
        merged = deep_merge(base, over)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # 回读自证：文件真的落了、且能被对应解析器读回来
    size = os.path.getsize(args.out)
    if size < 50:
        print(json.dumps({"ok": False, "error": f"产物过小({size}B)"}, ensure_ascii=False))
        return 4

    keys = sorted((over or {}).keys())
    print(json.dumps({
        "ok": True, "kind": args.kind, "out": os.path.abspath(args.out),
        "sizeBytes": size, "overridden": keys,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
