#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_unk.py —— 给 GPT-SoVITS 打「消 UNK」文本补丁（幂等、可回滚、带校验）。

背景（实测，2026-10-24 同批 383 条语料）：

1) `japanese.py` 的 g2p 会把 `「」『』～―（）` 这类符号直接落到 symbols2 之外。
   symbols2.py 的符号表是 732 项，这些字符一个都不在里面，于是 cleaner 把它们
   统一降级成 `UNK`。它们本来就没有音素，是纯粹的噪声条件。

2) `japanese.py` 的 prosody 输出里有 `#`（accent phrase border），同样不在 732 项里，
   每一次词边界都会变成 UNK。后果是所有词边界共享同一个 embedding，模型分不清
   「短停」和「连读」。音高重音由 `[` `]` 编码，这两个在符号表里，删 `#` 不损发音。

★ 不能把 `#` 或这些符号加进 symbols2：732 → 733 会让底模 embedding 维度不匹配，
  等于要重训底模。

★ 打了补丁**必须重训**。实测「只打补丁不重训」是负收益（PER 0.197 → 0.236）：
  推理端文本条件变了、权重还是旧条件训的，两头不一致。
  同一个补丁通过 GSV_LEGACY_UNK=1 可还原上游行为，用于给旧权重做对照推理——
  否则 A/B 不成立。

用法：
    python patch_unk.py --gsv-root D:/AI-Voice/02_GPT-SoVITS            # 只报计划
    python patch_unk.py --gsv-root ... --apply                          # 真打
    python patch_unk.py --gsv-root ... --revert                         # 回滚
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys

MARK = "[DSH-UNK]"
# 无音值符号。半角双引号也剥：日语台词里常出现英文引号，同样不在符号表。
UNVOICED = "「」『』～―（）()\""
SYM_HASH = "#"

ANCHOR_JP = "    text = replace_consecutive_punctuation(text)"
ANCHOR_CL = '    phones = ["UNK" if ph not in symbols else ph for ph in phones]'

INS_JP = '''    # [DSH-UNK] 剥离无音值符号：pyopenjtalk 不会把它们转成音素，会直落 symbols2 之外变成 UNK。
    # 只删没有音值的标点，不改变任何有音素对应字符的 word2ph / 对齐。
    # GSV_LEGACY_UNK=1 还原上游行为——旧权重是在带 UNK 的文本条件下训的，
    # 推理端换条件就与训练不一致，做 A/B 时必须还原。
    # 代价：383 行里 7 行(1.83%)读音会变（4 例「有地さん（ありちさん）」→「ゆうちさん」），
    # 这是 pyopenjtalk 在短句上重新断词造成的，换成顿号也修不好。
    if os.environ.get("GSV_LEGACY_UNK") != "1":
        text = "".join(c for c in text if c not in _UNVOICED)
'''

INS_CL = '''    # [DSH-UNK] 日语 g2p 输出声调短语边界 "#"，symbols2 的 732 项里没有它，
    # 下一行会把每次词边界降级成 UNK（所有词边界共享同一个 embedding）。
    # 它没有音值，删掉不损发音——音高重音仍由 "[" "]" 编码。
    # 不能把 "#" 塞进符号表：732→733 会让底模 embedding 维度不匹配。
    # GSV_LEGACY_UNK=1 还原上游行为。
    if language in ("ja", "jp") and os.environ.get("GSV_LEGACY_UNK") != "1":
        phones = [ph for ph in phones if ph != "#"]
'''

DEF_JP = '''# [DSH-UNK] 无音值符号：不在 symbols2.py 的 732 项里，留着会被 cleaner 降级成 UNK。
# 拆成三行是为了在等宽字体下能看清每个字符，别把全角符号挤成一团。
_QUOTES = "「」『』"
_DASHES = "～―"
_PARENS = "（）()"
_UNVOICED = _QUOTES + _DASHES + _PARENS + '"'


'''


def read(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write(p: str, s: str) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def patch_japanese(src: str):
    """返回 (新源码, 说明)；已打过或锚点缺失时抛 ValueError。"""
    if MARK in src:
        raise ValueError("already")
    # 已有等价修复（不是本工具打的）：手工补丁、别的脚本、或上游自带。
    # 关键词是 GSV_LEGACY_UNK —— 只有真正做了「可回退的 UNK 剥离」才会出现它。
    # 这种情况必须跳过：再插一遍会得到两份剥离逻辑，代码脏且行为难追。
    if "GSV_LEGACY_UNK" in src or "_UNVOICED" in src:
        raise ValueError("already-external")
    if src.count(ANCHOR_JP) != 1:
        raise ValueError(f'锚点不唯一或缺失：{ANCHOR_JP!r} 出现 {src.count(ANCHOR_JP)} 次')
    if "def text_normalize(text):" not in src:
        raise ValueError("找不到 text_normalize，可能不是 GPT-SoVITS 的 japanese.py")
    out = src
    # os 必须在作用域内（上游正常有 import os；没有就在文件头补）
    if not any(l.strip() == "import os" for l in out.splitlines()[:40]):
        out = "import os\n" + out
    # 模块级常量插在 text_normalize 之前（若已有同名定义则跳过定义、只插调用）
    if "_UNVOICED" not in out:
        idx = out.index("def text_normalize(text):")
        out = out[:idx] + DEF_JP + out[idx:]
    out = out.replace(ANCHOR_JP, ANCHOR_JP + "\n" + INS_JP.rstrip("\n"), 1)
    return out, "text_normalize 增加 UNK 剥离"


def patch_cleaner(src: str):
    if MARK in src:
        raise ValueError("already")
    if "GSV_LEGACY_UNK" in src:
        raise ValueError("already-external")
    if src.count(ANCHOR_CL) != 1:
        raise ValueError(f'锚点不唯一或缺失：{ANCHOR_CL!r} 出现 {src.count(ANCHOR_CL)} 次')
    if not any(l.strip() == "import os" for l in src.splitlines()[:40]):
        src = "import os\n" + src
    out = src.replace(ANCHOR_CL, INS_CL.rstrip("\n") + "\n" + ANCHOR_CL, 1)
    return out, "clean_text 过滤日语 # 边界"


def check_symbols(gsv_root: str):
    """
    保护性校验：要剥掉的字符必须真的不在当前 symbols2 表里。

    如果哪天上游把它们加进了符号表，剥离就会删掉**有音素**的符号 —— 那比 UNK 更糟。
    所以这里直接拒绝，而不是照打。

    用 importlib 按文件加载 symbols2.py，不走 `from text import symbols2`：
    那个包初始化会牵连一堆依赖（本机实测会 circular import），而符号表本身是纯数据。
    """
    import importlib.util

    path = os.path.join(gsv_root, "GPT_SoVITS", "text", "symbols2.py")
    if not os.path.isfile(path):
        return {"checked": False, "reason": f"找不到 symbols2.py: {path}"}
    try:
        spec = importlib.util.spec_from_file_location("_dsh_symbols2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)              # type: ignore[union-attr]
        table = set(mod.symbols)
    except Exception as e:                                        # noqa: BLE001
        return {"checked": False, "reason": f"无法加载 symbols2: {type(e).__name__}: {e}"}
    bad = [c for c in UNVOICED if c in table] + ([SYM_HASH] if SYM_HASH in table else [])
    return {"checked": True, "total": len(table), "conflicts": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS 消 UNK 文本补丁")
    ap.add_argument("--gsv-root", required=True, help="GPT-SoVITS 工程根（含 GPT_SoVITS/text/）")
    ap.add_argument("--apply", action="store_true", help="真的写盘（默认只报计划）")
    ap.add_argument("--revert", action="store_true", help="从备份恢复")
    ap.add_argument("--report", default="")
    args = ap.parse_args(argv)

    jp = os.path.join(args.gsv_root, "GPT_SoVITS", "text", "japanese.py")
    cl = os.path.join(args.gsv_root, "GPT_SoVITS", "text", "cleaner.py")
    missing = [p for p in (jp, cl) if not os.path.isfile(p)]
    if missing:
        print(json.dumps({"ok": False, "error": "目标文件不存在，确认 --gsv-root 指向 GPT-SoVITS 根目录",
                          "missing": missing}, ensure_ascii=False))
        return 2

    if args.revert:
        return do_revert(jp, cl, args.report)

    sym = check_symbols(args.gsv_root)
    if sym.get("checked") and sym.get("conflicts"):
        print(json.dumps({"ok": False,
                          "error": "当前 symbols2 已包含待剥离字符，剥离会删掉有效符号，拒绝打补丁",
                          "conflicts": sym["conflicts"]}, ensure_ascii=False))
        return 4

    files = []
    errors = []
    plan = [(jp, patch_japanese, "japanese.py"), (cl, patch_cleaner, "cleaner.py")]
    built = {}
    for path, fn, label in plan:
        src = read(path)
        try:
            new, note = fn(src)
            ast.parse(new)                      # 语法不过就别写
            built[path] = (new, note)
            files.append({"file": label, "path": path, "action": "patch" if args.apply else "would-patch",
                          "note": note, "backup": path + ".bak_dsh_unk"})
        except ValueError as e:
            if str(e) == "already":
                files.append({"file": label, "path": path, "action": "already", "note": "已打过本工具的补丁"})
            elif str(e) == "already-external":
                files.append({"file": label, "path": path, "action": "already-external",
                              "note": "已存在等价的 UNK 修复（手工补丁或其它脚本），本工具不重复插入；"
                                      "如需回滚请用你自己的备份"})
            else:
                errors.append({"file": label, "error": str(e)})
        except SyntaxError as e:
            errors.append({"file": label, "error": f"打好后语法不通过，已放弃：{e}"})

    if errors:
        print(json.dumps({"ok": False, "error": "补丁失败", "files": files, "errors": errors},
                         ensure_ascii=False))
        return 3

    if args.apply:
        for path, (new, _note) in built.items():
            bak = path + ".bak_dsh_unk"
            if not os.path.isfile(bak):
                shutil.copy2(path, bak)
            write(path, new)

    out = {
        "ok": True,
        "gsv_root": os.path.abspath(args.gsv_root),
        "applied": bool(args.apply),
        "files": files,
        "symbols": sym,
        "next": ("必须重训（s2 → s1）。只打补丁不重训是负收益："
                 "实测 PER 0.197→0.236，因为推理条件变了、权重还是旧条件训的。"
                 "给旧权重做对照推理时设 GSV_LEGACY_UNK=1 还原上游行为。"
                 if args.apply else "这是计划；加 --apply 才会写盘"),
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
    return 0


def do_revert(jp: str, cl: str, report: str) -> int:
    files = []
    for path, label in ((jp, "japanese.py"), (cl, "cleaner.py")):
        bak = path + ".bak_dsh_unk"
        if not os.path.isfile(bak):
            files.append({"file": label, "action": "no-backup", "note": "没有 .bak_dsh_unk，跳过"})
            continue
        shutil.copy2(bak, path)
        os.remove(bak)
        files.append({"file": label, "action": "reverted", "path": path})
    out = {"ok": True, "reverted": True, "files": files}
    if report:
        with open(report, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
