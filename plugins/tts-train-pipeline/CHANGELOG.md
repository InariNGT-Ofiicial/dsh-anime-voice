# 变更记录

## 0.2.0 — 2026-10-25

首次进入 `dsh-anime-voice` 仓库（此前只在本机开发）。

### 新增：`tts_patch_text`（消 UNK 文本补丁）

`symbols2.py` 的符号表是 732 项，而 `japanese.py` 的 g2p 会输出两类不在表里的东西：
`「」『』～―（）()` 这类无音值符号，以及 `#`（声调短语边界，每处词边界一个）。
`cleaner.py` 会把它们统一降级成 `UNK`，后果是所有词边界共享同一个 embedding。

工具改两处（`japanese.py` 剥符号 + `cleaner.py` 过滤 `#`），并带三条约束：

- 幂等（按 `[DSH-UNK]` 标记与 `GSV_LEGACY_UNK` 双重识别，已有等价补丁不重复插入）；
- 自动备份 `<file>.bak_dsh_unk`，`revert=true` 完全回滚；
- 打前核对符号表：若上游哪天把这些字符收进表里，剥离会删掉**有音素**的符号，工具直接拒绝。

`GSV_LEGACY_UNK=1` 可还原上游行为，用于给旧权重做对照推理。
**打了必须重训**——实测只打补丁不重训是负收益（PER 0.197 → 0.236）。

### 新增：`tts_build` 的尾音体检与留尾参数

- 输出 `tail_stats`：收尾电平中位（真人自然收尾约 −24dB）、尾部静音中位、收尾 > −12dB 的占比、判词。
- 新参数 `tailPad`（默认 0.16s）与 `trimTail`（复现旧数据集，会打警告）。

配套修正 `README §4.1`：此前把 `areverse + silenceremove` 标成「✅ 正确」，实测它会把尾部
250~400ms 真实语音一起削掉（`yos001_001.ogg`：3.04s → 2.64s，尾静音 82ms → 0ms，
收尾电平 −42.6dB → +0.5dB）。正确做法是只裁头 + 保留原衰减 + 兜静音底。

### 变更

- 工具数 13 → 14。
- 仓库不再包含 `experiments/`、`baseline/`、`test/*.py` 等开发期实验产物与基线音频；
  `npm test` 仍等于 `smoke`。

## 0.1.0

- 本地开发版：`tts_doctor` / `tts_plan` / `tts_script` / `tts_scan` / `tts_build` / `tts_asr` /
  `tts_preprocess` / `tts_run` / `tts_status` / `tts_stop` / `tts_infer` / `tts_register` / `tts_overview`。
