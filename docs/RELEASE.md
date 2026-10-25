# 二次元语音双技能 v0.1.0

正式版本，同时支持 **DSH 0.2.1-rc.2 与 0.2.1-rc.3**。原有配置、工具名和人工设置保持兼容，无需宿主版本豁免。

两个文件系统技能首次提供版本化 ZIP 发布：`anime-voice-line-extractor`、`anime-voice-clone-training`。保留 `install.ps1 -SkillRoot ... [-Update]` 安装方式；已有安装更新前自动备份。

Windows / Linux × rc.2 / rc.3 的原生宿主兼容 CI 通过；本轮触发的原有回归 CI 全部通过。本地安装包还经过两个官方 npm 运行时及已安装 rc.3 Desktop 内置模块验证。

附件包含技能 ZIP 和 `SHA256SUMS.txt`。解压后按 README 安装；Python、GPT-SoVITS、MSST 和模型仍需使用者自行配置。

本次兼容验收未运行 GPU 训练/翻唱、模型 API 或真实平台写入；宿主模块测试不代表 Desktop UI 全业务验收。验证边界与复现见 [DSH_COMPATIBILITY.md](docs/DSH_COMPATIBILITY.md)。
