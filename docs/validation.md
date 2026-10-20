# 验证记录（2026-10-20，Asia/Shanghai）

本记录区分独立 CLI、真实 DSH 宿主和模型质量。尚未完成的项目不会标为通过。

## 环境

| 项目 | 实测值 |
|---|---|
| DSH 桌面与运行时 | 0.2.1-rc.2 |
| GPT-SoVITS 本地上游提交 | `1d9fe63d2875238bc240bd5e7b8dacaee73f8396` |
| Python | 3.12，现有上游 venv |
| torch / CUDA | 2.7.1+cu128 / 12.8 |
| numpy / soundfile | 1.26.4 / 0.14.0 |
| faster-whisper / funasr | 1.2.1 / 1.3.30 |
| modelscope / pyworld / transformers | 1.40.2 / 0.3.5 / 4.57.6 |
| GPU | RTX 5060 Laptop，8151 MiB；先用 CPU，收到 GPU 释放通知并确认显存空闲后补做一次短 GPU 推理 |

以上是本机版本记录，不代表所有依赖功能都被本轮执行。

## 已执行的独立验证

- 两个 `SKILL.md` 通过 skill-creator 的结构检查，使用 Python UTF-8 模式。
- 9 项 CPU 回归通过：显式实验/数据集、清单阶段不调用预处理、跨目录重名不覆盖、数据指纹改变时拒绝复用旧特征、旧实验无指纹时拒绝复用、流水线真实调用训练驱动、导出器拒绝多角色误标，以及推理成功/异常时恢复上游模型选择文件。
- A 从 12 条已有真实片段执行了一次局部导出：保留 8 条、23.6 秒。该次独立导出缺词边界缓存，因此不能据此证明边界正确；DSH 复测指定含词边界缓存的 source。
- B 对新数据集执行 `--stage check --exp kurumi_v3 --trainset <NEW_TRAINSET>`：291 条。数据指纹 `cc6453fa5c06269b13f1f8529da143bc172eda3a637a279389d1fdabc8b721fd`。`check` 没有训练或写入该旧实验。
- 已有 v3 检查点的独立 CPU 单句推理通过：3.42 秒、32 kHz、109440 个样本、种子 7，进程退出码 0，含加载耗时约 67.9 秒。目标「今日は新しい一日が始まります。」经 CPU faster-whisper 强制日语复核后文字一致（忽略标点字错率 0），峰值 0.2361、RMS 0.0330、样本均为有限值。这不是盲听或音色质量验收。
- CPU 推理首次尝试报 `Invalid device id`。已修复可移植运行时提前初始化 CUDA 的影响：以启动前已屏蔽 CUDA 的子解释器运行，连同上游 `_CUDA_VISIBLE_DEVICES` 覆盖项一起设置；第二次实际成功。推理前后上游 `weight.json` SHA-256 一致：`df2e107529f4f71995cfa5ab7cc95741fdce59fe8f1cc06edb0c61aaa9b6b5c3`。
- 首个发布提交 `390810a797fb1a334ac8ecdc1e27ee56299333fb` 的 [GitHub Actions CPU interfaces](https://github.com/InariNGT-Ofiicial/dsh-anime-voice/actions/runs/39369768676) 已通过。
- 补测安装版 `run_pipeline.py --stages train,report --train-stage list`：读取 A 的 8 条真实导出片段，实际生成 `gw_list.txt` 和 HTML 报告，两个子命令退出码均为 0，`pipeline-status.json` 两阶段均为 `passed`，没有创建训练实验目录。`list` 写清单，**不是只读查询**；只读校验用 `check`。旧版误将 `list` 路由到 `prep` 的问题已修复。
- GPU 释放后独立短推理通过：CUDA 模式，3.46 秒、32 kHz、110720 样本，含加载约 23.0 秒，退出码 0。CPU ASR 复核忽略标点字错率为 0，峰值 0.2354、RMS 0.03354；上游模型选择文件恢复一致。见 [GPU 推理证据](evidence/cli-gpu-inference.json) 和 [实际清单/报告调用](evidence/cli-pipeline.json)。这些都是 CLI 成绩。

## 真实 DSH 第一次尝试

通过桌面界面新建独立会话，使用自然语言要求执行本地测试说明。宿主模型为 DeepSeek-V4.1-Flash，High。

| 项目 | 结果 |
|---|---|
| 读取本地测试说明 | 通过：宿主 `read` 工具 |
| 发现并读取 A 技能 | 通过：宿主 `skill` 工具 |
| 发现并读取 B 技能 | 通过：宿主 `skill` 工具 |
| shell 工具进入插件执行 | 未通过：授权阶段报错，尚未执行插件代码 |
| 具体错误 | `SetNamedSecurityInfoW failed (Win32 5): grantWrite(<WORK>)` |
| 自动补救 | 宿主提出 ACL 修复并等待用户审批；本任务未批准该权限修改 |
| 导出、报告、配音产物 | 本次尝试尚无产物 |

已保留完整原始会话作为本机证据，不上传。[脱敏事件摘录](evidence/dsh-attempt1.json) 仅保留选定工具名、调用标识和规范化错误。仅有技能加载成功不足以宣称 DSH 全链路通过。独立 CLI 推理与 ASR 的数字来自本节前一部分，不能算宿主成绩。后续真实复测结果将在本文件追加。

宿主重启方案尚待用户明确允许，当前不宣称 ACL 问题已修复。前一天的另一 Cordis TTS/训练工具插件不属于本仓库，其 doctor/status 结果没有计入本次验证。

## 本轮不覆盖

三集完整分离/ASR、全量训练、其它 GPT-SoVITS 版本、冷启动自动下载、跨平台、其它角色与语种，以及盲听质量判断。当前 17.9 分数据集和 v3 检查点的存在不等于旧声纹/F0结果对它仍有效。没有“0.807不可区分”或固定分钟数达标结论。
