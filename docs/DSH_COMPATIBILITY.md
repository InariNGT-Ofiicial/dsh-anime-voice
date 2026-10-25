# DSH 新旧宿主兼容

本次维护同时支持 **DSH 0.2.1-rc.2 与 0.2.1-rc.3**。保留原有工具名、配置键、数据目录和人工设置；已有安装可原位升级，无需版本豁免，也不要求升级 DSH。未重跑模型训练、音频推理或真实平台写入。

本仓库分发两个文件系统技能，保留现有 `install.ps1 -SkillRoot ... [-Update]` 安装方式，不改成 Cordis 插件。兼容测试通过官方 filesystem skill provider、SkillRegistry、skill 工具和 ToolRuntime，验证两个技能均可发现、读取、渲染，且资源目录结构正确。测试不包含模型自主选择技能、训练或配音。

## 复现

```powershell
node scripts/install-dsh-test-host.mjs 0.2.1-rc.2
node scripts/test-dsh-host.mjs
node scripts/install-dsh-test-host.mjs 0.2.1-rc.3
node scripts/test-dsh-host.mjs
```

测试宿主仅安装在仓库的 `work/dsh-host`，不读写用户 DSH profile、账号、凭据或会话。可用 `DSH_HOST_MODULE_ROOT` 指向已有官方运行时模块根目录；`DSH_TEST_PLUGIN_ROOT` 可指向解包后的发行包。宿主模块版本必须一致，测试不接受未声明的宿主版本。
