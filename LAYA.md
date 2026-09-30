# Laya 本地语义唤醒

[返回 README](README.md)

Laya 是可选的本地语义判断后端。Laya Python 源码已包含在插件的 `vendor/laya` 中。选择 `semantic_backend = laya` 后，插件使用 AstrBot 主程序的 Python（`sys.executable`）启动子进程加载模型，后续群消息复用该进程；不调用 Jev，也不需要 Jev API Key。@、引用和关键词仍可直接唤醒。

## 配置

在插件配置的「语义唤醒」中选择 **Laya 本地**，并在 `laya_model` 中选择模型，无需配置 Python 路径。模型目录由插件统一管理：

```text
Laya 模型选择     multilingual（默认）
模型目录          data/plugin_data/astrbot_plugin_qq_group_enhance/models/
计算设备          cuda
```

AstrBot 的 Python 环境须安装 PyTorch、Transformers、safetensors、huggingface_hub、NumPy 等模型依赖，无需另外安装 `laya`。worker 始终从插件自带目录读取 Laya 源码。支持 `english`（`convaiinnovations/laya`）、`multilingual`（`convaiinnovations/laya-multilingual`）和 `typed-decisions`（`convaiinnovations/laya-typed-decisions`）。从 AstrBot WebUI 侧边栏进入 **模型实验场**，在 **Laya** 分组即可下载模型并选择测试；下载同样使用 AstrBot 的 Python 环境，权重只保存到 `plugin_data/astrbot_plugin_qq_group_enhance/models/`，不会写入插件源码目录。指定的设备不可用时，启动或推理会报错并暂停语义判断。下载后可直接测试群聊唤醒或自定义选项，使用方式见 [页面测试](README.md#-页面测试)。

旧的 `semif_python` 和 `laya_python` 配置已移除，直接传入会报错；请删除这些字段，并将模型依赖安装到 AstrBot 的 Python 环境。

每次推理默认最多 8 题，显存紧张时可设为 2。三种后端共用的 `semantic_threshold` 接话概率阈值默认 0.7，整批判断超时默认 30 秒，模型加载超时默认 180 秒。配置缺失、加载失败、超时或评分失败都不会切换到其他后端；直接唤醒仍可用。

## 判断方式与输入限制

每条候选消息分别生成正反顺序的两道 `choice` 题，用中性 **A/B** 标签标出「应接话」与「不应接话」两项完整陈述，取两次“应接话”概率的平均值。没有使用 `noul` 的 `false/true` 标签，因为修正评测报告（`D:\benchmarks\laya_semif_20260924\corrected_01\FINAL_REPORT.md`）观察到这种标签可能主导结果。报告的自建事实题成绩不代表群聊唤醒准确率；Laya 概率也未针对本场景校准，建议依据实际群聊误唤醒、漏唤醒样本调节阈值。

Laya multilingual 权重当前的输入配置为 `max_len=1024`、`head_max_len=256`。worker 在调用模型前按 Laya 自身的分词与序列构建逻辑逐题预检；超限时只省略最早的历史，并记录省略数。如果题目或候选消息本身超限，整批判断失败并记录错误，不让模型静默截断目标内容。重载或关闭插件时结束模型进程。
