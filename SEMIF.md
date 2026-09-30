# SemIf 本地语义唤醒

[返回 README](README.md)

SemIf 是可选的本地语义判断后端。SemIf Python 源码已包含在插件的 `vendor/semif_phase1` 中。选择 `semantic_backend = semif` 后，插件使用 AstrBot 主程序的 Python（`sys.executable`）启动子进程，加载一次模型，后续群消息复用该进程；不会调用 Jev，也不需要 Jev API Key。@、引用和关键词仍直接唤醒。

## 配置

在 AstrBot 的 Python 环境中安装 PyTorch、Transformers 等模型依赖，无需另外安装 `semif-phase1`。在插件配置的「语义唤醒」中选择 **SemIf 本地**，并选择预设模型或自定义模型，无需配置 Python 路径。从 AstrBot WebUI 侧边栏进入 **模型实验场**，在 **SemIf** 分组即可下载预设模型并选择用于测试。下载同样使用 AstrBot 的 Python 环境。下载完成可直接在统一页面测试群聊唤醒、Choice、Score 或 Noul，见 [页面测试](README.md#-页面测试)；测试页面的模型选择不会修改群聊后端配置。下载目录为 `data/plugin_data/astrbot_plugin_qq_group_enhance/models/semif/<模型名>/`。

| 预设 | Hugging Face 模型 | 说明 |
| --- | --- | --- |
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | 较小，资源需求较低 |
| Qwen3.5 0.8B | `Qwen/Qwen3.5-0.8B` | 较小，但已有选项顺序偏置观察 |
| MiniCPM5 2B | `openbmb/MiniCPM5-2B` | 中等规模 |
| Qwen3.5 4B | `Qwen/Qwen3.5-4B` | 质量相对较稳，显存需求较高 |

预设均固定到已验证的 Hugging Face 提交号，并使用插件自带的 SemIf 源码。选择 **自定义模型** 时，继续填写现有的模型目录或 Hugging Face ID，以及固定模型版本：

| 配置项 | 用途 |
| --- | --- |
| SemIf 模型目录或 Hugging Face ID | 自定义模式下填写已下载的本地模型目录，或 `owner/repo` |
| SemIf 固定模型版本 | 自定义模式下填写本地模型的来源 revision；远程模型必须使用 40 位提交号 |
| 计算设备和精度 | RTX 2060 使用 `cuda`、`float16`；CPU 使用 `cpu`、`float32` |
| 单题输入 Token 上限 | 默认 4096；超限先省略最早历史，候选仍超限则报告失败 |
| 单次共享评分题数 | 默认 8；每条候选做正反顺序两题，显存紧张可设为 2 |
| 接话概率阈值 | 三种后端共用的 `semantic_threshold`，默认 0.7；请用群聊样本检查并调整 |
| 单批判断超时／模型加载超时 | 默认 30／180 秒；超时不会切换到 Jev |

自定义模型示例：

```text
模型目录              D:\SemIf-OpenJev\models\Qwen3.5-0.8B
固定模型版本          2fc06364715b967f1860aea9cf38778875588b17
计算设备／精度         cuda／float16
```

worker 始终从插件自带目录读取 SemIf 源码。旧的 `semif_python`、`laya_python` 和 `semif_source` 配置已移除，直接传入会报错；请删除这些字段，并将模型依赖安装到 AstrBot 的 Python 环境。预设模型权重只写入 `plugin_data`，不复制进插件源码目录；自定义模型路径也继续可用。配置缺失、模型加载失败或评分失败会明确记录错误，语义判断不回退 Jev；直接唤醒仍可用。

## 判断方式与边界

插件复用 Jev 的群聊历史与待判消息结构，把每条消息转换为两项完整陈述：「目标消息是在对机器人说话，机器人应当接话」和「目标消息不是在对机器人说话，机器人不应因它接话」。SemIf 读取 A/B 选项 logits，不生成自由文本。每条消息分别用两种选项顺序评分，取“接话”概率的平均值，以减轻位置偏置。批内各题优先共用同一群聊状态的模型前缀；若分词后的前缀不一致，则用同一模型逐题评分，并记录日志。

这里的概率是**给定两项之间的条件分数，并未针对本插件群聊唤醒场景校准**。你提供的修正评测报告（`D:\benchmarks\laya_semif_20260924\corrected_01\FINAL_REPORT.md`）在自建事实题上发现 0.8B 模型的选项顺序偏置；那份结果不代表群聊接话准确率。使用时应观察日志中的逐条分数，收集实际群聊误唤醒和漏唤醒样本，再决定阈值与模型。

本机另有 4B 模型；它在独立重测的自建题上更稳定，但报告记录的显存峰值约 10 GiB。若要试用，请先确认它与 AstrBot 的其他模型能共享显存，并从较小的 `semif_batch_rows` 开始。

SemIf 的默认单题输入上限为 4096 Token，通常比 Jev 的请求预算小。worker 会保留全部待判消息，只逐条省略最早历史，并在日志中报告省略数。如果候选本身仍超限，整批判定失败并归档，不截断目标内容，也不静默切换后端。模型进程按插件实例常驻；重载或关闭插件时结束。
