# FrontierLab：前沿 Paper 阅读、方法对比与复现实验库

> 仓库定位：沉淀前沿论文的阅读分析、方法谱系、横向对比、复现实验记录与测试结论。  
> 目标：以后可以按**方向 / 方法族 / 数据集 / 任务 / 复现状态**快速索引已经分析过的论文，并把相似方向的方法放到同一张对比表里持续更新。

## 快速入口

| 入口 | 用途 |
|---|---|
| [`indices/directions.md`](indices/directions.md) | 按研究方向查找论文、方法、对比报告与复现记录 |
| [`indices/papers.md`](indices/papers.md) | 论文清单：状态、方向、Git 地址、是否开源、是否开源训练、复现进度 |
| [`indices/methods.md`](indices/methods.md) | 方法族索引：同类方法、关键差异、适用场景 |
| [`indices/comparisons.md`](indices/comparisons.md) | 横向对比报告索引 |
| [`papers/_template.md`](papers/_template.md) | 单篇论文分析模板 |
| [`comparisons/_template.md`](comparisons/_template.md) | 多方法对比模板 |
| [`reproductions/_template.md`](reproductions/_template.md) | 复现实验记录模板 |
| [`reports/`](reports/) | 已有或历史分析报告 |
| [`indices/timeline.md`](indices/timeline.md) | 时间线：按年看方向演进与划时代方法（自动生成） |
| [`indices/landmarks.md`](indices/landmarks.md) | 划时代方法及其谱系（自动生成） |
| [`dashboards/`](dashboards/) | Obsidian Dataview 动态视图 |
| [`learning/`](learning/) | 各方向由浅入深学习路线 |
| [`KB-MEMORY.md`](KB-MEMORY.md) | 知识库跨会话记忆：约定、缺口、待办、决策 |

## 推荐目录结构

```text
FrontierLab/
├── README.md
├── indices/                 # 总索引层：按方向、论文、方法族、对比报告查找
│   ├── directions.md
│   ├── papers.md
│   ├── methods.md
│   └── comparisons.md
├── papers/                  # 单篇 paper 阅读分析；按方向分目录
│   ├── _template.md
│   └── <direction>/
│       └── <year>-<paper-slug>.md
├── comparisons/             # 多篇/多方法横向对比；按方向或主题分目录
│   ├── _template.md
│   └── <direction>/
│       └── <topic>.md
├── reproductions/           # 复现计划、环境、运行结果、失败原因
│   ├── _template.md
│   └── <direction>/<method>/
│       ├── README.md
│       └── runs/
│           └── <yyyy-mm-dd>-<experiment>.md
├── tests/                   # 轻量验证脚本、benchmark、sanity check 说明
├── datasets/                # 数据集说明、获取方式、许可与预处理记录；不提交大数据
├── assets/                  # 图、表、截图、小型可视化资源
├── notes/                   # 临时想法、会议/阅读札记，沉淀后迁移到 papers/comparisons
└── reports/                 # 既有报告或长文分析归档
```

> 新增/修改论文后运行 `python3 scripts/build_indices.py` 重建索引（timeline、landmarks 等自动生成内容）。

## 命名规范

### 方向 slug

用短横线小写命名，保持稳定，例如：

- `3d-reconstruction`
- `vision-foundation-models`
- `self-supervised-learning`
- `dense-vision`
- `world-models`
- `multimodal-learning`
- `robotics-autonomous-driving`
- `visual-localization`
- `image-matching`
- `generation-diffusion`
- `reasoning-agents`
- `efficient-training-inference`
- `evaluation-benchmarks`

### 单篇论文文件

```text
papers/<direction>/<year>-<short-title>.md
```

示例：

```text
papers/3d-reconstruction/2025-map-anything.md
papers/world-models/2026-example-world-model.md
```

### 对比报告文件

```text
comparisons/<direction>/<comparison-topic>.md
```

示例：

```text
comparisons/3d-reconstruction/feedforward-3d-reconstruction.md
comparisons/reasoning-agents/test-time-scaling-methods.md
```

### 复现实验文件

```text
reproductions/<direction>/<method>/README.md
reproductions/<direction>/<method>/runs/<yyyy-mm-dd>-<experiment>.md
```

## 每篇论文建议保留的元数据

每个 `papers/<direction>/*.md` 顶部使用 YAML frontmatter，便于搜索和后续自动化：

```yaml
---
type: paper-analysis
title: ""
year:
venue: ""
arxiv: ""
code: ""
github: ""
open_source: unknown      # true | false | unknown
license: ""
training_open_source: "unknown"  # true | false | unknown | "\\": 公开仓库主要是推理/demo/模型代码，未提供训练代码
direction: []
method_family: []
tasks: []
datasets: []
metrics: []
status: triage        # triage | reading | analyzed | compared | reproducing | reproduced | archived
reproduction: none    # none | planned | running | blocked | reproduced
confidence: medium    # low | medium | high
landmark: false          # true = 划时代/奠基方法，时间线与 landmarks 索引加 ★
org: []                  # 机构/团队，如 [Meta, ByteDance]
key_idea: ""             # 一句话核心，用于索引表与悬浮预览
supersedes: []           # 取代/改进的前作 slug，如 [2024-roma]
builds_on: []            # 依赖但不取代的前作 slug，如 [2023-dinov2]
updated: 2026-05-08
---
```

## 建议工作流

1. **新论文进入**：复制 [`papers/_template.md`](papers/_template.md) 到 `papers/<direction>/<year>-<slug>.md`。
2. **完成初读**：补齐“核心问题、方法、贡献、局限、与已有方法关系、可复现性”。
3. **更新索引**：在 [`indices/papers.md`](indices/papers.md) 增加一行；必要时更新 [`indices/directions.md`](indices/directions.md) 和 [`indices/methods.md`](indices/methods.md)。
4. **同方向方法超过 2 个**：创建或更新 `comparisons/<direction>/<topic>.md`，把方法放进统一维度表。
5. **准备复现**：复制 [`reproductions/_template.md`](reproductions/_template.md)，记录环境、commit、数据、指标、失败点。
6. **有测试或 benchmark**：把脚本/说明放到 `tests/<direction>/<method>/`；大数据、权重、缓存不要提交到仓库。
7. **重建索引**：新增或修改论文后运行 `python3 scripts/build_indices.py`，重新生成 timeline、landmarks 等自动索引。

## 检索方式

推荐优先用 `rg`：

```bash
# 查方向
rg "3d-reconstruction|world-models" indices papers comparisons reproductions reports

# 查某个方法或作者
rg "MapAnything|VGGT|Hunyuan" .

# 查复现状态
rg "reproduction: (planned|running|blocked|reproduced)" papers reproductions

# 查某类方法族
rg "method_family:|feed-forward|3DGS|test-time" papers comparisons
```

没有 `rg` 时可用 `grep -R` 替代。

## 维护原则

- **索引优先**：新增 paper / comparison / reproduction 后，同步更新 `indices/`。
- **代码状态显式记录**：`indices/papers.md` 必须维护 Git 地址、是否开源、是否开源训练；公开仓库若只是推理/demo/模型代码而非训练代码，在“是否开源训练”列写 `\`。
- **结论先行**：每份分析开头给出 3-5 条可复用结论，再展开证据。
- **证据与推断分开**：明确哪些来自论文/代码/实验，哪些是个人判断。
- **复现诚实记录**：失败、环境不一致、指标无法对齐都要保留。
- **对比维度稳定**：同方向对比尽量复用相同维度，方便持续追加新方法。
- **不提交大文件**：数据集、模型权重、缓存只记录路径、来源、hash、许可证。

## 当前已有资料

- [`papers/vision-foundation-models/2021-dino.md`](papers/vision-foundation-models/2021-dino.md)：DINO v1 / Emerging Properties in Self-Supervised Vision Transformers 论文分析。
- [`papers/vision-foundation-models/2023-dinov2.md`](papers/vision-foundation-models/2023-dinov2.md)：DINOv2 / Learning Robust Visual Features without Supervision 论文分析。
- [`papers/vision-foundation-models/2025-dinov3.md`](papers/vision-foundation-models/2025-dinov3.md)：DINOv3 论文分析。
- [`comparisons/vision-foundation-models/dino-family.md`](comparisons/vision-foundation-models/dino-family.md)：DINO 家族横向对比与初学者教学：v1 / v2 / v3。
- [`papers/3d-reconstruction/2026-lingbot-map.md`](papers/3d-reconstruction/2026-lingbot-map.md)：LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction 论文分析。
- [`papers/3d-reconstruction/2025-depth-anything-3.md`](papers/3d-reconstruction/2025-depth-anything-3.md)：Depth Anything 3 / Recovering the Visual Space from Any Views 论文分析。
- [`papers/3d-reconstruction/2026-vggt-omega.md`](papers/3d-reconstruction/2026-vggt-omega.md)：VGGT-Ω 论文分析：VGGT scaling、register attention、动态视频重建与空间 token 表征。
- [`papers/3d-reconstruction/2026-pi3.md`](papers/3d-reconstruction/2026-pi3.md)：Pi3 / π³: Permutation-Equivariant Visual Geometry Learning 论文分析。
- [`papers/world-models/2026-xiaomi-auto-world-model.md`](papers/world-models/2026-xiaomi-auto-world-model.md)：Xiaomi Auto World Model / JointWM 论文分析：WorldRec 稀疏 3D query 前馈 3DGS、WorldGen 因果 DiT、重建-生成联合自动驾驶世界模型。
- [`papers/world-models/2026-x-world.md`](papers/world-models/2026-x-world.md)：X-World 论文分析：XPeng 可控自车中心 7 相机世界模型，面向闭环评测、online RL、场景编辑与数据合成。
- [`papers/efficient-training-inference/2026-x-cache.md`](papers/efficient-training-inference/2026-x-cache.md)：X-Cache 论文分析：few-step AR 世界模型跨 chunk block residual caching，报告约 71% skip 与 2.6-2.7x DiT 加速。
- [`papers/world-models/2026-x-foresight.md`](papers/world-models/2026-x-foresight.md)：X-Foresight 论文分析：预测式世界模型接入 VLA/LDM，chunk-wise 长 horizon 未来视觉与动作预测。
- [`papers/world-models/2026-dreamzero.md`](papers/world-models/2026-dreamzero.md)：DreamZero 论文分析：World Action Model / video-diffusion robot policy，联合预测未来视频和动作，公开 DROID/AgiBot 权重、DROID 数据与训练脚本。
- [`comparisons/world-models/xpeng-x-series-world-models.md`](comparisons/world-models/xpeng-x-series-world-models.md)：XPeng X 系列自动驾驶世界模型横向对比：X-World / X-Cache / X-Foresight。
- [`comparisons/world-models/robot-world-action-models.md`](comparisons/world-models/robot-world-action-models.md)：Robot WAM / video-diffusion robot policies 横向对比：DreamZero / GR00T N1.x / pi_0 / V-JEPA 2 / Genie / X-Foresight。
- [`papers/visual-localization/2026-reloc-vggt.md`](papers/visual-localization/2026-reloc-vggt.md)：Reloc-VGGT 论文分析：VGGT early multi-view fusion、source pose token、sparse mask attention；当前代码未发布。
- [`papers/visual-localization/2025-reloc3r.md`](papers/visual-localization/2025-reloc3r.md)：Reloc3r 论文分析：scene-agnostic relative pose regression、motion averaging、视觉重定位可跑 baseline。
- [`papers/visual-localization/2024-marepo.md`](papers/visual-localization/2024-marepo.md)：MARePo 论文分析：map-relative pose regression、ACE scene coordinate map 条件化 APR/SCR 混合路线。
- [`comparisons/3d-reconstruction/streaming-3d-reconstruction.md`](comparisons/3d-reconstruction/streaming-3d-reconstruction.md)：Streaming 3D Reconstruction 方法对比：LingBot-Map / Stream3R / Wint3R / TTT3R / CUT3R 等。
- [`comparisons/3d-reconstruction/visual-geometry-foundation-models.md`](comparisons/3d-reconstruction/visual-geometry-foundation-models.md)：Any-view Visual Geometry Foundation Models 横向对比：DA3 / VGGT-Ω / VGGT / Pi3 / MapAnything / LingBot-Map。
- [`comparisons/visual-localization/feed-forward-visual-relocalization.md`](comparisons/visual-localization/feed-forward-visual-relocalization.md)：前馈式视觉重定位方法对比：Reloc-VGGT / Reloc3r / MARePo / FastForward / ACE / MASt3R。
- [`reports/feedforward_3d_reconstruction_compare.md`](reports/feedforward_3d_reconstruction_compare.md)：前馈式三维重建方法对比：MapAnything / Pi3 / HunyuanWorld-Mirror / OmniVGGT。
- [`papers/image-matching/2024-roma.md`](papers/image-matching/2024-roma.md)：RoMa / Robust Dense Feature Matching 论文分析：DINOv2 coarse + ConvNet fine 的稠密 coarse-to-fine 匹配。
- [`papers/image-matching/2025-romav2.md`](papers/image-matching/2025-romav2.md)：RoMa v2 论文分析：DINOv3 + Attention coarse matcher、两阶段训练、预测协方差、困难/跨域稠密匹配 SOTA。
- [`papers/image-matching/2023-lightglue.md`](papers/image-matching/2023-lightglue.md)：LightGlue 论文分析：SuperGlue 后继，adaptive depth/width 稀疏匹配，速度大幅领先。
- [`papers/image-matching/2024-gim.md`](papers/image-matching/2024-gim.md)：GIM 论文分析：从互联网视频自训练的可泛化匹配框架（架构无关），提出 ZEB 零样本基准。
- [`papers/image-matching/2026-loma.md`](papers/image-matching/2026-loma.md)：LoMa 论文分析：LightGlue 式稀疏匹配 + 数据/模型 scaling；训练代码与 HardMatch 数据未公开。
- [`comparisons/image-matching/image-matching-methods.md`](comparisons/image-matching/image-matching-methods.md)：图像匹配方法横向对比：RoMa / RoMa v2 / LightGlue / GIM / LoMa，重点标出相同数据集/指标的可比结果。
