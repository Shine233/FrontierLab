---
type: method-comparison
title: "DINO Family: DINO v1 vs DINOv2 vs DINOv3"
direction: [vision-foundation-models, self-supervised-learning, dense-vision]
methods: [DINO, DINOv2, DINOv3]
status: completed
confidence: high
updated: 2026-05-08
---

# DINO 家族横向对比：v1 / v2 / v3

> 证据范围：本对比基于三篇 arXiv 论文、Hugging Face paper metadata 与 facebookresearch 官方 GitHub 仓库 README/file tree。尚未在本仓库独立复跑训练或 benchmark。

## 结论先行

| 推荐排序 | Method | 推荐场景 | 主要理由 | 主要风险 |
|---:|---|---|---|---|
| 1 | **DINOv3** | 需要最强 dense/high-resolution foundation features：分割、深度、tracking、3D correspondence、遥感 | Gram anchoring 修复大规模长训中的 dense feature degradation；ViT-7B/16 + LVD-1689M/SAT-493M；公开训练/评测代码和多尺寸模型生态 | 自定义 DINOv3 License；旗舰预训练数据私有；完整训练需 256 GPU 级别资源 |
| 2 | **DINOv2** | 通用 frozen visual backbone：分类、检索、分割、深度 baseline；工程可用性/许可证更稳 | LVD-142M curated data；DINO+iBOT+KoLeo recipe；Apache-2.0；训练/评测代码和多尺寸权重成熟 | LVD-142M 不可完整复刻；dense 上限和高分辨率适配弱于 v3 |
| 3 | **DINO v1** | 入门理解自监督 ViT、teacher-student 自蒸馏、attention 语义涌现 | 方法最简洁，能清楚看到 EMA teacher、multi-crop、centering/sharpening 如何防塌缩 | 不再是最强 backbone；代码环境较旧；规模和下游覆盖有限 |

## 1. 先把 DINO 讲透：它到底在学什么？

### 1.1 监督学习 vs 自监督学习

普通监督分类像老师直接告诉你：“这张图是狗”。模型容易学到对分类有用的全局线索，但未必关心狗的边界、腿和背景的关系。

DINO 的自监督训练没有人工标签。它只知道：“这几个裁剪都来自同一张图，语义上应该对齐。”于是模型必须自己发现稳定对象、局部-全局关系和跨增强不变性。

### 1.2 Teacher-student 不是玄学

DINO 的 teacher/student 可以这样理解：

- **student**：当前正在学习，会被梯度更新。
- **teacher**：student 历史版本的移动平均，更稳定。
- **目标**：student 对一个视角的输出，要接近 teacher 对另一个视角的输出。
- **为什么有效**：teacher 像“慢变量”，不被单个 batch 噪声影响；student 追随更稳定的目标。

这和普通知识蒸馏不同：普通蒸馏常有一个已训练好的大 teacher；DINO 的 teacher 是训练过程中动态生成的，没有标签，所以叫 self-distillation with no labels。

### 1.3 `[CLS]` token 和 patch token

ViT 把图像切成 patch，每个 patch 是一个 token；再加一个 `[CLS]` token 用于聚合整张图信息。

| Token | 类比 | 常见用途 |
|---|---|---|
| `[CLS]` token | 读完整张图后的摘要 | 分类、检索、全局 embedding |
| Patch tokens | 图像每个小块的局部理解 | 分割、深度、跟踪、对应、3D、可视化 |

DINO v1 最重要的发现是：自监督 ViT 不只让 `[CLS]` 好用，还让 patch/attention 自然对齐物体区域。DINOv2 把这件事规模化，DINOv3 则专门保护 patch tokens 在超大规模训练中不退化。

### 1.4 什么是 global features，什么是 dense features？

- **Global features**：整张图一个向量。适合“这是什么图？”、“哪张图相似？”
- **Dense features**：每个 patch/像素附近一个向量。适合“哪里是物体？”、“这个点在另一帧对应哪里？”、“每个位置深度是多少？”

DINO 家族的演进主线就是：

```text
DINO v1：发现自监督 ViT 的 global + patch 语义涌现
DINOv2：把它变成可靠的通用 frozen visual backbone
DINOv3：在更大规模下稳住并强化 dense features
```

## 2. 一页总表

| 维度 | DINO v1 | DINOv2 | DINOv3 |
|---|---|---|---|
| 论文 | Emerging Properties in Self-Supervised Vision Transformers | DINOv2: Learning Robust Visual Features without Supervision | DINOv3 |
| 年份 | 2021 | 2023/2024 | 2025 |
| 核心定位 | 发现型 SSL ViT 方法 | 通用自监督视觉 backbone | 大规模 dense vision foundation model |
| 基本范式 | Self-distillation without labels | DINO + iBOT + KoLeo + curated data + distillation | DINOv2-style scaling + Gram anchoring + high-res adaptation + distillation |
| 主要输入 | 无标签 ImageNet 图像 | 大规模 curated 图像 LVD-142M | LVD-1689M web 图像 + SAT-493M 卫星图像等 |
| 代表模型 | ViT-S/8、ViT-S/16、ViT-B | ViT-S/B/L/g/14 | ViT-S/B/L/H+/7B/16、ConvNeXt、satellite variants |
| 模型规模重点 | 证明 ViT 小 patch 有语义 | ViT-g/14 teacher 约 1B 级，蒸馏小模型 | ViT-7B/16 约 6.7B，蒸馏多尺寸模型 |
| 关键创新 | EMA teacher、multi-crop、centering/sharpening；发现 attention object masks | LVD-142M 数据 pipeline；DINO+iBOT+KoLeo；大模型训练和蒸馏 | Gram anchoring 保护 patch 相似结构；高分辨率适配；多学生蒸馏 |
| 主要强项 | 教学清晰、机制可解释、attention 语义涌现 | 工程可用、Apache-2.0、通用 frozen features 强 | Dense/high-res/多任务上限高，覆盖遥感/ConvNeXt/text alignment |
| 主要短板 | 规模小、现代任务覆盖有限 | 数据不可完整复刻；dense scaling 问题未彻底解决 | License/数据/算力门槛更高 |
| Git 地址 | <https://github.com/facebookresearch/dino> | <https://github.com/facebookresearch/dinov2> | <https://github.com/facebookresearch/dinov3> |
| 是否开源 | 是（Apache-2.0） | 是（Apache-2.0） | 是（DINOv3 License，需审查） |
| 是否开源训练 | 是 | 是（训练代码开源；LVD-142M 不可完整复刻） | 是（训练代码/配置开源；旗舰私有数据不可复刻） |
| 预训练权重 | 官方下载/torch hub | 官方下载/torch hub/HF 生态 | Meta 下载/HF collection/Transformers/timm，部分需申请 |
| 训练可复现性 | 中：ImageNet + 多 GPU 可做 | 中低：小/公开数据可复现，大 teacher 不现实 | 低：小实验可做，7B exact setup 不现实 |
| 推荐用途 | 学原理、做轻量 SSL 复现实验 | 默认视觉 backbone baseline | 高上限 dense backbone / 遥感 / high-res vision |

## 3. 版本逐个分析

### 3.1 DINO v1：为什么它是“发现”而不是单纯刷榜？

DINO v1 的价值不只是 ImageNet linear eval 数字，而是它揭示了一个现象：**当 ViT 用无标签自蒸馏训练时，注意力头会自然学到对象区域。**

它的训练非常适合教学：

1. 同一张图做不同裁剪。
2. teacher 看 global crop，student 看 global + local crops。
3. student 预测 teacher 的 soft distribution。
4. teacher 用 EMA 更新。
5. centering + sharpening 防止所有样本输出一样。

这一套解释了现代很多视觉 SSL 的底层思想：不是硬背标签，而是在增强视角之间保持语义一致。

### 3.2 DINOv2：为什么它是“工程化基础模型”？

DINOv2 的重点不是发明一个完全不同的 loss，而是回答“怎么把 DINO 训练到大规模还稳定？”

关键在三件事：

- **数据**：LVD-142M curated dataset，说明 SSL 也需要数据质量和覆盖平衡。
- **目标**：DINO 管全局，iBOT 管 patch，KoLeo 管特征分布。
- **部署**：训练大 teacher 后蒸馏出小模型，让社区能真正使用。

如果今天要在项目里选一个默认 frozen visual backbone，DINOv2 仍然很实用：许可证清晰、生态成熟、模型尺寸多、示例多。

### 3.3 DINOv3：为什么需要 Gram anchoring？

DINOv3 发现：继续长时间训练大模型时，global 分类能力可能越来越强，但 dense tasks 反而下降。这说明“整图变强”和“局部变强”不是同一件事。

Gram anchoring 的关键洞察是：dense features 的质量很大程度体现在 patch 之间的关系上。比如同一物体内部 patch 应相似，边界两侧应有区分。直接约束每个 feature vector 可能太死，但约束 Gram matrix 可以保留这种结构。

因此 DINOv3 不是简单的“DINOv2 + 更多数据 + 更大模型”，而是加了一个专门为 dense feature scaling 设计的稳定器。

## 4. 场景化选择

| 场景 | 推荐 | 原因 |
|---|---|---|
| 初学者想真正理解自监督 ViT | DINO v1 | 方法最小闭环清楚，teacher/student、multi-crop、防塌缩都能看懂 |
| 需要一个稳妥 baseline backbone | DINOv2 | Apache-2.0、生态成熟、模型尺寸多、frozen feature 泛化好 |
| 分割/深度/跟踪/3D correspondence | DINOv3 | Gram anchoring 和 high-res adaptation 明确服务 dense features |
| 需要文本 zero-shot / 开放词表 | CLIP/SigLIP 或 DINOv3 dino.txt | 原始 DINOv2/v3 是纯视觉 SSL；文本能力需要额外 alignment |
| 算力有限做复现实验 | DINO v1 或 DINOv2 small config | 7B/大数据训练不现实；先复现 loss/attention/k-NN 更有学习收益 |
| 商业/产品集成 | 优先 DINOv2，谨慎 DINOv3 | DINOv2 Apache-2.0 更清晰；DINOv3 自定义 license 需法务审查 |
| 遥感/森林高度/卫星方向 | DINOv3 satellite / CHMv2 | DINOv3 提供 SAT-493M 和 CHMv2 生态 |

## 5. 初学者学习路线

### 第 1 步：先懂 ViT token

- 图像被切成 patch。
- 每个 patch 变成 token。
- Transformer 让每个 token 看其他 token。
- `[CLS]` token 汇总全图。

理解这一点后，就能分清：分类看 `[CLS]`，分割/深度看 patch tokens。

### 第 2 步：再懂自监督目标

不要先陷入公式，先问：没有标签时，模型如何知道自己学对了？

DINO 的回答是：同一张图的不同视角应该有一致语义。teacher 给稳定目标，student 去追。

### 第 3 步：理解防塌缩

所有 SSL 方法都在防止“输出全一样”。DINO v1 用 centering/sharpening + EMA teacher；DINOv2 加更强的 centering/regularization；DINOv3 在 dense 层面防止 patch 结构退化。

### 第 4 步：理解 scaling 的代价

规模化不是只加 GPU：

- 数据要去重、清洗、平衡；
- loss 要兼顾 global 和 local；
- 训练要能分布式稳定；
- 大模型要蒸馏成可用小模型；
- dense feature 还要额外保护。

这就是 v1 -> v2 -> v3 的真正演进。

## 6. 复现优先级

1. **DINO v1 教学复现**：用官方权重跑 attention visualization、k-NN；再用小数据验证 training 不塌缩。
2. **DINOv2 baseline 复现**：加载 `dinov2_vits14` / `dinov2_vitb14` 抽 frozen features，跑一个线性分类或简单分割/深度 probe。
3. **DINOv3 dense probe**：加载小模型或 ConvNeXt tiny，做 patch PCA 可视化、ADE20K/NYUv2 线性 probe，和 DINOv2 比较边界/局部一致性。
4. **不要优先从零训练 DINOv3 7B**：除非研究目标就是大规模 SSL 系统，否则投入产出比极低。

## 7. 已确认事实、推断与待验证

### 已确认事实

- 三个官方 GitHub 仓库均公开。
- DINO v1 与 DINOv2 仓库包含训练代码，许可证为 Apache-2.0。
- DINOv3 仓库包含训练/评测/蒸馏代码，许可证为 DINOv3 License。
- DINOv2 论文使用 LVD-142M；DINOv3 论文使用 LVD-1689M 和 SAT-493M，并提出 Gram anchoring。

### 推断

- 对普通研究/工程项目，DINOv2 是默认性价比最高的 baseline；DINOv3 是高上限 baseline。
- DINOv3 的 license 和权重获取流程会影响商业落地速度。
- DINO v1 最适合做教学和机制验证，而不是作为最新 SOTA backbone。

### 待验证

- 在本仓库目标任务（如 3D reconstruction / robotics / autonomous driving）中，DINOv3 patch features 相比 DINOv2 对几何任务的实际收益。
- DINOv3 小模型（ViT-S/B/L、ConvNeXt）在本地 GPU 上的速度、显存和 batch size。
- DINOv2 / DINOv3 作为 geometry backbone 接入 MapAnything/LingBot-Map 类任务的成本。

## Sources

- DINO paper: <https://arxiv.org/abs/2104.14294>
- DINO GitHub: <https://github.com/facebookresearch/dino>
- DINOv2 paper: <https://arxiv.org/abs/2304.07193>
- DINOv2 GitHub: <https://github.com/facebookresearch/dinov2>
- DINOv3 paper: <https://arxiv.org/abs/2508.10104>
- DINOv3 GitHub: <https://github.com/facebookresearch/dinov3>
- DINOv3 project page: <https://ai.meta.com/dinov3/>
