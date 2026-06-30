---
type: learning-path
direction: self-supervised-learning
maturity: draft
updated: 2026-06-30
---

# 自监督学习（Self-Supervised Learning）学习路线

## 0. 这个方向在解决什么
结论先行：自监督学习要解决的是「在没有人工标签的情况下，从海量原始数据中学到可迁移的表征」。视觉一侧的核心难题是防塌缩（collapse）——没有标签约束时，模型容易把所有输入映射到同一个点。DINO 系列给出的答案是自蒸馏（self-distillation）：用一个动量 teacher 产生目标、student 去匹配，再配合 centering + sharpening 把表征撑开。这条线把 SSL 从「对比学习需要大量负样本」推进到「无需显式负样本也能稳定训练」。

## 1. 前置知识
- 对比学习脉络：InfoNCE、MoCo（动量编码器 + 队列）、SimCLR（大 batch + 负样本）。
- 无负样本方法：BYOL、SimSiam（stop-gradient 如何防塌缩），以及 Barlow Twins / VICReg 的去冗余/方差正则视角。
- 掩码重建一脉：MAE / masked image modeling，作为与自蒸馏并行的另一条 SSL 主线。
- 评测：linear probe、k-NN、迁移到 dense 任务时的表现。

## 2. 奠基论文（仓库内）
- ★ [DINO / Emerging Properties in Self-Supervised Vision Transformers](../papers/vision-foundation-models/2021-dino.md) — 从 SSL 视角看，这是「self-distillation without labels」的奠基工作：teacher-student 架构 + centering/sharpening 的防塌缩机制，是无标签表征学习的关键范式。
- [DINOv2: Learning Robust Visual Features without Supervision](../papers/vision-foundation-models/2023-dinov2.md) — 展示 SSL 在 curated-data scaling 下能产出真正通用、可冻结复用的表征，把 SSL 从「学术指标」推向「生产可用 backbone」。
- [DINOv3](../papers/vision-foundation-models/2025-dinov3.md) — 用 Gram anchoring 解决规模放大时 dense feature 退化的问题，是 SSL recipe 在更大尺度上的稳定性研究。

## 3. 近期 SOTA 与分支
- [DINO 家族横向对比：v1 / v2 / v3](../comparisons/vision-foundation-models/dino-family.md) — 从无标签表征/自蒸馏/防塌缩的视角串起三代：v1 给机制、v2 给规模化 recipe、v3 给 dense feature 的 collapse 控制。
- 视角提示：读这条线时关注「目标信号从哪来（teacher/EMA、掩码、聚类）」与「塌缩怎么防（centering、stop-gradient、方差/去冗余正则、Gram anchoring）」两个轴，便于把 DINO 与 MAE/VICReg 等横向归类。
- 通用 backbone 视角与下游应用见 [vision-foundation-models 学习路线](./vision-foundation-models.md)。

## 4. 动手：最小复现路径
- 最小可跑路径：在小数据集（如 CIFAR/ImageNet 子集）上跑一个最小自蒸馏循环，重点观察「关闭 centering/sharpening 会不会塌缩」，再对学到的特征做 k-NN/linear probe。
- 复现记录：planned，仓库暂无复现记录。建议先复现 DINO v1 的小规模训练以理解防塌缩机制，再用 DINOv2 权重做 probe；大规模预训练暂不优先。

## 5. 常见误区 / 我的判断
- 误区：以为「无负样本」就没有防塌缩约束。BYOL/SimSiam 靠 stop-gradient + predictor，DINO 靠 centering/sharpening，约束只是换了形式。
- 误区：把 SSL 当作只为分类预训练。它最大的价值在 dense/几何下游（匹配、分割、三维），这些任务很难标注。
- 判断：想学机制就从 DINO v1 的自蒸馏 + 防塌缩入手，可控且直观；想要可用表征直接拿 DINOv2；研究规模化稳定性再看 v3 的 Gram anchoring。
