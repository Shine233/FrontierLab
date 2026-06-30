---
type: learning-path
direction: vision-foundation-models
maturity: draft
updated: 2026-06-30
---

# 视觉基础模型（Vision Foundation Models）学习路线

## 0. 这个方向在解决什么
结论先行：视觉基础模型要解决的是「训练一个无需任务标签、可被下游任意复用的通用视觉骨干」。一旦有了强 frozen backbone，分类、检测、分割、深度、匹配、三维重建都能从同一组特征出发，省掉为每个任务单独标注与从零训练。DINO 系列是这条路线最有代表性的实现：用自蒸馏（无标签）学到既有全局语义又有 dense patch 结构的表征，并随数据与模型规模放大持续变强。

## 1. 前置知识
- ViT 基础：patch embedding、self-attention、[CLS] token 与 patch token 的角色。
- 对比学习 / 自监督脉络：MoCo、SimCLR、BYOL（理解「无负样本如何防塌缩」的动机）。
- 评测方式：linear probing、k-NN、attention map 可视化、dense 任务（分割/深度）的 frozen-feature probe。
- 相关方向：自蒸馏机制细节见本仓库 [self-supervised-learning 学习路线](./self-supervised-learning.md)。

## 2. 奠基论文（仓库内）
- ★ [DINO / Emerging Properties in Self-Supervised Vision Transformers](../papers/vision-foundation-models/2021-dino.md) — 奠基之作。提出 self-distillation without labels（teacher-student + centering/sharpening 防塌缩），并首次展示自监督 ViT 自发涌现的语义分割注意力图，是整个 DINO 家族的起点。
- [DINOv2: Learning Robust Visual Features without Supervision](../papers/vision-foundation-models/2023-dinov2.md) — 把 DINO 路线工程化、规模化（curated-data scaling + 蒸馏），产出可直接当通用 backbone 的 frozen features，被 RoMa、众多三维/匹配方法当骨干使用。
- [DINOv3](../papers/vision-foundation-models/2025-dinov3.md) — 进一步放大并用 Gram anchoring 稳定 dense feature，强调高分辨率稠密表征的可扩展性。

## 3. 近期 SOTA 与分支
- [DINO 家族横向对比：v1 / v2 / v3](../comparisons/vision-foundation-models/dino-family.md) — 三代演进的主入口：v1 重机制、v2 重通用 backbone、v3 重 dense feature scaling。
- 应用外溢分支：VGGT-Ω 把「reconstruction as spatial pretraining」与 register scene tokens 引入几何骨干（见 [3d-reconstruction 学习路线](./3d-reconstruction.md)），可作为基础模型向三维延伸的观察点。
- 下游耦合：图像匹配（RoMa/RoMa v2 用 DINOv2/v3 冻结特征）是基础模型价值最直接的体现，见 [image-matching 学习路线](./image-matching.md)。

## 4. 动手：最小复现路径
- 最小可跑路径：加载 DINOv2 预训练权重，对一张图抽 patch features，做 PCA 可视化 + 一个 linear probe（如分类或分割）。不建议从零复刻大规模预训练。
- 复现记录：planned，仓库暂无复现记录。优先 inference / feature extraction / linear probe / spatial-token probe；大规模预训练复现成本高，暂不优先。

## 5. 常见误区 / 我的判断
- 误区：把 DINO 当成「另一个分类模型」。它的价值在 frozen feature 的可迁移性与 dense 结构，而非某个 benchmark 上的 top-1。
- 误区：默认更高版本一定更适合你的任务。v1 适合教学/理解机制，v2 是稳妥的通用 backbone，v3 才在高分辨率 dense 任务上有明显优势。
- 判断：做下游应用直接用 DINOv2 frozen feature 起步，性价比最高；要 dense/高分辨率再上 DINOv3；要理解「为什么自监督能涌现语义」回到 v1 读机制。
