---
type: learning-path
direction: image-matching
maturity: draft
updated: 2026-06-30
---

# 图像匹配（Image Matching）学习路线

## 0. 这个方向在解决什么
结论先行：图像匹配要解决的是「给定两张（或多张）有重叠的图，找到像素/特征之间的对应关系」，它是 SfM、视觉定位、三维重建、SLAM 的公共前端。核心矛盾在两条路线之间：稀疏关键点匹配（detector + descriptor + matcher，快、可控、易接 PnP/几何求解）与稠密匹配（detector-free，鲁棒、跨域强、但算力贵）。近几年的进展集中在：用冻结的视觉基础模型特征（DINOv2/DINOv3）做骨干、coarse-to-fine 解码、把匹配建模成回归 + 不确定性预测，以及用互联网视频做自训练以换取零样本泛化。

## 1. 前置知识
- 多视图几何基础：对极几何、基础/本质矩阵、PnP、RANSAC（推荐 Hartley & Zisserman《Multiple View Geometry》前 9 章）。
- 经典局部特征：SIFT、ORB，以及学习型 SuperPoint + SuperGlue（理解 detector-based 流水线的瓶颈）。
- 评测协议：MegaDepth-1500 / ScanNet-1500 的相对位姿 AUC@5/10/20、ZEB 零样本 Mean AUC，理解为什么不同论文的数字常不可直接横比。
- 视觉基础模型：DINOv2/DINOv3 的 frozen feature 概念（见本仓库 [vision-foundation-models 学习路线](./vision-foundation-models.md)）。

## 2. 奠基论文（仓库内）
- ★ [RoMa: Robust Dense Feature Matching](../papers/image-matching/2024-roma.md) — 稠密匹配的代表作，确立了「DINOv2 冻结粗特征 + 专用精特征 + 回归 match decoder + 预测协方差」的范式，是后续 detector-free 稠密匹配的事实基线。
- ★ [LightGlue: Local Feature Matching at Light Speed](../papers/image-matching/2023-lightglue.md) — 稀疏匹配一侧的奠基/转折点，用自适应深度与宽度（早停、剪枝）把 SuperGlue 的注意力匹配做到实时，确立了「轻量、可部署的稀疏 matcher」标杆。
- [GIM: Learning Generalizable Image Matcher From Internet Videos](../papers/image-matching/2024-gim.md) — 提出用互联网视频 + 自训练/标签传播获得跨域泛化的 matcher-agnostic 框架，是「数据驱动泛化」分支的代表。

## 3. 近期 SOTA 与分支
- [图像匹配方法横向对比：RoMa / RoMa v2 / LightGlue / GIM / LoMa](../comparisons/image-matching/image-matching-methods.md) — 一份报告覆盖本方向全部五篇，建议作为 SOTA 全景的主入口。重点看第 3 节标注的「协议一致 vs 不可跨论文比较」。
- 稠密分支演进：RoMa → [RoMa v2: Harder Better Faster Denser Feature Matching](../papers/image-matching/2025-romav2.md)（更快更密，注意其训练代码未在 README 提供）。
- 泛化/数据分支：[GIM](../papers/image-matching/2024-gim.md) 的自训练，以及 [LoMa: Local Feature Matching Revisited](../papers/image-matching/2026-loma.md)（训练代码与 HardMatch 数据未公开）。
- 稀疏分支：[LightGlue](../papers/image-matching/2023-lightglue.md) 仍是部署侧首选，可与 glue-factory 训练栈搭配。

## 4. 动手：最小复现路径
- 最小可跑路径：先做 inference sanity check 而非从头训练。RoMa（PyPI `romatch`，自动下载 `roma_outdoor`/`roma_indoor`，含 XFeat 轻量版 `tiny_roma_v1`）与 LightGlue（自带预训练权重）都开箱即用，跑一对 MegaDepth 图像出匹配点 + 估计相对位姿即可。
- 复现记录：planned，仓库暂无复现记录。可先复现 RoMa / LightGlue 推理，再尝试 GIM（+glue-factory）训练；RoMa v2 / LoMa 训练因代码或数据未公开暂 blocked，只做推理。

## 5. 常见误区 / 我的判断
- 误区：直接横比不同论文的 AUC 数字。各家在分辨率、RANSAC 阈值、关键点上限、评测子集上不一致，跨论文比较常无意义；以对比报告第 3 节标出的「协议一致」项为准。
- 误区：以为稠密一定强于稀疏。稠密匹配（RoMa 系）在低重叠/跨域更鲁棒，但算力与延迟高；实时定位/SLAM 前端 LightGlue 这类稀疏 matcher 仍是更现实的选择。
- 判断：当前性价比最高的起点是「DINOv2/v3 冻结特征 + coarse-to-fine」这条线；要可部署选 LightGlue，要鲁棒/跨域选 RoMa，要泛化数据驱动看 GIM。先把推理 + 几何求解链路跑通，再谈训练复现。
