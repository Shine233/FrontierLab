---
type: learning-path
direction: visual-localization
maturity: draft
updated: 2026-06-30
---

# 视觉重定位（Visual Localization）学习路线

## 0. 这个方向在解决什么
结论先行：视觉重定位要回答「这张/这组图是在哪里拍的、相机位姿是多少」。传统做法依赖每个场景单独建图 + 特征匹配 + PnP；当前前沿是「前馈式、map-free / map-light」——用一个泛化网络直接回归相对或绝对位姿，免去每场景训练。主流分支有：相对位姿回归（relative pose regression）、绝对/场景坐标回归（APR/SCR），以及把几何基础模型（VGGT 类）当骨干做重定位。核心权衡是泛化能力、是否需要场景地图、以及 metric 精度。

## 1. 前置知识
- 几何基础：PnP、本质矩阵分解、motion averaging、RANSAC。
- 经典定位流水线：image retrieval + local feature matching + PnP（理解前馈方法替代了哪一步）。
- 上游依赖：图像匹配（见 [image-matching 学习路线](./image-matching.md)）与三维几何骨干（见 [3d-reconstruction 学习路线](./3d-reconstruction.md)）。
- 评测：ScanNet1500、7-Scenes、Cambridge、Wayspots、Map-Free，指标含 AUC@5/10/20、median pose error、mapping/query time。

## 2. 奠基论文（仓库内）
- ★ [Reloc3r: Large-Scale Training of Relative Camera Pose Regression](../papers/visual-localization/2025-reloc3r.md) — 大规模训练的相对位姿回归代表作，无需每场景训练即可泛化，是前馈重定位的事实基线与最易复现的起点。
- [Reloc-VGGT: Visual Re-localization with Geometry Grounded Transformer](../papers/visual-localization/2026-reloc-vggt.md) — 用 VGGT 早期多视图融合的 geometry-grounded transformer 做重定位，代表「几何基础模型即定位骨干」分支（注意代码/权重未公开）。
- [MARePo / Map-Relative Pose Regression for Visual Re-Localization](../papers/visual-localization/2024-marepo.md) — map-relative pose regression，有 scene map 条件下的 metric 定位对照，代表 scene-coordinate-conditioned APR 一脉。

## 3. 近期 SOTA 与分支
- [前馈式视觉重定位方法横向对比](../comparisons/visual-localization/feed-forward-visual-relocalization.md) — 本方向 SOTA 全景主入口，覆盖 relative pose regression、map-relative、VGGT-grounded 等分支与评测协议。
- 分支轴：按「是否需要场景地图」分（map-free 的 Reloc3r/Reloc-VGGT vs map-relative 的 MARePo），按「位姿是相对还是绝对」分。
- 骨干外溢：Reloc-VGGT 直接复用三维几何基础模型，理解它需要先看 [3d-reconstruction 学习路线](./3d-reconstruction.md) 中的 VGGT 类骨干。

## 4. 动手：最小复现路径
- 最小可跑路径：Reloc3r 是最现实的 baseline——加载公开权重，对一对/一组图回归相对位姿，在 ScanNet1500 或 7-Scenes 子集上对 AUC@5/10/20 做 sanity check。
- 复现记录：planned，仓库暂无复现记录。Reloc3r / MARePo inference 复现 planned；Reloc-VGGT 因代码未公开暂 blocked。

## 5. 常见误区 / 我的判断
- 误区：把相对位姿回归当绝对定位用。relative pose 需要参考帧/已知锚点才能落到世界系，map-free 不等于「不需要任何参考」。
- 误区：忽视 metric 尺度。多数前馈方法位姿尺度需对齐，跨方法比 median pose error 前先确认尺度与评测子集一致。
- 判断：起步首选 Reloc3r（可跑、泛化好、易复现）；要 metric/有地图场景看 MARePo；想跟几何基础模型最新进展看 Reloc-VGGT，但要接受其复现 blocked。
