---
type: learning-path
direction: 3d-reconstruction
maturity: draft
updated: 2026-06-30
---

# 三维重建（3D Reconstruction）学习路线

## 0. 这个方向在解决什么
结论先行：当前三维重建最活跃的子方向是「前馈式视觉几何基础模型」——给若干张（甚至无序、任意视角、视频流）图像，用一个网络一次前馈直接吐出相机位姿、深度/点图、乃至 metric 尺度的几何，取代传统 SfM/MVS 的迭代优化管线。它的吸引力在于：快、对低纹理/弱重叠鲁棒、可端到端接入定位与机器人/自动驾驶建图。代价是 metric 尺度、动态场景、长序列在线建图仍是开放难题。

## 1. 前置知识
- 经典管线：SfM（COLMAP）、MVS、bundle adjustment，理解前馈方法在替代/补足哪一段。
- 表征基础：depth map、point map、camera ray、3D Gaussian Splatting 初始化的概念。
- 骨干：DINOv2/v3 frozen features 与 ViT，见 [vision-foundation-models 学习路线](./vision-foundation-models.md)。
- 评测协议：相机位姿、深度、point map reconstruction 在 7Scenes/ETH3D/DTU/Sintel 等数据集上的指标差异。

## 2. 奠基论文（仓库内）
- ★ [Depth Anything 3: Recovering the Visual Space from Any Views](../papers/3d-reconstruction/2025-depth-anything-3.md) — 任意视角视觉几何的代表作，输出 depth/ray/geometry 并配 DA3-BENCH，是「any-view geometry backbone」的奠基级参考。
- ★ [VGGT-Ω](../papers/3d-reconstruction/2026-vggt-omega.md) — 把 reconstruction 当作 spatial pretraining，引入 register scene tokens，是 RGB-only/video geometry backbone 与空间表征的关键候选。
- [Pi3 / π³: Permutation-Equivariant Visual Geometry Learning](../papers/3d-reconstruction/2026-pi3.md) — 置换等变、reference-free 的无序多视图几何学习，是「不依赖参考帧的鲁棒 baseline」。
- [LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction](../papers/3d-reconstruction/2026-lingbot-map.md) — 面向连续视频的在线流式建图/VO，是 streaming reconstruction 分支代表。

## 3. 近期 SOTA 与分支
- [Any-view Visual Geometry Foundation Models 横向对比](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md) — 任意视角几何基础模型的全景对比，建议作为主入口。
- [Streaming 3D Reconstruction 方法横向对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) — 在线/流式建图分支（含 LingBot-Map）的横向梳理。
- [前馈式三维重建方法对比：MapAnything / Pi3 / HunyuanWorld-Mirror / OmniVGGT](../reports/feedforward_3d_reconstruction_compare.md) — 跨方法专题报告，覆盖多先验/metric 重建分支（MapAnything、HunyuanWorld-Mirror 等仓库暂无单篇分析，先经此报告了解）。
- 向定位延伸：Reloc3r / Reloc-VGGT 把同类几何骨干用于重定位，见 [visual-localization 学习路线](./visual-localization.md)。

## 4. 动手：最小复现路径
- 最小可跑路径：选一个有公开权重的前馈模型（如 Depth Anything 3 或 Pi3），喂 2-8 张同场景图，跑出 point map + 相机位姿并可视化；先确认推理链路与可视化，再谈 benchmark 子集。
- 复现记录：planned，仓库暂无复现记录。VGGT-Ω / Depth Anything 3 / Pi3 / LingBot-Map inference-level sanity check 均 planned；MapAnything / HunyuanWorld-Mirror planned。

## 5. 常见误区 / 我的判断
- 误区：以为前馈输出自带可靠 metric 尺度。多数方法尺度仍需对齐或外部先验，跨方法比绝对误差前要先确认尺度协议。
- 误区：用静态多视图的结论外推到动态/长序列。streaming（LingBot-Map）与 any-view（DA3/Pi3）的难点与评测并不同。
- 判断：要强 RGB/video backbone 看 VGGT-Ω；要任意视角/NVS 看 Depth Anything 3；要无序/reference-free 鲁棒看 Pi3；要在线建图看 LingBot-Map。先把开权重模型的推理 + 可视化跑通，再进对比报告里挑指标。
