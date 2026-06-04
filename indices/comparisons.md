# 对比报告索引

| Topic | Direction | Methods | Status | Report | Key takeaway |
|---|---|---|---|---|---|
| DINO 家族横向对比 | vision-foundation-models / self-supervised-learning / dense-vision | DINO v1 / DINOv2 / DINOv3 | completed | [`comparisons/vision-foundation-models/dino-family.md`](../comparisons/vision-foundation-models/dino-family.md) | v1 是自监督 ViT 机制起点；v2 是当前稳妥通用 backbone；v3 用 Gram anchoring 把 DINO scaling 推向更强 dense/high-res features。 |
| Any-view Visual Geometry Foundation Models | 3d-reconstruction / dense-vision / robotics-autonomous-driving | Depth Anything 3 / VGGT-Ω / VGGT / Pi3 / MapAnything / LingBot-Map / Depth Anything 2 | initial-completed | [`comparisons/3d-reconstruction/visual-geometry-foundation-models.md`](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md) | VGGT-Ω 是动态/RGB-only feed-forward 强 backbone；DA3 是 any-view geometry + API/NVS 主候选；Pi3 是 reference-free / input-order-robust baseline；MapAnything 更适合 metric prompts；LingBot-Map 更适合 causal streaming。 |
| 前馈式视觉重定位方法对比 | visual-localization / robotics-autonomous-driving / evaluation-benchmarks | Reloc-VGGT / Reloc3r / MARePo / FastForward / ACE / MASt3R | initial-completed | [`comparisons/visual-localization/feed-forward-visual-relocalization.md`](../comparisons/visual-localization/feed-forward-visual-relocalization.md) | Reloc3r 是当前最可跑的 scene-agnostic RPR baseline；Reloc-VGGT 是高上限 early fusion 但代码未发布；MARePo 是有 scene coordinate map 条件的 map-relative APR/SCR hybrid；FastForward 值得跟踪但暂未开源代码。 |
| 前馈式三维重建方法对比 | 3d-reconstruction / robotics-autonomous-driving | MapAnything / Pi3 / HunyuanWorld-Mirror / OmniVGGT / VGGT-Ω supplement | completed | [`reports/feedforward_3d_reconstruction_compare.md`](../reports/feedforward_3d_reconstruction_compare.md) | MapAnything 更适合作为自动驾驶/机器人 metric 3D 重建主候选；Pi3/Pi3X 是 reference-free 与输入顺序鲁棒性 baseline；VGGT-Ω 是动态/RGB-only 强 backbone；HunyuanWorld-Mirror 适合多先验与 3DGS/NVS 分支。 |
| Streaming 3D Reconstruction 方法对比 | 3d-reconstruction / robotics-autonomous-driving | LingBot-Map / Stream3R / Wint3R / TTT3R / CUT3R / InfiniteVGGT | initial-completed | [`comparisons/3d-reconstruction/streaming-3d-reconstruction.md`](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) | LingBot-Map 是当前长视频 streaming 3D reconstruction 主候选；GCA 在 Oxford Spires dense 和多数据集 pose/reconstruction 上显著优于 streaming baselines，但训练/eval benchmark 尚未开源。 |

## 新增对比时的最小信息

- 对比目标：为什么比较这些方法？
- 候选方法：至少 2 个。
- 固定维度：输入、输出、训练、开源、许可证、数据集、指标、工程成本、适用场景。
- 结论：推荐排序 + 触发条件 + 风险。
