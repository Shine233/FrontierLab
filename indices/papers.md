# 论文索引

> 状态：`triage` 初筛；`reading` 阅读中；`analyzed` 已分析；`compared` 已纳入对比；`reproducing` 复现中；`reproduced` 已复现；`archived` 暂不跟进。
> 代码字段约定：`Git 地址` 记录主要公开仓库；`是否开源` 记录代码/仓库是否公开；`是否开源训练` 记录训练代码是否公开。公开仓库若主要是推理/demo/模型代码而不是训练代码，写 `\`。

| Paper | Year | Direction | Method family | Status | Reproduction | Git 地址 | 是否开源 | 是否开源训练 | Links | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|
| LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction | 2026 | 3d-reconstruction / robotics-autonomous-driving | streaming 3D reconstruction / geometric context attention | compared | planned | [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | 是（Apache-2.0） | `\` | [paper](../papers/3d-reconstruction/2026-lingbot-map.md), [comparison](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) | 长视频 streaming pose/depth/point cloud 主候选；训练/eval benchmark 未开源 |
| MapAnything | 2025 | 3d-reconstruction / robotics-autonomous-driving | feed-forward metric 3D reconstruction | compared | planned | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 是 | 是 | [comparison](../reports/feedforward_3d_reconstruction_compare.md) | 当前已有对比报告中推荐为自动驾驶三维重建主候选 |
| OmniVGGT | 2025 | 3d-reconstruction | VGGT geometry-prior adapter | compared | none | [Livioni/OmniVGGT-official](https://github.com/Livioni/OmniVGGT-official) | 是 | 待核验 | [comparison](../reports/feedforward_3d_reconstruction_compare.md) | 适合作为轻量 baseline 或 adapter 对照；训练完整性需逐 commit 验证 |
| HunyuanWorld-Mirror | 2025 | 3d-reconstruction / world-models | any-prior world reconstruction / 3DGS / NVS | compared | planned | [Tencent-Hunyuan/HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) | 是（非商用/受限许可） | 是 | [comparison](../reports/feedforward_3d_reconstruction_compare.md) | 适合多先验融合和可视化/重渲染分支，需关注许可和工程复杂度 |
