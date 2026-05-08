# 方法族索引

> 目标：把“相似方向的方法”放在一起，避免每次从零比较。

## Feed-forward metric 3D reconstruction

核心问题：从多视图图像与可选几何先验中，单次前向输出相机、深度、点云/点图或可用于重建的几何表示。

| Method | 定位 | 强项 | 风险 | 相关资料 |
|---|---|---|---|---|
| MapAnything | 主 backbone 候选 | metric scale、混合几何先验、长序列、多数据训练生态 | 动态物体需额外处理；非 raw LiDAR/Radar 输入 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |
| OmniVGGT | VGGT 几何先验增强 baseline | 轻量、易做 RGB+depth/camera prior 对照 | 生态较薄，长序列/大场景不是核心卖点 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |
| HunyuanWorld-Mirror | any-prior 世界重建与 3DGS/NVS 分支 | 多先验、输出丰富、适合渲染/资产生成 | 工程重，许可证需审查 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |

建议固定对比维度：输入模态、metric scale、长序列能力、训练开源度、许可证、自动驾驶适配、动态场景处理、推理成本、导出生态。

## Geometry-prior prompting / adapter

核心问题：将 camera、pose、depth、point map、SLAM/LiDAR 投影等几何先验编码为 prompt/token/condition，增强重建或世界模型稳定性。

- 已有关联：MapAnything、OmniVGGT、HunyuanWorld-Mirror。
- 待补充：新 paper 或代码库。

## 3DGS / NVS world reconstruction

核心问题：从图像和几何先验构建可重渲染场景表示，用于 novel view synthesis、仿真、数据生成或世界资产。

- 已有关联：HunyuanWorld-Mirror；MapAnything 可通过导出衔接 3DGS。
- 待补充：Gaussian Splatting 相关方法、动态 3DGS、驾驶场景 4D 表示。

## Test-time scaling / reasoning agents

核心问题：通过搜索、采样、自验证、工具调用或 agent pipeline 提升推理质量。

- 待补充。
