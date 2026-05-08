# 方法族索引

> 目标：把“相似方向的方法”放在一起，避免每次从零比较。
> 代码字段约定与 `indices/papers.md` 一致：公开仓库若主要是推理/demo/模型代码而不是训练代码，在“是否开源训练”列写 `\`。

## Feed-forward metric 3D reconstruction

核心问题：从多视图图像与可选几何先验中，单次前向输出相机、深度、点云/点图或可用于重建的几何表示。

| Method | 定位 | Git 地址 | 是否开源 | 是否开源训练 | 强项 | 风险 | 相关资料 |
|---|---|---|---|---|---|---|---|
| MapAnything | 主 backbone 候选 | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 是 | 是 | metric scale、混合几何先验、长序列、多数据训练生态 | 动态物体需额外处理；非 raw LiDAR/Radar 输入 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |
| OmniVGGT | VGGT 几何先验增强 baseline | [Livioni/OmniVGGT-official](https://github.com/Livioni/OmniVGGT-official) | 是 | 待核验 | 轻量、易做 RGB+depth/camera prior 对照 | 生态较薄，长序列/大场景不是核心卖点；训练完整性需核验 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |
| HunyuanWorld-Mirror | any-prior 世界重建与 3DGS/NVS 分支 | [Tencent-Hunyuan/HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) | 是（非商用/受限许可） | 是 | 多先验、输出丰富、适合渲染/资产生成 | 工程重，许可证需审查 | [对比](../reports/feedforward_3d_reconstruction_compare.md) |
| LingBot-Map | streaming visual mapping 补充分支 | [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | 是（Apache-2.0） | `\` | 长视频因果推理、实时 pose/depth、trajectory memory 降漂移 | 不是多先验模型；训练/eval benchmark 未开源；无显式 loop closure | [paper](../papers/3d-reconstruction/2026-lingbot-map.md), [streaming 对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |

建议固定对比维度：输入模态、metric scale、长序列能力、训练开源度、许可证、自动驾驶适配、动态场景处理、推理成本、导出生态。

## Streaming 3D reconstruction / online visual mapping

核心问题：视频帧持续到达时，因果估计 camera trajectory、depth/point cloud，并在长序列中控制显存、延迟和漂移。

| Method | 上下文/状态策略 | Git 地址 | 是否开源 | 是否开源训练 | 当前定位 | 相关资料 |
|---|---|---|---|---|---|---|
| LingBot-Map | Anchor context + local pose-reference window + trajectory memory；FlashInfer paged KV cache | [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | 是（Apache-2.0） | `\` | 长视频 streaming 3D reconstruction 主候选；优先做 inference sanity check | [paper](../papers/3d-reconstruction/2026-lingbot-map.md), [对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |
| Stream3R / Stream3R-w | Causal/sequential transformer 类 streaming | 待补充 | 待补充 | 待补充 | 重要 baseline；Tanks and Temples 指标较强 | [对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |
| Wint3R | Window-based streaming reconstruction | 待补充 | 待补充 | 待补充 | 重要 baseline；ETH3D reconstruction 较强 | [对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |
| TTT3R | Test-time-training 3D reconstruction | 待补充 | 待补充 | 待补充 | TTT baseline；短序列表现接近但有额外开销 | [对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |
| CUT3R / InfiniteVGGT / SLAM3R / Spann3R / StreamVGGT | Recurrent state、cache 或早期 streaming adaptation | 待补充 | 待补充 | 待补充 | 历史/消融 baseline | [对比](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md) |

## Geometry-prior prompting / adapter

核心问题：将 camera、pose、depth、point map、SLAM/LiDAR 投影等几何先验编码为 prompt/token/condition，增强重建或世界模型稳定性。

- 已有关联：MapAnything、OmniVGGT、HunyuanWorld-Mirror。
- LingBot-Map 当前不属于多先验 prompting 主线；它的 anchor/window/memory 是 streaming 状态设计，不是 raw LiDAR/IMU 或 depth prompt 融合。
- 待补充：新 paper 或代码库。

## 3DGS / NVS world reconstruction

核心问题：从图像和几何先验构建可重渲染场景表示，用于 novel view synthesis、仿真、数据生成或世界资产。

- 已有关联：HunyuanWorld-Mirror；MapAnything 可通过导出衔接 3DGS。
- LingBot-Map 可输出/渲染点云和轨迹，但不是 3DGS/NVS 主模型；可作为在线建图前端或几何初始化来源候选。
- 待补充：Gaussian Splatting 相关方法、动态 3DGS、驾驶场景 4D 表示。

## DINO family / self-supervised visual foundation models

核心问题：不用人工标签，训练可迁移的图像级与 patch 级视觉表征；在 scaling 时同时保持 global 和 dense feature 质量。

| Method | 定位 | Git 地址 | 是否开源 | 是否开源训练 | 强项 | 风险 | 相关资料 |
|---|---|---|---|---|---|---|---|
| DINO v1 | 自监督 ViT 语义涌现与 self-distillation 教学起点 | [facebookresearch/dino](https://github.com/facebookresearch/dino) | 是（Apache-2.0） | 是 | teacher-student、multi-crop、防塌缩机制清楚；attention object masks 易理解 | 代码环境较旧；不是最新强 backbone | [paper](../papers/vision-foundation-models/2021-dino.md), [对比](../comparisons/vision-foundation-models/dino-family.md) |
| DINOv2 | 通用 frozen visual backbone baseline | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 是（Apache-2.0） | 是（训练代码开源；LVD-142M 不可完整复刻） | curated data scaling、DINO+iBOT+KoLeo、蒸馏模型生态成熟 | 完整大规模数据不可复刻；dense/high-res 上限弱于 v3 | [paper](../papers/vision-foundation-models/2023-dinov2.md), [对比](../comparisons/vision-foundation-models/dino-family.md) |
| DINOv3 | 高上限 dense/high-resolution vision foundation model | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | 是（DINOv3 License，需审查） | 是（训练代码/配置开源；旗舰私有数据不可复刻） | Gram anchoring 修复 dense feature degradation；ViT-7B、LVD-1689M、SAT-493M、多尺寸蒸馏 | 自定义 license；权重申请/私有数据/超大算力；完整训练不可复刻 | [paper](../papers/vision-foundation-models/2025-dinov3.md), [对比](../comparisons/vision-foundation-models/dino-family.md) |

建议固定对比维度：global vs dense feature、训练数据可得性、训练代码开源度、权重/许可证、输入分辨率适配、下游任务、推理成本、与 CLIP/SAM/3D backbones 的互补关系。

## Dense visual foundation features

核心问题：提供可冻结使用的 patch/像素级表征，用于 segmentation、depth、tracking、3D correspondence、robotics perception 等。

- 已有关联：DINOv3、DINOv2；后续可加入 SAM/SAM2、SigLIP/PEspatial、AM-RADIO、Depth Anything、各类 3D foundation model backbones。
- 当前结论：DINOv3 是 dense feature scaling 的重点候选；DINOv2 是更稳妥的工程 baseline；DINO v1 是机制教学起点。
- 待补充：在自动驾驶/机器人/3D reconstruction 任务中，用 DINOv2/v3 patch features 做统一 probe。

## Test-time scaling / reasoning agents

核心问题：通过搜索、采样、自验证、工具调用或 agent pipeline 提升推理质量。

- 待补充。
