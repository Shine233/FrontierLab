# 方向索引

> 维护方式：每新增一篇 paper、一个对比报告或一个复现实验，都在相关方向下追加链接。

## 3d-reconstruction

关注：前馈式三维重建、多视图几何、metric reconstruction、3DGS/NVS、自动驾驶/机器人场景重建。

- Paper：[`LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction`](../papers/3d-reconstruction/2026-lingbot-map.md)
- 对比报告：[`前馈式三维重建方法对比：MapAnything / OmniVGGT / HunyuanWorld-Mirror`](../reports/feedforward_3d_reconstruction_compare.md)
- 对比报告：[`Streaming 3D Reconstruction 方法横向对比`](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md)
- 相关方法族：feed-forward 3D reconstruction、streaming 3D reconstruction、geometric context attention、geometry-prior prompting、3D Gaussian Splatting 初始化、多视图 metric depth / point map。
- 复现状态：LingBot-Map inference 复现 planned；MapAnything / HunyuanWorld-Mirror planned。

## vision-foundation-models

关注：视觉基础模型、自监督视觉预训练、frozen backbone、ViT/ConvNeXt 表征、dense feature、跨任务迁移。

- Paper：[`DINO / Emerging Properties in Self-Supervised Vision Transformers`](../papers/vision-foundation-models/2021-dino.md)
- Paper：[`DINOv2: Learning Robust Visual Features without Supervision`](../papers/vision-foundation-models/2023-dinov2.md)
- Paper：[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- 对比报告：[`DINO 家族横向对比：v1 / v2 / v3`](../comparisons/vision-foundation-models/dino-family.md)
- 相关方法族：DINO family、self-distillation without labels、self-supervised Vision Transformer、curated-data scaling、Gram anchoring、dense visual foundation features、model distillation。
- 复现状态：DINO v1/v2/v3 均 planned；优先做 inference / feature extraction / linear probe，不优先从零复刻大规模预训练。

## self-supervised-learning

关注：无标签表征学习、自蒸馏、masked image modeling、feature regularization、collapse avoidance、预训练 recipe。

- Paper：[`DINO`](../papers/vision-foundation-models/2021-dino.md)、[`DINOv2`](../papers/vision-foundation-models/2023-dinov2.md)、[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- 对比报告：[`DINO family`](../comparisons/vision-foundation-models/dino-family.md)
- 关联：DINO v1 适合机制教学；DINOv2 适合通用 SSL backbone；DINOv3 适合 dense feature scaling。

## dense-vision

关注：像素/patch 级视觉表征、语义分割、深度估计、tracking、3D correspondence、高分辨率 dense backbone。

- Paper：[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- 对比报告：[`DINO family`](../comparisons/vision-foundation-models/dino-family.md)
- 关联：DINOv3 的 Gram anchoring 是当前 dense feature scaling 的重点候选；可后续与 SAM、SigLIP/PEspatial、DINOv2 registers、3D reconstruction backbones 横向比较。

## world-models

关注：世界模型、视频/交互预测、场景生成、仿真环境、机器人/自动驾驶闭环。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## multimodal-learning

关注：视觉-语言-动作、多传感器/几何先验、多模态基础模型、跨模态对齐。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## robotics-autonomous-driving

关注：机器人、自动驾驶、感知/规划/控制结合、真实尺度与传感器先验。

- Paper：[`LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction`](../papers/3d-reconstruction/2026-lingbot-map.md)
- 对比：[`Streaming 3D Reconstruction 方法横向对比`](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md)
- 关联：LingBot-Map 更适合作为连续视频在线建图/VO 候选；MapAnything/HunyuanWorld-Mirror 更适合多先验/metric reconstruction 或重渲染分支。
- 复现：LingBot-Map inference-level sanity check planned。

## generation-diffusion

关注：扩散模型、视频生成、3D/4D 生成、可控生成与数据合成。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## reasoning-agents

关注：推理模型、test-time compute、agent framework、工具使用、自我改进与评测。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## efficient-training-inference

关注：训练效率、推理加速、量化、蒸馏、长上下文、系统优化。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## evaluation-benchmarks

关注：benchmark、数据集、指标、评测协议、复现实验可信度。

- Paper / Benchmark：待补充。
- 对比：待补充。
- 复现：待补充。
