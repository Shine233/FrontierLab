# 方向索引

> 维护方式：每新增一篇 paper、一个对比报告或一个复现实验，都在相关方向下追加链接。

## 3d-reconstruction

关注：前馈式三维重建、多视图几何、metric reconstruction、3DGS/NVS、自动驾驶/机器人场景重建。

- Paper：[`Depth Anything 3: Recovering the Visual Space from Any Views`](../papers/3d-reconstruction/2025-depth-anything-3.md)
- Paper：[`VGGT-Ω`](../papers/3d-reconstruction/2026-vggt-omega.md)
- Paper：[`Pi3 / π³: Permutation-Equivariant Visual Geometry Learning`](../papers/3d-reconstruction/2026-pi3.md)
- Paper：[`LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction`](../papers/3d-reconstruction/2026-lingbot-map.md)
- Paper：[`Xiaomi Auto World Model / JointWM`](../papers/world-models/2026-xiaomi-auto-world-model.md)
- 对比报告：[`前馈式三维重建方法对比：MapAnything / Pi3 / HunyuanWorld-Mirror / OmniVGGT`](../reports/feedforward_3d_reconstruction_compare.md)
- 对比报告：[`Streaming 3D Reconstruction 方法横向对比`](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md)
- 对比报告：[`Any-view Visual Geometry Foundation Models 横向对比`](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)
- 相关方法族：any-view visual geometry foundation model、dynamic feed-forward reconstruction、register attention、reconstruction as spatial pretraining、permutation-equivariant visual geometry、reference-free reconstruction、depth-ray representation、feed-forward 3D reconstruction、streaming 3D reconstruction、geometric context attention、sparse-scene-query 3DGS、geometry-prior prompting、3D Gaussian Splatting 初始化、多视图 metric depth / point map。
- 复现状态：VGGT-Ω / Depth Anything 3 / Pi3 / LingBot-Map inference 复现 planned；MapAnything / HunyuanWorld-Mirror planned；Xiaomi Auto World Model 因代码/权重未公开暂 blocked。

## vision-foundation-models

关注：视觉基础模型、自监督视觉预训练、frozen backbone、ViT/ConvNeXt 表征、dense feature、跨任务迁移。

- Paper：[`DINO / Emerging Properties in Self-Supervised Vision Transformers`](../papers/vision-foundation-models/2021-dino.md)
- Paper：[`DINOv2: Learning Robust Visual Features without Supervision`](../papers/vision-foundation-models/2023-dinov2.md)
- Paper：[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- Paper：[`VGGT-Ω`](../papers/3d-reconstruction/2026-vggt-omega.md)
- 对比报告：[`DINO 家族横向对比：v1 / v2 / v3`](../comparisons/vision-foundation-models/dino-family.md)
- 相关方法族：DINO family、self-distillation without labels、self-supervised Vision Transformer、curated-data scaling、Gram anchoring、dense visual foundation features、model distillation、reconstruction as spatial pretraining、register scene tokens。
- 复现状态：DINO v1/v2/v3 与 VGGT-Ω 均 planned；优先做 inference / feature extraction / linear probe / spatial-token probe，不优先从零复刻大规模预训练。

## self-supervised-learning

关注：无标签表征学习、自蒸馏、masked image modeling、feature regularization、collapse avoidance、预训练 recipe。

- Paper：[`DINO`](../papers/vision-foundation-models/2021-dino.md)、[`DINOv2`](../papers/vision-foundation-models/2023-dinov2.md)、[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- 对比报告：[`DINO family`](../comparisons/vision-foundation-models/dino-family.md)
- 关联：DINO v1 适合机制教学；DINOv2 适合通用 SSL backbone；DINOv3 适合 dense feature scaling。

## dense-vision

关注：像素/patch 级视觉表征、语义分割、深度估计、tracking、3D correspondence、高分辨率 dense backbone。

- Paper：[`DINOv3`](../papers/vision-foundation-models/2025-dinov3.md)
- Paper：[`VGGT-Ω`](../papers/3d-reconstruction/2026-vggt-omega.md)
- Paper：[`Depth Anything 3`](../papers/3d-reconstruction/2025-depth-anything-3.md)
- Paper：[`Pi3 / π³`](../papers/3d-reconstruction/2026-pi3.md)
- 对比报告：[`DINO family`](../comparisons/vision-foundation-models/dino-family.md)
- 对比报告：[`Any-view Visual Geometry Foundation Models`](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)
- 关联：DINOv3 的 Gram anchoring 是 dense feature scaling 重点候选；VGGT-Ω 是 reconstruction-as-spatial-pretraining 和 register scene tokens 重点候选；Depth Anything 3 是 dense depth/ray/geometry 输出重点候选；Pi3 是 reference-free local pointmap / pose 输出重点候选；可后续与 SAM、SigLIP/PEspatial、DINOv2 registers、3D reconstruction backbones 横向比较。

## world-models

关注：世界模型、视频/交互预测、场景生成、仿真环境、机器人/自动驾驶闭环。

- Paper：[`Xiaomi Auto World Model / JointWM`](../papers/world-models/2026-xiaomi-auto-world-model.md)
- 关联：WorldRec 用稀疏 3D scene queries 做前馈式 3DGS 重建；WorldGen 用双向预训练 + 因果微调 + ODE distillation + DMD 做长时序驾驶视频生成；JointWM 用 WorldRec 渲染先验约束 WorldGen，面向 closed-loop simulation、data synthesis、end-to-end training。
- 对比：待创建 driving world models / reconstruction-generation hybrid 横向对比。
- 复现：Xiaomi Auto World Model 当前无公开 GitHub、权重或训练数据，复现 blocked。

## multimodal-learning

关注：视觉-语言-动作、多传感器/几何先验、多模态基础模型、跨模态对齐。

- Paper：待补充。
- 对比：待补充。
- 复现：待补充。

## robotics-autonomous-driving

关注：机器人、自动驾驶、感知/规划/控制结合、真实尺度与传感器先验。

- Paper：[`Depth Anything 3`](../papers/3d-reconstruction/2025-depth-anything-3.md)
- Paper：[`VGGT-Ω`](../papers/3d-reconstruction/2026-vggt-omega.md)
- Paper：[`Pi3 / π³`](../papers/3d-reconstruction/2026-pi3.md)
- Paper：[`LingBot-Map / Geometric Context Transformer for Streaming 3D Reconstruction`](../papers/3d-reconstruction/2026-lingbot-map.md)
- Paper：[`Xiaomi Auto World Model / JointWM`](../papers/world-models/2026-xiaomi-auto-world-model.md)
- 对比：[`Streaming 3D Reconstruction 方法横向对比`](../comparisons/3d-reconstruction/streaming-3d-reconstruction.md)
- 对比：[`Any-view Visual Geometry Foundation Models`](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)
- 关联：VGGT-Ω 更适合作为强 RGB-only/video geometry backbone 和空间 token 表征候选；Depth Anything 3 更适合作为任意视角视觉几何/NVS backbone；Pi3 更适合作为无序多视图/reference-free robustness baseline；LingBot-Map 更适合作为连续视频在线建图/VO 候选；Xiaomi Auto World Model 更适合作为自动驾驶重建-生成联合世界模型参考；MapAnything/HunyuanWorld-Mirror 更适合多先验/metric reconstruction 或重渲染分支。
- 复现：VGGT-Ω / Depth Anything 3 / Pi3 / LingBot-Map inference-level sanity check planned；Xiaomi Auto World Model 当前 blocked。

## generation-diffusion

关注：扩散模型、视频生成、3D/4D 生成、可控生成与数据合成。

- Paper：[`Xiaomi Auto World Model / JointWM`](../papers/world-models/2026-xiaomi-auto-world-model.md)
- 关联：WorldGen 属于 autonomous-driving video generation / causal DiT world model；使用 bidirectional pretraining、Teacher Forcing、ODE distillation、DMD，报告 nuScenes FVD 64.97、FID 7.04、81 frames、0.19s/frame。
- 对比：待补充 MagicDrive / MagicDrive-V2 / Epona / Genesis / GAIA-2 / WorldGen。
- 复现：无公开代码/权重，暂 blocked。

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

- Paper / Benchmark：[`VGGT-Ω`](../papers/3d-reconstruction/2026-vggt-omega.md)；论文评测覆盖 7Scenes、NRGBD、ETH3D、DyCheck、Sintel、TUM-Dynamic 的 camera/depth evaluation，并报告 LIBERO / language retrieval 表征实验。
- Paper / Benchmark：[`Depth Anything 3`](../papers/3d-reconstruction/2025-depth-anything-3.md)；DA3-BENCH 覆盖 HiRoom、ETH3D、DTU、7Scenes、ScanNet++ 的 pose/reconstruction evaluation。
- Paper / Benchmark：[`Pi3 / π³`](../papers/3d-reconstruction/2026-pi3.md)；evaluation branch 覆盖 monocular/video depth、relative camera pose、multi-view point map reconstruction。
- Paper / Benchmark：[`Xiaomi Auto World Model / JointWM`](../papers/world-models/2026-xiaomi-auto-world-model.md)；论文报告 Waymo / nuScenes 的 WorldRec PSNR、SSIM，以及 nuScenes WorldGen FID、FVD、frames、inference time；缺少可复跑代码和几何/闭环数值协议。
- 对比：[`Any-view Visual Geometry Foundation Models`](../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)
- 复现：Depth Anything 3 benchmark 子集 planned。
