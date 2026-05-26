---
type: paper-analysis
title: "Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving"
short_name: "Xiaomi Auto World Model / JointWM"
year: 2026
venue: "arXiv technical report"
arxiv: "2605.18137v3"
paper_url: "https://arxiv.org/abs/2605.18137"
pdf_url: "https://arxiv.org/pdf/2605.18137"
code: ""
github: ""
project_page: "https://JointWM.github.io"
open_source: false
license: "Unknown; paper/project page do not specify code, model, or data license."
training_open_source: false
weights_availability: "No public weights found in the paper or project page as of 2026-05-26."
data_availability: "Evaluation uses Waymo and nuScenes plus private data; training data and reconstructed-prior fine-tuning data are not released in the paper/project page."
direction: [world-models, 3d-reconstruction, robotics-autonomous-driving, generation-diffusion, evaluation-benchmarks]
method_family: [joint-world-model, feed-forward-3dgs-reconstruction, sparse-scene-query, autonomous-driving-video-generation, reconstruction-conditioned-generation]
tasks: [feed-forward-3d-gaussian-reconstruction, novel-view-synthesis, driving-video-generation, long-horizon-autoregressive-generation, multi-view-consistency, closed-loop-simulation, data-synthesis]
datasets: [Waymo, nuScenes, private-driving-data]
metrics: [PSNR, SSIM, FID, FVD, reconstruction-time, inference-time, frames, temporal-consistency, multi-view-spatial-consistency, multi-run-stability]
status: analyzed
reproduction: blocked
confidence: high
updated: 2026-05-26
---

# Xiaomi Auto World Model：重建与生成联合的自动驾驶世界模型

## 结论先行

- **一句话定位**：这是一份小米汽车世界模型技术报告，提出 `WorldRec + WorldGen + Joint World Model`：先用稀疏 3D scene queries 做前馈式 3DGS 重建，再用因果 DiT 做长时序多视角驾驶视频生成，最后把重建出的 4D Gaussian 场景作为生成模型的几何锚点。
- **核心方法**：WorldRec 不走每帧像素级 DPT Gaussian head，而是在世界坐标中初始化稀疏 3D queries，跨相机、跨时间采样/聚合特征，再解码 Gaussian 属性；WorldGen 先双向预训练，再用 Teacher Forcing、ODE distillation、DMD 三阶段转成在线因果生成器。
- **实验结论**：论文报告 WorldRec 在 Waymo / nuScenes 的 NVS PSNR、SSIM 上超过 MVSSplat、NoPoSplat、DepthSplat、STORM、DGGT；WorldGen 在 nuScenes 上 FVD 64.97、FID 7.04，可生成 81 帧，单视角 0.19s/frame。
- **工程亮点**：报告声称 10 秒视频片段约 10 秒完成重建，而逐场景优化约 4 小时；WorldGen 通过 50 -> 4 denoising steps 的 ODE distillation 获得约 12x 采样加速。
- **开源状态**：论文和项目页公开，但截至 2026-05-26 未看到 GitHub、训练代码、推理代码、权重或数据下载入口；因此当前只能做 paper-level 分析，不能做 inference-level 复现。
- **与本仓库已有主线关系**：WorldRec 和 HunyuanWorld-Mirror / DA3 / 3DGS/NVS 主线相关，但更明确面向自动驾驶多相机时序场景；WorldGen 属于 driving world model / generation-diffusion；JointWM 是本仓库 `world-models` 方向的第一篇核心条目。

## 1. 这篇报告解决什么问题？

### 已确认的论文事实

- **问题定义**：自动驾驶世界模型需要两类能力：世界表征（可显式表示和渲染观测场景）与世界生成（能预测未来、补全未观测区域、生成长尾场景）。
- **系统组成**：
  - `WorldRec`：前馈式 sparse-query 3D Gaussian reconstruction。
  - `WorldGen`：基于 DiT 的多模态条件驾驶视频生成模型。
  - `Joint World Model / JointWM`：把 WorldRec 的 4D scene representation 作为 WorldGen 的几何先验，提升长时序稳定性、跨视角一致性和视觉保真度。
- **输入条件**：论文描述中包含多相机、多时刻图像、ego trajectory、camera intrinsics / extrinsics、layout maps、free-form text，以及 WorldRec 渲染出来的 prior image condition。
- **输出**：WorldRec 输出紧凑 3D Gaussian 场景表示，可做 novel view synthesis；WorldGen 输出多视角驾驶视频；JointWM 支持 closed-loop simulation、data synthesis 和 end-to-end training。

### 我的理解

这篇报告的关键不是单纯“又做一个 3DGS”或“又做一个视频生成模型”，而是把两个方向耦合：

- 重建模型提供确定性、几何一致的 scene memory，抑制生成漂移。
- 生成模型补足重建模型看不到的区域、未来状态和长尾事件。
- 对自动驾驶来说，这比纯 NVS 或纯 video generation 更接近闭环仿真的真实需求。

## 2. WorldRec：稀疏查询驱动的前馈式 3DGS 重建

### 2.1 动机

论文指出传统 per-scene 3DGS optimization 质量高，但每个新 sequence 需要几十分钟到数小时训练；已有 feed-forward 3DGS 方法常用 DPT head 逐像素预测每帧 Gaussians，拼接后容易产生 ghosting、layered surface duplication，而且一个 clip 会累积到数亿 primitives，渲染开销很重。

WorldRec 的替代思路是：**用稀疏 scene tokens / 3D queries 表达场景，而不是每帧生成密集像素级 Gaussian primitives。**

### 2.2 架构

| 模块 | 论文描述 | 作用 |
|---|---|---|
| Multi-scale feature extraction | 多相机多时刻图像经 shared visual backbone 得到多尺度特征 | 同时保留纹理细节和大尺度语义结构 |
| 3D query initialization | 在世界坐标中初始化 `N` 个稀疏 3D queries，每个 query 有参考坐标 `p=[X,Y,Z]^T` | 让 Gaussian primitive 从一开始就绑定到 3D 空间位置 |
| Projection sampling | 将 3D query 投影到各相机、各尺度 feature map，用双线性插值采样特征 | 从多视角观测收集同一空间点的信息 |
| Cross-view / cross-temporal aggregation | visibility-aware weighted aggregation 融合多相机、多时间特征 | 抑制遮挡、反光、低质量视角，强化跨帧一致性 |
| Gaussian attribute decoding | MLP 解码位置偏移、RGB、opacity、scale、rotation quaternion | 得到最终 3D Gaussian primitives |
| Rendering supervision | Gaussian rasterization 渲染目标视角，用 pixel loss + perceptual loss 监督 | 用多视角渲染误差逼迫几何和外观一致 |

### 2.3 与已有前馈重建方法的区别

- 与 STORM / DGGT 等逐帧 Gaussian 预测相比，WorldRec 的 primitive 不直接绑定单帧像素，而是绑定 3D query，因此天然更容易跨视角融合。
- 与 DA3 / VGGT-Ω / Pi3 这类 visual geometry backbone 相比，WorldRec 目标更偏 **driving 3DGS/NVS world representation**，不是通用 pose/depth/point cloud 输出。
- 与 HunyuanWorld-Mirror 类 any-prior world reconstruction 相比，WorldRec 更强调自动驾驶多相机时序结构和稀疏查询压缩。

## 3. WorldGen：双向预训练到因果生成的驾驶视频模型

### 已确认的论文事实

WorldGen 使用 Diffusion Transformer / DiT backbone，条件包括：

- multi-view first frame；
- layout condition；
- ego trajectory；
- camera intrinsics / extrinsics；
- free-form text；
- JointWM 阶段额外加入 WorldRec 渲染出的 prior images。

训练分两大阶段：

1. **Bidirectional pre-training**：先用全时序双向注意力训练 DiT，学习完整驾驶场景分布和强生成先验。
2. **Causal fine-tuning**：再施加 causal attention mask，把模型转成在线自回归生成器，并依次做：
   - Teacher Forcing：让当前帧只能看当前 noisy frame 和过去 clean frames；
   - ODE distillation：把约 50 步采样蒸馏成 4 步，论文称约 12x 加速；
   - DMD：训练时用模型自己生成的历史帧替代 GT 历史帧，缓解 exposure bias 和长时序漂移。

### 我的理解

WorldGen 的设计逻辑很实用：

- 先用双向模型学“画得好”，因为双向 attention 学分布更容易。
- 再用 causal fine-tuning 学“在线生成”，满足闭环仿真不能看未来的约束。
- 最后用 distillation 和 DMD 分别解决“太慢”和“越滚越偏”的问题。

## 4. Joint World Model：重建约束生成，生成补全重建

### 已确认的论文事实

JointWM 对两个模块做了接口改造：

- **WorldRec incremental scene reconstruction**：将新观测 token 与 cached scene tokens 通过 cross-attention fusion 融合，持续扩展/更新一个全局一致的 4D Gaussian representation。
- **WorldGen rendered-prior conditioning**：先把 WorldRec 的 scene tokens rasterize 到目标相机视角，生成可能有空洞/遮挡缺失的 partial reference images，再作为额外条件输入 DiT。
- **训练数据**：为 rendered-prior conditioning 构造了专门训练数据，即从 reconstructed scenes 在 held-out target poses 上渲染 reference images，再 fine-tune WorldGen。

### 我的判断

这部分是报告里最值得关注的系统点。纯生成模型容易出现跨帧漂移、跨相机不一致、多次采样结构不稳定；WorldRec 的 4D scene state 等价于一个共享空间记忆，可以约束车道线、边界、静态物体位置和纹理。反过来，生成模型可以补 WorldRec 没见过的区域和未来动态。

## 5. 实验与证据

### 5.1 WorldRec 定量结果

论文 Table 2 在 Waymo 和 nuScenes 上报告 NVS / reconstruction rendering 指标：

| Method | Waymo PSNR ↑ | Waymo SSIM ↑ | nuScenes zero-shot PSNR ↑ | nuScenes zero-shot SSIM ↑ | nuScenes fine-tune PSNR ↑ | nuScenes fine-tune SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| MVSSplat | 20.56 | 0.697 | 17.84 | 0.563 | - | - |
| NoPoSplat | 24.31 | 0.751 | 19.75 | 0.545 | - | - |
| DepthSplat | 23.26 | 0.696 | 19.52 | 0.601 | - | - |
| STORM | 26.38 | 0.794 | 17.77 | 0.669 | 24.54 | 0.784 |
| DGGT | 27.41 | 0.846 | 25.31 | 0.794 | 26.63 | 0.813 |
| **WorldRec** | **28.48** | **0.861** | **26.54** | **0.821** | **27.50** | **0.826** |

论文还称：10 秒视频片段约 10 秒完成重建，而 per-scene optimization 约 4 小时。

### 5.2 WorldGen 定量结果

论文 Table 3 在 nuScenes 上比较 driving world models：

| Model | Bi / AR | FID ↓ | FVD ↓ | Frames | Infer. Time |
|---|---|---:|---:|---:|---:|
| MagicDrive | Bi | 16.20 | - | 1 | - |
| MagicDrive-V2 | Bi | 20.91 | 94.84 | 16 | - |
| Vista | Bi | 6.9 | 89.4 | 16 | - |
| DiVE | Bi | 7.14 | 68.4 | 8 | - |
| Delphi | Bi | 15.08 | 113.5 | 8 | - |
| UniScene | Bi | 6.12 | 70.52 | 8 | - |
| Genesis | Bi | 6.45 | 67.87 | 16 | - |
| Epona | AR | 7.5 | 82.8 | 16 | 1.06s/frame |
| **WorldGen** | **AR** | **7.04** | **64.97** | **81** | **0.19s/frame** |

论文正文还说明：

- H20 GPU 上单视角 0.19s/frame，三视角 0.46s/frame。
- 支持 10fps / 30fps、最长约 1 分钟的可控长时序生成。
- 4-step efficient sampling 来自 ODE distillation。

### 5.3 JointWM 结果

JointWM 的评估主要是 qualitative / visual evidence，分为：

- long-horizon temporal consistency；
- multi-view spatial consistency；
- multi-run stability。

论文展示结果认为 WorldRec 的几何先验能抑制 WorldGen 的 drift、hallucination 和多次采样结构方差。

## 6. 局限与风险

### 论文 / 项目页已确认的限制

- 当前公开材料没有 GitHub、推理代码、训练代码、权重、license 或数据下载入口。
- WorldRec 定量表主要是 PSNR / SSIM；没有报告 3D 几何指标如 Chamfer、F-score、depth/pose error、occupancy IoU 或下游感知指标。
- JointWM 的三项核心收益主要通过 qualitative figures 展示，缺少可复跑的 temporal consistency / cross-view consistency / multi-run stability 数值协议。
- 训练数据包含 private data，WorldGen rendered-prior fine-tuning 数据也未公开。

### 我的推断风险

- **复现风险极高**：没有代码/权重时无法做 inference sanity check；即使后续开源，WorldRec + WorldGen + JointWM 三模块系统很可能依赖大规模内部驾驶数据和算力。
- **指标覆盖不足**：WorldRec 用 PSNR/SSIM 证明渲染质量，但自动驾驶闭环还需要几何尺度、动态主体一致性、occupancy/BEV/感知闭环收益等指标。
- **生成评测风险**：FID/FVD 对驾驶安全关键长尾事件不够充分；动物、极端天气、长时序控制等结果目前主要靠可视化展示。
- **工程集成复杂**：WorldRec 的 incremental scene fusion、WorldGen rendered-prior conditioning、DiT causal generation 和 3DGS rasterization 串在一起，部署成本远高于单个 feed-forward geometry model。
- **商业/许可未知**：项目页未列许可证，不能假定代码、权重、样例视频或模型输出可商用。

## 7. 与相似方法关系

| Method | 相同点 | 不同点 | 何时参考 |
|---|---|---|---|
| HunyuanWorld-Mirror | 都面向 world reconstruction / 3DGS / NVS，并可服务仿真和资产生成 | Xiaomi Auto World Model 更强调自动驾驶多相机时序场景和 WorldRec/WorldGen 联合；HunyuanWorld-Mirror 更偏 any-prior world reconstruction 公开系统 | 需要多先验、公开工程和 3DGS/NVS 生态时优先看 Hunyuan；研究重建-生成耦合看 JointWM |
| Depth Anything 3 | 都可服务前馈几何和 3DGS/NVS | DA3 是通用 any-view depth/ray/pose/point backbone，公开 API/benchmark/权重；WorldRec 是 driving-specific sparse-query 3DGS reconstruction，当前不开源 | 需要可跑 baseline 选 DA3；研究 driving 3DGS sparse query 架构看 WorldRec |
| VGGT-Ω | 都关注 feed-forward reconstruction 和长序列/动态视频能力 | VGGT-Ω 输出 camera/depth/register tokens，强调 RGB geometry backbone scaling；WorldRec 直接输出 3D Gaussian scene representation | 强 RGB geometry baseline 看 VGGT-Ω；渲染/仿真场景表示看 WorldRec |
| LingBot-Map | 都和长视频/自动驾驶在线空间理解相关 | LingBot-Map 是 causal streaming pose/depth/point cloud；JointWM 是重建+生成联合世界模型 | 做在线 VO/建图先看 LingBot-Map；做闭环仿真和视频预测看 JointWM |
| MagicDrive / Epona / Genesis | 都是 driving video/world generation | WorldGen 是 AR 模型，使用双向预训练 + 因果微调 + 4-step distillation，并接入 WorldRec 几何先验 | 做 driving video generation benchmark 时作为同类生成 baseline |
| STORM / DGGT | 都是 driving feed-forward 3DGS reconstruction baseline | STORM / DGGT 属于更直接的前馈 4D reconstruction；WorldRec 用稀疏 3D queries 减少每帧高斯爆炸和 ghosting | 后续若做 world-model reconstruction 对比，STORM/DGGT/WorldRec 应放在同表 |

## 8. 复现判断

- Paper：<https://arxiv.org/abs/2605.18137>
- PDF：<https://arxiv.org/pdf/2605.18137>
- Project：<https://JointWM.github.io>
- GitHub：未发现公开 GitHub。
- 是否开源：否。当前只看到论文和项目页。
- 是否开源训练：否。未发现训练代码、训练配置、数据处理 pipeline。
- 权重可用性：未发现公开权重。
- 数据可获得性：Waymo / nuScenes 可按原始数据集协议获取；private data、训练集、WorldGen rendered-prior fine-tuning data 未公开。
- 许可证：未知。项目页未给出代码/权重/数据许可证。
- 最小复现路径：当前不可复现。若后续开源，建议先验证：
  1. WorldRec 在 Waymo mini split 上的 PSNR/SSIM 与 10s/clip reconstruction time；
  2. WorldGen 在 nuScenes 上 81 frames、0.19s/frame、FID/FVD；
  3. JointWM 对同一 prompt / trajectory 的 multi-run variance、cross-view consistency 和长时序 drift。
- 是否值得跟进：值得作为 `world-models` 与 `driving reconstruction-generation hybrid` 方向重点跟踪；短期不适合作为工程 baseline，因为缺少代码和权重。

## 9. 后续动作

- [x] 创建 Xiaomi Auto World Model 单篇论文分析
- [x] 更新 `README.md`
- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [ ] 若后续释放代码/权重，创建 `reproductions/world-models/xiaomi-auto-world-model/README.md`
- [ ] 后续可创建 driving world models 横向对比：WorldRec / STORM / DGGT / HunyuanWorld-Mirror / WorldGen / Epona / Genesis / MagicDrive-V2

## Sources

- Paper: <https://arxiv.org/abs/2605.18137>
- PDF: <https://arxiv.org/pdf/2605.18137>
- Project page: <https://JointWM.github.io>
