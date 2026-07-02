---
type: paper-analysis
title: "Depth Anything 3: Recovering the Visual Space from Any Views"
short_name: "Depth Anything 3"
year: 2025
venue: "arXiv"
arxiv: "2511.10647v1"
paper_url: "https://arxiv.org/abs/2511.10647"
pdf_url: "https://arxiv.org/pdf/2511.10647"
code: "https://github.com/ByteDance-Seed/Depth-Anything-3"
github: "https://github.com/ByteDance-Seed/Depth-Anything-3"
project_page: "https://depth-anything-3.github.io/"
open_source: true
license: "Code Apache-2.0; weights are mixed: DA3-BASE/SMALL/METRIC/MONO Apache-2.0, DA3-GIANT/LARGE/NESTED CC BY-NC 4.0"
training_open_source: "\\"
direction: [3d-reconstruction, dense-vision, robotics-autonomous-driving, evaluation-benchmarks]
method_family: [any-view-visual-geometry-foundation-model, depth-ray-representation, feed-forward-3d-reconstruction, feed-forward-3dgs, monocular-depth]
tasks: [camera-pose-estimation, multi-view-depth-estimation, pose-conditioned-depth-estimation, monocular-depth-estimation, metric-depth-estimation, point-cloud-reconstruction, feed-forward-novel-view-synthesis, feed-forward-3d-gaussian-estimation]
datasets: [HiRoom, ETH3D, DTU, 7Scenes, ScanNet++, DL3DV, Tanks-and-Temples, MegaDepth, KITTI, NYU, SINTEL, DIODE, SUN-RGBD]
metrics: [AUC3, AUC30, F1, Chamfer-Distance, delta1, AbsRel, SqRel, PSNR, SSIM, LPIPS, FPS]
status: compared
reproduction: planned
confidence: high
org: [ByteDance]
key_idea: "用单个预训练 plain transformer + depth-ray 双目标，把任意视角视觉几何压成两张 dense map"
builds_on: [2024-depth-anything-v2, 2023-dinov2, 2024-vggt, 2024-dust3r]
updated: 2026-07-02
---

# Depth Anything 3：Recovering the Visual Space from Any Views

## 结论先行

- **一句话定位**：Depth Anything 3（DA3）把 Depth Anything 系列从单目深度扩展成任意视角视觉几何基础模型；输入可以是一张图、多张图或视频帧，并可选输入相机位姿，输出一致的 depth、ray/camera、点云和 3DGS/NVS 相关结果。
- **核心方法**：论文的关键是“少做任务头，做好几何表示”：用单个 DINOv2 风格 plain transformer + input-adaptive cross-view attention + dual-DPT head，同时预测 depth 与 per-pixel ray map；ray map 隐式表达 camera pose，避免显式旋转矩阵约束和多任务冗余。两张 map 逐像素组合即得世界坐标点云。
- **实验结论**：论文报告 DA3-Giant 在 5 个 visual geometry benchmark 上多数 pose / reconstruction 指标超过 VGGT、Pi3、MapAnything、Fast3R、DUSt3R；DA3-Large 仅约 0.36B 参数，也能在多项指标超过 1.19B 的 VGGT。关于相对 VGGT 的平均提升，摘要存在两个版本：arXiv 摘要页（/abs）写 pose 平均提升 44.3%、geometric 25.1%，而 HTML/PDF 摘要写 35.7% / 23.6%；正文（Table 3 附近）明确写 reconstruction 平均相对 VGGT 提升 25.1%、相对 Pi3 提升 21.5%。本笔记以正文可核对的 25.1% 作为 reconstruction 主数字，pose 平均提升因两版本不一致标为“待核验”。
- **开源状态**：GitHub 仓库公开，代码 Apache-2.0，提供 Python package、CLI/API、Gradio、benchmark evaluator、DA3-Streaming 推理管线和 HuggingFace 权重；但仓库未发现训练脚本 / 训练 recipe，因此按本仓库约定 `training_open_source` 记为 `\`。
- **复现优先级**：值得先做 inference + benchmark sanity check，尤其是 DA3-LARGE/BASE 与 VGGT/MapAnything 在本地多视图 / 行车片段上的 pose/depth/point-cloud 质量；不建议把“完整训练复现”作为近期目标，因为训练 DA3-Giant 需 128 H100 约 10 天，且训练代码未公开。
- **工程风险**：大模型权重许可证并不都宽松，Giant/Large/Nested 系列权重为 CC BY-NC 4.0；只有 Base/Small/Metric/Mono 权重在 README model card 中标为 Apache 2.0。商业或闭源使用不能只看代码许可证。

## 1. 这篇论文解决什么问题？

### 已确认的论文事实

- **问题定义**：从任意数量视觉输入中恢复一致的 3D visual space，覆盖单图、多视图集合、视频输入，并支持已知或未知相机位姿。
- **输入 / 输出**：输入为 $N$ 张 RGB 图像，可选相机内参 / 外参；输出为每张图的 dense depth map、dense ray map，并可从 ray map 推出 camera pose，进一步融合为 point cloud 或用于 feed-forward 3D Gaussian / novel view synthesis。
- **目标场景**：通用 3D perception、robotics、mixed reality、visual geometry estimation、monocular / multi-view depth、pose estimation、feed-forward NVS。
- **与现有方法差异**：VGGT / DUSt3R 系方法常用 point map、显式 camera head 或多阶段 / 冗余任务组合；DA3 主张用 depth + ray 作为最小充分目标，并尽量复用完整预训练 plain transformer。

### 我的理解

DA3 的价值不在于“发明了一个复杂 3D 模块”，而是把多视图几何问题压回到两个 dense prediction target：

- depth 决定每个像素沿相机射线走多远；
- ray map 决定每个像素在世界坐标系里的射线原点和方向；
- 两者逐像素组合即可得到世界坐标点。

这样做的好处是任务接口统一：单目、多视图、posed / unposed、点云融合、3DGS 都可以从同一套几何输出延展，避免了 VGGT 那种“depth head + point head + camera head 各自训练、彼此冗余”的结构负担。

## 2. 方法概览

- **核心想法**：不堆专用几何模块，而是把“任意视角几何恢复”重新表述成两张逐像素 dense map（depth + ray）的联合回归，用一个完整预训练的 plain ViT 作为骨干，让预训练视觉先验直接迁移到几何任务。
- **一句话 pipeline**： $N$ 张图 → patch embed → 单 plain transformer（层间交替 within-view / cross-view self-attention）→ dual-DPT head → per-view depth map + 半分辨率 ray map → 逐像素 unproject 成统一世界坐标点云 / camera pose / 3DGS。

### 2.1 架构解析

![Depth Anything 3 整体架构与数据流（来源：arXiv 2511.10647, Depth Anything 3, Fig. 2 pipeline）](../../assets/3d-reconstruction/2025-depth-anything-3/pipeline.png)

**模块分解与数据流**（自左向右）：

| 模块 | 作用 | 论文中的关键信息 |
|---|---|---|
| Image Patch Embed | 把 $N$ 张 RGB 切成 patch token | 所有视角共享同一 embed；不同视角 token 拼在同一序列里 |
| Plain ViT backbone（Vanilla DINO） | 复用预训练视觉特征做几何推理 | 单个 DINOv2 风格 ViT；避免 VGGT 式多 transformer 堆叠 |
| Input-adaptive cross-view attention | 任意视角数量的信息交换 | 前 $L_s$ 层做 within-view attention，后续 $L_g$ 层交替 cross-view / within-view，默认比例 $L_s : L_g = 2 : 1$ ，靠 token rearrangement 实现，无需新增参数 |
| Camera condition token | 兼容 posed / unposed 输入 | 有相机参数时由编码器 $\mathcal{E}_c$ （MLP）编码成 camera token 加到序列；无参数时用 learnable token |
| Dual-DPT head | 联合输出 depth 和 ray | 共享 reassembly modules，depth / ray 分支用不同 fusion layers；depth 出 $N \times H \times W$ ，ray 出半分辨率 $N \times \tfrac{1}{2}H \times \tfrac{1}{2}W \times 6$ |
| Optional camera head $\mathcal{D}_c$ | 快速输出 camera params | 从 camera token 解码 FOV、quaternion、translation（图中右下角相机 frustum）；论文称额外计算约为主 backbone 的 0.1% |

**关键设计选择及理由**：

- **单 plain transformer 而非多分支**：DA3 把 cross-view 交互塞进 backbone 内部的 attention（靠 token 重排），而不是像 VGGT 那样外挂 alternating-attention 模块。这样 backbone 可以整段加载 DINOv2 预训练权重，几何能力“长”在预训练视觉表示上。
- **depth + ray 双输出而非 point map**：point map 把“相机几何”和“场景深度”耦合进一个 3D 向量里，梯度互相干扰；depth + ray 把两者解耦——ray 只负责相机侧（原点 + 方向），depth 只负责沿射线的距离，几何含义清晰且可分别监督。
- **ray map 半分辨率**：相机几何在空间上是低频信号（同一相机的射线场平滑），半分辨率足够表达且省算力，而 depth 保留全分辨率以保住细节。

### 2.2 核心原理

- **为什么这样设计 work**：论文实验（Table 7）显示，用同参数量的 VGGT-style 多阶段架构替换 DA3 的单 plain transformer，性能掉到约 79.8%。这说明在 any-view geometry 上，**预训练完整性比架构复杂度更重要**——一个能整段继承 DINOv2 权重的朴素 ViT，比把预训练破坏掉再堆更多未初始化 block 更强。
- **关键机制 / 归纳偏置**：
  - *Input-adaptive attention* 让同一套权重自然处理 1 到 N 张图：单图时 cross-view 退化为 within-view，多图时通过 token 重排实现视角间信息流，无需为不同视角数训练不同模型。
  - *depth-ray 分解* 是核心归纳偏置：它假设“世界点 = 相机射线 + 沿射线深度”，把 9-DoF 全局相机位姿分摊成每像素对齐的 dense ray，位姿估计因此变成一个良态的 dense regression，而非病态的全局参数拟合。
- **与前作的本质区别**：DUSt3R / VGGT 输出 point map 并显式回归 camera；DA3 不直接回归全局位姿，而是回归 ray map，再由 ray map 通过闭式几何（求相机中心、homography、RQ 分解）反推 intrinsics/rotation。位姿从“网络硬记的参数”变成“从 dense 几何解出来的量”，泛化更稳。

### 2.3 关键公式解析

**公式 (1)：经典逐像素 unproject（相机系 → 世界系）**

$$P = R_i \left( D_i(u,v) \cdot K_i^{-1} p \right) + t_i$$

- 符号： $p$ 为像素齐次坐标 $(u,v,1)$ ； $K_i^{-1}$ 把像素反投影到归一化相机射线； $D_i(u,v)$ 为该像素深度； $R_i, t_i$ 为第 $i$ 张图的旋转与平移。
- 作用：这是传统做法——要显式知道 $K_i, R_i, t_i$ 才能拿到世界点 $P$ ，位姿是显式变量。DA3 想绕开它。

**公式 (2)：depth-ray 表示（DA3 的核心简化）**

$$P = t + D(u,v) \cdot d$$

- 符号： $t \in \mathbb{R}^3$ 为该像素射线的原点（相机中心）， $d \in \mathbb{R}^3$ 为世界系下的归一化射线方向，二者即 ray map 的 6 个通道； $D(u,v)$ 仍是标量深度。
- 作用：把公式 (1) 里的 $R_i, K_i^{-1} p$ 全部吸收进 per-pixel 的 $(t, d)$ 。网络不再需要输出全局 $R_i, t_i$ ，只要逐像素预测 $(t, d, D)$ ，世界点就是一次逐元素运算。这是 DA3 统一各任务接口的关键。（已按 HTML 正文核对：world point 表达式为 $P = t + D(u,v) \cdot d$ 。）

**公式 (3)：从 ray map 解 homography（闭式恢复相机参数）**

$$H^\ast = \arg\min_{\lVert H \rVert = 1} \sum_{h,w} \left\lVert H\, p_{h,w} \times M(h,w,3{:}) \right\rVert$$

- 符号： $M$ 为预测的 ray map， $M(h,w,3{:})$ 取该像素的方向分量 $d$ ； $p_{h,w}$ 为像素坐标； $H$ 为待求单应，约束 $\lVert H \rVert = 1$ 避免平凡解； $\times$ 为叉积（方向对齐残差）。
- 作用：用 Direct Linear Transform 从整幅 ray map 最小二乘解出 $H = KR$ ，再对 $H$ 做 RQ 分解得到 intrinsics 与 rotation。这让“没有 camera head 也能拿到位姿”，camera head 只是加速推理的旁路。

**公式 (4)：训练总损失（概念形式）**

$$\mathcal{L} = \mathcal{L}_{\text{depth}} + \lambda_r \mathcal{L}_{\text{ray}} + \lambda_p \mathcal{L}_{\text{point}} + \lambda_g \mathcal{L}_{\text{grad}} + \lambda_c \mathcal{L}_{\text{cam}}$$

- 符号： $\mathcal{L}_{\text{depth}}$ 为带 confidence 加权的深度 L1； $\mathcal{L}_{\text{ray}}$ 为 ray map 回归 L1； $\mathcal{L}_{\text{point}}$ 为按公式 (2) 组合出的 3D 点重投影一致性 L1； $\mathcal{L}_{\text{grad}}$ 为水平 / 垂直有限差分梯度正则（保边缘）； $\mathcal{L}_{\text{cam}}$ 为可选的 quaternion + translation 位姿损失。
- 作用：depth / ray / point 三项互相约束——point loss 强迫 depth 与 ray 的组合在 3D 里自洽，而不是各自独立收敛，这是 depth-ray 双目标“对齐”的直接监督来源。（各项权重 $\lambda$ 的具体数值论文正文未给全，此处以符号占位，待核验。）

### 2.4 训练与推理细节

已确认的论文事实：

- **规模与算力**：DA3-Giant 使用 128 H100 GPU、200k steps（含 8k warmup）、约 10 天；peak lr $2 \times 10^{-4}$ 。
- **多分辨率**：基础分辨率 504，训练混合 504×504、504×378、504×336、504×280、336×504、896×504、756×504、672×504 等长宽比；batch size 动态调整以维持 token 数恒定。
- **视角采样**：504×504 时每图 view 数从 $[2, 18]$ 均匀采样，让模型天然适配任意视角数。
- **teacher-student**：真实 depth 常稀疏 / 有噪；先训一个只用合成数据（Hypersim、TartanAir、IRS、vKITTI2、BlendedMVS 等）的单目 relative-depth teacher，teacher 用 exponential depth（而非 disparity）以增强近处判别，并带 distance-weighted surface-normal loss、sky/object mask、global-local ROE 对齐；再用 RANSAC scale-shift 把 teacher 伪深度对齐到真实稀疏 / 噪声测量。
- **监督切换**：训练前 120k steps 用 ground-truth depth 监督，之后切换到 teacher 伪标签；pose conditioning 以 0.2 概率随机启用，使模型既能 posed 也能 unposed 推理。
- **数据来源**：700k+ 场景，混合 synthetic、LiDAR、3D reconstruction、COLMAP 多来源（含 Objaverse ~505k、Trellis ~557k、AriaSyntheticENV ~99k 等）。

**推理流程**：图像 patch embed → plain transformer（交替 attention）→ dual-DPT 出 depth + ray → 若需位姿，走 camera head 或从 ray map 闭式解 → 逐像素 unproject 融合成点云；接 GS-DPT head 可 feed-forward 出 3D Gaussian 做 NVS。

我的判断：

- 这套 recipe 的关键不是“数据全公开就容易复现”，而是 pseudo-label 生成与数据清洗很关键。论文附录披露了部分合成数据清洗规则，但公开仓库没有训练脚本，无法确认完整训练 pipeline。
- teacher-student 解决的是 real-world depth quality 问题：用 synthetic teacher 保留细节，再用真实 sparse/noisy depth 对齐尺度和几何。

## 3. 关键贡献

1. **depth + ray 最小几何目标**：相比 depth + point cloud + camera 或 depth + camera，depth + ray 在多数据集 pose/reconstruction ablation 中更强，且概念更统一。
2. **plain transformer 作为 any-view geometry backbone**：证明不一定需要 VGGT 那样的多阶段专用架构；完整预训练的单 transformer 比同参数量的 VGGT-style 堆叠更有效（Table 7 掉到 79.8%）。
3. **统一覆盖单目、多视图、posed/unposed、3DGS/NVS**：主模型支持多种视觉几何任务，额外 GS-DPT head 可用于 feed-forward 3DGS。
4. **Visual Geometry Benchmark**：提出覆盖 pose、reconstruction、visual rendering 的 benchmark；仓库已发布 evaluator 和 DA3-BENCH 预处理数据。
5. **完整工程接口**：公开 repo 提供 package、CLI、API、Gradio、benchmark evaluator、多格式导出和 DA3-Streaming 推理管线，适合做 inference-level 复现。

## 4. 实验与证据

| 维度 | 内容 |
|---|---|
| Pose / geometry benchmark | HiRoom、ETH3D、DTU、7Scenes、ScanNet++；覆盖 object、indoor、outdoor 场景 |
| NVS benchmark | DL3DV 140 scenes、Tanks and Temples 6 scenes、MegaDepth 19 scenes；每场景约 300 sampled frames，12 context views |
| Baseline | DUSt3R、Fast3R、MapAnything、Pi3、VGGT；NVS 对比 pixelSplat、MVSplat、DepthSplat，以及替换 Fast3R / MV-DUSt3R / VGGT / DA3 backbone 的统一 3DGS 框架 |
| Pose 指标 | AUC@3、AUC@30，基于 relative rotation / translation accuracy |
| Reconstruction 指标 | F1-score；DTU 用 Chamfer Distance / Overall |
| Monocular depth 指标 | delta1、AbsRel、SqRel |
| NVS 指标 | PSNR、SSIM、LPIPS |
| Efficiency | 最大输入图片数、参数量、A100 FPS |

### 4.1 效果与性能解析

![点云重建质量对比（来源：arXiv 2511.10647, Depth Anything 3, Fig. 5）](../../assets/3d-reconstruction/2025-depth-anything-3/results-pointcloud.png)

| 结果 | 论文证据 | 解读 |
|---|---|---|
| DA3-Giant pose 几乎全面领先 | Table 2 中 DA3-Giant 在 HiRoom/ETH3D/DTU/7Scenes/ScanNet++ 的 AUC3/AUC30 多数第一；正文称 AUC3 相对所有对手至少 8% 相对提升，ScanNet++ 相对次优约 33% 相对增益 | pose-free camera estimation 是 DA3 强项，尤其复杂室内 ScanNet++ |
| reconstruction 全面强于 VGGT/Pi3 | Table 3 中 DA3-Giant 在 5 个 pose-free 设置全领先；正文称平均相对 VGGT 提升 25.1%、相对 Pi3 提升 21.5% | depth+ray 输出能直接转成更干净点云（Fig. 5 目视也更完整少飘点） |
| DA3-Large 性价比高 | 约 0.36B 参数，论文称在 10 个 reconstruction settings 中 5 个超过 1.19B VGGT | 工程上优先试 DA3-LARGE/BASE，而非直接上 Giant |
| 单目深度超过 DA2/VGGT | Table 4 中 DA3 rank 2.20、DA2 2.60、VGGT 3.75；teacher rank 1.00 | 多视图能力没有牺牲单目 depth，仍继承 Depth Anything 系优势 |
| NVS backbone 最强 | Table 5 中 DAv3 作 3DGS backbone 在 DL3DV/T&T/MegaDepth 的 PSNR/SSIM/LPIPS 均优于 VGGT backbone | 几何 backbone 质量直接转化为 feed-forward rendering 质量 |
| depth + ray ablation 强 | Table 6 中明显强于 depth + cam、depth + pcd + cam；正文称 AUC3 相对 depth + cam 接近翻倍 | depth-ray 是核心方法，不是装饰性 head |
| 单 plain transformer > VGGT-style | Table 7 中 VGGT-style 同参数掉到约 79.8% | 预训练完整性比堆更多未预训练 block 更重要 |

**速度 / 显存 / 参数量**（Table 8，A100）：DA3-Giant 单卡最多约 900–1000 张图、37.6 FPS；DA3-Large 约 1500–1600 张图、78.37 FPS；VGGT reference 约 400–500 张图、34.1 FPS。DA3 在“最大可处理视角数”和吞吐上都优于 VGGT，说明单 plain transformer + 半分辨率 ray 不仅精度好，显存和长序列扩展性也更友好。（表内具体数值来自 WebFetch 抽取，若做正式引用建议复跑或比对 PDF 表格。）

**消融揭示的关键因素**：

- **Depth-ray representation**：depth + ray 是最小充分组合；额外加显式 cam head 主要是推理便利，精度不一定继续提升。
- **Dual-DPT head**：两个完全独立 DPT head 会降指标；共享 reassembly 再分支融合更好，印证 depth 与 ray 需要“相互对齐”而非各管各的。
- **Teacher labels**：对 HiRoom、7Scenes、ScanNet++ 等细节 / 真实场景有明显收益；DTU 上去掉 teacher 有时略好，说明 teacher supervision 并非对所有数据集单调改善。
- **Pose conditioning**：有 GT pose 时，pose condition 明显提升 posed reconstruction。
- **Data / teacher design**：V3 datasets + 多分辨率策略、depth target、完整 teacher loss 均优于替代设置。

**可比性与协议一致性**：DA3 自建 Visual Geometry Benchmark 并统一了各 baseline 的评测协议（同一批 sampled frames / context views、同一 evaluator），横向数字可比性较好；但 NVS 部分是把不同 backbone 接进同一 3DGS 框架比较，衡量的是“几何 backbone 对 rendering 的贡献”而非各方法原生 NVS 能力，解读时需注意这一口径差异。

![深度图细节质量对比（来源：arXiv 2511.10647, Depth Anything 3, Fig. 6）](../../assets/3d-reconstruction/2025-depth-anything-3/results-depth.png)

## 5. 局限与风险

### 论文明确承认或可见的限制

- 未来工作包括 dynamic scenes、language / interaction cues、更大规模预训练；说明当前模型主要还是静态 / 准静态视觉几何。
- Feed-forward NVS 仍需额外 GS-DPT head fine-tune；不是通用 4D 动态世界模型。
- 完整 DA3-Giant 训练成本很高：128 H100 约 10 天。

### 已确认的代码 / 仓库事实

- GitHub repo 公开，代码 Apache-2.0，默认分支 `main`。
- 仓库包含 `src/depth_anything_3/model/`、`api.py`、CLI、Gradio app、benchmark evaluator、docs、configs、DA3-Streaming。
- 仓库含 `src/depth_anything_3/bench/evaluator.py` 与 `docs/BENCHMARK.md`，可下载 `depth-anything/DA3-BENCH` 跑 pose/reconstruction evaluation。
- 仓库未发现 `train.py`、training scripts、optimizer/scheduler recipe 或 dataset preprocessing pipeline；公开内容主要是 inference、API/CLI、benchmark evaluation、model definitions 和 streaming inference。

### 我推断的风险

- **训练不可复刻风险高**：即使 raw datasets 全是公开 academic datasets，缺训练脚本、teacher pseudo-label 生成代码和完整清洗流程，论文主结果无法端到端复现。
- **许可证风险**：代码 Apache-2.0 不等于所有权重可商用；Giant/Large/Nested 权重为 CC BY-NC 4.0，商业闭环应优先看 DA3-BASE/SMALL/METRIC/MONO 或重新训练。
- **动态场景风险**：自动驾驶 / 机器人场景里移动物体、遮挡、rolling shutter、曝光变化会影响点云和 pose；需要 mask、dynamic filtering 或后端优化。
- **metric scale 风险**：DA3Nested/Metric 提供 metric 路径，但 image-only 任意视角几何仍不应直接替代带 LiDAR/IMU/pose 的 metric pipeline。
- **benchmark 依赖重**：DA3-BENCH 数据约 40GB 量级，full evaluation 需多 GPU / 较大显存才合理。

### Unknowns / to verify

- 摘要 pose 平均提升数字在两个官方来源不一致：arXiv 摘要页（/abs）显示 44.3%（geometric 25.1%），HTML/PDF 摘要显示 35.7%（geometric 23.6%）；正文可核对的 reconstruction 平均提升为 25.1%。pose 的“平均”口径待官方勘误或复跑确认，本笔记暂以正文数字为准。
- 损失函数 (4) 是概念形式：各项存在（depth 带 confidence、ray、point、grad、cam），但各项权重 $\lambda$ 的具体数值论文正文未给全，标为待核验。
- 公式 (1)(2)(3) 的确切写法已核对 HTML 正文与图注（ $P = t + D(u,v) \cdot d$ 、homography DLT、经典 unproject），但未逐字节比对 PDF LaTeX 源，个别下标 / 归一化约束的精确记法可能与原文略有出入。
- DA3-Streaming 是仓库后续发布的长视频推理管线，基于 VGGT-Long 思路；不是论文主训练方法的一部分，需单独做工程验证。
- HuggingFace 上部分模型卡带 `-1.1` 后缀，README 称修复 training bug 后应优先使用；这些权重对应的论文表格数值是否完全一致，需官方说明或复跑确认。
- 方法谱系里引用的前作 slug（2023-dinov2 / 2024-depth-anything-v2 / 2024-vggt / 2024-dust3r）按常见命名假定，未逐一 grep 校验本仓库实际文件名是否完全一致，以 `indices/methods.md` 为准。

## 方法谱系

- 基于（backbone / 表示先验）：[DINOv2](../3d-reconstruction/2023-dinov2.md)（plain ViT 预训练权重直接迁移）、[Depth Anything 2](../3d-reconstruction/2024-depth-anything-v2.md)（单目 depth 泛化与 teacher 思路）。
- 对标 / 改进（同代 any-view geometry）：[VGGT](../3d-reconstruction/2024-vggt.md)（DA3 用单 plain transformer + depth-ray 取代其多阶段架构与 point map）、[DUSt3R](../3d-reconstruction/2024-dust3r.md)（DA3 用 ray map 替代 point map 的显式位姿回归）。

> 谱系链接为方向内相对路径，若目标文件 slug 与此处不一致以 `indices/methods.md` 为准。

## 6. 与相似方法对比

| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| VGGT | 都是任意视角视觉几何 foundation model，输出 pose/depth/3D geometry | DA3 用单 plain transformer + depth-ray；VGGT 架构更复杂、输出 point map，DA3 在多项 pose/reconstruction 指标超过它 | 需要成熟 baseline 或复现 VGGT 生态选 VGGT；追求更强 any-view geometry 和更简接口优先 DA3 |
| Pi3 | 都做 feed-forward visual geometry，支持无序 / 多视图几何 | Pi3 更强调 permutation-equivariant、affine/scale-invariant camera；DA3 更强调 depth-ray 和预训练 DINO backbone | 研究 unordered/scale-invariant formulation 选 Pi3；要工程可用模型和 benchmark pipeline 选 DA3 |
| MapAnything | 都是 feed-forward 多视图几何模型，可利用相机 /pose/depth 类信息 | MapAnything 更强调 metric reconstruction 和任意几何 constraints/prompt；DA3 更强调最小 depth-ray target 和 Depth Anything 系 depth 泛化 | 自动驾驶已有标定 /pose/LiDAR depth prompt 时 MapAnything 更像主 metric backbone；纯视觉 any-view geometry / 3DGS/NVS backbone 优先 DA3 |
| LingBot-Map | 都输出 pose/depth/point cloud，服务机器人 / 自动驾驶视觉建图 | LingBot-Map 是 causal streaming 架构；DA3 本体是任意视角全局 feed-forward，repo 另有 DA3-Streaming 推理管线 | 在线长视频 VO / 建图优先 LingBot-Map 或 DA3-Streaming；离线多视图 / posed-unposed 统一推理优先 DA3 |
| Depth Anything 2 | 都来自 Depth Anything 系列，单目 depth 泛化强 | DA2 主要是单目相对 / metric depth；DA3 扩到多视图、pose、ray、3DGS | 只做单图 depth 且需轻量生态用 DA2；需要多视图一致性和相机 / 点云输出选 DA3 |

更详细的同类横向对比见：[`../../comparisons/3d-reconstruction/visual-geometry-foundation-models.md`](../../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)。

## 7. 复现判断

- Git 地址：<https://github.com/ByteDance-Seed/Depth-Anything-3>
- 是否开源：是。代码仓库公开，GitHub 页面与 `pyproject.toml` 均显示 Apache-2.0。
- 是否开源训练：`\`。公开仓库主要提供 inference/API/CLI/app/benchmark/model code，未发现训练代码。
- 权重可用性：HuggingFace model zoo 提供 DA3NESTED-GIANT-LARGE、DA3-GIANT、DA3-LARGE、DA3-BASE、DA3-SMALL、DA3METRIC-LARGE、DA3MONO-LARGE 等；README 建议优先用 `-1.1` refreshed checkpoints（对 Giant/Large/Nested）。
- 权重许可证：Giant/Large/Nested 为 CC BY-NC 4.0；Base/Small/Metric/Mono 为 Apache 2.0。
- 数据可获得性：论文称训练使用公开 academic datasets；仓库发布 processed benchmark `depth-anything/DA3-BENCH`，但训练数据清洗 / teacher label 生成 pipeline 未开源。
- 预计环境成本：基础推理需 Python 3.9–3.13、PyTorch >=2、xformers、Open3D、pycolmap 等；3DGS 需 `gsplat`；benchmark 数据约 40GB，full eval 建议多 GPU。
- 最小复现路径：
  1. 安装 `pip install -e .`，先用 `DA3-BASE` 或 `DA3-LARGE-1.1` 跑 `assets/examples/SOH`。
  2. 导出 `glb` / `npz` / depth images，检查 pose、intrinsics、depth、confidence shape 和点云质量。
  3. 下载 `depth-anything/DA3-BENCH` 中的 HiRoom 或 7Scenes 子集，跑 `python -m depth_anything_3.bench.evaluator model.path=... eval.datasets=[hiroom] eval.modes=[pose]`。
  4. 与 VGGT/MapAnything 在同一小场景上做 qualitative + AUC/F1 sanity check。
- 是否值得复现：值得做 inference-level 和 benchmark-level 复现；完整训练暂不现实。

## 8. 后续动作

- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [x] 创建 `comparisons/3d-reconstruction/visual-geometry-foundation-models.md`
- [ ] 若后续做复现，创建 `reproductions/3d-reconstruction/depth-anything-3/README.md`

## Sources

- Paper: <https://arxiv.org/abs/2511.10647>
- HTML: <https://arxiv.org/html/2511.10647>
- PDF: <https://arxiv.org/pdf/2511.10647>
- Hugging Face paper metadata: <https://huggingface.co/papers/2511.10647>
- GitHub: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Project page: <https://depth-anything-3.github.io/>
- Benchmark dataset: <https://huggingface.co/datasets/depth-anything/DA3-BENCH>
