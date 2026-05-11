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
updated: 2026-05-09
---

# Depth Anything 3：Recovering the Visual Space from Any Views

## 结论先行

- **一句话定位**：Depth Anything 3（DA3）把 Depth Anything 系列从单目深度扩展成任意视角视觉几何基础模型；输入可以是一张图、多张图或视频帧，并可选输入相机位姿，输出一致的 depth、ray/camera、点云和 3DGS/NVS 相关结果。
- **核心方法**：论文的关键是“少做任务头，做好几何表示”：用单个 DINOv2 风格 plain transformer + input-adaptive cross-view attention + dual-DPT head，预测 depth + per-pixel ray map；ray map 隐式表达 camera pose，避免显式旋转矩阵约束和多任务冗余。
- **实验结论**：论文报告 DA3-Giant 在 5 个 visual geometry benchmark 上多数 pose/reconstruction 指标超过 VGGT、Pi3、MapAnything、Fast3R、DUSt3R；DA3-Large 仅约 0.36B 参数，也能在多项指标超过 1.19B VGGT。论文 HTML/PDF 摘要写相对 VGGT 平均提升约 35.7% pose accuracy、23.6% geometry accuracy；表格正文写 reconstruction 平均相对 VGGT 提升 25.1%。
- **开源状态**：GitHub 仓库公开，代码 Apache-2.0，提供 Python package、CLI/API、Gradio、benchmark evaluator、DA3-Streaming 推理管线和 HuggingFace 权重；但仓库未发现训练脚本/训练 recipe，因此按本仓库约定 `training_open_source` 记为 `\`。
- **复现优先级**：值得先做 inference + benchmark sanity check，尤其是 DA3-LARGE/BASE 与 VGGT/MapAnything 在本地多视图/行车片段上的 pose/depth/point-cloud 质量；不建议把“完整训练复现”作为近期目标，因为论文训练 DA3-Giant 需要 128 H100 约 10 天，且训练代码未公开。
- **工程风险**：大模型权重许可证并不都宽松，Giant/Large/Nested 系列权重为 CC BY-NC 4.0；只有 Base/Small/Metric/Mono 权重在 README model card 中标为 Apache 2.0。商业或闭源使用不能只看代码许可证。

## 1. 这篇论文解决什么问题？

### 已确认的论文事实

- **问题定义**：从任意数量视觉输入中恢复一致的 3D visual space，覆盖单图、多视图集合、视频输入，并支持已知或未知相机位姿。
- **输入 / 输出**：输入为 `N` 张 RGB 图像，可选相机内参/外参；输出为每张图的 dense depth map、dense ray map，并可从 ray map 推出 camera pose，进一步融合为 point cloud 或用于 feed-forward 3D Gaussian / novel view synthesis。
- **目标场景**：通用 3D perception、robotics、mixed reality、visual geometry estimation、monocular / multi-view depth、pose estimation、feed-forward NVS。
- **与现有方法差异**：VGGT/DUSt3R 系方法常用 point map、显式 camera head 或多阶段/冗余任务组合；DA3 主张用 depth + ray 作为最小充分目标，并尽量复用完整预训练 plain transformer。

### 我的理解

DA3 的价值不在于“发明了一个复杂 3D 模块”，而是把多视图几何问题压回到两个 dense prediction target：

- depth 决定每个像素沿相机射线走多远；
- ray map 决定每个像素的世界坐标射线原点和方向；
- 两者逐像素组合即可得到世界坐标点。

这样做的好处是任务接口统一：单目、多视图、posed/unposed、点云融合、3DGS 都可以从同一套几何输出延展。

## 2. 方法概览

### 2.1 Depth-ray representation

论文用每像素 ray map 表示相机几何。每个像素的 ray 有 6 个数：3D 原点 `t` 和 3D 方向 `d`。深度 `D(u, v)` 与 ray 组合后得到世界坐标点：

```text
P = t + D(u, v) * d
```

这种表示把 camera pose 从“一个全局 9-DoF 参数”变成“每像素都对齐的 dense ray prediction”。论文还说明可以通过 ray map 估计 camera center、homography，再经 RQ decomposition 得到 intrinsics/rotation。出于推理效率，最终模型仍加了轻量 camera head，但 ablation 显示核心收益来自 depth + ray。

### 2.2 网络结构

| 模块 | 作用 | 论文中的关键信息 |
|---|---|---|
| Plain ViT backbone | 复用预训练视觉特征 | 使用 DINOv2 风格 Vision Transformer；避免 VGGT 式多 transformer 堆叠 |
| Input-adaptive cross-view attention | 任意视角数量的信息交换 | 前面层做 per-image attention，后面层在 selected layers 交替做 cross-view / within-view attention，通过 token rearrangement 实现 |
| Camera condition token | 兼容 posed / unposed 输入 | 有相机参数时由 MLP 编码成 camera token；无相机参数时使用 learnable token |
| Dual-DPT head | 联合输出 depth 和 ray | 共享 reassembly modules，depth/ray 分支用不同 fusion layers，鼓励两个目标对齐但避免完全独立头的冗余 |
| Optional camera head | 快速输出 camera params | 从 camera tokens 预测 FOV、quaternion、translation；论文称额外计算成本约为主 backbone 的 0.1% |

### 2.3 训练策略

已确认的论文事实：

- DA3-Giant 训练使用 128 H100 GPUs、200k steps、约 10 天。
- 基础分辨率为 504，训练时混合多种长宽比；504x504 时 view 数从 `[2, 18]` 均匀采样。
- 训练数据来自 synthetic、LiDAR/depth capture、COLMAP/3D reconstruction 等多来源。
- 真实世界 depth 常有噪声或稀疏，作者先训练一个 synthetic-only monocular teacher，再把 teacher pseudo-depth 用 RANSAC scale-shift 对齐到真实稀疏/噪声 depth。
- 训练 120k steps 后 supervision 从 ground-truth depth 转为 teacher labels；pose conditioning 以 0.2 概率启用。

我的判断：

- 这套训练 recipe 的关键不是“数据全公开就容易复现”，而是 pseudo-label 和数据清洗很关键。论文附录披露了部分 synthetic 数据清洗规则，但公开仓库没有训练脚本，无法确认完整训练 pipeline。
- Teacher-student 解决的是 real-world depth quality 问题：用 synthetic teacher 保留细节，再用真实 sparse/noisy depth 对齐尺度和几何。

## 3. 关键贡献

1. **depth + ray 最小几何目标**：相比 depth + point cloud + camera 或 depth + camera，depth + ray 在多数据集 pose/reconstruction ablation 中更强，且概念更统一。
2. **plain transformer 作为 any-view geometry backbone**：证明不一定需要 VGGT 那样的多阶段专用架构；完整预训练的单 transformer 可能比同参数量的 VGGT-style 堆叠更有效。
3. **统一覆盖单目、多视图、posed/unposed、3DGS/NVS**：主模型支持多种视觉几何任务，额外 GS-DPT head 可用于 feed-forward 3DGS。
4. **Visual Geometry Benchmark**：提出覆盖 pose、reconstruction、visual rendering 的 benchmark；仓库已发布 evaluator 和 DA3-BENCH 预处理数据。
5. **完整工程接口**：公开 repo 提供 package、CLI、API、Gradio、benchmark evaluator、多格式导出和 DA3-Streaming 推理管线，适合做 inference-level 复现。

## 4. 实验与证据

| 维度 | 内容 |
|---|---|
| Pose / geometry benchmark | HiRoom、ETH3D、DTU、7Scenes、ScanNet++；共覆盖 object、indoor、outdoor 场景 |
| NVS benchmark | DL3DV 140 scenes、Tanks and Temples 6 scenes、MegaDepth 19 scenes；每个场景约 300 sampled frames，12 context views |
| Baseline | DUSt3R、Fast3R、MapAnything、Pi3、VGGT；NVS 对比 pixelSplat、MVSplat、DepthSplat，以及替换 Fast3R / MV-DUSt3R / VGGT / DA3 backbone 的统一 3DGS 框架 |
| Pose 指标 | AUC@3、AUC@30，基于 relative rotation / translation accuracy |
| Reconstruction 指标 | F1-score；DTU 用 Chamfer Distance / Overall |
| Monocular depth 指标 | delta1、AbsRel、SqRel |
| NVS 指标 | PSNR、SSIM、LPIPS |
| Efficiency | 最大输入图片数、参数量、A100 FPS |

### 4.1 论文报告的主要结果

| 结果 | 论文证据 | 解读 |
|---|---|---|
| DA3-Giant pose 几乎全面领先 | Table 2 中 DA3-Giant 在 HiRoom/ETH3D/DTU/7Scenes/ScanNet++ 的 AUC3/AUC30 多数第一；例如 ScanNet++ AUC3 85.0，高于 VGGT 62.6 | pose-free camera estimation 是 DA3 的强项，尤其复杂室内 ScanNet++ |
| DA3-Giant reconstruction 全面强于 VGGT/Pi3 | Table 3 中 DA3-Giant 在 pose-free reconstruction 5 个设置都领先；正文称平均相对 VGGT 提升 25.1%、相对 Pi3 提升 21.5% | depth+ray 输出能直接转化成更干净点云 |
| DA3-Large 有较好性价比 | DA3-Large 约 0.36B 参数，论文称在 10 个 reconstruction settings 中 5 个超过 1.19B VGGT | 工程上优先试 DA3-LARGE/BASE，而不是直接上 Giant |
| 单目深度超过 DA2/VGGT | Table 4 中 DA3 rank 2.20，DA2 2.60，VGGT 3.75；teacher rank 1.00 | DA3 不是只会多视图，单目 depth 也继承 Depth Anything 系优势 |
| NVS backbone 最强 | Table 5 中 DAv3 作为 3DGS backbone 在 DL3DV/T&T/MegaDepth 的 PSNR/SSIM/LPIPS 均优于 VGGT backbone | 几何 backbone 质量能直接转化为 feed-forward rendering 质量 |
| depth + ray ablation 强 | Table 6 中 depth + ray 明显强于 depth + cam、depth + pcd + cam；正文称 AUC3 相对 depth + cam 接近翻倍 | depth-ray 是核心方法，不是装饰性 head |
| 单 plain transformer 强于 VGGT-style | Table 7 中 VGGT-style 同参数 ablation 明显下降；正文称约为 proposed arch 的 79.8% | 预训练完整性比堆更多未预训练 block 更重要 |
| 推理效率不错 | Table 8 中 DA3-Giant 900-1000 images / 37.6 FPS；DA3-Large 1500-1600 images / 78.37 FPS；VGGT reference 400-500 images / 34.1 FPS | DA3 不只是精度强，长 view count 和 FPS 也有优势 |

### 4.2 消融结论

- **Depth-ray representation**：depth + ray 是最小充分组合；加显式 cam head 主要是推理便利，精度不一定继续提升。
- **Dual-DPT head**：两个完全独立 DPT heads 会降指标；共享 reassembly 再分支融合更好。
- **Teacher labels**：对 HiRoom、7Scenes、ScanNet++ 等细节/真实场景有明显收益；DTU 上去掉 teacher 有时略好，说明 teacher supervision 不是所有数据集都单调改善。
- **Pose conditioning**：有 GT pose 时，pose condition 明显提升 posed reconstruction。
- **Data scaling / teacher design**：V3 datasets + multi-resolution strategy、depth target、完整 teacher loss 都优于替代设置。

## 5. 局限与风险

### 论文明确承认或可见的限制

- 未来工作包括 dynamic scenes、language / interaction cues、更大规模预训练；这说明当前模型主要还是静态/准静态视觉几何。
- Feed-forward NVS 仍是通过额外 GS-DPT head fine-tune；不是通用 4D 动态世界模型。
- 完整 DA3-Giant 训练成本非常高：128 H100 约 10 天。

### 已确认的代码/仓库事实

- GitHub repo 公开，Apache-2.0，默认分支 `main`，我检查到 HEAD 为 `41736238f5bced4debf3f2a12375d2466874866d`。
- 仓库包含 `src/depth_anything_3/model/`、`api.py`、CLI、Gradio app、benchmark evaluator、docs、configs、DA3-Streaming。
- 仓库包含 `src/depth_anything_3/bench/evaluator.py` 和 `docs/BENCHMARK.md`，可下载 `depth-anything/DA3-BENCH` 跑 pose/reconstruction evaluation。
- 仓库未发现 `train.py`、training scripts、optimizer/scheduler recipe 或 dataset preprocessing pipeline；公开内容主要是 inference、API/CLI、benchmark evaluation、model definitions 和 streaming inference。

### 我的推断风险

- **训练不可复刻风险高**：即使所有 raw datasets 是公开 academic datasets，没有训练脚本、teacher pseudo-label 生成代码和完整清洗流程，论文主结果无法端到端复现。
- **许可证风险**：代码 Apache-2.0 不等于所有权重可商用；Giant/Large/Nested 权重为 CC BY-NC 4.0，工程方案若要商业闭环应优先看 DA3-BASE/SMALL/METRIC/MONO 或重新训练。
- **动态场景风险**：自动驾驶/机器人场景里移动物体、遮挡、rolling shutter、曝光变化会影响点云和 pose；需要 mask、dynamic filtering 或后端优化。
- **metric scale 风险**：DA3Nested/Metric 提供 metric 路径，但 image-only 任意视角几何仍不应直接替代带 LiDAR/IMU/pose 的 metric pipeline。
- **benchmark 依赖重**：DA3-BENCH 数据约 40GB 量级，full evaluation 需要多 GPU/较大显存才合理。

### Unknowns / to verify

- arXiv API summary 与 HTML/PDF 可见摘要对 VGGT pose 平均提升数字不一致：API summary 显示 44.3%，HTML/PDF 文本显示 35.7%；本笔记正文采用论文 HTML/PDF 和表格正文可见数字。
- DA3-Streaming 是仓库后续发布的长视频推理管线，基于 VGGT-Long 思路；它不是论文主训练方法的一部分，需单独做工程验证。
- HuggingFace 上部分模型卡带 `-1.1` 后缀，README 称修复 training bug 后应优先使用；这些权重对应的论文表格数值是否完全一致需要官方说明或复跑确认。

## 6. 与相似方法对比

| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| VGGT | 都是任意视角视觉几何 foundation model，输出 pose/depth/3D geometry | DA3 用单 plain transformer + depth-ray；VGGT 架构更复杂，论文中 DA3 在多项 pose/reconstruction 指标超过 VGGT | 需要成熟 baseline 或复现 VGGT 生态时选 VGGT；追求更强 any-view geometry 和更简接口时优先 DA3 |
| Pi3 | 都做 feed-forward visual geometry，支持无序/多视图几何 | Pi3 更强调 permutation-equivariant、affine/scale-invariant camera；DA3 更强调 depth-ray 和预训练 DINO backbone | 要研究 unordered/scale-invariant formulation 选 Pi3；要工程可用模型和 benchmark pipeline 选 DA3 |
| MapAnything | 都是 feed-forward 多视图几何模型，并可利用相机/pose/depth 类信息 | MapAnything 更强调 metric reconstruction 和任意几何 constraints/prompt；DA3 更强调最小 depth-ray target 和 Depth Anything 系 depth generalization | 自动驾驶已有标定/pose/LiDAR depth prompt 时 MapAnything 更像主 metric backbone；纯视觉 any-view geometry / 3DGS/NVS backbone 可优先 DA3 |
| LingBot-Map | 都输出 pose/depth/point cloud，服务机器人/自动驾驶视觉建图 | LingBot-Map 是 causal streaming 架构；DA3 本体是任意视角全局 feed-forward，repo 另有 DA3-Streaming 推理管线 | 在线长视频 VO/建图优先 LingBot-Map 或 DA3-Streaming；离线多视图/posed-unposed 统一推理优先 DA3 |
| Depth Anything 2 | 都来自 Depth Anything 系列，单目 depth 泛化强 | DA2 主要是单目相对/metric depth；DA3 扩到多视图、pose、ray、3DGS | 只做单图 depth 且需要轻量生态可用 DA2；需要多视图一致性和相机/点云输出选 DA3 |

更详细的同类横向对比见：[`../../comparisons/3d-reconstruction/visual-geometry-foundation-models.md`](../../comparisons/3d-reconstruction/visual-geometry-foundation-models.md)。

## 7. 复现判断

- Git 地址：<https://github.com/ByteDance-Seed/Depth-Anything-3>
- 是否开源：是。代码仓库公开，GitHub API 和 `pyproject.toml` 均显示 Apache-2.0。
- 是否开源训练：`\`。公开仓库主要提供 inference/API/CLI/app/benchmark/model code，未发现训练代码。
- 权重可用性：HuggingFace model zoo 提供 DA3NESTED-GIANT-LARGE、DA3-GIANT、DA3-LARGE、DA3-BASE、DA3-SMALL、DA3METRIC-LARGE、DA3MONO-LARGE 等；README 建议优先用 `-1.1` refreshed checkpoints（对 Giant/Large/Nested）。
- 权重许可证：Giant/Large/Nested 为 CC BY-NC 4.0；Base/Small/Metric/Mono 为 Apache 2.0。
- 数据可获得性：论文称训练使用公开 academic datasets；仓库发布 processed benchmark `depth-anything/DA3-BENCH`，但训练数据清洗/teacher label 生成 pipeline 未开源。
- 预计环境成本：基础推理需 Python 3.9-3.13、PyTorch >=2、xformers、Open3D、pycolmap 等；3DGS 需 `gsplat`；benchmark 数据约 40GB，full eval 建议多 GPU。
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
- PDF: <https://arxiv.org/pdf/2511.10647>
- Hugging Face paper metadata: <https://huggingface.co/papers/2511.10647>
- GitHub: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Project page: <https://depth-anything-3.github.io/>
- Benchmark dataset: <https://huggingface.co/datasets/depth-anything/DA3-BENCH>
