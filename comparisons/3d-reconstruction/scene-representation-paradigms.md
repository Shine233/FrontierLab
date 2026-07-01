---
type: method-comparison
title: "场景表示范式对比：经典几何 / 神经渲染 / 显式高斯"
direction: [3d-reconstruction, novel-view-synthesis, dense-vision]
methods: [COLMAP, NeRF, 3DGS]
status: initial-completed
confidence: medium
updated: 2026-07-01
---

# 场景表示范式对比：COLMAP / NeRF / 3D Gaussian Splatting

> 证据范围：以三者原论文与官方仓库为主（见各自分析文件）。三者不是同类可换算的方法，而是**从图像到三维/新视角**这条链上三种互补范式；本对比讲清各自定位、依赖关系与选型。

## 结论先行

| 推荐定位 | Method | 主要理由 | 主要风险 |
|---|---|---|---|
| **几何/位姿事实标准** | **COLMAP** | 无序图像 → 相机位姿 + 稀疏/稠密点云的经典 SfM+MVS 管线；至今是 NeRF/3DGS 的输入与 benchmark 真值来源 | 迭代优化慢、对弱纹理/重复纹理敏感、无学习先验 |
| **神经隐式表示分水岭** | **NeRF** | 坐标 MLP + 体渲染，连续隐式场，新视角合成质量里程碑；催生整条 neural rendering 谱系 | 训练/渲染慢（原版每场景数小时、渲染非实时）；需已知相机（通常来自 COLMAP） |
| **实时显式表示主流** | **3DGS** | 各向异性 3D 高斯 + 可微分光栅化，1080p 实时（论文实测 130-200 FPS）高质量 NVS；训练分钟级 | 显存/存储大、几何非表面精确、动态/稀疏视图需扩展；权重许可非商用 |

## 1. 三者定位（不是替代，是链条）

- **COLMAP**（2016，经典几何）：从图像**恢复相机与稀疏/稠密几何**。它是上游——NeRF 与 3DGS 训练前几乎都先用 COLMAP 估相机位姿和点云初始化。
- **NeRF**（2020，神经隐式）：给定 posed 图像，用坐标 MLP 隐式编码辐射场，体渲染出新视角。开创 neural rendering，但慢。
- **3DGS**（2023，显式高斯）：同样是 posed 图像 → 新视角，但用**显式**高斯点集 + 光栅化替代隐式 MLP + 体渲染，把质量与实时性同时拉起来，成为当前 NVS 主流表示。

典型 pipeline：`图像 → COLMAP（位姿 + 点云）→ NeRF 或 3DGS（新视角合成 / 场景表示）`。

## 2. 一页总表

| 维度 | COLMAP | NeRF | 3DGS |
|---|---|---|---|
| 年份 / venue | 2016 / CVPR+ECCV | 2020 / ECCV (Oral) | 2023 / SIGGRAPH (TOG) |
| 范式 | 增量式 SfM + PatchMatch MVS（经典几何优化） | 坐标 MLP + 神经辐射场 + 体渲染（隐式） | 各向异性 3D 高斯 + 可微分 tile 光栅化（显式） |
| 输入 | 无序/无标定图像集合 | posed 图像（相机通常来自 COLMAP） | posed 图像 + 点云初始化（通常来自 COLMAP） |
| 输出 | 相机内外参 + 稀疏点云 + 稠密深度/点云 | 连续辐射场 → 新视角图像 + 深度 | 高斯点集 → 新视角图像（实时） |
| 场景表示 | 点云（离散、无外观模型） | 隐式（MLP 权重里） | 显式（高斯参数：位置/协方差/颜色/透明度） |
| 速度 | 重建慢（迭代 BA/PatchMatch） | 训练慢（原版每场景数小时）、渲染非实时 | 训练分钟级、渲染 1080p 实时（论文实测 130-200 FPS） |
| 是否需要相机先验 | 否（自己估） | 是（依赖 COLMAP 等） | 是（依赖 COLMAP 等，点云初始化） |
| 主要用途 | 位姿/几何真值、SfM/MVS、下游初始化 | 高保真 NVS、隐式几何、研究基座 | 实时 NVS、可编辑显式场景、下游 3D 资产 |
| 代码 / 许可证 | [colmap/colmap](https://github.com/colmap/colmap)；new BSD | [bmild/nerf](https://github.com/bmild/nerf)；MIT | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)；非商用研究许可 |

## 3. 关键差异

- **隐式 vs 显式**：NeRF 把场景压进 MLP 权重（隐式、紧凑但渲染需逐点采样、慢）；3DGS 用显式高斯点（占存储但光栅化快、可直接编辑/操作）。这是 2020→2023 从"神经隐式"转向"显式可微渲染"的核心转折。
- **优化 vs 学习先验**：COLMAP 是纯几何优化、无数据先验，泛化靠工程鲁棒性；NeRF/3DGS 是每场景过拟合式优化（不是跨场景学习模型），仍依赖 COLMAP 提供初值。
- **与前馈重建的关系**：本表三者都是**每场景优化**范式；而 [visual-geometry-foundation-models 对比](visual-geometry-foundation-models.md) 里的 DUSt3R/VGGT/DA3 等是**前馈**范式（一次推理、跨场景泛化），近年正试图用前馈模型替代 COLMAP 的位姿估计、并直接前馈输出 3DGS（如 DA3 的 feed-forward 3DGS）。两张表合起来才是完整图景。

## 4. 场景化选择

| 场景 | 推荐 | 原因 |
|---|---|---|
| 只要相机位姿 / 点云真值 | COLMAP | 事实标准，下游 NeRF/3DGS 的前置 |
| 高保真离线 NVS、研究隐式表示 | NeRF（或其后继 Instant-NGP/Mip-NeRF） | 隐式场质量高；加速版解决慢的问题 |
| 实时 NVS、可编辑场景、3D 资产 | 3DGS | 实时 + 高质量 + 显式可操作 |
| 想跳过每场景优化、直接前馈 | 见前馈几何模型对比 | DUSt3R/VGGT/DA3 前馈估位姿甚至直接出 3DGS |

## 5. 证据与不确定性

- 已确认：三者的范式、输入输出、依赖关系、许可证来自各自论文与官方仓库（见 Sources）。
- 3DGS 实时帧率（130-200 FPS）为论文在特定数据集/硬件上的实测，跨场景会变化。
- 本对比不含定量 PSNR/SSIM 横评——三者评测协议与场景不完全一致，跨论文数值不可直接比较。
- NeRF/3DGS 后继工作（Instant-NGP、Mip-NeRF、2DGS、动态 4DGS 等）众多，本表只锚定三个范式起点。

## 6. 后续动作

- [ ] 更新 `indices/comparisons.md`（脚本自动追加清单）
- [ ] 视需要补 Instant-NGP / Mip-NeRF（NeRF 加速线）与 2DGS/4DGS（3DGS 扩展）

## Sources

- COLMAP 分析：[`../../papers/3d-reconstruction/2016-colmap.md`](../../papers/3d-reconstruction/2016-colmap.md)
- NeRF 分析：[`../../papers/3d-reconstruction/2020-nerf.md`](../../papers/3d-reconstruction/2020-nerf.md)
- 3DGS 分析：[`../../papers/3d-reconstruction/2023-3dgs.md`](../../papers/3d-reconstruction/2023-3dgs.md)
- 前馈几何模型对比：[`visual-geometry-foundation-models.md`](visual-geometry-foundation-models.md)
- 发展全景：[`development-survey.md`](development-survey.md)
