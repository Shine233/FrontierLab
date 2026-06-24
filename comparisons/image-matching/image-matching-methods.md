---
type: method-comparison
title: "图像匹配方法横向对比：RoMa / RoMa v2 / LightGlue / GIM / LoMa"
direction: [image-matching, visual-localization, 3d-reconstruction, evaluation-benchmarks]
methods: [RoMa, RoMa-v2, LightGlue, GIM, LoMa]
status: initial-completed
confidence: medium
updated: 2026-06-24
---

# 图像匹配方法横向对比：RoMa / RoMa v2 / LightGlue / GIM / LoMa

> 用户需求：仔细分析每篇论文，并做横向对比，**如果有在相似数据集或相同指标上的结果，重点标出**。本报告把可直接对比的数字放在第 2、3 节，并明确区分「同协议可比」与「不可直接比较」。

## 结论先行

| 推荐排序 | Method | 推荐场景 | 主要理由 | 主要风险 |
|---:|---|---|---|---|
| 1 | **RoMa v2** | 高精度 / 困难/跨域稠密匹配主路线 | 困难场景 EPE 大降、比 v1 快约 1.7×、位姿 SOTA | 训练代码疑未公开、依赖自定义 CUDA kernel、DINOv3 非 MIT |
| 2 | **LightGlue** | 速度/低显存优先的稀疏匹配工程默认 | 比 SuperGlue 快约 2–4×、易部署、生态成熟 | 依赖前端检测器、许可碎片化、难对精度逊于 dense |
| 3 | **GIM** | 需要单一通用「零样本」匹配器 | 用互联网视频自训练，架构无关，ZEB 全面提升 | 训练门槛高、视频数据未公开、版权/可复现性风险 |
| 4 | **RoMa (v1)** | 稠密匹配精度基准 / v2 对照 | 2024 强 SOTA，训练开源、许可友好 | 比 v2 慢、极端模态略弱 |
| 5 | **LoMa** | 想要更鲁棒的稀疏 matcher（仅推理） | LightGlue 式 + 数据/模型 scaling，5 基准大幅提升 | 训练代码与 HardMatch 数据未公开、多项细节 to verify |

**一句话**：稠密路线选 **RoMa v2**（精度上限）；稀疏路线选 **LightGlue**（速度/工程）；要跨域泛化套 **GIM**；**LoMa** 是值得跟踪的稀疏 scaling 新作但训练不可复现。

## 1. 对比目标

- **为什么比较**：这 5 个方法都是两视图图像/特征匹配，但分属三条路线，直接同表比较容易混淆「谁和谁可比」。
- **目标任务**：两视图匹配 → 相对位姿 / homography / 视觉定位 / SfM 前端。
- **约束**：必须区分稀疏 vs 稠密、新架构 vs 训练框架，并区分指标协议是否一致。

### 三条路线（先分类再比较）

| 路线 | 方法 | 性质 |
|---|---|---|
| **稠密 dense matching** | RoMa、RoMa v2 | 全像素 warp 回归，精度上限高、计算重 |
| **稀疏 sparse matching** | LightGlue、LoMa | 关键点 + matcher，速度快、依赖检测器 |
| **训练/数据框架（架构无关）** | GIM | 不是新架构，包裹上面的 backbone 提升泛化 |

> ⚠️ **GIM 与其他四者不是同级对象**：GIM 是训练框架，可作用于 RoMa/DKM/LoFTR/LightGlue。比较 GIM 应看「同一 backbone 加 GIM 前后」的提升，而不是把 GIM 当成独立架构和 RoMa v2 比绝对精度。

## 2. 一页总表

| 维度 | RoMa | RoMa v2 | LightGlue | GIM | LoMa |
|---|---|---|---|---|---|
| 基本范式 | 稠密 coarse-to-fine | 稠密 + Attention + 协方差 | 稀疏 attention matcher | 自训练框架（架构无关） | 稀疏 matcher（LightGlue 式）+ scaling |
| 输入 | 两张图像 | 两张图像 | 关键点+描述子 | 取决于 backbone | 关键点 |
| 输出 | 稠密 warp + certainty | 稠密对应 + 2×2 精度矩阵 | 部分赋值 + matchability | 取决于 backbone | 稀疏匹配 |
| backbone | 冻结 DINOv2 + VGG19 | 冻结 DINOv3 ViT-L | Transformer（无视觉 backbone） | 原 backbone | 类 LightGlue |
| 年份/venue | CVPR 2024 | 2025 arXiv | ICCV 2023 | ICLR 2024 Spotlight | 2026 arXiv |
| 数据集（评测） | MegaDepth-1500、ScanNet-1500、IMC2022、WxBS | MegaDepth-1500、ScanNet-1500、WxBS、SatAst、6 个稠密基准、Hypersim | HPatches、MegaDepth-1500/1800、Aachen、InLoc、IMC | **ZEB**（12 域）、位姿、homography | HardMatch、WxBS、InLoc、RUBIK、IMC2022 |
| 指标 | AUC@5/10/20、mAA | AUC、EPE、PCK、mAA | AUC、homography、runtime | ZEB Mean AUC@5、mAA | mAA、AUC、loc recall |
| Git 地址 | [Parskatt/RoMa](https://github.com/Parskatt/RoMa) | [Parskatt/romav2](https://github.com/Parskatt/romav2) | [cvg/LightGlue](https://github.com/cvg/LightGlue) | [xuelunshen/gim](https://github.com/xuelunshen/gim) | [davnords/LoMa](https://github.com/davnords/LoMa) |
| 是否开源 | 是 | 是 | 是 | 是 | 是（推理） |
| 是否开源训练 | 是 | `\`（未提供 / to verify） | `\`（在 glue-factory） | 是（独立分支） | 否 |
| 预训练权重 | roma_outdoor/indoor/tiny | 自动下载（无显式 URL） | SuperPoint/DISK/ALIKED/SIFT | gim_roma/dkm/loftr/lightglue | LoMa-B/B128/L/G/R |
| 许可证 | MIT（DINOv2 Apache-2.0） | MIT（DINOv3 custom） | Apache-2.0（组件混合，SuperPoint 受限） | MIT | MIT + LightGlue Apache-2.0 |
| 推理成本 | 重（~199ms/对 RTX6000） | 中（比 v1 快 ~1.7×） | 轻（数十 ms） | 同 backbone | 轻（稀疏） |
| 工程风险 | 显存/速度 | CUDA kernel、DINOv3 许可 | 检测器依赖、许可碎片 | 数据采集/算力门槛 | 训练&数据未公开 |

## 3. ⭐ 重点：相同数据集 / 相同指标的直接对比

### 3.1 MegaDepth-1500，相对位姿 AUC@5°/10°/20°（最常见的可比基准）

| Method | AUC@5° | AUC@10° | AUC@20° | 来源/协议 |
|---|---|---|---|---|
| **RoMa v2** | **62.8** | **77.0** | **86.6** | RoMa v2 论文 |
| RoMa (v1) | 62.6 | 76.7 | 86.3 | RoMa v2 论文复算（与 v1 论文一致） |
| DKM | 60.4 | 74.9 | 85.1 | RoMa v2 论文 |
| LoFTR | 52.8 | 69.2 | 81.2 | RoMa v2 论文 |
| MASt3R | 42.4 | 61.5 | 76.9 | RoMa v2 论文 |
| LightGlue（RANSAC） | 49.9 | 67.0 | 80.1 | LightGlue 论文 |
| LightGlue（RoMa 论文报告值） | 51.0 | 68.1 | 80.7 | RoMa v1 论文 |

**怎么读**：
- RoMa v2 ≈ RoMa v1 在 MegaDepth-1500 上**几乎持平**——这个基准已饱和，不要据此判断 v2 的真实进步（v2 的提升在第 3.3 节的难基准上）。
- 稠密（RoMa/DKM/LoFTR）整体 AUC 高于稀疏（LightGlue），符合「稠密精度上限更高」的直觉。
- ⚠️ LightGlue 在两篇论文里的数字（49.9 vs 51.0）不同，**协议/RANSAC 设置不同**，跨论文比较只能看大致档位。

### 3.2 ScanNet-1500，相对位姿 AUC@5°/10°/20°（室内）

| Method | AUC@5° | AUC@10° | AUC@20° |
|---|---|---|---|
| **RoMa v2** | **33.6** | **56.2** | **73.8** |
| RoMa (v1) | 31.8 | 53.4 | 70.9 |
| DKM | 29.4 | 50.7 | 68.3 |
| LoFTR | 22.1 | 40.8 | 57.6 |
| ASpanFormer | 25.6 | 46.0 | 63.3 |

**怎么读**：在室内 ScanNet-1500 上 **RoMa v2 比 v1 有明显提升**（+1.8/+2.8/+2.9），与 MegaDepth 的持平形成对比，说明 v2 的增益主要在「更难/分布外」场景。RoMa v1 论文称其首次让 AUC@20° 超过 70。

### 3.3 RoMa v2 的真正主场：困难/跨域稠密匹配 EPE（越低越好）

| 基准 | RoMa (v1) EPE | RoMa v2 EPE | 备注 |
|---|---|---|---|
| TartanAir Wide-Baseline | 60.61 | **13.82** | 宽基线 |
| MegaDepth | 2.34 | **1.47** | |
| ScanNet++ v2 | 27.52 | **4.00** | |
| FlyingThings3D | 5.68 | **0.93** | |
| AerialMegaDepth | 25.05 | **4.12** | 论文称 -84% EPE |
| MapFree | 8.55 | **2.03** | |

→ 这才是 RoMa v2「Harder/Denser」标题的依据：困难场景 EPE 成倍下降。

### 3.4 ⭐ ZEB 零样本基准（GIM 的主场，Mean AUC@5°，越高越好）

| Backbone | 原版 | + GIM | 训练时长 |
|---|---|---|---|
| RoMa | 48.8 | **53.3 (GIM_RoMa)** | — |
| DKM | 45.8 | **51.2 (GIM_DKM)** | 100h |
| LoFTR | 33.1 | **39.1 (GIM_LoFTR)** | 50h |
| LightGlue | 31.7 | **38.3 (GIM_LightGlue)** | 100h |
| SuperGlue | — | 34.3 (GIM_SuperGlue) | 50h |
| RootSIFT（参照） | 31.8 | — | — |

**怎么读**：
- GIM 对**每个 backbone 都有提升**（+4.5～+6.6 Mean AUC），印证「GIM 是叠加在现有架构上的训练框架」。
- 即便在跨域 ZEB 上，**稠密 backbone（RoMa/DKM）仍排在稀疏（LightGlue）之上**，与第 3.1 节一致。
- ⚠️ DKM/LightGlue 用 100h、LoFTR/SuperGlue 用 50h，**训练时长口径不同**，比较 GIM 增益时要带上时长。

### 3.5 ⚠️ 不可直接比较 / 协议冲突的指标（重点提醒）

- **WxBS（多模态）**：RoMa v1 论文报 mAA@10px = **80.1**；RoMa v2 论文报 RoMa = **60.8** → v2 = **55.4**。两套数字差距巨大，说明**WxBS 版本/协议不同**（v2 用更难的设定），**绝对值不可跨论文比较**。在 v2 自己的协议下，v2 相对 v1 在 WxBS 上**退步**（极端模态鲁棒性下降）。
- **LoMa 的数字是「相对提升」而非绝对值**：HardMatch +18.6 mAA、WxBS +29.5 mAA、InLoc +21.4 @(1m,10°)、RUBIK +24.2 AUC、IMC2022 +12.4 mAA，**全部相对 ALIKED+LightGlue baseline**。这些基准（HardMatch/RUBIK）与上面几张表不重叠，且缺论文绝对值，**当前无法与 RoMa/LightGlue 同表对齐**（to verify，需全文绝对数）。
- **IMC2022**：RoMa v1 报 mAA@10 = 88.0；LightGlue / LoMa 在 IMC 上用的 split/年份/指标不完全一致，谨慎对齐。

## 4. 方法逐个分析（要点）

### RoMa
- 核心：冻结 DINOv2 coarse + ConvNet fine，transformer anchor decoder，RbC+robust regression。
- 优势：2024 稠密 SOTA，许可友好，训练开源。
- 局限：稠密成本高（~199ms/对），依赖有监督对应。

### RoMa v2
- 核心：DINOv3 + Attention coarse matcher（替代 GP）+ 两阶段训练 + 预测协方差 + CUDA kernel。
- 优势：困难场景 EPE 大降、快 1.7×、协方差提升位姿求解。
- 局限：训练代码未提供、CUDA kernel 可移植性、DINOv3 许可、极端模态略退。

### LightGlue
- 核心：adaptive depth/width + matchability head（替代 Sinkhorn）+ rotary 编码。
- 优势：比 SuperGlue 快 2–4×、易部署、训练快。
- 局限：依赖检测器、许可碎片、难对精度逊于 dense。

### GIM
- 核心：互联网视频自训练 + 标签传播，架构无关。
- 优势：把专用匹配器变成通用零样本匹配器，ZEB 全面提升。
- 局限：训练算力门槛高、视频数据未公开、依赖 teacher 标签质量。

### LoMa
- 核心：LightGlue 式稀疏 matcher + 数据/模型/算力 scaling。
- 优势：5 基准相对 ALIKED+LightGlue 大幅提升，多规格权重（含旋转不变）。
- 局限：训练代码 + HardMatch 数据未公开，架构/消融/训练集 to verify。

## 5. 场景化选择

| 场景 | 推荐 | 原因 |
|---|---|---|
| 高精度匹配 / 重建定位前端 | RoMa v2 | 难基准 EPE 最低，位姿 SOTA，提速 |
| 实时 / 低显存 / 端侧 | LightGlue | 稀疏、数十 ms、生态成熟 |
| 未知域 / 零样本泛化 | GIM（套在 RoMa/DKM 上） | 跨 12 域稳定提升 |
| 稳妥稠密基准 / v2 对照 | RoMa v1 | 训练开源、许可友好 |
| 更强稀疏 matcher（仅推理）/ 旋转不变/aerial | LoMa | LightGlue 式 + scaling，含 LoMa-R |
| 论文复现 | RoMa / LightGlue(+glue-factory) / GIM | 三者训练可复现；RoMa v2、LoMa 训练受限 |

## 6. 证据与不确定性

- **已确认事实**：5 篇论文标题/venue/arXiv/GitHub/许可/权重状态；MegaDepth-1500、ScanNet-1500、ZEB 的可比数字（见第 3 节）；GIM 是训练框架而非架构。
- **推断**：稠密 > 稀疏的精度档位在多个基准一致；RoMa v2 增益集中在难/跨域。
- **待验证（to verify）**：LoMa 全文（架构/消融/训练集/绝对数/ECCV 2026）；RoMa v2 训练代码是否存在与权重托管；WxBS 跨论文协议差异的精确定义；LightGlue 二手 ScanNet 数字。

## 7. 后续动作

- [x] 创建本对比报告
- [x] 更新 `indices/comparisons.md`
- [x] 更新 `indices/methods.md`
- [ ] 读 LoMa 全文补齐 to verify 项与绝对数字
- [ ] 若确定复现，优先 `reproductions/image-matching/{romav2,lightglue,gim}/README.md`
