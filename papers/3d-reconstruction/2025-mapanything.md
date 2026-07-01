---
type: paper-analysis
title: "MapAnything: Universal Feed-Forward Metric 3D Reconstruction"
short_name: "MapAnything"
year: 2025
venue: "3DV 2026（arXiv 2025-09）"
arxiv: "2509.13414"
paper_url: "https://arxiv.org/abs/2509.13414"
code: "https://github.com/facebookresearch/map-anything"
github: "https://github.com/facebookresearch/map-anything"
open_source: true
license: "代码 Apache 2.0；权重双版本：facebook/map-anything 为 CC-BY-NC 4.0（研究，论文全 13 数据集），facebook/map-anything-apache 为 Apache 2.0（商用，数据集为可商用子集，具体数量待核验）"
training_open_source: "true"
direction: [3d-reconstruction, dense-vision, depth-estimation]
method_family: [feed-forward-3d-reconstruction, promptable-reconstruction, factored-scene-representation, pointmap-regression, metric-3d]
tasks: [uncalibrated-sfm, calibrated-mvs, monocular-depth-estimation, camera-localization, depth-completion, metric-pointmap-estimation]
datasets: [ETH3D, ScanNet++, TartanAirV2-WB, KITTI, ScanNet, Robust-MVD]
metrics: [abs-rel, inlier-ratio, angular-error, tau]
status: analyzed
reproduction: none
confidence: medium
landmark: false
org: [Meta, Carnegie Mellon University]
key_idea: "一个统一前馈 Transformer 接收任意数量图像及可选先验（内参/位姿/深度/部分重建），用「深度图+局部 ray map+相机位姿+度量尺度」的因子化表示，单次前馈直接回归全局一致的度量 3D 几何与相机。"
supersedes: []
builds_on: [2023-dust3r, 2025-vggt]
updated: 2026-07-01
---

# MapAnything：通用前馈度量 3D 重建

## 结论先行
- MapAnything 把多种 3D 视觉任务（无标定 SfM、标定 MVS、单目深度、相机定位、深度补全）统一进「一个可 prompt 的前馈模型、一次前馈」框架：输入是 1~N 张图像 + 任意可选先验（内参/位姿/深度/部分重建），输出直接是度量尺度的全局一致 3D 几何与相机（证据：arXiv 2509.13414 摘要）。
- 关键在「因子化表示」——把场景拆成 深度图 + 局部 ray map + 相机位姿 + 单一度量尺度因子，再拼装为全局度量坐标系；这让同一模型既能吃图像、也能吃几何先验，先验越多精度越高（证据：2-view 无先验 inlier 53.6%，注入 内参+位姿+深度 后 abs-rel 降到 0.01、inlier 升到 92.1%，arXiv v1 HTML Table 2）。
- 它在多项基准上追平或超过专用前馈模型：2-view images-only inlier ratio 53.6% 高于 DUSt3R 43.9%/MASt3R 30.2%/VGGT 43.2%（abs-rel 与三者的 0.20/0.25/0.20 相当或更优）；单视图标定平均角误差 1.18° 优于 VGGT 4.00°、MoGe-2 1.95°、AnyCalib 2.01°（证据：arXiv v1 HTML 对比表）。作者定位为「追平或超过 specialist 模型，同时联合训练更高效」。
- 工程可用性高且明确面向落地：代码 Apache 2.0；权重双许可（CC-BY-NC 研究版含论文全部 13 数据集 / Apache 2.0 商用版仅含可商用数据集子集，子集数量待核验），训练代码、13 数据集处理管线均开源，权重上 HuggingFace。商用可直接用 Apache 版权重（这是它相较 VGGT 权重需申请的落地优势）。
- 对自动驾驶友好点在于「metric + promptable」：可注入车载已知内参/位姿/稀疏深度做深度补全与度量重建，且 memory-efficient 模式下宣称单卡 140GB 可处理约 2000 视图（证据：README；推断：车队多相机/长序列场景契合）。

## 1. 这篇论文解决什么问题？
- 问题定义：把分散在多个专用模型里的 3D 视觉任务（SfM、MVS、单目深度、定位、深度补全）统一成一个前馈模型，并支持度量尺度（metric）输出与可选几何先验注入。
- 输入 / 输出：输入 1~N 张 RGB 图像 + 可选先验（相机内参、相机位姿、深度图、部分重建）；输出各帧度量深度、局部 ray map、相机位姿与统一到全局度量坐标系的 3D 几何。
- 目标场景：通用多视图/单视图重建，尤其是能拿到部分标定信息的场景（机器人、自动驾驶多相机、已知内参的采集），可按可用先验灵活退化或增强。
- 与现有方法的差异：DUSt3R/VGGT 等前馈模型多为「只吃图像、输出非度量或需对齐」的固定接口；MapAnything 以因子化表示统一接口，可 prompt 先验、直接给度量结果，一个模型覆盖多任务而非每任务一模型。

## 2. 方法概览
- 核心想法：用统一 Transformer + 因子化场景表示，把「图像 + 任意几何先验」映射到「度量 3D 几何 + 相机」，通过输入增强（随机丢弃/提供不同先验）训练出可 prompt 的单模型。
- 模型结构：基于 Transformer 的前馈骨干；输出被因子化为 逐帧深度图、局部 ray map（编码像素视线）、相机位姿、以及一个全局度量尺度因子，据此把各帧局部重建拼成全局一致的度量坐标系。
- 训练目标：跨 13 个数据集做标准化监督（统一坐标/尺度约定），配合灵活的输入增强——训练时随机提供或屏蔽内参/位姿/深度等先验，使推理期能接受任意先验组合。
- 推理流程：单次前馈；无先验时做无标定 SfM/单目深度，有先验时退化为标定 MVS/深度补全/定位。memory-efficient 模式支持大规模视图（README 称约 2000 视图 @140GB）。

## 3. 关键贡献
1. 提出统一的可 prompt 前馈度量 3D 重建模型，一个模型 + 一次前馈覆盖无标定 SfM、标定 MVS、单目深度、定位、深度补全五类任务。
2. 因子化场景表示（深度 + 局部 ray map + 相机位姿 + 度量尺度因子），既统一多任务输出、又天然支持注入任意几何先验并输出度量尺度。
3. 跨 13 数据集的标准化监督 + 输入增强训练范式，使单模型的联合训练比训练一堆专用模型更高效，且先验越多精度单调提升。
4. 全面开源（Apache 2.0 代码 + 训练代码 + 13 数据集处理管线 + 双许可权重），提供商用可用（Apache）权重。

## 4. 实验与证据
| 维度 | 内容 |
|---|---|
| 数据集 | 训练 13 个数据集（Apache 权重版为其中可商用子集，数量待核验）；评测含 ETH3D、ScanNet++ v2、TartanAirV2-WB、KITTI、ScanNet、Robust-MVD |
| Baseline | DUSt3R、MASt3R、Pow3R、VGGT、MV-DUSt3R+、Fast3R、π³、MoGe-2、AnyCalib、MUSt3R 等 |
| 指标 | abs-rel（绝对相对误差）、inlier ratio（1.03% 相对阈值）、单视图标定角误差、Robust-MVD 的 rel/τ |
| 主要结果 | 2-view images-only：inlier 53.6%（DUSt3R 43.9%，MASt3R 30.2%，VGGT 43.2%），abs-rel 与 DUSt3R/VGGT 的 0.20、MASt3R 的 0.25 相当或更优（MapAnything 具体 abs-rel 因 Table 2 多子指标堆叠、跨来源解析不一致，此处不引单一数值）；2-view + 内参+位姿+深度：abs-rel 0.01 / inlier 92.1%（Pow3R 0.03/89.0）。多视图（50 views，跨 ETH3D/ScanNet++v2/TartanAirV2-WB 平均）images-only abs-rel 0.16 / inlier 40.7%，+内参+度量位姿后 0.05 / 57.8%。单视图标定平均角误差 1.18°（VGGT 4.00°，MoGe-2 1.95°，AnyCalib 2.01°）。 |
| 消融 | 先验消融即核心卖点：随注入先验增多，abs-rel/inlier 单调改善（2-view inlier 53.6%→92.1%，abs-rel 降至 0.01） |
| 失败案例 | 论文未在摘要给出显式失败案例（推断：强动态/非刚性、无任何先验的大尺度户外、极端外推视角仍受前馈范式与训练分布限制） |

（注：以上数值来自 arXiv 2509.13414 v1 HTML 的对比表，已联网核实；Table 2 单表内堆叠多个子指标（Scale/Points/Depth/rel/τ 等），MapAnything 的 2-view images-only abs-rel 跨解析来源读数不一致，故只保留稳定的 inlier 与 baseline 对比。不同表格的视图数/先验组合不同，引用时需连同设定一起看，部分方法在其原设定下评测。）

## 5. 局限与风险
- 论文明确承认：定位为「追平或超过」专用模型而非全面碾压，个别单任务上专用模型仍可能领先；度量精度强依赖可用先验，纯 images-only 下误差明显高于带先验设定（如 2-view 无先验 inlier 53.6% vs 带全先验 92.1%）。
- 我推断的风险：训练覆盖 13 数据集但仍是特定分布，强动态/非刚性场景、无任何先验的开放世界外推一致性存疑；ray map + 尺度因子的度量尺度在训练域外可能漂移。
- 工程落地风险：大规模视图（~2000）需 140GB 级显存，普通单卡需用 memory-efficient 模式或降视图数；先验质量差（错误内参/位姿）时是否稳健需评估。
- 许可证 / 数据风险：完整 13-数据集权重为 CC-BY-NC 4.0（不可商用），商用须用仅含可商用数据集子集训练的 Apache 版权重（子集数量待核验），二者精度可能有差（需自测）；这是相较「代码开源即可商用」的关键陷阱。

## 6. 与相似方法对比
| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| VGGT | 单次前馈、多帧、多几何量回归 | VGGT 只吃图像、非可 prompt、度量性弱；MapAnything 可注入内参/位姿/深度并直接输出度量，单视图标定 1.18° vs VGGT 4.00° | 只有图像、要几何基础 backbone/点跟踪时选 VGGT；能拿到先验、要度量结果时选 MapAnything |
| DUSt3R | 前馈点图回归、去 SfM 优化 | DUSt3R 成对 + 需全局对齐、非度量；MapAnything 原生多视图 + 度量 + promptable，2-view images-only inlier 53.6% vs DUSt3R 43.9% | 轻量成对/作谱系参照时选 DUSt3R |
| Pow3R | 同样支持注入相机/深度先验 | Pow3R 沿 DUSt3R 成对范式；MapAnything 统一多任务 + 度量因子化表示，覆盖更广 | 只需 2-view 可 prompt 重建时可比较 Pow3R |
| π³ / MV-DUSt3R+ | 前馈多视图重建 | 多为特定任务专用；MapAnything 单模型联合训练、接口统一 | 单一任务追极致精度时可比对应专用模型 |

## 方法谱系
- 基于：[VGGT](../3d-reconstruction/2025-vggt.md)（延续纯前馈、多帧、多几何量单次回归的范式，扩展为可 prompt 先验 + 度量输出）
- 基于：[DUSt3R](../3d-reconstruction/2023-dust3r.md)（继承前馈 pointmap 回归、去几何优化管线的思路，并统一到多任务度量框架）

## 7. 复现判断
- Git 地址：https://github.com/facebookresearch/map-anything
- 是否开源：是（Apache 2.0 代码）。
- 是否开源训练：是，仓库含完整训练指令、脚本、配置及 13 数据集的数据处理管线。
- 代码可用性：完整（推理 + 训练）。
- 权重可用性：HuggingFace 双版本——`facebook/map-anything`（CC-BY-NC 4.0，论文全 13 数据集，研究用）与 `facebook/map-anything-apache`（Apache 2.0，可商用数据集子集，商用）。
- 数据可获得性：训练数据为公开数据集组合，完整复现需自行下载并按官方管线处理。
- 预计环境成本：推理单卡可跑（大视图数需大显存或 memory-efficient 模式）；从头训练成本高（大模型 + 13 数据集），一般直接用官方权重。
- 最小复现路径：clone 仓库 → 装依赖 → 加载 HF 权重（商用选 apache 版）→ 在自有图像上前馈推理，逐步注入内参/位姿/深度先验验证精度单调提升。
- 是否值得复现：推理级复现值得（可 prompt + 度量，适合有先验的机器人/自驾场景）；从头训练非必要，除非做数据/架构实验。商用务必用 Apache 版权重。

## 8. 后续动作
- [ ] 更新 `indices/papers.md`
- [ ] 更新 `indices/directions.md`
- [ ] 更新 `reports/feedforward_3d_reconstruction_compare.md` 中的 MapAnything 条目
- [ ] 若计划复现，创建 `reproductions/3d-reconstruction/mapanything/README.md`
