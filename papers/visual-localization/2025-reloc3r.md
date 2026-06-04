---
type: paper-analysis
title: "Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization"
short_name: "Reloc3r"
year: 2025
venue: "CVPR 2025 / arXiv"
arxiv: "2412.08376"
paper_url: "https://arxiv.org/abs/2412.08376"
pdf_url: "https://arxiv.org/pdf/2412.08376"
code: "https://github.com/ffrivera0/reloc3r"
github: "https://github.com/ffrivera0/reloc3r"
open_source: true
license: "CC BY-NC-SA 4.0"
training_open_source: true
weights_availability: "Hugging Face checkpoints reloc3r-224 and reloc3r-512 are linked from the README."
data_availability: "Training uses CO3Dv2, ScanNet++, ARKitScenes, BlendedMVS, MegaDepth, DL3DV, RealEstate10K; repo provides preprocessing scripts and pair-list archive, while users must download source datasets under their licenses."
direction: [visual-localization, 3d-reconstruction, robotics-autonomous-driving, evaluation-benchmarks]
method_family: [feed-forward-visual-relocalization, relative-pose-regression, motion-averaging, dust3r-initialized-pose-regression]
tasks: [relative-camera-pose-estimation, visual-relocalization, absolute-camera-pose-estimation, map-free-localization]
datasets: [ScanNet1500, RealEstate10K, ACID, CO3Dv2, 7-Scenes, Cambridge-Landmarks, MegaDepth1500]
metrics: [AUC@5, AUC@10, AUC@20, RRA@15, RTA@15, mAA@30, median-translation-error, median-rotation-error, FPS]
status: compared
reproduction: planned
confidence: high
updated: 2026-06-04
---

# Reloc3r：大规模训练的相对姿态回归视觉重定位

## 结论先行

- **一句话定位**：Reloc3r 是当前最实用的 scene-agnostic RPR baseline 之一：训练一次，在新场景中只需要 posed database images 和 retrieval，就能用 pair-wise relative pose regression + motion averaging 输出 query absolute pose。
- **核心方法**：用 DUSt3R-style ViT encoder-decoder 回归双向相对姿态，不直接学习 metric relative scale；多张 retrieved database images 的结果通过 rotation averaging 和 camera-center triangulation 得到绝对位姿。
- **论文证据**：Reloc3r-512 在 ScanNet1500 上 AUC@5/10/20 为 34.79/58.37/75.56，论文报告 24 FPS；7-Scenes 平均 0.04m/1.02deg；Cambridge Average(4) 0.38m/0.52deg。
- **代码状态**：GitHub 已公开推理、评测、训练代码、dataset preprocessing scripts、Reloc3r-224/512 checkpoint 链接和 wild image/video demo；按仓库约定 `training_open_source: true`。
- **工程判断**：Reloc3r 很适合作为 Reloc-VGGT 和 FastForward 的可跑 baseline，但它的 late fusion / motion averaging 在共线、多 source 弱几何、retrieval outliers 或尺度退化场景中仍有风险。

## 1. 这篇论文解决什么问题？

### 已确认的论文事实

- **问题定义**：传统 visual localization 需要显式 3D map 或每场景训练；APR scene-specific 且数据饥渴；RPR 可泛化但精度长期不足。Reloc3r 想通过大规模、多域 posed image-pair training 提高 RPR 的泛化和精度。
- **输入 / 输出**：Relative pose module 输入两个 images，输出双向 relative poses；visual localization 时 query 与 top-K retrieved database images 配对，最终输出 query absolute pose。
- **目标场景**：pair-wise relative pose、multi-view relative pose、indoor/outdoor visual localization。
- **训练数据**：论文称约 8M posed image pairs，覆盖 object-centric、indoor、outdoor 数据；README 列出 CO3Dv2、ScanNet++、ARKitScenes、BlendedMVS、MegaDepth、DL3DV、RealEstate10K。

### 我的理解

Reloc3r 的价值在于它把 “pose regression 不能泛化” 这个印象往前推了一步：与其给每个场景训练 APR，不如训练一个跨场景的相对位姿模型，然后在新场景里借助 database poses 把 relative pose 转成 absolute pose。

它仍不是传统几何定位的替代品。Reloc3r 不输出显式 correspondences，也不构建 3D map；它的尺度来自多 source views 的 motion averaging，而不是直接由网络可靠估计。因此在强几何约束和大场景精定位中，仍应和 ACE/MASt3R/FastForward/feature matching 方法同表比较。

## 2. 方法概览

### 2.1 Relative pose regression network

- 采用 DUSt3R/CroCo 风格 ViT encoder-decoder。
- 两个 image branches 使用对称设计，共享 decoder 和 prediction head，减少 image order bias。
- 预测双向相对姿态 `P_I1,I2` 和 `P_I2,I1`。
- 旋转用 continuous 9D-to-SO(3) 表示。
- 默认不回归 metric relative scale，论文消融显示直接学习 metric pose 会降低泛化。

### 2.2 Motion averaging for absolute localization

Visual localization 时：

1. 用 NetVLAD 等 retrieval 找 query 的 top-K database images。
2. 对每个 database-query pair 回归相对 pose。
3. 用 database image 的已知 pose 得到多个 query pose 候选。
4. Rotation averaging 融合旋转。
5. Camera center triangulation 从多条 translation direction 中恢复 query camera center 和 metric scale。

这个模块极简，但它解释了 Reloc3r 为什么能在不学习 metric relative scale 的情况下输出 metric absolute pose：metric 信息来自 database poses 的几何关系。

## 3. 关键贡献

1. **大规模 RPR 训练**：用约 8M posed image pairs 覆盖 object-centric、indoor、outdoor 数据域。
2. **对称 relative pose architecture**：共享 decoder/head，避免输入顺序偏置，训练更简洁。
3. **极简 motion averaging**：不依赖复杂后端优化，把 top-K relative predictions 融合成 absolute pose。
4. **强泛化实证**：在 ScanNet1500、RealEstate10K、ACID、CO3Dv2、7-Scenes、Cambridge 等数据上验证。

## 4. 实验与证据

| 维度 | 内容 |
|---|---|
| 数据集 | ScanNet1500、RealEstate10K、ACID、CO3Dv2、7-Scenes、Cambridge Landmarks、MegaDepth1500 |
| Baseline | Map-free、ExReNet、Efficient LoFTR、ROMA、DUSt3R、MASt3R、NoPoSplat、APR/RPR methods |
| 指标 | AUC@5/10/20、RRA@15、RTA@15、mAA@30、median translation/rotation error、FPS |
| 主要结果 | ScanNet1500 AUC@5/10/20：Reloc3r-512 34.79/58.37/75.56；CO3Dv2 multi-view：RRA@15 95.8%、RTA@15 93.7%、mAA@30 82.9%；7-Scenes 平均 0.04m/1.02deg；Cambridge Average(4) 0.38m/0.52deg |
| 速度 | README/论文报告 Reloc3r-512 在 RTX 4090 上 fp32 约 24 FPS；开启 fp16 `--amp=1` 可到约 40 FPS |
| 消融 | 对称结构优于非对称；不直接学习 metric scale 更好；DUSt3R initialization 对性能重要；训练数据规模和多样性提升精度 |
| 失败案例 | 论文明确指出完全共线图像会让 motion averaging 的 metric scale 不可解；大规模场景精度仍可能落后强 feature matching / solver 方法 |

## 5. 已确认的代码/仓库事实

- GitHub：<https://github.com/ffrivera0/reloc3r>
- 2026-06-04 read-only check：`git ls-remote` HEAD 为 `761fac648e9c21fd7dcda01ab2ccd4fc20058102`。
- License：CC BY-NC-SA 4.0。
- 公开文件包括 `train.py`、`eval_relpose.py`、`eval_visloc.py`、`wild_relpose.py`、`wild_visloc.py`、`visualization.py`、`scripts/train.sh`、`scripts/train_small.sh`、7Scenes/Cambridge/ScanNet1500/MegaDepth1500 eval scripts。
- README 已标记 release training code and data，并链接 Reloc3r-224 / Reloc3r-512 Hugging Face checkpoints。
- `datasets_preprocess/` 包含 CO3D、ScanNet++、ARKitScenes、BlendedMVS、MegaDepth、DL3DV preprocessing scripts。

## 6. 局限与风险

### 论文/代码层面已确认

- CC BY-NC-SA 4.0，商业使用受限。
- 完整训练需要自行下载多个大规模数据集并遵守各自许可证。
- README 说明官方 `train.sh` 与实际训练不严格等价，但足够接近；因此完整论文数字仍需谨慎复刻。
- Wild video demo 用第一帧/最后帧作为 database，要求 overlap，不支持 linear motion，这说明极简 demo 不代表完整系统鲁棒性。

### 我的推断风险

- Motion averaging 是强假设模块：source views 过少、近共线、translation baseline 不足或 retrieval outliers 会影响 metric scale。
- Reloc3r 不输出 2D-3D correspondences，所以难以接 PnP/RANSAC 做几何 outlier rejection；可解释性弱于 FastForward/ACE/MASt3R。
- 对自动驾驶/机器人系统，Reloc3r 更适合做 quick global pose proposal 或 no-map baseline，不能替代带地图约束的定位后端。

## 7. 与相似方法对比

| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| Reloc-VGGT | 都是 scene-agnostic RPR / feed-forward relocalization | Reloc3r late fusion；Reloc-VGGT early fusion with VGGT + pose tokens | 当前复现选 Reloc3r；研究 Reloc3r 上限选 Reloc-VGGT |
| FastForward | 都用 posed mapping/database images，不做每场景训练 | FastForward 预测 query 3D coordinates 并 PnP；Reloc3r 直接 relative pose regression + averaging | 需要可解释 correspondences/scale transfer 选 FastForward；需要可跑 RPR baseline 选 Reloc3r |
| MARePo | 都是 feed-forward pose output | MARePo 依赖 scene-specific coordinate map；Reloc3r 不需要 per-scene training | 有 map/ACE head 选 MARePo；无场景训练选 Reloc3r |
| Map-free / ExReNet | 都属于 RPR/relative localization 族 | Reloc3r 大规模训练、DUSt3R init、更强泛化 | 历史 baseline 保留 Map-free/ExReNet；新 baseline 用 Reloc3r |
| MASt3R / DUSt3R | 都来自 3D foundation model 生态 | MASt3R/DUSt3R 更偏 matching/reconstruction/solver；Reloc3r 是 direct pose regression | 做强非 PR baseline 选 MASt3R；做 RPR 选 Reloc3r |

## 8. 复现判断

- Git 地址：<https://github.com/ffrivera0/reloc3r>
- 是否开源：是。
- 是否开源训练：是。
- 代码可用性：可安装环境，跑 wild examples、relative pose eval、visual localization eval、train scripts。
- 权重可用性：Reloc3r-224 / Reloc3r-512 Hugging Face checkpoints。
- 数据可获得性：需要从各官方来源下载；repo 提供 preprocessing scripts 和 pair lists。
- 预计环境成本：推理可在单 RTX 4090 级别；完整训练 Reloc3r-512 README 给出 8 H800 GPU 脚本，`train_small.sh` 可在 RTX 3090 上做小训练。
- 最小复现路径：
  1. 安装 conda env，递归拉取 submodules。
  2. 下载 checkpoint 或让 eval 自动下载。
  3. 跑 `wild_relpose.py` 和 `wild_visloc.py` 做 sanity check。
  4. 下载 7-Scenes 或 Cambridge，跑 `scripts/eval_7scenes.sh` / `scripts/eval_cambridge.sh`。
  5. 若做训练复现，先跑 `scripts/train_small.sh`，再评估 ScanNet1500 subset。
- 是否值得复现：值得。它是 Reloc-VGGT 当前最直接、最可运行的 baseline。

## 9. 后续动作

- [x] 创建 Reloc3r 单篇论文分析
- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [x] 创建 visual localization 横向对比
- [ ] 若开始复现，创建 `reproductions/visual-localization/reloc3r/README.md`

## Sources

- Paper: <https://arxiv.org/abs/2412.08376>
- PDF: <https://arxiv.org/pdf/2412.08376>
- GitHub: <https://github.com/ffrivera0/reloc3r>
- Reloc3r-224 checkpoint: <https://huggingface.co/siyan824/reloc3r-224>
- Reloc3r-512 checkpoint: <https://huggingface.co/siyan824/reloc3r-512>
