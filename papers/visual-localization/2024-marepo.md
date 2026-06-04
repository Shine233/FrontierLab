---
type: paper-analysis
title: "Map-Relative Pose Regression for Visual Re-Localization"
short_name: "MARePo"
year: 2024
venue: "CVPR 2024 Highlight"
arxiv: "2404.09884"
paper_url: "https://arxiv.org/abs/2404.09884"
pdf_url: "https://arxiv.org/pdf/2404.09884"
code: "https://github.com/nianticlabs/marepo"
github: "https://github.com/nianticlabs/marepo"
project_page: "https://nianticlabs.github.io/marepo/"
open_source: true
license: "Niantic non-commercial marepo License"
training_open_source: true
weights_availability: "Pre-trained ACE heads and marepo paper models are linked from the README; ace_encoder_pretrained.pt and dummy ACE head are in the repo."
data_availability: "Map-Free training scenes archive and setup scripts for 7-Scenes, 12-Scenes, Wayspots are linked/provided; users must check dataset licenses."
direction: [visual-localization, robotics-autonomous-driving, evaluation-benchmarks]
method_family: [map-relative-pose-regression, absolute-pose-regression, scene-coordinate-regression, ace-conditioned-localization]
tasks: [visual-relocalization, absolute-camera-pose-estimation, scene-coordinate-conditioned-pose-regression]
datasets: [Map-Free, 7-Scenes, 12-Scenes, Niantic-Wayspots]
metrics: [median-translation-error, median-rotation-error, accuracy-under-threshold, throughput-fps, mapping-time]
status: compared
reproduction: planned
confidence: high
updated: 2026-06-04
---

# MARePo：带场景坐标图条件的 Map-Relative Pose Regression

## 结论先行

- **一句话定位**：MARePo 是 APR/SCR 混合路线：query 时是单帧前馈 pose regression，但输入不是纯 RGB，而是 scene-specific coordinate map，因此 pose 是相对于目标地图坐标系的 metric pose。
- **核心差异**：Reloc3r/Reloc-VGGT 只要求 posed database images 和 retrieval；MARePo 要先为每个新场景准备 ACE-style scene coordinate head/map representation。它不是完全 map-free，但能利用显式 3D 几何先验提升 metric pose 稳定性。
- **论文证据**：7-Scenes 上 MARePo 平均 3.9cm/1.68deg，scene-specific fine-tuned MARePoS 平均 3.2cm/1.54deg；ACE 作为 SCR baseline 平均 2.8cm/0.93deg。MARePo 推理 55.6 fps，且相对 APR 方法精度大幅提升。
- **代码状态**：GitHub 开源训练、测试、预处理脚本和预训练模型链接，但许可证是 Niantic non-commercial license；商业或公司内部产品化需单独授权。
- **工程判断**：如果你能接受每个新场景训练/准备 ACE head，MARePo 是强 pose-regression baseline；如果目标是“完全不为新场景训练”，就应优先比较 Reloc3r、Reloc-VGGT、FastForward。

## 1. 这篇论文解决什么问题？

### 已确认的论文事实

- **问题定义**：APR 直接从 RGB 回归 pose，但通常需要每场景大量训练数据且精度低；SCR/ACE 精度高，但需要预测 scene coordinates 并通过 RANSAC/PnP solver。MARePo 尝试保留 APR 的 feed-forward simplicity，同时把 scene-specific geometry 作为条件输入。
- **输入 / 输出**：输入 query image 经 scene-specific coordinate regressor `G_S` 输出的 scene coordinate map，以及 camera intrinsics；map-relative regressor `M` 输出相对于 scene coordinate system 的 6DoF metric pose。
- **目标场景**：indoor/outdoor visual relocalization，特别是 7-Scenes 与 Niantic Wayspots。
- **训练设定**：`M` 在 Map-Free dataset 上训练一次；新场景需要训练 ACE-style scene coordinate head `G_S`，可选择对 `M` 做 1-15 分钟 scene-specific fine-tuning。

### 我的理解

MARePo 的关键不是“又一个 APR 网络”，而是把 pose regression 的坐标系问题外包给 scene coordinate map。只看 query 推理，它是直接输出 pose；但从系统角度，它仍然依赖每个场景的 map representation。

这使 MARePo 和 Reloc3r/Reloc-VGGT 的比较必须很小心：MARePo 在新场景中不是零准备，它需要 ACE head 或等价 SCR map；Reloc3r/Reloc-VGGT 则只需要 posed database views。MARePo 的优势是 metric coordinate frame 稳定，缺点是新场景准备成本和许可证限制。

## 2. 方法概览

### 2.1 Scene-specific coordinate regressor `G_S`

- MARePo 使用 ACE-style scene coordinate regression 作为 scene-specific map representation。
- 对每个新场景训练一个 coordinate head，输入 RGB，输出每个像素对应的 3D scene coordinates。
- 这些 scene coordinates 不直接走 RANSAC/PnP，而是作为 map-relative pose regressor 的输入。

### 2.2 Scene-agnostic map-relative regressor `M`

- `M` 是 transformer-based pose regressor，训练一次后跨场景使用。
- 输入 scene coordinate map、camera-aware 2D positional embedding、3D positional embedding。
- 输出 10D pose representation：translation 用 homogeneous coordinates，rotation 用 6D representation。
- 训练时每 4 个 transformer blocks 后加 auxiliary pose losses；推理只用最后输出。

### 2.3 Optional scene-specific fine-tuning

- `marepo`：通用 `M` 直接用于新场景。
- `marepoS`：在新场景 mapping images 上 fine-tune `M`，论文称 Wayspots 上约 1 分钟，7-Scenes 最多约 15 分钟。
- 这提升精度，但更偏 scene-specific APR。

## 3. 关键贡献

1. **提出 map-relative pose regression**：让 APR 输出相对于显式 scene coordinate map 的 pose，而不是把整个场景隐式塞进网络权重。
2. **把 scene-specific geometry 与 scene-agnostic pose regressor 分离**：`G_S` 每场景训练，`M` 跨场景学习 scene coordinate map 到 pose 的通用关系。
3. **缩短 APR 新场景准备时间**：相对传统 APR 几小时/几天训练，MARePo 依赖 ACE-style head 后可在分钟级准备。
4. **在 indoor/outdoor 上验证**：7-Scenes 与 Wayspots 显示 MARePo 显著强于 PoseNet/MS-Transformer/DFNet 等 APR。

## 4. 实验与证据

| 维度 | 内容 |
|---|---|
| 数据集 | Map-Free training set、7-Scenes、Niantic Wayspots、12-Scenes supplementary |
| Baseline | DSAC*、ACE、PoseNet、MapNet、Direct-PN、MS-Transformer、DFNet、LENS 等 |
| 指标 | Median translation/rotation error；accuracy under 10cm/5deg、0.5m/5deg；mapping time；throughput |
| 主要结果 | 7-Scenes：MARePo 3.9cm/1.68deg，MARePoS 3.2cm/1.54deg；ACE 2.8cm/0.93deg；PoseNet 44cm/10.4deg。Wayspots：MARePo 推理 55.6 fps，显著优于传统 APR，并接近/对照 SCR 路线 |
| 训练成本 | `M` 训练使用 Map-Free 数据，论文提到约 10 天；每新场景 `G_S` 训练约 5 分钟，`marepoS` fine-tune 约 1-15 分钟 |
| 消融 | Re-Attention、dynamic positional encoding、model dimension、training strategy、auxiliary losses、scene coordinate backbone、coordinate noise |
| 失败案例 | 强依赖 scene coordinate map 质量；新场景需要 coordinate head；不是 zero-shot posed-database-only 方法 |

## 5. 已确认的代码/仓库事实

- GitHub：<https://github.com/nianticlabs/marepo>
- 2026-06-04 read-only check：`git ls-remote` HEAD 为 `22ef5df8e137e68e0104369de3526fdbad5b445b`。
- License：Niantic non-commercial marepo License，专利待定，非商业使用限定明确。
- 公开文件包括 `train_marepo.py`、`test_marepo.py`、`preprocess_marepo.py`、`train_ace.py`、`test_ace.py`、`scripts/train_marepo.sh`、`scripts/train_mapfree.sh`、`scripts/test_7scenes.sh`、`scripts/test_wayspots.sh` 等。
- README 链接 pre-trained ACE heads、MARePo paper models、Map-Free training scenes archive。
- 仓库包含 `ace_encoder_pretrained.pt`、`ace_head_dummy.pt`，以及 DSAC* C++ bindings 相关文件。

## 6. 局限与风险

### 论文/代码层面已确认

- 非商用许可证，商业或公司内产品化使用需联系 Niantic。
- 每个新场景需要 scene coordinate head；这不是无地图/无场景训练方法。
- Map-Free augmented training set 可能需要至少 4TB 存储。
- `M` 从头训练推荐至少 8 V100 或同等硬件，并非轻量复现。
- 训练/评估依赖 ACE/DSAC* 生态和多个数据集脚本，环境复杂度高于 Reloc3r 推理。

### 我的推断风险

- MARePo 直接 pose regression 不使用 RANSAC/PnP solver，可能牺牲部分 outlier rejection；虽然输入是 scene coordinates，但错误 coordinate map 会直接影响 pose。
- 与 ACE 相比，MARePo 在 7-Scenes 平均精度仍略低于 ACE；它的优势是 feed-forward pose regression 和 APR 对照，而不是绝对精度压倒 SCR。
- 对动态/长期变化场景，scene coordinate head 需要更新；否则 map-relative pose 可能过期。

## 7. 与相似方法对比

| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| ACE | 都依赖 scene coordinate regression | ACE 用 scene coordinates + RANSAC/PnP；MARePo 用 scene coordinates + pose regressor | 追求成熟 SCR 精度选 ACE；研究 direct pose regression with geometry condition 选 MARePo |
| Reloc3r | 都能 query-time feed-forward 输出 pose | Reloc3r 无每场景训练，依赖 posed database images；MARePo 要 scene coordinate map | 无场景训练/快速 baseline 选 Reloc3r；有目标场景 map 选 MARePo |
| Reloc-VGGT | 都直接回归 query pose | Reloc-VGGT early-fuses source poses/images；MARePo conditions on scene coordinate map | 关注 RPR early fusion 选 Reloc-VGGT；关注 map-conditioned APR 选 MARePo |
| FastForward | 都利用 mapping poses / scene geometry | FastForward 预测 2D-3D correspondences 再 PnP，不训练 per-scene network；MARePo 训练 scene coordinate head 并直接回归 pose | 要 map-light / no per-scene training 选 FastForward；已有 ACE map 选 MARePo |
| PoseNet / MS-Transformer / DFNet | 都属于 APR 或 APR-like pose regression | MARePo 显式条件化 scene coordinate map，精度高很多 | 历史 APR baseline 可保留；新工作更应比较 MARePo |

## 8. 复现判断

- Git 地址：<https://github.com/nianticlabs/marepo>
- 是否开源：是，非商用。
- 是否开源训练：是。
- 代码可用性：提供训练、测试、数据预处理和预训练模型链接。
- 权重可用性：pre-trained ACE heads、MARePo paper models、ACE encoder。
- 数据可获得性：Map-Free archive、7-Scenes/12-Scenes/Wayspots setup scripts；需检查数据许可证。
- 预计环境成本：推理/评测可单 GPU；训练 `M` 约 8 V100 / 10 天量级；每场景 ACE head 约分钟级。
- 最小复现路径：
  1. 创建 `public_marepo` conda env。
  2. 下载 7-Scenes 或 Wayspots。
  3. 下载 pre-trained ACE heads 和 MARePo paper models 到 `logs/`。
  4. 跑 `scripts/test_7scenes.sh` 或 `scripts/test_wayspots.sh`。
  5. 若研究新场景，先训练 ACE head，再生成 MARePo preprocess data，最后测试/fine-tune `marepoS`。
- 是否值得复现：值得作为 map-conditioned APR/SCR hybrid baseline；但不要和 zero-shot/RPR 方法混为同一类。

## 9. 后续动作

- [x] 创建 MARePo 单篇论文分析
- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [x] 创建 visual localization 横向对比
- [ ] 若开始复现，创建 `reproductions/visual-localization/marepo/README.md`

## Sources

- Paper: <https://arxiv.org/abs/2404.09884>
- PDF: <https://arxiv.org/pdf/2404.09884>
- Project page: <https://nianticlabs.github.io/marepo/>
- GitHub: <https://github.com/nianticlabs/marepo>
- ACE project/repo: <https://nianticlabs.github.io/ace/>, <https://github.com/nianticlabs/ace>
