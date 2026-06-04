---
type: method-comparison
title: "Feed-forward Visual Relocalization"
direction: [visual-localization, robotics-autonomous-driving, evaluation-benchmarks]
methods: [Reloc-VGGT, Reloc3r, MARePo, FastForward, ACE, Map-Free, ExReNet, MASt3R]
status: initial-completed
confidence: medium
updated: 2026-06-04
---

# 前馈式视觉重定位方法横向对比

> 证据范围：本报告基于 Reloc-VGGT / Reloc3r / MARePo / FastForward 论文、公开项目页、公开 GitHub 仓库和本地 read-only repo inspection；尚未本地复跑 benchmark。结论用于方法分型、复现优先级和实验设计，不替代正式复现实验。

## 结论先行

| 推荐排序 | Method | 推荐场景 | 主要理由 | 主要风险 |
|---:|---|---|---|---|
| 1 | Reloc3r | 当前可跑的 scene-agnostic RPR baseline | 代码、训练脚本、eval scripts、weights 已公开；不需要新场景训练；速度快 | Late fusion / motion averaging 对共线、弱 overlap、retrieval outlier 和尺度退化敏感 |
| 2 | FastForward | map-light feed-forward localization，高精度候选 | 用 posed mapping features 预测 2D-3D correspondences 再 PnP，scale transfer 更自然；论文报告在 Wayspots/Indoor6/Cambridge 强于 RPR | 截至 2026-06-04 项目页未提供可用 GitHub 链接；不能作为当前复现代码 baseline |
| 3 | MARePo | 有 scene map / ACE head 的 map-conditioned pose regression | 直接 metric pose，训练/测试代码与模型公开；7-Scenes 上显著强于传统 APR | 需要每个场景 ACE-style coordinate map；非商用许可证；不是 zero-shot/no-scene-training |
| 4 | Reloc-VGGT | Reloc3r 的高上限 early fusion 研究路线 | 把 source poses/images 放入 VGGT 内部 early fusion，论文指标优于 Reloc3r | 当前仓库只有占位 README，无代码/权重/license；复现 blocked |
| 5 | ACE / MASt3R / feature matching | 强几何或 SCR 对照 | ACE 是 scene-coordinate + RANSAC 强 baseline；MASt3R/feature matching 可构成非 PR 强对照 | ACE 需要每场景训练；MASt3R/feature matching 通常 mapping/solver 成本更高 |

## 1. 对比目标

- **我们为什么比较这些方法**：Issue 目标是分析 Reloc-VGGT、Reloc3r、MARePo，并加入相近的前馈式视觉重定位方法，避免把“无新场景训练”和“有 scene-specific map 条件”的方法混为一类。
- **目标任务 / 应用场景**：给定 query image，估计其在目标场景坐标系下的 6DoF camera pose；场景包括室内 7-Scenes、室外 Cambridge/Wayspots、机器人/AR/自动驾驶候选定位场景。
- **约束**：优先考虑 query-time feed-forward 或少后处理；显式记录是否需要新场景训练、是否需要 3D map/scene coordinate head、是否开源训练、许可证与复现可行性。

## 2. 一页总表

| 维度 | Reloc3r | Reloc-VGGT | MARePo | FastForward | ACE / MASt3R 对照 |
|---|---|---|---|---|---|
| 基本范式 | Scene-agnostic relative pose regression + motion averaging | VGGT early multi-view fusion + source pose tokens | Map-relative APR conditioned on scene coordinate map | Posed mapping features -> query scene coordinates -> PnP | ACE: SCR + RANSAC/PnP；MASt3R: matching/reconstruction + solver |
| 输入 | Query + top-K posed database images | Query + K source images + source poses | Query image -> scene-specific coordinate map + intrinsics | Query + retrieved posed mapping images/features | ACE: query + scene-specific head；MASt3R: query/reference images |
| 输出 | Relative poses, fused absolute pose | Direct query pose | Direct metric pose in scene coordinate system | 2D-3D correspondences, then query pose via PnP | ACE outputs pose via solver；MASt3R outputs matches/points for solver |
| 是否需要新场景训练 | 否 | 论文声称否 | 是，需要 ACE-style coordinate head；可选 fine-tune `M` | 否，使用 mapping images/features/retrieval | ACE 是；MASt3R 通常否但需 matching/SfM/PnP |
| 是否需要显式 map | 需要 posed database images，不需要 3D map | 需要 posed source images，不需要 3D map | 需要 scene coordinate map / ACE head | 需要 posed mapping images 和 feature map representation | ACE 需要 scene coordinate map；MASt3R 可用 image set + solver |
| 真实尺度 / 校准 | Metric scale 来自 database poses + triangulation | Metric scale 来自 source poses / early fusion | Metric scene coordinate system 直接给 scale | Mapping poses + scene normalization transfer scale | ACE scale 由 scene map 给；MASt3R 需 solver/scale handling |
| 数据集 | ScanNet1500、RE10K、ACID、CO3Dv2、7-Scenes、Cambridge | ScanNet1500、CO3Dv2、7-Scenes、Cambridge | Map-Free、7-Scenes、Wayspots、12-Scenes | ARKitScenes、WildRGBD、ScanNet++、MegaDepth、BlenderMVS、Map-Free；eval Wayspots/Indoor6/RIO10/Cambridge/7-Scenes | ACE: 7/12-Scenes、Cambridge、Wayspots；MASt3R 多几何/定位评测 |
| 指标 | AUC@5/10/20、median pose error、FPS | AUC@5/10/20、RRA/RTA/mAA、median pose error、time | Median pose error、threshold accuracy、fps、mapping time | Median pose error、threshold accuracy、latency、mapping prep time | Median pose error、inlier/threshold accuracy、mapping/solver time |
| Git 地址 | [ffrivera0/reloc3r](https://github.com/ffrivera0/reloc3r) | [dtc111111/Reloc-VGGT](https://github.com/dtc111111/Reloc-VGGT) | [nianticlabs/marepo](https://github.com/nianticlabs/marepo) | Project page only: <https://nianticspatial.github.io/fastforward/> | [nianticlabs/ace](https://github.com/nianticlabs/ace), [naver/mast3r](https://github.com/naver/mast3r) |
| 是否开源 | 是 | 否，当前只有占位 README | 是，非商用 | 未发现可用代码仓库 | ACE 是，非商用；MASt3R 是 |
| 是否开源训练 | 是 | 否 | 是 | 否 | ACE 是；MASt3R 需单独核验 |
| 预训练权重 | Reloc3r-224/512 HF links | 未公开 | ACE heads / MARePo paper models links | 未公开 | ACE encoder/head links；MASt3R weights 需单独核验 |
| 推理成本 | Reloc3r-512 论文/README 报告 24 FPS fp32、40 FPS fp16 on RTX 4090 | ScanNet1500 pair-wise 45ms；mask 版更快但略降精度 | 论文表 Wayspots 55.6 fps；每新场景准备约分钟级 | Outdoor top-20 + N=3000 约 0.49s/query on V100 including PnP | ACE query 快但每场景训练；MASt3R solver pipeline 成本高 |
| 许可证 | CC BY-NC-SA 4.0 | unknown | Niantic non-commercial | unknown | ACE non-commercial；MASt3R 需单独核验 |
| 工程风险 | Late fusion、尺度退化、无 correspondences | 无代码/权重，无法复现 | 新场景 map 成本、非商用、数据/环境复杂 | 无开源代码，无法复现；仍需 PnP/RANSAC | ACE 每场景训练；MASt3R/SfM/solver 工程链较长 |

## 3. 方法逐个分析

### Reloc3r

- **核心思路**：训练一个跨场景 relative pose regression network；query 时对 top-K database images 分别回归相对姿态，再用 motion averaging 得到 absolute pose。
- **优势**：可复现度最高；训练/eval/demo/weights 已公开；不需要新场景训练；速度快；是 Reloc-VGGT 直接 baseline。
- **局限**：多视图信息只在网络外 late fusion；没有 2D-3D correspondences；共线/弱 baseline/错误 retrieval 会影响 scale。
- **复现要点**：优先跑 `wild_relpose.py` / `wild_visloc.py`，再跑 7-Scenes/Cambridge；记录 retrieval top-K、AMP、GPU、checkpoint。

### Reloc-VGGT

- **核心思路**：用 VGGT backbone 让 query/source image tokens 与 source pose tokens 在网络内部 early fusion，直接预测 query pose；sparse mask attention 降低多帧 attention 成本。
- **优势**：论文结果优于 Reloc3r；相较 late fusion，更可能处理复杂多 source 空间关系。
- **局限**：截至 2026-06-04 无可用代码/权重/license；不能安排短期复现。
- **复现要点**：等待 release 后先核验公开 checkpoint 是否对应论文表；重点复跑 Reloc3r 同样 retrieval 设置，比较 7-Scenes/Cambridge 与 source-pose noise robustness。

### MARePo

- **核心思路**：用 ACE-style scene coordinate head 输出 query 的 scene coordinate map，再用 scene-agnostic transformer regressor 直接输出 map-relative pose。
- **优势**：metric coordinate frame 稳定；训练/测试/模型链接公开；对传统 APR 提升很大；可作为 map-conditioned APR/SCR hybrid baseline。
- **局限**：需要每个新场景训练/准备 scene coordinate head；非商用；不是 Reloc3r/Reloc-VGGT 那种 no per-scene training。
- **复现要点**：先用 README 的 pre-trained ACE heads + paper models 跑 7-Scenes/Wayspots；不要把 ACE-head training time 从系统成本里抹掉。

### FastForward

- **核心思路**：把多张 posed mapping images 表示为一组 3D-anchored image features；query 与 fixed-size map feature set 通过 Transformer 交互，输出 query image-to-scene correspondences，再用 PnP-RANSAC 求 pose。
- **优势**：和 Reloc3r 一样不需要每场景训练，但比 pair-wise RPR 更直接利用 multi-view map representation；论文报告 Wayspots median translation 0.17m，Indoor6/7-Scenes/Cambridge 对 RPR 有强对照价值。
- **局限**：官方 project page 当前只提供 paper 和 citation，没有实际 GitHub 链接；不能作为马上可跑的 baseline。
- **复现要点**：等待代码 release；如果自行实现，必须缓存 mapping features、锁定 retrieval top-K、N feature count、PnP confidence threshold。

### ACE / MASt3R / feature-matching 对照

- **核心思路**：ACE 是 scene coordinate regression + robust pose solver；MASt3R/feature matching 是通过 correspondences / 3D points / solver 得 pose。
- **优势**：几何约束强，可解释，通常是定位任务强 baseline。
- **局限**：不是纯 pose regression；ACE 需要每场景训练；MASt3R/SfM/PnP 工程成本和 latency 可能高。
- **复现要点**：作为“非 PR / solver-based”上限对照；明确 mapping time、storage、solver time、query latency。

## 4. 场景化选择

| 场景 | 推荐 | 原因 |
|---|---|---|
| 快速可跑 baseline | Reloc3r | 代码/权重/训练/eval 都公开，最直接验证 issue 目标 |
| 高上限 RPR 研究 | Reloc-VGGT + Reloc3r | Reloc-VGGT 论文展示 early fusion 优势，但当前只能记录和等待 release |
| 有目标场景地图 / 可训练 ACE head | MARePo + ACE | 显式 scene coordinates 提供稳定 metric pose |
| 无新场景训练但希望几何可解释 | FastForward | 论文路线输出 2D-3D correspondences + PnP，比 direct RPR 更可解释；等待代码 |
| 大场景高精度对照 | MASt3R / feature matching / ACE | Solver-based 方法应作为强上限或工程对照 |
| 自动驾驶/机器人定位前端 | Reloc3r 做 proposal，MARePo/ACE/FastForward 做有地图对照 | Reloc3r 快；map-conditioned 或 correspondence-based 方法更适合 metric map consistency |

## 5. 证据与不确定性

### 已确认事实

- Reloc3r 仓库公开推理、评测、训练脚本、dataset preprocessing scripts、Reloc3r-224/512 checkpoint 链接，license 为 CC BY-NC-SA 4.0。
- Reloc-VGGT 仓库截至 2026-06-04 只有标题 README，无代码/权重/license。
- MARePo 仓库公开训练/测试/预处理脚本和模型下载链接，license 为 Niantic non-commercial。
- FastForward project page 公开 ICLR 2026 paper 和方法说明，但页面中的 GitHub 按钮被注释，未提供实际 code link。

### 论文确认的关键结果

- Reloc-VGGT vs Reloc3r：ScanNet1500 AUC@5/10/20 为 36.35/58.62/75.90 vs 34.79/58.37/75.56；7-Scenes 平均 0.031m/0.896deg vs 0.041m/1.035deg；Cambridge Average(4) 0.32m/0.37deg vs 0.38m/0.52deg。
- Reloc3r：CO3Dv2 multi-view RRA@15 95.8%、RTA@15 93.7%、mAA@30 82.9%；ScanNet1500 24 FPS fp32 on RTX 4090。
- MARePo：7-Scenes MARePo 3.9cm/1.68deg，MARePoS 3.2cm/1.54deg；ACE 2.8cm/0.93deg；PoseNet 44cm/10.4deg。
- FastForward：论文报告 Wayspots median translation 0.17m，且在 7-Scenes / Indoor6 / RIO10 / Cambridge 中对 RPR 和部分 SCR/feature-matching baseline 有优势或可比表现；outdoor top-20 + N=3000 约 0.49s/query on V100 including PnP。

### 推断

- Reloc-VGGT 的真正价值取决于 release 后的 retrieval/source-pose robustness，而不仅是论文表格。
- FastForward 可能是比 Reloc-VGGT 更工程友好的“无新场景训练 + 几何可解释”方向，因为它保留 correspondences/PnP；但当前无代码。
- MARePo 不应该与 Reloc3r/Reloc-VGGT 归为同一列“无场景训练”方法，应标为 map-conditioned pose regression。

### 待验证

- Reloc-VGGT 是否会释放完整代码、权重、license、训练数据或 eval protocol。
- FastForward 是否会释放 GitHub 代码和模型，许可证是否可用。
- Reloc3r 在本地自有视频 / 行车片段上 top-K retrieval、source pose noise、linear motion、low-overlap 的失败边界。
- MARePo 在新场景 ACE head 质量较差或长期变化下的鲁棒性。

## 6. 后续动作

- [x] 创建 Reloc-VGGT / Reloc3r / MARePo 单篇分析
- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [x] 更新 `indices/comparisons.md`
- [ ] 后续优先创建 `reproductions/visual-localization/reloc3r/README.md`
- [ ] 监控 Reloc-VGGT / FastForward 代码 release

## Sources

- Reloc-VGGT paper: <https://arxiv.org/abs/2512.21883>
- Reloc-VGGT GitHub: <https://github.com/dtc111111/Reloc-VGGT>
- Reloc3r paper: <https://arxiv.org/abs/2412.08376>
- Reloc3r GitHub: <https://github.com/ffrivera0/reloc3r>
- MARePo paper: <https://arxiv.org/abs/2404.09884>
- MARePo project/GitHub: <https://nianticlabs.github.io/marepo/> / <https://github.com/nianticlabs/marepo>
- ACE paper/project/GitHub: <https://arxiv.org/abs/2305.14059>, <https://nianticlabs.github.io/ace/>, <https://github.com/nianticlabs/ace>
- FastForward paper/project: <https://arxiv.org/abs/2510.00978>, <https://nianticspatial.github.io/fastforward/>
- Map-Free paper/repo: <https://arxiv.org/abs/2210.05494>, <https://github.com/nianticlabs/map-free-reloc>
- MASt3R paper/repo: <https://arxiv.org/abs/2406.09756>, <https://github.com/naver/mast3r>
