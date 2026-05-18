---
type: method-comparison
title: "Any-view Visual Geometry Foundation Models"
direction: [3d-reconstruction, dense-vision, robotics-autonomous-driving]
methods: [Depth-Anything-3, VGGT-Omega, VGGT, Pi3, MapAnything, LingBot-Map, Depth-Anything-2]
status: initial-completed
confidence: medium
updated: 2026-05-18
---

# Any-view Visual Geometry Foundation Models 横向对比

> 证据范围：本对比以 Depth Anything 3 / VGGT-Ω / Pi3 论文、公开 GitHub 仓库、README/model cards、DA3-BENCH 文档以及 FrontierLab 已有 MapAnything/LingBot-Map 资料为主；尚未独立复跑 benchmark。结论用于复现优先级和方法分工，不替代正式实验报告。

## 结论先行

| 推荐定位 | Method | 主要理由 | 主要风险 |
|---|---|---|---|
| **any-view geometry 主候选** | **Depth Anything 3** | depth-ray representation 统一 depth/pose/point cloud；单 plain transformer 复用 DINOv2 预训练；公开 CLI/API/benchmark/weights；论文表中 pose、reconstruction、NVS 多项强于 VGGT/Pi3/MapAnything | 训练代码未开源；大模型权重 CC BY-NC 4.0；完整训练成本极高；动态场景仍需额外处理 |
| **dynamic/RGB-only 强 backbone** | **VGGT-Ω** | VGGT scaling 版，在静态+动态 camera/depth benchmark 上大幅超过 VGGT/Pi3/DA3/MegaSaM；register attention 降训练成本，scene tokens 可服务 VLA/语言对齐；长帧数推理效率强 | 公开仓库只有 inference/demo/model code；HF 权重 gated；FAIR Noncommercial Research License；完整训练依赖 internal data/128 H100/10B 未公开 |
| **metric prompt 主候选** | **MapAnything** | 更强调 metric reconstruction 和 camera/pose/depth/point-map prompts；对自动驾驶已有标定、ego pose、LiDAR depth 更直接 | 不是 3DGS/NVS 主模型；需要外部动态物体处理；现有报告已覆盖，需与 DA3 实测分工 |
| **经典强 baseline** | **VGGT** | 任意视角视觉几何强 baseline，生态和论文引用度高 | DA3 论文中多项指标被 DA3 超过；架构更重，训练/推理冗余更高 |
| **reference-free 鲁棒性 baseline** | **Pi3** | permutation-equivariant / affine-invariant formulation 去掉固定参考帧；对输入顺序和 reference view 选择更稳；training/evaluation 分支已公开 | 原始 Pi3 不是 metric 主线；权重 CC BY-NC 4.0；完整训练含 internal dataset；Pi3X 与论文主结果需分开验证 |
| **online visual mapping 主候选** | **LingBot-Map** | causal streaming、长视频、实时 pose/depth/point cloud 明确；与 DA3 的离线 any-view 范式互补 | 训练代码未开源；没有显式 loop closure；不是多先验 prompt 模型 |
| **单目 depth baseline** | **Depth Anything 2** | 轻量、成熟、单图相对/metric depth 仍实用 | 不输出多视图一致 pose/ray/point cloud；不能替代 DA3 的 any-view geometry |

## 1. 方法分工

| 场景 | 优先方法 | 原因 |
|---|---|---|
| 无相机位姿的多图/视频帧，想直接恢复 pose + depth + point cloud | DA3 | 论文目标正是 posed/unposed any-view geometry，仓库提供完整 inference API |
| 静态+动态 RGB/video reconstruction，需要强 feed-forward baseline | VGGT-Ω | 论文在 7Scenes/NRGBD/ETH3D/DyCheck/Sintel/TUM-Dynamic 上整体超过 VGGT/Pi3/DA3/MegaSaM；register tokens 还可作为空间表征 |
| 有相机标定、ego pose、LiDAR/stereo depth，追求真实尺度 | MapAnything + DA3Metric/DA3Nested 对照 | MapAnything 的 metric priors 更自然；DA3Nested/Metric 可作为强视觉几何对照 |
| 长视频 online / streaming mapping | LingBot-Map 或 DA3-Streaming | LingBot-Map 是专门 streaming 架构；DA3-Streaming 是仓库推理管线但需单独验证 |
| feed-forward 3DGS / NVS | DA3-Giant / DA3Nested | 论文 Table 5 显示 DA3 backbone 替代 VGGT/Fast3R/MV-DUSt3R 后 NVS 指标更强 |
| VLA/语言对齐中的空间 token | VGGT-Ω | 论文显示冻结 scene tokens 接入 OpenVLA-OFT 提升 LIBERO success rate，text-aligned checkpoint 支持 language retrieval |
| 单图相对深度、部署简单 | DA2 或 DA3MONO-LARGE | DA2 生态成熟；DA3MONO 在论文 Table 10 中比 DA2 student 更强 |
| 研究架构机制 | DA3 vs VGGT vs Pi3 | DA3 的“plain transformer + depth-ray”与 VGGT/Pi3 是清晰对照 |

## 2. 一页总表

| 维度 | Depth Anything 3 | VGGT | VGGT-Ω | Pi3 | MapAnything | LingBot-Map |
|---|---|---|---|---|---|---|
| 核心范式 | Plain transformer + depth-ray + dual-DPT | 多任务视觉几何 transformer | VGGT scaling + DINOv3 + register attention + dynamic data | permutation-equivariant / affine-invariant geometry；reference-free local pointmap | metric promptable feed-forward reconstruction | causal streaming 3D reconstruction |
| 输入 | 任意数量 RGB；可选 camera pose | 任意数量 RGB | 多帧 RGB images/video；公开接口不以 camera/depth prompt 为主 | 单图、视频或无序 RGB views；Pi3X 可选 camera/intrinsics/depth 条件 | RGB + camera/pose/depth/point priors | RGB video stream |
| 输出 | depth、ray、camera、point cloud、3DGS | camera、depth、3D points | camera encoding、depth、confidence、camera/register tokens；可视化 point cloud | camera poses、local point maps、confidence、global point cloud | metric pose/depth/point map | streaming pose/depth/point cloud |
| 是否 posed/unposed 兼容 | 是 | 是 | 主要 unposed RGB/video；auxiliary inputs 不是预训练主线 | 是 | 是，且 prompt 更灵活 | 主要 unposed/streaming |
| 是否 metric 主线 | 部分：DA3Metric/DA3Nested | 部分 | 不是 prompt-first metric 主线；更适合作强 RGB geometry backbone | 原始 Pi3 偏 scale-invariant；Pi3X 支持近似 metric scale | 是 | 可估计轨迹/深度但更偏 VO/relative mapping |
| 是否 3DGS/NVS | 是，GS-DPT / DA3-Giant/Nested | 可作 backbone | 可作为几何初始化/point cloud backbone，论文不以 3DGS 为主线 | 可作 backbone | 可导出衔接 | 不是主线 |
| 是否 streaming | repo 有 DA3-Streaming 推理管线 | 需要长视频改造 | batch feed-forward；论文/README 显示长帧数推理能力强，但不是 causal online state | 需要长视频改造 | 非主线 | 是，核心目标 |
| 代码许可证 | Apache-2.0 | 待按具体仓库核验 | FAIR Noncommercial Research License | main BSD 3-Clause；evaluation branch 写 academic BSD-2/commercial contact，需按分支核验 | 已有报告记录 | Apache-2.0 |
| 权重许可证 | Base/Small/Metric/Mono Apache-2.0；Giant/Large/Nested CC BY-NC 4.0 | 待按模型卡核验 | Gated HF；FAIR Noncommercial Research License；公开 README 只列 1B checkpoints | Pi3 / Pi3X weights CC BY-NC 4.0 | 已有报告记录 | 仓库 Apache-2.0；权重许可证需按模型卡核验 |
| 训练代码 | 未开源 | 待补充 | `\`；公开仓库主要是 inference/demo/model code | 是，training branch；完整训练数据/算力不可完整复刻 | 是 | `\` |
| 当前复现建议 | inference + DA3-BENCH 子集 | baseline 对照 | 申请 HF 权重后做 inference + 动态视频小评测；测 register tokens downstream probe | inference + evaluation branch；重点测输入顺序鲁棒性与 Pi3/Pi3X 差异 | metric prompt 对照 | streaming 对照 |

## 3. 关键差异

### 3.1 VGGT-Ω vs DA3 / VGGT / Pi3

- VGGT-Ω 是 VGGT 的 scaling 路线：用 DINOv3 初始化、register attention、单 dense head + 多任务 loss、大规模动态视频数据和 teacher-student 自监督，把 camera/depth reconstruction 推向更强动态 benchmark。
- 与 DA3 相比，VGGT-Ω 不以 depth-ray representation、3DGS/NVS API 和 benchmark 工程生态为核心；它更强调 RGB/video reconstruction 的规模化和 scene/register tokens 的空间表征价值。
- 与 Pi3 相比，VGGT-Ω 不是 reference-free/local pointmap 论文；它仍有 camera token，并且主要解决 scaling、dynamic data、训练效率和 representation reuse。研究输入顺序鲁棒性仍应保留 Pi3/Pi3X。
- 与原 VGGT 相比，VGGT-Ω 在论文表中显著提高静态和动态 camera/depth 指标，并在公开 README 中给出更长帧数推理显存表；但公开仓库没有训练/evaluation pipeline，复现难度高于普通开源 baseline。
- 自动驾驶里，VGGT-Ω 适合作为强 RGB-only geometry backbone 或初始化来源；若需要标定、ego pose、LiDAR depth 等真尺度约束，MapAnything/DA3Metric/Pi3X 仍应一起作为主对照。

### 3.2 DA3 vs VGGT

- 共同点：都是任意视角 visual geometry foundation model，都能从多图估计 camera/depth/3D geometry。
- DA3 差异：用完整预训练的单 DINOv2-style transformer，加 token rearrangement 做 cross-view attention；不用 VGGT-style 多 transformer 冗余堆叠。
- 论文证据：DA3 ablation 中 VGGT-style 同参数设计降到 proposed architecture 约 79.8%；Table 8 中 DA3-Giant 支持更多 images，FPS 也高于 VGGT reference。
- 工程判断：VGGT 仍是必须保留的 baseline；但如果只选一个新模型做复现，DA3 优先级更高。

### 3.3 DA3 vs Pi3

- Pi3 的研究点是 permutation-equivariant 和 affine/scale-invariant geometry，对 unordered views、reference-view bias、输入顺序鲁棒性研究重要。
- DA3 更工程化：公开权重、API、CLI、benchmark evaluator、3DGS/NVS head，更适合快速做应用验证。
- Pi3 代码生态已经补上 training/evaluation 分支，且 main 分支发布 Pi3X：convolutional head 降低 grid artifacts，支持 camera/intrinsics/depth 条件注入和近似 metric scale。
- 自动驾驶/机器人里若需要严格 metric scale，原始 Pi3 仍需外部尺度锚定；DA3Nested/Metric、Pi3X、MapAnything 应放在同一小场景中实测。
- 复现时不要混淆论文 Pi3 和工程更新 Pi3X：论文主要证据来自 reference-free formulation；Pi3X 是后续工程增强。

### 3.4 DA3 vs MapAnything

- MapAnything 更像“带几何先验的 metric reconstruction 框架”：camera、pose、depth、point-map prompts 是核心能力。
- DA3 更像“通用视觉几何 backbone”：即使没有显式 metric priors，也能通过 depth-ray 统一 pose/depth/point geometry。
- 推荐组合：MapAnything 做主 metric prompt baseline；DA3 做纯视觉 any-view geometry 和 NVS backbone baseline；两者在同一行车片段上比较 pose、scale、点云重影和动态物体鲁棒性。

### 3.5 DA3 vs LingBot-Map / DA3-Streaming

- LingBot-Map 是专门为 causal streaming 设计；论文中的 GCA 管理 anchor、local window、trajectory memory。
- DA3 本体更像离线或 batch any-view geometry 模型；当前 GitHub repo 后续加入 DA3-Streaming，通过 chunk streaming 管理长视频推理，README 报告 KITTI/TUM 对 VGGT-Long/Pi-Long 有速度和 ATE 优势。
- DA3-Streaming 是工程推理管线，不等价于 DA3 论文训练方法；后续复现应单独记录。

## 4. 复现优先级

1. **VGGT-Ω gated checkpoint sanity check**：申请 `facebook/VGGT-Omega` 权重，先跑 3-10 张图和一个短视频，导出 depth / cameras / GLB，记录 1B-512 显存和速度。
2. **DA3 inference sanity check**：`DA3-BASE` 或 `DA3-LARGE-1.1` 跑官方 SOH 示例，导出 depth / npz / glb，记录显存、速度、点云质量。
3. **动态视频小评测**：在 Sintel/TUM-Dynamic 或自有行车片段上比较 VGGT-Ω、DA3、Pi3/Pi3X、VGGT 的 camera/depth/qualitative artifacts。
4. **DA3-BENCH mini evaluation**：只下载 HiRoom 或 7Scenes，跑 `pose` mode，确认 evaluator 可用。
5. **Pi3 / Pi3X order robustness sanity check**：同一组多视图输入随机打乱顺序，比较 camera pose、point cloud、confidence 和 grid artifacts；记录 Pi3 与 Pi3X 的差异。
6. **MapAnything metric prompt 横向小场景**：同一组多视图输入，加/不加 camera/pose/LiDAR depth，比较 pose、scale、depth 和点云 F1。
7. **DA3-Streaming 单独评估**：用 KITTI/TUM 或自有长视频测试 chunk size、overlap、VRAM、ATE；不要与 DA3 paper core metrics 混在一起。
8. **许可证筛选**：商业相关实验优先使用 Apache 权重或仅记录 CC BY-NC / FAIR Noncommercial 权重为 research-only。

## 5. 不确定性

- 本对比尚未复跑 DA3-BENCH；所有 DA3 定量结论来自论文和官方 benchmark docs。
- VGGT-Ω 尚未本地复跑；论文 10B 主结果与公开 1B gated checkpoints 需要分开记录。
- VGGT 当前仓库/权重许可证未在本次任务逐项核验；后续若做正式复现实验，需要锁定 commit 和 model card。
- Pi3 main README 与 evaluation branch README 的许可证措辞不同；商用或再发布前需逐分支核验。
- DA3 README 提到 `-1.1` 权重修复 training bug；论文表格与 refreshed checkpoints 的精确对应关系需复跑确认。

## Sources

- VGGT-Ω analysis note: [`../../papers/3d-reconstruction/2026-vggt-omega.md`](../../papers/3d-reconstruction/2026-vggt-omega.md)
- VGGT-Ω paper: <https://arxiv.org/abs/2605.15195>
- VGGT-Ω GitHub: <https://github.com/facebookresearch/vggt-omega>
- Depth Anything 3 paper: <https://arxiv.org/abs/2511.10647>
- Depth Anything 3 GitHub: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Depth Anything 3 analysis note: [`../../papers/3d-reconstruction/2025-depth-anything-3.md`](../../papers/3d-reconstruction/2025-depth-anything-3.md)
- DA3-BENCH: <https://huggingface.co/datasets/depth-anything/DA3-BENCH>
- Pi3 analysis note: [`../../papers/3d-reconstruction/2026-pi3.md`](../../papers/3d-reconstruction/2026-pi3.md)
- Pi3 GitHub: <https://github.com/yyfz/Pi3>
- Pi3 OpenReview: <https://openreview.net/forum?id=DTQIjngDta>
- LingBot-Map analysis note: [`../../papers/3d-reconstruction/2026-lingbot-map.md`](../../papers/3d-reconstruction/2026-lingbot-map.md)
- Existing feed-forward comparison: [`../../reports/feedforward_3d_reconstruction_compare.md`](../../reports/feedforward_3d_reconstruction_compare.md)
