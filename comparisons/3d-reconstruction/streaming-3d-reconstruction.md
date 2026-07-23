---
type: method-comparison
title: "Streaming 3D Reconstruction Methods"
direction: [3d-reconstruction, robotics-autonomous-driving]
methods: [LingBot-Map, Stream3R, Wint3R, TTT3R, CUT3R, InfiniteVGGT, SLAM3R, Spann3R, StreamVGGT, MAGiSt3R]
status: initial-completed
confidence: medium
updated: 2026-07-23
---

# Streaming 3D Reconstruction 方法横向对比

> 证据范围：本对比主要基于 LingBot-Map 论文中的统一 benchmark 表与公开 GitHub 仓库状态；尚未独立复跑全部 baseline。结论适合作为 FrontierLab 的初始方向索引和复现优先级判断。

## 结论先行

| 推荐排序 | Method | 推荐场景 | 主要理由 | 主要风险 |
|---:|---|---|---|---|
| 1 | **LingBot-Map** | 长视频、机器人/自动驾驶在线视觉建图、实时或接近实时 streaming pose + depth | GCA 同时保留 anchor、local window、trajectory memory；Oxford Spires dense 3,840 帧 ATE 7.11、20.29 FPS；GitHub Apache-2.0 且有 demo/weights | 训练代码和 evaluation benchmark 尚未开源；没有显式 loop closure；未融合 LiDAR/IMU；超长 VO mode 有窗口对齐漂移 |
| 2 | **Stream3R / Stream3R-w** | streaming 3D reconstruction baseline，尤其是需要 causal transformer 对照 | 论文表中在 Tanks and Temples 上是最强 baseline 之一，且 7-Scenes ATE 接近 LingBot-Map | 长序列 Oxford Spires drift 大；F1/ATE 多数指标落后 |
| 3 | **Wint3R / TTT3R** | windowed streaming 或 test-time-training baseline | Wint3R 在 ETH3D reconstruction baseline 中较强；TTT3R 在若干短序列指标接近 | Wint3R 速度低于 LingBot-Map；TTT 类方法有额外 test-time overhead；长序列稳定性不足 |
| 4 | **CUT3R / InfiniteVGGT / SLAM3R / Spann3R / StreamVGGT** | 消融/历史 baseline | 覆盖 recurrent state、causal attention/cache、early streaming adaptations 等设计点 | Oxford Spires 和复杂长序列下显著漂移；部分方法显存/速度/状态遗忘问题明显 |
| 5 | **MAGiSt3R** | **多机器人/多设备协作建图**（多个智能体各自采集视频、需融合成一张全局一致图）；本表中唯一原生支持多智能体的方法 | 首个明确"多智能体协作"的前馈式分支：MAGMA 学习式合并 + 显式 SE(3) 位姿图优化(PGO) + 跨智能体回环检测；ReplicaMultiagent 上 Acc 5.37/Comp 3.36/ATE-RMSE 3.87，全面优于 RANSAC+ICP 拼接的多智能体基线 | 未开源（code coming soon）；数值来自论文自建的 ReplicaMultiagent/AriaMultiagent，评测协议与本表其余单智能体方法（Oxford Spires/ETH3D/7-Scenes/T&T）不同，**不可直接跨表比较绝对数值**；本表其余方法均为单智能体，若只需单机流式建图不需要它 |

## 1. 对比目标

Streaming 3D reconstruction 的核心目标是：在视频帧持续到达时，因果地估计 camera trajectory、depth / point cloud，并尽量保持长程一致性、低延迟和可控显存。

这与本仓库已有的前馈式三维重建报告互补：

- [`reports/feedforward_3d_reconstruction_compare.md`](../../reports/feedforward_3d_reconstruction_compare.md) 更关注 **多视图 / 多先验 / metric 3D reconstruction**，候选包括 MapAnything、Pi3/Pi3X、HunyuanWorld-Mirror、OmniVGGT。
- 本报告关注 **online streaming visual mapping**，候选包括 LingBot-Map、Stream3R、Wint3R、TTT3R、CUT3R 等。

## 2. 一页总表

| 维度 | LingBot-Map | Stream3R | Wint3R | TTT3R | CUT3R / InfiniteVGGT 等 | MAGiSt3R |
|---|---|---|---|---|---|---|
| 基本范式 | Feed-forward streaming 3D foundation model + GCA | Causal transformer streaming reconstruction | Window-based streaming reconstruction | Test-time-training streaming reconstruction | Recurrent state 或 causal/cache streaming adaptation | Feed-forward 骨干(VGGT) + 可学习合并(MAGMA) + 经典后端(PGO) 的多智能体协作重建 |
| 输入 | RGB video stream | RGB sequence | RGB sequence | RGB sequence | RGB sequence | **多条**独立 RGB 序列（每个智能体一条） |
| 输出 | Pose、depth、point cloud；demo/renderer 可视化 | Pose/depth/point cloud 类输出 | Pose/depth/point cloud 类输出 | Pose/depth/point cloud 类输出 | Pose/depth/point cloud 类输出 | 各智能体位姿 + 全局统一坐标系稠密点图 |
| 核心状态管理 | Anchor context + pose-reference window + trajectory memory | Causal/sequential context | Window/camera token pool | TTT 更新以适配 sequence | 历史缓存或 recurrent 压缩 | 无持久隐状态；submap 级前馈 + MAGMA 显式合并进全局坐标系 |
| 是否 test-time optimization / training | 否；纯 feed-forward | 否/弱 | 否/弱 | 是，TTT 是核心 | 多数否，CUT3R 是 recurrent 状态 | 否（推理期不做 TTT）；但后端 PGO 是显式优化 |
| 是否多智能体协作 | 否（单智能体流式） | 否 | 否 | 否 | 否 | **是**——本表唯一原生支持，intra-agent 增量合并 + inter-agent 回环融合共用同一 MAGMA 模块 |
| 漂移抑制手段 | Windowed VO + Sim(3) overlap alignment（隐式对齐，无显式 PGO） | 因果注意力/上下文，无显式全局优化 | 滑窗 + 相机 token pool，无显式全局优化 | Test-time training 更新权重，无显式位姿图 | Recurrent 状态压缩或历史缓存，无显式全局优化 | **显式**：学习式 MAGMA 合并 + $SE(3)$ 位姿图 LM 优化(PGO)；消融显示 PGO 比骨干选择更关键（去 PGO 后 RMSE 3.87→5.65） |
| 是否回环检测(loop closure) | 否 | 否 | 否 | 否 | 否 | **是**——跨智能体描述子相关度触发回环，融合进全局图 |
| 长序列策略 | Direct mode 到约 3,000 帧；超长用 windowed VO + Sim(3) overlap alignment | 顺序处理，长程漂移较明显 | Window-based | TTT 缓解但有开销 | 状态遗忘或历史冗余问题 | 逐 submap（$m$=10 帧、步长 5）前馈 + 增量合并进统一坐标系，长程一致性交给后端 PGO |
| Git 地址 | <https://github.com/robbyant/lingbot-map> | 待补充 | 待补充 | 待补充 | 待补充 | 无仓库；项目页 <https://zorangong.github.io/magist3r_page/> 标注 "Code (coming soon)" |
| 是否开源 | 是，Apache-2.0 | 待补充 | 待补充 | 待补充 | 待补充 | **否**（截至 2026-07 核实） |
| 是否开源训练 | `\`：公开 repo 当前主要是 inference/demo/model source | 待补充 | 待补充 | 待补充 | 待补充 | 未知（代码/训练脚本均未发布） |
| 预训练权重 | HuggingFace / ModelScope | 待补充 | 待补充 | 待补充 | 待补充 | 不可用 |
| 推理成本 | 论文报告 518×378、20 FPS 级别；FlashInfer paged KV cache 推荐 | Oxford dense 表中 Stream3R-w 13.66 FPS | Oxford dense 表中 3.88 FPS | Oxford dense 表中 28.97 FPS，但含 TTT 成本需看设置 | CUT3R 29.21 FPS；InfiniteVGGT 7.78 FPS | 约 10 FPS（两智能体，ReplicaMultiagent，论文 Tab.8）；**未在 Oxford Spires/ETH3D 等本表统一 benchmark 上跑过，不可直接与左侧数值并列比较** |
| 主要风险 | 训练未开源；无显式 loop closure；无 raw LiDAR/IMU | 长序列漂移 | 慢且长序列漂移 | TTT overhead 和工程复杂度 | 状态遗忘或显存/上下文膨胀 | 未开源、无权重；评测仅限室内合成/自采多智能体数据，多智能体基线为作者自建(RANSAC+ICP 拼接)，非原生实现；>3 智能体未验证 |

## 3. 关键 benchmark 对比

> 注：MAGiSt3R 未参与本节任一表格，因为其论文只报告 ReplicaMultiagent / AriaMultiagent（多智能体协作设定）的结果，与本表 Oxford Spires / ETH3D / 7-Scenes / Tanks & Temples（均为单智能体）不是同一协议，跨表数值不可直接比较；MAGiSt3R 自身的数值见第 2 节总表最后一列的"推理成本"行与结论先行第 5 条。

### 3.1 Oxford Spires dense：长序列 streaming 压力测试

| Method | ATE sparse ↓ | ATE dense ↓ | ΔATE | FPS ↑ | 解读 |
|---|---:|---:|---:|---:|---|
| LingBot-Map | 6.42 | 7.11 | +0.69 | 20.29 | 长序列增长 12× 后退化极小，速度仍可用 |
| TTT3R | 19.35 | 25.05 | +5.70 | 28.97 | FPS 高，但长程轨迹明显落后 |
| Wint3R | 21.10 | 32.90 | +11.80 | 3.88 | 长序列漂移和速度都劣势明显 |
| InfiniteVGGT | 30.49 | 31.75 | +1.26 | 7.78 | 退化小但绝对误差高 |
| Stream3R-w | 33.03 | 33.73 | +0.70 | 13.66 | 退化小但绝对误差高 |
| CUT3R | 18.16 | 32.47 | +14.31 | 29.21 | sparse 不差，但 dense 漂移很重 |

### 3.2 多数据集 pose 泛化

| Method | ETH3D ATE ↓ | 7-Scenes ATE ↓ | Tanks & Temples ATE ↓ | 备注 |
|---|---:|---:|---:|---|
| LingBot-Map | 0.22 | 0.08 | 0.20 | 三个数据集均最好 |
| Stream3R | 1.67 | 0.10 | 0.76 | Tanks & Temples 是最强 baseline |
| Wint3R | 0.86 | 0.12 | 0.88 | ETH3D baseline 较强 |
| TTT3R | 1.22 | 0.10 | 0.66 | 7-Scenes / T&T baseline 竞争力较强 |
| InfiniteVGGT | 1.46 | 0.12 | 1.00 | 中等 |
| CUT3R | 1.43 | 0.29 | 1.79 | 7-Scenes / T&T 明显落后 |

### 3.3 点云重建质量

| Method | ETH3D F1 ↑ | 7-Scenes F1 ↑ | NRGBD F1 ↑ | 解读 |
|---|---:|---:|---:|---|
| LingBot-Map | 98.98 | 80.39 | 64.26 | 三个 reconstruction benchmark 均最好 |
| Wint3R | 77.28 | 78.81 | 56.96 | 最强重建 baseline 之一 |
| Stream3R | 72.87 | 78.79 | 54.07 | 7-Scenes 接近 Wint3R |
| TTT3R | 68.48 | 77.25 | 53.55 | 中等偏强 |
| CUT3R | 67.63 | 58.98 | 32.22 | 房间/NRGBD 重建落后 |

## 4. 与 MapAnything / HunyuanWorld-Mirror / OmniVGGT 的关系

| Method | 所属主线 | 与 LingBot-Map 的关系 | 选择建议 |
|---|---|---|---|
| MapAnything | 多视图 feed-forward metric 3D reconstruction | 更适合给定 image set + camera/depth/pose prompt 的 metric reconstruction；不是主打严格视频流 | 自动驾驶中若已有相机标定、ego pose、LiDAR depth prompt，优先做主 reconstruction backbone；LingBot-Map 可做在线 VO/建图分支 |
| HunyuanWorld-Mirror | Any-prior world reconstruction / 3DGS / NVS | 更适合多先验、重渲染、3DGS/NVS 和世界资产生成 | 如果目标是仿真/渲染/资产，Hunyuan 更直接；如果目标是在线轨迹+点云，LingBot-Map 更直接 |
| OmniVGGT | VGGT geometry-prior adapter | 更适合作为轻量几何先验 baseline | 用于验证 camera/depth prior adapter；不适合作为长视频主模型 |
| LingBot-Map | Streaming 3D reconstruction / online visual mapping | 补上“长视频、实时、causal、稳定状态管理”这条主线 | 自动驾驶/机器人连续视频建图的新主候选 |

## 5. 复现优先级

1. **LingBot-Map inference sanity check**：下载 `lingbot-map-long`，跑官方 demo scenes 和一个自有长视频；记录 FPS、显存、pose collapse 阈值、windowed mode 对齐质量。
2. **Oxford / outdoor sequence 小型 benchmark**：等官方 evaluation benchmark 或自行接入 Oxford Spires / KITTI-360 / Waymo 片段，先测 ATE/轨迹可视化。
3. **与 MapAnything 分工验证**：同一段行车视频，比较 LingBot-Map streaming pose/depth 与 MapAnything 多视图/先验重建结果。
4. **动态场景过滤测试**：加入 sky mask、dynamic object mask，看点云重影与轨迹稳定性变化。

## 6. 不确定性

- 本报告未独立复跑论文表；数值来自 LingBot-Map 论文。
- 其他 streaming baseline 的 Git 地址、许可证、训练开源状态待后续逐个补齐。
- LingBot-Map 的 training/eval benchmark release 可能之后更新；需要定期检查 GitHub README / file tree。

## Sources

- LingBot-Map paper: <https://arxiv.org/abs/2604.14141>
- LingBot-Map GitHub: <https://github.com/robbyant/lingbot-map>
- LingBot-Map analysis note: [`../../papers/3d-reconstruction/2026-lingbot-map.md`](../../papers/3d-reconstruction/2026-lingbot-map.md)
- Existing FrontierLab comparison: [`../../reports/feedforward_3d_reconstruction_compare.md`](../../reports/feedforward_3d_reconstruction_compare.md)
- MAGiSt3R paper: <https://arxiv.org/abs/2607.15211>
- MAGiSt3R project page: <https://zorangong.github.io/magist3r_page/>
- MAGiSt3R analysis note: [`../../papers/3d-reconstruction/2026-magist3r.md`](../../papers/3d-reconstruction/2026-magist3r.md)
