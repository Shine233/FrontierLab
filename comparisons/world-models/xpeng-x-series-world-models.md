---
type: method-comparison
title: "XPeng X 系列自动驾驶世界模型横向对比：X-World / X-Cache / X-Foresight"
direction: [world-models, robotics-autonomous-driving, generation-diffusion, efficient-training-inference]
methods: [X-World, X-Cache, X-Foresight, Xiaomi Auto World Model]
status: initial-completed
updated: 2026-06-08
---

# XPeng X 系列自动驾驶世界模型横向对比

## 结论先行

- **X-World 是核心生成式仿真器**：它把 7 相机历史和未来自车动作转成未来多相机视频，目标是 closed-loop evaluation、online RL 和可控场景编辑。
- **X-Cache 是 X-World 的推理加速层**：不训练、不改权重，利用连续 chunk 的 DiT block residual 冗余，报告 71% skip 和 2.6-2.7x DiT wall-clock speedup。
- **X-Foresight 是 VLA/LDM 内部预测式世界模型**：把 long-horizon future prediction 接入动作控制，报告 collision rate 相对 baseline 下降 16.2%，并用 X-World 初始化 renderer。
- **三者是系统链路而非互斥 baseline**：X-World 提供可交互世界，X-Cache 让它更快，X-Foresight 把“预见未来”变成规划模型的训练/推理能力。
- **共同限制**：三篇都是 2026 arXiv/项目页技术报告，代码、权重、训练数据、license 均未公开；当前复现全部 blocked，只适合方法跟踪和设计借鉴。

## 方法定位

| Method | 主问题 | 输入 | 输出 | 关键方法 | 当前角色 |
|---|---|---|---|---|---|
| [X-World](../../papers/world-models/2026-x-world.md) | 可控多相机闭环世界仿真 | 7 相机历史、未来 ego actions、dynamic/static controls、camera params、text prompt | 未来 7 相机视频 | WAN 2.2 / DiT latent video、view-temporal attention、decoupled conditions、causal chunk、4-step denoising、rolling KV cache | XPeng X 系列核心生成式 simulator |
| [X-Cache](../../papers/efficient-training-inference/2026-x-cache.md) | few-step AR 世界模型推理太慢 | 当前 chunk block input、上一 chunk cached residual/fingerprint、action condition、KV update state | 跳过/计算 DiT block 的决策与近似 residual 输出 | cross-chunk residual caching、dual-metric gating、KV-update protection、adaptive threshold | X-World 部署加速层 |
| [X-Foresight](../../papers/world-models/2026-x-foresight.md) | VLA 缺少未来预测和长时因果 | 多相机历史、instructions、action/state tokens、query/camera tokens | 未来动作/轨迹、camera tokens、未来多相机图像 | chunk-wise AR predictive world modeling、CLEF、TIS、X-World-initialized diffusion renderer | VLA/Large Drive Model 的预测式世界知识模块 |
| [Xiaomi Auto World Model / JointWM](../../papers/world-models/2026-xiaomi-auto-world-model.md) | 重建与生成联合的自动驾驶世界模型 | 多相机时序、轨迹、地图/文本、WorldRec rendered priors | 4D Gaussian scene representation 和未来多相机视频 | WorldRec sparse 3D queries + WorldGen causal DiT + rendered-prior conditioning | 同方向外部参照：重建-生成耦合路线 |

## 开源与复现状态

| Method | GitHub | 是否开源 | 是否开源训练 | 权重 | 数据 | License | 复现状态 |
|---|---|---|---|---|---|---|---|
| X-World | 无公开 GitHub | 否 | 否 | 未公开 | XPeng internal | Unknown | blocked-code-unavailable |
| X-Cache | 无公开 GitHub | 否 | 否 | 不适用/依赖 X-World 权重，未公开 | XPeng internal held-out split | Unknown | blocked-code-unavailable |
| X-Foresight | 无公开 GitHub | 否 | 否 | 未公开 | XPeng internal 280k hours / 34M clips / 13.8T tokens | Unknown | blocked-code-unavailable |
| Xiaomi Auto World Model | 无公开 GitHub | 否 | 否 | 未公开 | Waymo / nuScenes + private data；训练数据未公开 | Unknown | blocked-code-unavailable |

## 关键指标和证据强度

| Method | 论文报告的强证据 | 证据强度 | 主要缺口 |
|---|---|---|---|
| X-World | 7 相机 12 FPS 数据 schema；81 帧 Stage-I；4-step Stage-II；24s multi-camera rollout；动作/动态体/静态元素/外观控制 demos | 中：系统设计清楚，但主要是 qualitative | 缺公开 benchmark 数值表、闭环指标协议、代码/权重 |
| X-Cache | 13 个 22s clips；71.3-71.6% skip；2.65-2.70x DiT speedup；7-cam SSIM 约 0.999；KV protection ablation | 高：对加速目标有明确表格和 ablation | 只对 full-compute reference fidelity，不证明世界真实性；内部数据/硬件不可复跑 |
| X-Foresight | H=1/6/21、CL/CLEF/TIS、1024 GPU production comparison；collision 0.228% -> 0.191%；Vision Renderer 6s FID/FVD 2.84/29.52 | 高：有规划和渲染表格 | 内部 baseline/CCES 不透明；数据和模型不可复跑 |
| JointWM | WorldRec Waymo/nuScenes PSNR/SSIM；WorldGen nuScenes FID/FVD/frames/time；JointWM qualitative consistency | 中高：重建/生成有表格，联合部分偏 qualitative | 无代码/权重；JointWM 稳定性缺数值协议 |

## 工程链路理解

```text
X-World:
  history cameras + future actions + scene controls
        -> controllable future multi-camera videos
        -> closed-loop simulator / data factory

X-Cache:
  X-World few-step AR DiT inference
        -> cross-chunk block residual reuse
        -> lower DiT latency for interactive simulation

X-Foresight:
  VLA/LDM + predictive future camera/action tokens
        -> diffusion renderer initialized from X-World
        -> future-aware planning and closed-loop visual feedback
```

## 设计取舍

| 维度 | X-World | X-Cache | X-Foresight |
|---|---|---|---|
| 训练目标 | 学可控未来视频分布 | 无训练；复用现有 DiT residual | 学未来视觉/动作 token 和规划相关世界因果 |
| 是否生成像素 | 是，核心输出 | 不直接生成，只近似中间 block residual | 是，renderer 输出像素；LDM 更偏 latent/camera token |
| 闭环位置 | 外部 simulator，接收 policy action | simulator 内部推理优化 | policy/model 内部的 foresight 机制 |
| 风险重点 | 视频真实性和物理一致性 | 缓存误差污染 KV / 分布外失效 | 预测幻觉反馈进 VLA；内部指标可解释性 |
| 最小可借鉴点 | 多条件注入、chunk-wise causal conversion、rolling KV | cross-chunk cache、action-aware fingerprint、KV-update protection | long-horizon chunk supervision、CLEF、TIS、renderer 解耦 |

## 与 Xiaomi JointWM 的横向关系

- **X-World vs JointWM**：X-World 主要是视频空间可控生成仿真；JointWM 试图用 WorldRec 的 4D Gaussian scene state 约束 WorldGen，更强调显式重建先验。
- **X-Foresight vs JointWM**：X-Foresight 关注 VLA 内部世界知识和 planning gains；JointWM 关注重建-生成联合的仿真/数据合成。
- **X-Cache vs JointWM**：X-Cache 是系统加速技巧，可迁移到任何 few-step AR DiT world model；如果 JointWM/WorldGen 有类似 AR DiT 推理，也可能受益。

## 推荐跟进顺序

1. **设计层面优先读 X-World**：理解 XPeng X 系列的 simulator abstraction、condition schema 和 causal conversion。
2. **工程层面优先读 X-Cache**：如果我们要部署 AR video diffusion world model，KV-update protection 是必须注意的风险点。
3. **VLA 训练层面读 X-Foresight**：重点关注 chunk-wise long-horizon supervision、CLEF 和 TIS，而不是试图复刻 1024 GPU 系统。
4. **与 JointWM 一起看**：判断“纯视频生成式仿真”和“重建先验约束生成”两条路线在自动驾驶闭环中的取舍。

## 后续待补

- [ ] 若 XPeng 公开代码/权重，补复现计划和 license 审查。
- [ ] 补 Waymo World Model、GAIA-2、MagicDrive-V2、Epona、Genesis 等公开/半公开 driving world model 到同一表。
- [ ] 建立统一评估维度：动作跟随、跨视角一致性、长 rollout drift、闭环安全指标、可控编辑、推理延迟、许可证、数据可得性。

## Sources

- X-World: <https://arxiv.org/abs/2603.19979>, <https://x-world-1.github.io>
- X-Cache: <https://arxiv.org/abs/2604.20289>, <https://x-cache-1.github.io/en/>
- X-Foresight: <https://arxiv.org/abs/2605.24892>, <https://x-foresight-1.github.io/en/>
- Xiaomi Auto World Model / JointWM: <https://arxiv.org/abs/2605.18137>, <https://JointWM.github.io>
