---
type: learning-path
direction: world-models
maturity: draft
updated: 2026-06-30
---

# 世界模型（World Models）学习路线

## 0. 这个方向在解决什么
结论先行：世界模型要学一个「可预测、可交互的环境动力学」——给定历史观测和动作，预测未来的观测（视频/场景）甚至联合预测动作。它服务两类落地：自动驾驶的可控生成式仿真（造数据、闭环评测）与机器人的 video-diffusion 策略（把生成模型当 zero-shot policy）。当前主战场是把扩散/DiT 视频生成器做成 action-conditioned、长 horizon、多相机、可实时推理。最大现实障碍是：这一方向多数工作代码/权重/数据未公开，复现门槛高。

## 1. 前置知识
- 生成模型：扩散模型（DDPM/DDIM）、latent diffusion、DiT 架构、few-step/AR 采样。
- 视频生成基础：时序一致性、causal attention、camera/action conditioning（如何把控制信号注入生成）。
- 机器人/驾驶背景：VLA（vision-language-action）、闭环仿真、cross-embodiment transfer 概念。
- 推理加速：见 [efficient-training-inference 学习路线](./efficient-training-inference.md)（X-Cache 即针对世界模型推理）。

## 2. 奠基论文（仓库内）
- ★ [DreamZero / World Action Models are Zero-shot Policies](../papers/world-models/2026-dreamzero.md) — 把 video diffusion backbone 转成机器人 World Action Model，联合预测未来视频与动作，并公开 DROID 权重/数据，是本方向最可上手的奠基级参考。
- [X-World](../papers/world-models/2026-x-world.md) — XPeng 的可控 7 相机 action-conditioned streaming 驾驶仿真器（WAN 2.2/DiT latent video），代表生成式驾驶世界模型主线。
- [Xiaomi Auto World Model / JointWM](../papers/world-models/2026-xiaomi-auto-world-model.md) — 用 WorldRec 稀疏 3D scene queries + WorldGen 因果 DiT 做重建-生成联合，是「重建×生成」联合世界模型代表。
- [X-Foresight](../papers/world-models/2026-x-foresight.md) — 把 predictive world modeling 接入 VLA/LDM，用长 horizon chunk 预测提升规划安全。

## 3. 近期 SOTA 与分支
- [XPeng X 系列自动驾驶世界模型横向对比](../comparisons/world-models/xpeng-x-series-world-models.md) — 串起 X-World（生成式仿真）、[X-Cache](../papers/efficient-training-inference/2026-x-cache.md)（推理加速）、X-Foresight（预测式 VLA）三条链路的主入口。
- [Robot WAM / video-diffusion robot policies 横向对比](../comparisons/world-models/robot-world-action-models.md) — 机器人侧 video-diffusion policy 分支（含 DreamZero）的横向对比。
- 分支轴：按落地分「驾驶仿真（XPeng X 系列、Xiaomi JointWM）」vs「机器人策略（DreamZero）」；按目标分「只生成观测」vs「联合生成观测+动作」。

## 4. 动手：最小复现路径
- 最小可跑路径：DreamZero-DROID 是唯一现实选择——它公开了 DROID 权重与数据，可做 inference / fine-tune sanity check（跑出未来视频 + 动作预测）。
- 复现记录：planned，仓库暂无复现记录。DreamZero-DROID inference / fine-tune sanity check planned；Xiaomi Auto World Model / X-World / X-Cache / X-Foresight 当前无公开代码、权重或训练数据，复现 blocked。

## 5. 常见误区 / 我的判断
- 误区：把论文报表的 FID/FVD/成功率当可复现结论。本方向大量指标基于内部数据/协议，缺公开代码，跨工作几乎无法横比。
- 误区：以为「世界模型 = 视频生成」。关键差异在动作条件与闭环可控性，纯生成质量高不代表能当仿真器或策略用。
- 判断：想真正动手只能从 DreamZero-DROID 起步；XPeng X 系列与 Xiaomi JointWM 当前更适合作为「读论文理解设计」的对象，等代码/权重 release 再谈复现。
