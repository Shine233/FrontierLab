---
type: method-comparison
title: "Robot World Action Models / video-diffusion robot policies 横向对比"
direction: [world-models, robotics-autonomous-driving, generation-diffusion, evaluation-benchmarks]
methods: [DreamZero, GR00T N1.x, pi_0, pi_0.5, V-JEPA 2, Genie]
status: initial-completed
updated: 2026-06-11
---

# Robot World Action Models / video-diffusion robot policies 横向对比

## 结论先行

- **DreamZero 是当前最值得优先跟进的 robot WAM 路线**：它把 pretrained video diffusion backbone 改成 joint video-action policy，并公开代码、DROID/AgiBot 权重、DROID 预处理数据和训练脚本；但 AgiBot 主训练数据未公开，完整论文级复现不可直接完成。
- **和 VLA baseline 的核心差异**：GR00T N1.x、pi_0 / pi_0.5 更偏“视觉语言输入到动作”的策略模型；DreamZero 把未来视频作为显式中间世界计划，再和动作联合去噪。
- **证据强项在真实机器人泛化**：DreamZero 论文报告 DROID unseen task 49% task progress / 22.5% success，优于 GR00T N1.6 和 pi_0.5；AgiBot unseen task 也显著高于 pretrained VLA baseline。
- **最大风险不是代码不可得，而是复现范围**：公开 DROID 分支可做 inference / fine-tune sanity check；AgiBot 500h internal data、真实机器人 rollout 环境和 GB200 低延迟路径仍难外部复刻。
- **研究判断**：如果目标是机器人策略泛化，DreamZero 应和 pi_0.5 / GR00T 一起作为策略 baseline；如果目标是世界模型机制，则应和 V-JEPA 2、Genie、X-Foresight 这类“未来预测/交互世界”路线对照。

## 方法定位

| Method | 主问题 | 输入 | 输出 | 关键机制 | 当前角色 |
|---|---|---|---|---|---|
| [DreamZero](../../papers/world-models/2026-dreamzero.md) | 让 pretrained video diffusion 成为零样本机器人策略 | 视觉历史、语言指令、proprioceptive state | 未来视频帧 + action chunks | Wan video diffusion backbone、joint video-action denoising、chunk-wise AR、真实观测写回 KV cache、DreamZero-Flash | robot WAM / video-diffusion policy 主候选 |
| GR00T N1.x | 通用 humanoid / robot foundation policy | 多模态观测、语言、机器人状态 | 动作 / action tokens | VLA/action model scaling、机器人数据混合训练 | 强机器人策略 baseline；DreamZero 论文中的 DROID 对照之一 |
| pi_0 / pi_0.5 | 通用 VLA / action expert policy | 图像、语言、机器人状态 | 动作序列 | VLM/VLA backbone + action expert / diffusion action modeling | 开源生态和评测常用 baseline；DreamZero 论文中的 DROID 对照之一 |
| V-JEPA 2 | latent predictive world model for robotics | 图像/视频上下文、任务相关条件 | latent future / representation，结合控制头使用 | 自监督 latent predictive learning | 世界模型表征路线参照；不是 pixel video-action joint denoising |
| Genie / video world models | 从视频学习可交互环境动态 | 视频/动作或 latent interaction condition | 未来视频/交互世界 rollout | generative interactive world model | 世界模拟和交互预测参照；通常不直接输出真实机器人动作 |
| X-Foresight | 自动驾驶 VLA/LDM 内部预测式世界模型 | 多相机历史、指令、动作/状态 tokens | 未来动作、camera tokens、未来图像 | chunk-wise AR foresight、CLEF、TIS、diffusion renderer | 自动驾驶 future-prediction policy 参照，不是机械臂 manipulation 主线 |

## 开源与复现状态

| Method | GitHub / Project | 是否开源 | 是否开源训练 | 权重 | 数据 | License | 复现状态 |
|---|---|---|---|---|---|---|---|
| DreamZero | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) | 是 | 是 | DreamZero-DROID / DreamZero-AgiBot public on Hugging Face | DROID preprocessed public；AgiBot internal data 未释放 | Code Apache-2.0；DROID weights CC-BY-NC-4.0；AgiBot weights Apache-2.0 | planned |
| GR00T N1.x | NVIDIA project / repo，需单独按版本核验 | 部分公开 | 需按具体版本核验 | 需按具体版本核验 | 训练数据完整性需核验 | 需按具体版本核验 | baseline-only |
| pi_0 / pi_0.5 | Physical Intelligence / openpi 生态，需按具体版本核验 | 部分公开 | 需按具体版本核验 | 需按具体版本核验 | 训练数据完整性需核验 | 需按具体版本核验 | baseline-only |
| V-JEPA 2 | Meta / project，需单独建条目核验 | 待核验 | 待核验 | 待核验 | 待核验 | 待核验 | triage |
| Genie / video world models | 多个实现，需拆分具体论文 | 待核验 | 待核验 | 待核验 | 待核验 | 待核验 | triage |
| X-Foresight | [project](https://x-foresight-1.github.io/en/)（无公开 GitHub） | 否 | 否 | 未公开 | XPeng internal | Unknown | blocked-code-unavailable |

## 关键指标和证据强度

| Method | 论文/公开材料中的强证据 | 证据强度 | 主要缺口 |
|---|---|---|---|
| DreamZero | AgiBot seen task 62.2% average task progress；AgiBot unseen task 39.5%；DROID unseen task 49% task progress / 22.5% success；Flash 1-step 74% vs vanilla 1-step 52% | 高：真实机器人主表、消融和代码/权重较完整 | AgiBot internal data 未公开；真实 rollout 环境不可完全复刻；README 与脚本命名有漂移 |
| GR00T N1.x | DreamZero 论文中 DROID unseen task 对照：GR00T N1.6 31% task progress / 12.5% success | 中：作为 DreamZero 论文 baseline 有数值 | 需要回到 GR00T 原始材料核验训练数据、模型版本和开源范围 |
| pi_0.5 | DreamZero 论文中 DROID unseen task 对照：33% task progress / 7.5% success | 中：作为 DreamZero 论文 baseline 有数值 | 需要回到 pi_0.5 原始材料核验具体 checkpoint、数据和评测协议 |
| V-JEPA 2 | 世界模型/预测表征方向与 robotics 有关联 | 待补 | 尚未在本库建单篇分析，不能和 DreamZero 指标直接等价比较 |
| Genie / video world models | 可交互视频世界建模思想与 DreamZero 有方法关联 | 待补 | 多数不是直接机器人 action policy，需要拆具体论文和任务 |
| X-Foresight | 自动驾驶 predictive world model 报告 collision rate 0.228% -> 0.191% | 高：原论文有规划/渲染表格 | 内部数据、代码、权重未公开；任务域和机器人 manipulation 不同 |

## 方法差异

| 维度 | DreamZero | VLA/action-policy baseline | Latent/video world model baseline |
|---|---|---|---|
| 中间表示 | 显式未来视频 + action chunks | 通常直接预测动作或 action tokens | 未来 latent/video，可不直接输出动作 |
| 训练信号 | video/action 联合 flow matching，teacher forcing | 行为克隆、action expert、diffusion/flow action modeling 等 | 自监督预测、视频生成或交互 rollout |
| 泛化来源 | pretrained video diffusion 的物理/动态先验 + 机器人动作对齐 | VLM/VLA 语义先验 + 机器人 demonstration scaling | 视频/世界动态表征，但控制闭环需另接 policy |
| 闭环执行 | 每个动作 chunk 后注入真实观测，更新 KV cache | 取决于具体 policy loop | 多数先做 rollout/representation，控制闭环需额外设计 |
| 失败模式 | 错误视频计划会被动作忠实执行 | 语义理解对但动作/接触失败，或未见任务泛化弱 | 预测可视化合理但动作不可执行，或 latent 与控制脱节 |
| 工程成本 | 高：video diffusion + 多 GPU + 机器人数据 | 中到高：取决于模型尺寸和 policy stack | 中到高：常需要额外 policy/control bridge |

## 推荐使用方式

1. **做公开可复现 robot WAM**：优先 DreamZero-DROID。先跑 inference sanity check，再做 LoRA / small-step fine-tune，不从 AgiBot 主结果开始。
2. **做机器人策略 baseline**：DreamZero 应与 GR00T N1.x、pi_0/pi_0.5、OpenVLA 类模型同表比较，维度固定为 task progress、success、训练数据公开度、动作频率和硬件需求。
3. **做世界模型机制研究**：把 DreamZero 和 V-JEPA 2、Genie、X-Foresight、X-World 分开对照，重点比较“未来预测是否直接参与动作决策”。
4. **做系统落地**：必须单独审查 license、Wan base model、DROID checkpoint 非商用限制、GPU/latency 路径和真实机器人安全边界。

## 后续待补

- [ ] 为 GR00T N1.x、pi_0.5、V-JEPA 2 分别建立单篇分析，核验开源训练、权重、license 和原始指标。
- [ ] 建立 DreamZero-DROID 复现记录：环境、checkpoint、WebSocket inference、sim/API sanity check、训练脚本最小步数。
- [ ] 把 OpenVLA、RDT、Diffusion Policy、Octo 等机器人策略 baseline 纳入同一张更完整的 robot policy 表。
- [ ] 单独做 DreamZero license audit：代码、DROID weights、AgiBot weights、DROID data、Wan base weights 的组合使用边界。

## Sources

- DreamZero paper: <https://arxiv.org/abs/2602.15922>
- DreamZero project: <https://dreamzero0.github.io/>
- DreamZero GitHub: <https://github.com/dreamzero0/dreamzero>
- DreamZero-DROID checkpoint: <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>
- DreamZero-AgiBot checkpoint: <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>
- DreamZero-DROID-Data: <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>
- X-Foresight paper/project: <https://arxiv.org/abs/2605.24892>, <https://x-foresight-1.github.io/en/>
