---
type: paper-analysis
title: "World Action Models are Zero-shot Policies"
short_name: "DreamZero"
year: 2026
venue: "arXiv technical report"
arxiv: "2602.15922v1"
paper_url: "https://arxiv.org/abs/2602.15922"
pdf_url: "https://arxiv.org/pdf/2602.15922"
code: "https://github.com/dreamzero0/dreamzero"
github: "https://github.com/dreamzero0/dreamzero"
project_page: "https://dreamzero0.github.io/"
open_source: true
license: "Code: Apache-2.0. DreamZero-DROID weights: CC-BY-NC-4.0 on Hugging Face. DreamZero-AgiBot weights: Apache-2.0 on Hugging Face. Base Wan and dataset licenses still require downstream review."
training_open_source: true
weights_availability: "DreamZero-DROID and DreamZero-AgiBot checkpoints are public on Hugging Face; DROID model is not gated, AgiBot model is not gated. Verified 2026-06-11."
data_availability: "Preprocessed DROID dataset is public as GEAR-Dreams/DreamZero-DROID-Data; raw DROID 1.0.1 conversion path is documented. AgiBot ~500h internal dataset is not released yet; paper says release is planned."
direction: [world-models, robotics-autonomous-driving, generation-diffusion, evaluation-benchmarks]
method_family: [world-action-model, video-diffusion-policy, vision-language-action, robot-foundation-model, cross-embodiment-transfer]
tasks: [language-conditioned-manipulation, zero-shot-task-generalization, robot-policy-learning, cross-embodiment-transfer, closed-loop-control, video-action-prediction]
datasets: [DROID, DreamZero-DROID-Data, AgiBot-G1-internal, YAM-play-data, human-egocentric-video]
metrics: [task-progress, success-rate, inference-latency, control-frequency, denoising-steps]
status: compared
reproduction: planned
confidence: high
updated: 2026-06-11
---

# DreamZero：World Action Models are Zero-shot Policies

## 结论先行

- **一句话定位**：DreamZero 是把 pretrained video diffusion backbone 改造成机器人策略的 World Action Model (WAM)：同一个模型同时预测未来视频和动作，让视频预测像“视觉计划”，动作头像“从视觉计划反推控制”的 inverse dynamics model。
- **最关键结果**：论文在 AgiBot G1 和 DROID-Franka 上报告，DreamZero 相比 VLA baselines 在未见环境、未见物体和未见任务上有明显更高 task progress；AgiBot 未见任务平均 39.5%，DROID 未见任务 49% task progress / 22.5% success rate。
- **工程价值**：不是只有论文。`dreamzero0/dreamzero` 已公开 Apache-2.0 代码、推理服务、DROID/AgiBot 权重入口、DROID 预处理数据、DROID 转换脚本和多种训练脚本；但 full reproduction 需要多 GPU、Wan 权重、DROID 大数据和较复杂环境。
- **核心创新不是“视频生成+动作头”这么简单**：它强调 joint video-action denoising、chunk-wise autoregressive training、闭环执行时用真实观测替换生成帧进入 KV cache、以及 DreamZero-Flash 的视频/动作解耦噪声日程。
- **最大风险**：失败案例显示机器人会忠实执行错误视频计划；也就是说上限高度绑定视频模型的语言跟随和物理预测质量。长时记忆任务没有显式评估，高精度操作和边缘部署仍是弱项。
- **复现优先级**：先跑 DROID sim/API 或单机 inference sanity check，再考虑 DROID LoRA/full fine-tune；不要从 AgiBot 主结果开始，因为 500h AgiBot 数据未公开。

## 1. 这篇论文解决什么问题？

### 已确认的论文事实

- 论文认为 VLA 的强项是语义泛化，但对**未见物理动作、未见环境和新技能**的泛化不足。典型原因是 VLA 从静态图文/VLM 继承语义知识，却没有足够强的时空动态先验。
- DreamZero 将策略学习改成联合建模：
  - 未来视觉状态：模型预测接下来世界会怎样变化；
  - 未来动作：模型同时预测执行这些视觉变化需要的动作。
- 作者把这类模型称为 **World Action Model (WAM)**，并明确说 video 只是当前使用的世界建模目标，未来也可能是触觉、力反馈或 latent state。
- 实验覆盖：
  - AgiBot G1 mobile bimanual manipulator；
  - Franka / DROID；
  - YAM robot 与 human egocentric video 的跨 embodiment transfer；
  - seen tasks、unseen tasks、post-training tasks、few-shot new embodiment adaptation。

### 初学者解释

普通 VLA 更像“看到图像和语言，直接吐动作”。DreamZero 更像先在脑内生成一段“如果任务做对，画面应该怎么变化”的短视频，再把这段视觉未来转成动作。这样做的好处是：视频模型在大规模视频中已经学到很多物体运动、接触和场景变化规律；机器人数据只需要把这些视觉变化对齐到具体关节动作。

## 2. 方法概览

### 2.1 模型结构

已确认的论文事实：

- Backbone：Wan2.1-I2V-14B-480P video diffusion model；论文也做 5B vs 14B ablation。
- 输入：
  - visual context，经 VAE 编码；
  - language instruction，经 text encoder；
  - proprioceptive state，经 state encoder。
- 输出：
  - future video frames；
  - action chunks。
- 额外参数：主要是 state encoder、action encoder 和 action decoder；多视角机器人数据通过拼接多 camera view 到单帧，而不是大改 video backbone。
- 训练目标：flow matching + teacher forcing。当前 noisy chunk 可以看 clean previous chunks，联合预测 video/action 的 velocity。
- 推理：jointly denoise video and action chunks；闭环执行后，把真实观测替换生成帧写回 KV cache，以减少纯自回归视频生成常见的误差累积。

### 2.2 图示解析

| 图 | 论文中表达的机制 | 阅读结论 |
|---|---|---|
| Figure 1 Overview | WAM 通过联合预测 video/action 获得物理先验，支撑多样非重复数据学习、开放任务泛化、跨 embodiment 视频学习和少样本新机器人适配 | 这是论文总论点：不是把视频当辅助 loss，而是把视频未来作为策略的核心中间表示 |
| Figure 2 Joint Video and Action Prediction | DreamZero 生成的视频和预测动作在未见任务上保持对齐 | 这张图用来支撑“动作跟着视频计划走”；但它也暗示失败时动作会跟着错误视频走 |
| Figure 4 Model Architecture | VAE/text/state 编码后进入 autoregressive DiT，训练时当前 chunk 在 clean context 上 denoise，推理时异步执行并把真实观测写回 KV cache | 这是最值得复现时对照的主图：看代码里的 action/state registers、chunk/block、KV cache 是否按这个逻辑跑 |
| Figure 5 Decoupled Noise Schedules | DreamZero-Flash 让 video 处于更 noisy 的条件、action 保持 uniform noise，训练模型在嘈杂视觉条件下预测干净动作 | Flash 的关键是匹配 few-step/single-step inference；不是简单少采样 |
| Figure 6 AgiBot data distribution | AgiBot 预训练数据约 7.2K episodes / 500h，长 episode、多 subtask、技能分布宽 | 论文想证明 WAM 能吃“多样但非重复”的数据，而不是只靠大量同任务重复 demo |
| Figure 8/9 main eval | seen task 和 unseen task 上 DreamZero 优于 VLA baselines | 主要证据来自真实机器人 task progress，但外部目前只能完整复查 DROID 公开分支 |
| Figure 11/12 transfer | YAM/human 视频-only 数据提升 AgiBot 未见任务；30 分钟 YAM play data 可适配新机器人 | 跨 embodiment 的数据效率是论文亮点，但 AgiBot/YAM 原始数据不完全公开 |
| Figure 14 attention strategy | 训练时用 causal mask；推理时 conditional frames 的 KV cache 可复用，action token 能看历史视觉观测 | 这是代码阅读时理解 `num_frame_per_block`、`num_action_per_block`、action register 的关键 |
| Figure 16 failure analysis | 错误视频预测会导致机器人执行错误计划，例如先拿面包而不是先开烤箱 | 这是最重要的局限：WAM 的动作不是独立纠错器，它会忠实执行视觉未来 |

## 3. 关键贡献

1. **把 video diffusion backbone 变成 joint video-action policy**：不同于“先生成视频再另用 IDM”的两阶段路线，DreamZero 训练单一端到端模型，追求更深的 video/action alignment。
2. **用 chunk-wise autoregressive WAM 适配闭环控制**：AR 结构能保持原生帧率、利用 KV cache，并在每个动作 chunk 后注入真实观测，降低长 rollout 的生成误差累积。
3. **证明多样非重复机器人数据对 WAM 更友好**：论文中 500h diverse data 比 500h repetitive data 在 PnP Easy 上从 33% 提升到 50% task progress。
4. **系统级 real-time 化**：CFG parallelism、DiT caching、torch.compile/CUDA Graphs、kernel/scheduler 优化、GB200 NVFP4 quantization 和 DreamZero-Flash 共同把 GB200 latency 从 5.7s 降到约 150ms。
5. **跨 embodiment 视频-only transfer**：只用 10-20 分钟 YAM/human 视频数据，不用动作标签，也能提升 AgiBot 未见任务 task progress。

## 4. 实验与证据

| 维度 | 内容 |
|---|---|
| 数据集 | AgiBot G1 internal ~500h / 7.2K episodes；DROID；YAM 20min video-only；human egocentric 12min；YAM 30min play data |
| Baseline | GR00T N1.6、pi_0.5、from-scratch VLA、from-pretrained VLA、5B/14B WAM、AR/BD WAM |
| 指标 | task progress、success rate、inference latency、denoising steps、speedup |
| 主要结果 | AgiBot seen task DreamZero 62.2% average task progress；AgiBot unseen task 39.5%，best pretrained VLA 16.3%；DROID unseen task DreamZero 49% task progress / 22.5% success rate，GR00T N1.6 31% / 12.5%，pi_0.5 33% / 7.5% |
| 跨 embodiment | 9 个 AgiBot unseen tasks 上，DreamZero 38.3%；+Human2Robot 54.3%；+Robot2Robot 55.4% |
| Flash | Table bussing：4-step DreamZero 83%；1-step DreamZero 52%；1-step DreamZero-Flash 74%，约 2x faster |
| 消融 | diverse data 50% vs repetitive data 33%；14B WAM 50% vs 5B WAM 21%；VLA 5B/14B diverse data 均 0%；AR/BD task progress 相近但 AR 更平滑且 3-4x faster |
| 失败案例 | 论文展示 generated video 失败时，执行也跟随错误视频计划；作者认为改善语言跟随和视觉规划会直接改善动作执行 |

## 5. 代码 / 仓库事实

### 已确认的代码仓库事实

- GitHub：<https://github.com/dreamzero0/dreamzero>
- 本次核验 commit：`ab790c198fbce33503358efbbd4187ce9a89adf3`
- License：仓库 `LICENSE` 为 Apache-2.0，copyright NVIDIA Corporation。
- `pyproject.toml`：
  - package name: `dreamzero`
  - Python: `~=3.11,<3.13`
  - core dependencies include PyTorch 2.8.0、diffusers、deepspeed、transformers、ray、tensorrt、nvidia-modelopt 等。
- 推理相关：
  - `socket_test_optimized_AR.py`
  - `test_client_AR.py`
  - `eval_utils/serve_dreamzero_wan22.py`
  - `eval_utils/run_sim_eval.py`
  - `scripts/inference/build_trt_engine.sh`
- 训练/数据相关：
  - `scripts/train/droid_training_wan22.sh`
  - `scripts/train/droid_training_full_finetune.sh`
  - `scripts/train/droid_training_full_finetune_wan21.sh`
  - `scripts/train/droid_training_full_finetune_wan22.sh`
  - `scripts/train/droid_training_lora.sh`
  - `scripts/train/agibot_training.sh`
  - `scripts/train/yam_training.sh`
  - `scripts/data/convert_droid.py`
  - `scripts/data/convert_lerobot_to_gear.py`
  - `docs/DROID_CONVERSION.md`
  - `docs/DATASET_TO_GEAR_AND_TRAIN.md`
  - `docs/WAN22_BACKBONE.md`
- 模型结构入口：
  - `groot/vla/model/dreamzero/base_vla.py` 暴露 `get_action`、`joint_video_action`、`lazy_joint_video_action*` 等路径。
  - `docs/WAN22_BACKBONE.md` 明确说明 action/state registers、block/chunk、KV cache、Wan2.1 14B vs Wan2.2 5B 的差异。

### 代码/README 不一致点

- README 的 “Running Training” 示例写 `bash scripts/train/droid_training.sh`。
- 当前核验的 GitHub 快照中，`scripts/train/droid_training.sh` 不存在；实际存在 `droid_training_lora.sh`、`droid_training_wan22.sh`、`droid_training_full_finetune*` 等。
- 因此复现时应优先使用当前仓库真实脚本或 README 后续更新，不要机械复制该示例命令。

### Hugging Face 事实

- DreamZero-DROID：<https://huggingface.co/GEAR-Dreams/DreamZero-DROID>
  - public / non-gated；
  - license tag: `cc-by-nc-4.0`；
  - base model: `Wan-AI/Wan2.1-I2V-14B-480P`；
  - includes safetensors shards and TensorRT artifacts in repo metadata。
- DreamZero-AgiBot：<https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>
  - public / non-gated；
  - license tag: `apache-2.0`；
  - safetensors parameter count shown in HF API metadata。
- DreamZero-DROID-Data：<https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>
  - public / non-gated；
  - derived from DROID 1.0.1；
  - repository contains parquet episodes and three camera-video folders;
  - raw conversion path requires raw DROID 1.0.1 plus idle-frame filter JSON.

## 6. 局限与风险

### 论文明确承认 / 展示

- WAM scaling laws 仍缺系统证据，需要进一步研究模型大小、数据大小和训练 compute 的关系。
- 当前 human video transfer 只验证了小规模 in-lab egocentric data，不等价于已经证明可从大规模野外人类视频稳定学习机器人技能。
- DreamZero 在 2 个 GB200 上可达 7Hz，但相比一些 VLA 在消费级 GPU 上超过 20Hz，仍然很重。
- 当前架构主要是短时 System 1 policy；显式长时记忆任务没有训练/评估，6 秒视觉记忆仍不足以覆盖长 horizon。
- 高精度任务，如插钥匙或精密装配，可能需要更密集的精细 demo。

### 我推断的风险

- **视频计划错误会直接转成动作错误**：Figure 16 说明 action decoder 没有独立语义纠错能力；它更像忠实执行视觉未来的控制器。
- **真实机器人指标外部复核有限**：AgiBot/YAM 内部数据与真实 rollout 环境不可完全复现，公开复现主要落在 DROID/sim 分支。
- **许可组合复杂**：代码 Apache-2.0 不代表权重、DROID 数据、Wan base model 和下游使用都可商用；DROID checkpoint 明确是 CC-BY-NC-4.0。
- **工程门槛高**：Python 3.11、CUDA 12.9+、flash-attn、DeepSpeed、Wan weights、DROID 数据、TensorRT/GB200 路径都会增加复现成本。
- **README 与脚本命名有漂移**：公开仓库仍在快速迭代，复现脚本应按当前文件树和 docs 校验。

## 7. 与相似方法对比

| Method | 相同点 | 不同点 | 何时选它 |
|---|---|---|---|
| GR00T N1.x | 都是通用机器人策略 / VLA 方向 | GR00T 更偏 VLA/action model 路线；DreamZero 把 pretrained video diffusion 的世界动态作为核心先验 | 要 NVIDIA 机器人 foundation model baseline 看 GR00T；要研究 video prior -> action policy 看 DreamZero |
| pi_0 / pi_0.5 | 都是强 VLA/机器人策略 baseline | pi_0 系列偏 action expert / VLA scaling；DreamZero 论文认为 WAM 对多样非重复数据更友好 | 做公开 VLA baseline 和 DROID 对照时必须纳入 |
| V-JEPA 2 | 都利用世界模型思想做机器人 | V-JEPA 2 是 latent predictive model；DreamZero 是 pixel/video diffusion + action joint denoising | 追求高效 latent planning 看 V-JEPA；追求视频未来和动作强对齐看 DreamZero |
| Genie / video world model | 都从视频学动态 | Genie 更偏可交互/环境模拟；DreamZero 直接输出机器人动作 | 做仿真世界模型看 Genie；做真实机器人闭环策略看 DreamZero |
| X-Foresight / X-World | 都是 world model + action/control | XPeng 系列面向自动驾驶多相机未来预测/仿真；DreamZero 面向机器人 manipulation policy | 自动驾驶闭环看 X 系列；机械臂/移动双臂 manipulation 看 DreamZero |

## 8. 复现判断

- Git 地址：<https://github.com/dreamzero0/dreamzero>
- 是否开源：是。
- 是否开源训练：是。仓库提供 DROID、新 embodiment、AgiBot/YAM 相关训练脚本和数据转换文档；但 AgiBot 500h 内部数据未公开，不能完整复刻主 AgiBot 训练。
- 代码可用性：高，但环境重。
- 权重可用性：DreamZero-DROID 和 DreamZero-AgiBot 公开；DROID checkpoint 非商用 license。
- 数据可获得性：DROID preprocessed public；raw DROID conversion documented；AgiBot internal data not released。
- 预计环境成本：DROID inference 至少多 GPU；README 写 minimum 2 GPUs for distributed inference，测试过 GB200/H100；训练默认脚本多为 8 GPU 级别。
- 最小复现路径：
  1. 先用 Hugging Face 下载 DreamZero-DROID checkpoint；
  2. 跑 WebSocket inference server + `test_client_AR.py`；
  3. 若能拿到 API host，可跑 `eval_utils/run_sim_eval.py`；
  4. 下载 `DreamZero-DROID-Data`，用 `droid_training_wan22.sh` 或 `droid_training_lora.sh` 做小步数 sanity check；
  5. 后续再评估 full fine-tune 和 RoboArena/DROID real。
- 是否值得复现：值得。它是目前少数同时开源代码、权重、训练脚本和预处理 DROID 数据的 video-diffusion robot policy 路线；但完整论文级真实机器人结果无法完全外部复刻。

## 9. 后续动作

- [x] 创建 DreamZero 单篇论文分析
- [x] 更新 `indices/papers.md`
- [x] 更新 `indices/directions.md`
- [x] 更新 `indices/methods.md`
- [x] 创建 robot WAM / VLA policy 横向对比
- [ ] 后续创建 `reproductions/world-models/dreamzero/README.md`，先跑 DROID inference / sim sanity check
- [ ] 单独做 license audit：DreamZero code、DreamZero-DROID weights、DreamZero-AgiBot weights、DROID data、Wan base weights 的组合使用边界

## Sources

- Paper: <https://arxiv.org/abs/2602.15922>
- PDF: <https://arxiv.org/pdf/2602.15922>
- Project page: <https://dreamzero0.github.io/>
- GitHub: <https://github.com/dreamzero0/dreamzero>
- DreamZero-DROID checkpoint: <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>
- DreamZero-AgiBot checkpoint: <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>
- DreamZero-DROID-Data: <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>
