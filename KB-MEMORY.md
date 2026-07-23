# 知识库记忆（KB-MEMORY）

> 六个 skill（paper-read / paper-compare / repro-plan / deep-analysis / open-source-tracking / frontier-tracking）每次运行先读本文件，收尾按需更新。
> 分工：indices/ = 结构化机读事实；本文件 = 叙事性决策与待办。

## 约定与偏好
- 命名：`papers/<direction>/<year>-<slug>.md`，slug 全仓库唯一。
- 风格：结论先行；证据与推断分开标注；中文。
- 对比维度按方向稳定复用（见各 comparisons 文件末尾"建议固定对比维度"）。
- 谱系信息写两处：frontmatter 的 supersedes/builds_on + 正文「方法谱系」小节标准 markdown 链接。
- 新增/改动论文后运行 `python3 scripts/build_indices.py` 重建索引。
- **分析新论文必做横向对比判断**（paper-read SOP 步骤 7）：查同方向相似论文，决定纳入现有对比 / 新建对比 / 暂记待攒；正文「## 6.」不留空。

## 各方向缺口
- multimodal-learning / reasoning-agents：暂无论文，学习路线待方向有内容再建。
- 3d-reconstruction：历史脉络见 comparisons/3d-reconstruction/development-survey.md（2000-2026 全景）。
  - 已补（2026-07-01）：COLMAP/NeRF/DUSt3R/3DGS/VGGT + MASt3R/MapAnything/CUT3R；前馈链 DUSt3R→MASt3R→VGGT→MapAnything/CUT3R 成形。
  - **全 12 篇已深度化（2026-07-02）**：架构解析/核心原理/关键公式(LaTeX)/训练推理/效果性能小节 + 嵌入 arXiv 关键插图（存 assets/，标来源）。公式经 scripts/fix_inline_math.py 修复 GitHub 渲染。
  - **survey 缺口全部补齐（2026-07-10，+13 篇深度）**：SIFT/KinectFusion/Instant-NGP（历史 landmark）+ Spann3R/Fast3R/MV-DUSt3R/MonST3R/MegaSaM/Stream3R/Wint3R/TTT3R/OmniVGGT/HunyuanWorld-Mirror。development-survey.md 的 🟡/⬜ 已清零，本方向独立分析共 25 篇。注：MegaSaM 非 landmark（强但非奠基）；Fast3R 实为 CVPR 2025（slug 2025-fast3r）。
  - **+3 篇单目/多智能体（2026-07-23，deep-analysis workflow）**：MoGe-3（2026-moge3，单目几何 2D→3D 稀疏体素细化，未开源 code coming soon）、FoundationGeo（2026-foundationgeo，ECCV 2026，单目 metric 逐像素标定场，开源 MIT+训练+HF 权重）、MAGiSt3R（2026-magist3r，多智能体协作前馈重建+PGO，未开源）。本方向独立分析共 28 篇。对比处置：MoGe-3/FoundationGeo 纳入 visual-geometry-foundation-models.md 新增「§3.7 单目几何分支」；MAGiSt3R 纳入 streaming-3d-reconstruction.md 新增「多智能体协作」维度（该线首个多智能体分支）。
  - beginner 前置知识地图：learning/3d-reconstruction-prerequisites.md（几何数学→Transformer/ViT→自监督→SfM/BA/NeRF/3DGS，deep 格式 + matplotlib 自绘配图，脚本 scripts/gen_prereq_figs.py）。
- world-models：X-Mind（2026-x-mind，小鹏 X- 系列，arXiv 2606.28758）已入库并纳入 xpeng-x-series-world-models.md 对比。注意 arXiv 2604.20289 是 X-Cache（已在 efficient-training-inference），非新论文。
- 其余方向缺口随阅读补充。

## 未完成意图
- 复现 planned 列表见 indices/directions.md 各方向「复现状态」。

## 跟踪基础设施（2026-07-23 新增）
- **开源跟踪**：skill `open-source-tracking` + `scripts/scan_open_source.py`。从 frontmatter 的 open_source/training_open_source 分三档（未开源/仅推理/训练开源）写 `indices/open-source.md`；`--stale` 按 `oss_last_checked`（可选 frontmatter 字段）陈旧度列出待重审。每次被问到未开源论文时重新核实 GitHub/Releases，状态变了就回填 frontmatter + oss_last_checked。当前 45 篇：未开源 10 / 仅推理 10 / 训练开源 25。
- **前沿跟踪**：skill `frontier-tracking` + `scripts/frontier_scan.py`。search-history 差分：已见 id = 历史扫描记录 ∪ 已入库论文 frontmatter 的 arxiv；`diff` 只回没见过的，`record` 登记本次扫描窗口。状态存 `indices/frontier-scan-history.jsonl`（已 seed 初始基线 39 篇，window ..2026-07-23）。首扫用长窗、复扫从上次窗口末尾续。
- **深度分析**：skill `deep-analysis` 沉淀了本仓多次用过的 workflow SOP（`.superpowers/create-papers.js` 新建 / `deepen-papers.js` 加深，pipeline 起草→核查，效率 high）。注意 verify agent 可能直接落盘文件而非返回完整 markdown——收尾要核实 `papers/` 下文件实际存在与完整性。

## 近期决策记录
- 2026-06-30：确立 frontmatter 为唯一事实源 + 脚本生成静态索引 + Dataview 动态视图双轨方案。
