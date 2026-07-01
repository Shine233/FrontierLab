# 知识库记忆（KB-MEMORY）

> 三个 skill（paper-read / paper-compare / repro-plan）每次运行先读本文件，收尾按需更新。
> 分工：indices/ = 结构化机读事实；本文件 = 叙事性决策与待办。

## 约定与偏好
- 命名：`papers/<direction>/<year>-<slug>.md`，slug 全仓库唯一。
- 风格：结论先行；证据与推断分开标注；中文。
- 对比维度按方向稳定复用（见各 comparisons 文件末尾"建议固定对比维度"）。
- 谱系信息写两处：frontmatter 的 supersedes/builds_on + 正文「方法谱系」小节标准 markdown 链接。
- 新增/改动论文后运行 `python3 scripts/build_indices.py` 重建索引。

## 各方向缺口
- multimodal-learning / reasoning-agents：暂无论文，学习路线待方向有内容再建。
- 3d-reconstruction：历史脉络见 comparisons/3d-reconstruction/development-survey.md（2000-2026 全景）。
  - 已补（2026-07-01）：COLMAP/NeRF/DUSt3R/3DGS/VGGT + MASt3R/MapAnything/CUT3R；前馈链 DUSt3R→MASt3R→VGGT→MapAnything/CUT3R 成形。
  - 仍缺（对比报告提及但无独立分析）：OmniVGGT、HunyuanWorld-Mirror、MonST3R、MegaSaM、Stream3R、Wint3R、TTT3R、Spann3R、Fast3R、MV-DUSt3R。
- 其余方向缺口随阅读补充。

## 未完成意图
- 复现 planned 列表见 indices/directions.md 各方向「复现状态」。

## 近期决策记录
- 2026-06-30：确立 frontmatter 为唯一事实源 + 脚本生成静态索引 + Dataview 动态视图双轨方案。
