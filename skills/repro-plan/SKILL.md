---
name: repro-plan
description: 评估一篇论文的可复现性并产出复现计划。适用于「这个能复现吗」「写复现计划」。
---

# 复现判断 / 计划 SOP

## 何时用
要判断某方法可复现性，并产出 `reproductions/<direction>/<method>/README.md`。

## 步骤
1. 先读 `KB-MEMORY.md` 与该论文 `papers/` 文件的「复现判断」小节。
2. 核查仓库：代码是否公开、训练是否开源、权重/数据可得性、license、commit。
3. 复制 `reproductions/_template.md` 到 `reproductions/<direction>/<method>/README.md`。
4. 填：环境、目标 commit、最小复现路径、预计成本、是否值得复现。
5. 诚实记录 blocker（代码未发布 / 数据未公开 / 算力不足）。
6. 同步论文 frontmatter 的 `reproduction:` 字段（planned/running/blocked/reproduced）。
7. 收尾：运行 `python3 scripts/build_indices.py`。
8. 更新 `KB-MEMORY.md` 未完成意图。

## 验收
- reproduction README 含环境/commit/最小路径/是否值得复现。
- 论文 frontmatter reproduction 字段与结论一致。
- build_indices 跑通。
