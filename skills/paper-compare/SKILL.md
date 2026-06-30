---
name: paper-compare
description: 把同方向多篇论文整理成横向对比。适用于「对比这几个方法」「更新 xx 方向对比表」。
---

# 论文横向对比 SOP

## 何时用
同方向 ≥2 篇论文，要产出/更新 `comparisons/<direction>/<topic>.md`。

## 步骤
1. 先读 `KB-MEMORY.md` 与该方向已有 comparisons 文件末尾「建议固定对比维度」。
2. 复制 `comparisons/_template.md`（或在既有对比文件上追加）。
3. 复用稳定维度表；新方法按同维度补列。
4. 明确标注：协议一致可比 vs 不可跨论文直接比较的项。
5. 结论先行：给推荐排序 + 场景化选择。
6. 收尾：运行 `python3 scripts/build_indices.py`（刷新 comparisons 清单）。
7. 更新 `KB-MEMORY.md` 决策记录（如调整了维度口径）。

## 验收
- 对比文件 frontmatter 含 direction/methods。
- indices/comparisons.md 自动块出现该文件。
- 可比/不可比标注清晰。
