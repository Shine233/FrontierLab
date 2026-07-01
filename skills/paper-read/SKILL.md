---
name: paper-read
description: 读一篇前沿论文并按仓库规范产出分析文件。适用于「分析这篇论文」「读 arxiv xxxx」。
---

# 论文阅读分析 SOP

## 何时用
给定论文链接 / arxiv id / PDF，要在 `papers/<direction>/` 产出一篇符合仓库规范的分析。

## 步骤
1. 先读 `KB-MEMORY.md`：确认命名约定、该方向缺口、对比维度口径。
2. 确定 direction（用 README「方向 slug」列表）与 slug = `<year>-<short-title>`。
3. 复制 `papers/_template.md` 到 `papers/<direction>/<slug>.md`。
4. 填 frontmatter：必填 title/year/direction/method_family/status/reproduction；
   尽量填 org/key_idea；判断 landmark（是否奠基/划时代）；有谱系则填 supersedes/builds_on。
5. 填正文：结论先行 3-5 条；证据与推断分开标注。
6. 若有谱系，填正文「方法谱系」小节（标准 markdown 链接，与 frontmatter 一致）。
7. 收尾：运行 `python3 scripts/build_indices.py` 重建索引。
8. 更新 `KB-MEMORY.md`：补该方向缺口/决策（如有）。

## 验收
- frontmatter 能被 `scripts/lib_frontmatter.py` 解析（无 YAML 报错）。
- `python3 scripts/build_indices.py` 跑通，新论文出现在 indices/papers.md 与 timeline.md。
- 谱系 frontmatter 与正文链接一致。
