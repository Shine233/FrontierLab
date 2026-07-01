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
7. **横向对比判断**：查同方向是否已有相似论文，决定是否要横向对比。
   - 查证据：`grep` 该 direction 下 `papers/<direction>/` 已有文件，看 `indices/methods.md`（方法族）与 `indices/comparisons.md`（已有对比报告），并读 KB-MEMORY「各方向缺口」。判断标准：任务/输入输出相近、method_family 有交集、或解决同一问题的不同路线。
   - 三种处置：
     - **已有对应对比文件** → 把新论文追加进 `comparisons/<direction>/<topic>.md`（复用其稳定维度表），并在新论文正文「## 6. 与相似方法对比」里指向它。
     - **无对比文件但同方向相似方法已 ≥2 个** → 用 `paper-compare` skill 新建 `comparisons/<direction>/<topic>.md`（README 约定：同方向 ≥2 方法即建对比）。
     - **暂不足以对比**（该方向此前 <2 篇、或角度太独特）→ 不建对比，但在新论文「## 6.」表里列出最接近的 1-2 个方法+差异，并在 KB-MEMORY 记一句"该方向待攒够方法后建对比"。
   - 无论哪种，新论文的「## 6. 与相似方法对比」小节都要填（至少列相似方法与差异），不留空。
8. 收尾：运行 `python3 scripts/build_indices.py` 重建索引。
9. 更新 `KB-MEMORY.md`：补该方向缺口/决策（如有），含本次对比处置结论。

## 验收
- frontmatter 能被 `scripts/lib_frontmatter.py` 解析（无 YAML 报错）。
- `python3 scripts/build_indices.py` 跑通，新论文出现在 indices/papers.md 与 timeline.md。
- 谱系 frontmatter 与正文链接一致。
- 已做横向对比判断：正文「## 6.」非空；若纳入/新建了对比文件，其链接可解析且已同步 `indices/comparisons.md`。
