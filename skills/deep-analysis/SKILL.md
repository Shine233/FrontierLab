---
name: deep-analysis
description: 按仓库深度模板（架构/原理/公式/性能+配图）分析一篇或多篇前沿论文，含事实核查与 GitHub 公式渲染修复。适用于「deep analysis 这几篇」「深度分析 arxiv xxxx」「补全这批论文」。
---

# 深度分析 SOP（已沉淀的多次实践）

## 何时用
给定 1+ 篇论文（链接/arXiv id/PDF），要在 `papers/<direction>/` 产出**深度**分析——不是摘要，是架构解析 + 核心原理 + 逐符号公式 + 训练推理 + 性能解读 + 嵌入 arXiv 配图。单篇也可直接用 `paper-read`；**≥2 篇或要并行核查时用本 skill 的 workflow**。

## 深度标准（每篇必须达到）
对齐 `papers/_template.md`：
- **2.1 架构解析**：模块分解 + 数据流 + 关键设计选择，**配架构图**（arXiv HTML 取图）。
- **2.2 核心原理**：为什么 work、关键机制/归纳偏置、与前作本质区别。
- **2.3 关键公式解析**：LaTeX，逐符号解释 + 说明作用（2-4 个核心公式；论文无严格公式则形式化并注明）。
- **2.4 训练与推理细节**：损失/数据规模/超参/推理流程。
- **4.1 效果与性能解析**：解读结果（非搬数字）+ 速度/显存/参数量 + 消融关键因素 + 可比性。
- 结论先行 3-5 条；证据与推断分开标注；中文。

## 首选路径：workflow（起草→核查 两段 pipeline）
新建用 `.superpowers/create-papers.js`，加深已有用 `.superpowers/deepen-papers.js`。两者都是 `pipeline(draft→verify)`，各 agent effort=high，schema 强制返回 `{slug,dir,markdown,figures,uncertainties}`。

1. 先读 `KB-MEMORY.md`（命名约定、方向缺口、对比口径）。
2. 确定每篇的 `dir`（README 方向 slug）与 `slug_hint = <year>-<short>`。
3. 编辑脚本顶部：把 `REPO_SLUGS` 更新为当前 `papers/<那个方向>/` 真实 slug 列表（`ls papers/<dir>/*.md`），`updated:` 改今天日期——**谱系链接只许指向真实存在的 slug**。
4. 运行（ultracode 下）：`Workflow({scriptPath: ".superpowers/create-papers.js", args: [{name, dir, slug_hint, note}, ...]})`。`note` 写清背景（核心思想、org、arXiv、GitHub、开源状态待核实）。
5. 完成后从 `journal.jsonl` 取每个 agent 的 result dict（**不要**读被截断的 .output）；按 slug 去重，保留 verify（后出现）那份，加长度守卫 `len(markdown) >= 0.6*len(prev)` 防止拿到占位稿。
6. 落盘每篇 markdown 到 `papers/<dir>/<slug>.md`；核对 agent 下载的图片确实在 `assets/<dir>/<slug>/`（`test -f && file`），删掉编造/放错年份的孤儿目录。

## 手动路径（单篇/无 workflow）
直接走 `paper-read` skill 的步骤 1-10（同样的深度模板 + 取图 + 公式规范 + 横向对比 + 索引重建）。

## GitHub 公式渲染规范（务必，收尾必跑）
1. 行内 `$...$` 与中文之间留 ASCII 空格（`：$x$` 不渲染 → `： $x$`）。
2. math mode 内禁用 `#`（用 `\lvert\{\dots\}\rvert` 等替代）。
3. **同一行 ≥2 个含下划线的行内公式必须 `_`→`\_`**（GitHub 把同行多个 `_` 配对成斜体，吃掉下划线）。
4. `\text{}` 内不放中文；展示公式用成对 `$$`，花括号平衡。
5. 收尾跑 `python3 scripts/fix_inline_math.py <file> --write`，再复查残余为 0。

## 收尾（每次都做）
1. `python3 scripts/build_indices.py` 重建索引（确认新论文进 papers.md / timeline.md）。
2. `python3 scripts/scan_open_source.py` 刷新开源状态（新论文 frontmatter 的 open_source/training_open_source 要填对）。
3. 横向对比判断（paper-read 步骤 8）：纳入既有对比 / 新建对比 / 暂记待攒；正文「## 6.」不留空。
4. 更新 `KB-MEMORY.md`：方向缺口、对比处置、本批结论。
5. 无 broken 链接、`$` 成对、公式残余 0、图片路径可解析。

## 验收
- 每篇 2.1-2.4、4.1 非空且达深度标准；嵌入图片真实存在、图注标来源（arXiv id + 标题）。
- frontmatter 可被 `lib_frontmatter.py` 解析；谱系 frontmatter 与正文链接一致且指向真实 slug。
- `build_indices.py`、`scan_open_source.py`、`fix_inline_math.py` 均跑通、残余 0。
