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
5. 填正文：结论先行 3-5 条；证据与推断分开标注。**深度要求**（按 `_template.md` 新结构）：
   - 2.1 架构解析：模块分解 + 数据流，配架构图。
   - 2.2 核心原理：为什么 work、关键机制/归纳偏置。
   - 2.3 关键公式解析：LaTeX，逐符号解释 + 说明作用。**GitHub 渲染注意**：(a) 行内公式 `$...$` 与中文之间必须留 ASCII 空格（`：$x$` 不渲染，要写 `： $x$`）；(b) math mode 里禁用 `#`（用 `\lvert\{...\}\rvert` 等替代）；(c) `\text{}` 内不放中文。写完跑 `python3 scripts/fix_inline_math.py <file> --write` 自动补空格。
   - 2.4 训练与推理细节：损失/数据规模/超参/推理步骤。
   - 4.1 效果与性能解析：解读结果（非搬数字）+ 速度/显存/参数量 + 消融关键因素。
6. **关键插图提取**：从 arXiv HTML 版（`https://arxiv.org/html/<id>/xN.png`）下载 2-4 张关键图（架构图优先，其次核心结果/机制图）到 `assets/<direction>/<slug>/`，正文用相对路径 `![说明](../../assets/<direction>/<slug>/<name>.png)` 嵌入，图注标注来源（arXiv id + 论文标题）。取图方法见「取图步骤」。
7. 若有谱系，填正文「方法谱系」小节（标准 markdown 链接，与 frontmatter 一致）。
8. **横向对比判断**：查同方向是否已有相似论文，决定是否要横向对比。
   - 查证据：`grep` 该 direction 下 `papers/<direction>/` 已有文件，看 `indices/methods.md`（方法族）与 `indices/comparisons.md`（已有对比报告），并读 KB-MEMORY「各方向缺口」。判断标准：任务/输入输出相近、method_family 有交集、或解决同一问题的不同路线。
   - 三种处置：
     - **已有对应对比文件** → 把新论文追加进 `comparisons/<direction>/<topic>.md`（复用其稳定维度表），并在新论文正文「## 6. 与相似方法对比」里指向它。
     - **无对比文件但同方向相似方法已 ≥2 个** → 用 `paper-compare` skill 新建 `comparisons/<direction>/<topic>.md`（README 约定：同方向 ≥2 方法即建对比）。
     - **暂不足以对比**（该方向此前 <2 篇、或角度太独特）→ 不建对比，但在新论文「## 6.」表里列出最接近的 1-2 个方法+差异，并在 KB-MEMORY 记一句"该方向待攒够方法后建对比"。
   - 无论哪种，新论文的「## 6. 与相似方法对比」小节都要填（至少列相似方法与差异），不留空。
9. 收尾：运行 `python3 scripts/build_indices.py` 重建索引。
10. 更新 `KB-MEMORY.md`：补该方向缺口/决策（如有），含本次对比处置结论。

## 取图步骤

1. 定位 arXiv id，取 HTML 版：`https://arxiv.org/html/<id>`（或带版本 `.../html/<id>v1`）。
2. 找图 URL：HTML 里图为相对名 `x1.png`/`x2.png`...，完整地址 `https://arxiv.org/html/<id>/x1.png`。结合图注（Figure N）确认哪张是架构图/核心结果图。
3. 下载到 `assets/<direction>/<slug>/`，用语义化文件名（如 `arch.png`、`results.png`、`pipeline.png`）：
   `curl -sL "https://arxiv.org/html/<id>/x1.png" -o assets/<direction>/<slug>/arch.png`
4. 正文嵌入（论文文件在 `papers/<direction>/`，故相对路径为 `../../assets/...`）：
   `![架构图（来源：arXiv <id>, <论文短名>）](../../assets/<direction>/<slug>/arch.png)`
5. 挑图原则：架构/pipeline 图必选；再选 1-2 张最能说明核心机制或主结果的图；避免纯文字表格截图（表格用 markdown 重画）。每篇 2-4 张为宜，控制体积。
6. 若无 HTML 版（老论文），尝试项目主页或 GitHub README 的图；仍无则跳过，正文说明"原文图未获取"。

## 公式渲染规范（GitHub MathJax，务必遵守）

行内/展示公式在 GitHub 上有严格解析规则，违反会导致整块公式渲染成乱码或原样文本。写公式时必须：

1. **行内 `$...$` 与中文之间留 ASCII 空格**：`：$x$` 不渲染 → 写 `： $x$`；`（$m$）` → `（ $m$ ）`。这是最常见的坑。
2. **math mode 内禁用 `#`**：`#` 是 LaTeX 宏参数字符，会报 `macro parameter character` 并使整块 fallback。用 `\lvert\{\dots\}\rvert`（基数）等替代。
2b. **同一行多个含下划线的行内公式必须转义 `_` 为 `\_`**：GitHub 会把同行多个 `_` 两两配对当成斜体强调，吃掉下划线并破坏公式（如 `$\mathcal{L}_{\text{a}}$ ... $\mathcal{L}_{\text{b}}$` 变成乱码）。写成 `$\mathcal{L}\_{\text{a}}$`，GitHub 与 Obsidian 两端都正确。`fix_inline_math.py` 会自动处理（对含 ≥2 个下划线公式的行转义）。注意：不要用 `` $`...`$ ``（GitHub 官方推荐但 Obsidian 会显示反引号）。
3. **`\text{}` 内不放中文**：MathJax 渲染不了 CJK，中文说明写在公式外。
4. **展示公式**用独立成行的 `$$ ... $$`；确保 `$$` 成对、花括号平衡。
5. **收尾必跑**：`python3 scripts/fix_inline_math.py <file.md> --write` 自动在 CJK 与 `$` 间补空格（跳过代码块与 `$$` 块）；再复查残余为 0。

## 验收
- frontmatter 能被 `scripts/lib_frontmatter.py` 解析（无 YAML 报错）。
- `python3 scripts/build_indices.py` 跑通，新论文出现在 indices/papers.md 与 timeline.md。
- 谱系 frontmatter 与正文链接一致。
- 已做横向对比判断：正文「## 6.」非空；若纳入/新建了对比文件，其链接可解析且已同步 `indices/comparisons.md`。
- 深度小节（2.1-2.4、4.1）非空；嵌入的图片相对路径可解析（文件真实存在于 assets/），图注标注了来源。
- 公式可正常渲染：无 CJK 紧贴 `$...$`（跑 `scripts/fix_inline_math.py` 复查残余为 0）、math mode 无 `#`、`$$` 成对、花括号平衡。
