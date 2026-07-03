export const meta = {
  name: 'deepen-papers',
  description: '按深度模板重写论文（架构/原理/公式/性能）+ 下载嵌入插图 + 公式修复',
  phases: [
    { title: 'Deepen+Figures', detail: '每篇：深读 + 下图 + 深度重写' },
    { title: 'Verify', detail: '核查公式/数值/图片路径/渲染合规' },
  ],
}

// args: [{slug, dir, arxiv, name}, ...]
const PAPERS = args

const SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string' },
    dir: { type: 'string' },
    markdown: { type: 'string' },
    figures: { type: 'array', items: { type: 'string' } },
    uncertainties: { type: 'array', items: { type: 'string' } },
  },
  required: ['slug', 'dir', 'markdown', 'figures', 'uncertainties'],
}

const results = await pipeline(
  PAPERS,
  (p) => agent(
    `深度重写中文知识库 FrontierLab 里「${p.name}」的论文分析（方向 ${p.dir}），工作目录 /Workspace/FrontierLab。

先 Read 现有文件 papers/${p.dir}/${p.slug}.md —— 保留已核实的 frontmatter 与事实（arXiv、license、数值、方法谱系、横向对比指针），在此基础上**加深**，不推翻已核实内容。

深度模板（对齐 papers/_template.md，正文必须达到这个深度）：
- ## 2. 方法概览：核心想法 + 一句话 pipeline
  - ### 2.1 架构解析：模块分解 + 数据流 + 关键设计选择，**配架构图**
  - ### 2.2 核心原理：为什么 work、关键机制/归纳偏置、与前作本质区别
  - ### 2.3 关键公式解析：LaTeX，逐符号解释 + 作用（2-4 个核心公式；若论文无严格公式，用形式化表述并注明"论文未给严格公式"）
  - ### 2.4 训练与推理细节：损失、数据规模、超参、推理流程
- ## 4. 实验与证据：保留原表 + ### 4.1 效果与性能解析（解读结果非搬数字 + 速度/显存/参数量 + 消融关键因素 + 可比性）
- 其余小节（结论先行、1、3、5、方法谱系、6、7、8）保留并加厚。

**公式渲染规范（GitHub MathJax，必须遵守）**：
- 行内 $...$ 与中文之间留 ASCII 空格：写「： $x$ 」不要「：$x$」。
- math mode 内禁用 #（用 \\lvert\\{...\\}\\rvert 等替代）。
- \\text{} 内不放中文。
- 展示公式用独立成行 $$ ... $$，$$ 成对、花括号平衡。
- 返回前自查所有行内公式与中文间已留空格。

图片提取（自己用 Bash 执行下载）：
- arXiv：${p.arxiv}。先试 HTML：https://arxiv.org/html/${p.arxiv}
- 用 WebFetch 打开 HTML，找图 URL（相对名 x1.png/x2.png...，完整 https://arxiv.org/html/${p.arxiv}/xN.png），结合图注确认架构/pipeline 图。
- 下载到 assets/${p.dir}/${p.slug}/，语义名（arch.png/pipeline.png/results.png）：
  curl -sL "<图URL>" -o assets/${p.dir}/${p.slug}/arch.png
  下完 file 确认是有效 PNG/JPG。若 HTML 不存在或取图失败超过 2 次，果断跳过取图（正文注明"原文图未获取"），不要反复重试卡住。
- 每篇 2-4 张（架构图优先）。正文用相对路径嵌入（论文在 papers/${p.dir}/，用 ../../assets/${p.dir}/${p.slug}/xxx.png）：
  ![架构图（来源：arXiv ${p.arxiv}, ${p.name}）](../../assets/${p.dir}/${p.slug}/arch.png)
- 图注标来源。下不到就正文注明"原文图未获取"，不编造路径。

要求：用 WebSearch/WebFetch 深读论文正文，公式和数值准确；中文；结论先行；证据与推断分开。slug 固定「${p.slug}」，dir 固定「${p.dir}」；frontmatter year 保持整数、updated 改 2026-07-02。

返回：slug、dir、完整 markdown、figures（已下载图片相对路径+来源）、存疑点。`,
    { label: `deepen:${p.slug}`, phase: 'Deepen+Figures', schema: SCHEMA, effort: 'high' }
  ),
  (draft, p) => agent(
    `事实核查「${p.name}」深度重写稿（方向 ${p.dir}，/Workspace/FrontierLab）。

草稿 markdown：
${draft.markdown}

已下载图片：${JSON.stringify(draft.figures)}
存疑：${JSON.stringify(draft.uncertainties)}

任务：
1. 用 WebSearch/WebFetch 验证 2.3 公式正确性、4.1 性能数值、frontmatter 的 arXiv/license。修正错误；无法核实的数值改定性或标"待核验"。
2. 校验图片：对 markdown 里每个 ![...](../../assets/${p.dir}/${p.slug}/xxx)，用 Bash 确认 /Workspace/FrontierLab/assets/${p.dir}/${p.slug}/xxx 真实存在且是有效图片（test -f && file）。引用了不存在/无效的图就删除该引用或改"原文图未获取"，不新增编造图。
3. 公式渲染合规（GitHub MathJax）：行内 $...$ 与中文间有 ASCII 空格、math mode 无 #、\\text{} 无中文、$$ 成对。逐条修正。
4. 保持深度模板结构（2.1-2.4、4.1）与 frontmatter 完整。

返回修正后完整 markdown、最终有效 figures 列表、改动说明。slug 保持「${p.slug}」，dir 保持「${p.dir}」。`,
    { label: `verify:${p.slug}`, phase: 'Verify', schema: SCHEMA, effort: 'high' }
  )
)

return results.filter(Boolean).map(r => ({ slug: r.slug, dir: r.dir, markdown: r.markdown, figures: r.figures, uncertainties: r.uncertainties }))
