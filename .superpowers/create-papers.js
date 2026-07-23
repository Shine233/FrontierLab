export const meta = {
  name: 'create-papers',
  description: '新建论文分析（深度模板：架构/原理/公式/性能 + 插图），agent 自行核实 arxiv/year/venue',
  phases: [
    { title: 'Research+Draft', detail: '深读 + 下图 + 按深度模板起草' },
    { title: 'Verify', detail: '核查事实/公式/图片/渲染' },
  ],
}

// args: [{name, dir, slug_hint, note}, ...]  —— agent 核实确切 year/venue/arxiv，slug 用 <year>-<short>
const PAPERS = args

const SCHEMA = {
  type: 'object',
  properties: {
    slug: { type: 'string', description: '最终 slug = <year>-<short-title>，与背景提示 slug_hint 一致或按核实年份修正' },
    dir: { type: 'string' },
    markdown: { type: 'string' },
    figures: { type: 'array', items: { type: 'string' } },
    uncertainties: { type: 'array', items: { type: 'string' } },
  },
  required: ['slug', 'dir', 'markdown', 'figures', 'uncertainties'],
}

const REPO_SLUGS = '2004-sift,2011-kinectfusion,2016-colmap,2020-nerf,2022-instant-ngp,2023-3dgs,2023-dust3r,2024-mast3r,2024-mv-dust3r,2024-spann3r,2025-cut3r,2025-depth-anything-3,2025-fast3r,2025-hunyuanworld-mirror,2025-mapanything,2025-megasam,2025-monst3r,2025-omnivggt,2025-stream3r,2025-ttt3r,2025-vggt,2025-wint3r,2026-lingbot-map,2026-pi3,2026-vggt-omega'

const results = await pipeline(
  PAPERS,
  (p) => agent(
    `为中文知识库 FrontierLab **新建**「${p.name}」的论文分析，方向 ${p.dir}，工作目录 /Workspace/FrontierLab。这是一篇仓库尚无的论文。

背景提示：${p.note}
建议 slug：${p.slug_hint}（若核实的发表年份不同，改成正确的 <year>-<short-title>）。

步骤：
1. 用 WebSearch/WebFetch 核实：确切论文标题、发表年份、venue、arXiv id（经典论文可能无 arXiv，用官方/会议链接）、官方代码仓库、license、训练是否开源、权重/数据可用性，以及 2-3 个代表性数值/基准。不要凭记忆编数字。
2. 按深度模板写完整 markdown（含 YAML frontmatter，字段参照 papers/_template.md）：
   - frontmatter：type: paper-analysis；title/short_name/year(整数)/venue/arxiv/paper_url/code/github/open_source/license/training_open_source/direction(含${p.dir})/method_family/tasks/datasets/metrics/status: analyzed/reproduction/confidence/landmark(判断)/org/key_idea/supersedes/builds_on/updated: 2026-07-23。
   - 正文：结论先行(3-5)；1 问题；2 方法概览 + 2.1 架构解析(配图) + 2.2 核心原理 + 2.3 关键公式解析(LaTeX 逐符号；无严格公式则形式化并注明) + 2.4 训练推理；3 贡献；4 实验 + 4.1 效果性能解析；5 局限；方法谱系(只链接仓库真实存在的 slug：${REPO_SLUGS}，无则省略链接)；6 与相似方法对比(指向 comparisons/3d-reconstruction/ 下相关对比文件)；7 复现判断；8 后续动作。
3. 公式渲染规范(GitHub MathJax)：行内 $...$ 与中文间留 ASCII 空格(「： $x$ 」)；math mode 禁用 #；\\text{} 不放中文；$$ 成对、花括号平衡。返回前自查。
4. 图片：从 arXiv HTML(https://arxiv.org/html/<id>) 或项目页找架构/pipeline 图，下载到 assets/${p.dir}/<slug>/（curl -sL "<url>" -o assets/${p.dir}/<slug>/arch.png，file 确认有效）。取图失败超过 2 次就跳过，正文注明"原文图未获取"，不编造路径。每篇 1-3 张。正文用 ../../assets/${p.dir}/<slug>/xxx.png 嵌入并标来源。

中文；结论先行；证据与推断分开。返回：slug、dir、完整 markdown、figures、存疑点。`,
    { label: `draft:${p.slug_hint}`, phase: 'Research+Draft', schema: SCHEMA, effort: 'high' }
  ),
  (draft, p) => agent(
    `事实核查「${p.name}」新建稿（方向 ${p.dir}，/Workspace/FrontierLab）。

草稿 markdown：
${draft.markdown}
已下载图片：${JSON.stringify(draft.figures)}
存疑：${JSON.stringify(draft.uncertainties)}

任务：
1. WebSearch/WebFetch 验证 arXiv id/year/venue、license、训练开源、2.3 公式、4.1 数值。修正错误；无法核实改定性或标"待核验"。
2. 校验图片：markdown 里每个 ![...](../../assets/${p.dir}/<slug>/xxx) 用 Bash 确认文件存在且有效（test -f && file）；不存在就删引用或改"原文图未获取"，不新增编造图。
3. 公式渲染合规：行内 $...$ 与中文间 ASCII 空格、math mode 无 #、\\text{} 无中文、$$ 成对。逐条修正。
4. 谱系/对比链接只指向仓库真实存在的文件。

**关键：返回完整 markdown 字符串本身，不要写"见上"之类占位。** 返回修正后完整 markdown、最终 figures、slug、dir、改动说明。`,
    { label: `verify:${p.slug_hint}`, phase: 'Verify', schema: SCHEMA, effort: 'high' }
  )
)

return results.filter(Boolean).map(r => ({ slug: r.slug, dir: r.dir, markdown: r.markdown, figures: r.figures, uncertainties: r.uncertainties }))
