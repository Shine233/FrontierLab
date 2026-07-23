---
name: frontier-tracking
description: 追踪已有方向的前沿新论文与版本更新（CVPR/ICCV best paper、方法新版如 MoGe-3），用 search-history 差分只捞新增，判断哪些值得深度分析。适用于「有什么新论文」「扫一下 3d-recon 最近进展」「MoGe 出新版了吗」。
---

# 前沿跟踪 SOP（含 search-history 差分）

## 何时用
- 想知道某个已有方向（3d-reconstruction / world-models / image-matching ...）最近有什么新论文或新版本。
- 定期复扫：首次可能要扫过去 3 年，之后只扫过去 1 个月甚至 1 天——靠差分避免重复。

## 核心机制：差分
`scripts/frontier_scan.py` 记住「已见/已入库」的 arXiv id（历史扫描记录 ∪ 已入库论文 frontmatter 的 `arxiv`），复扫时只回没见过的。所以**已分析的论文会自动被过滤**，不需手工维护清单。

## 步骤
1. 先读 `KB-MEMORY.md` 与 `indices/frontier-scan-history.jsonl`（`python3 scripts/frontier_scan.py log` 看上次扫到哪、覆盖什么窗口）。
2. **定窗**：首扫用长窗（如过去 3 年）；复扫从上次窗口末尾续到今天（如 `2026-06-23..2026-07-23`）。
3. **搜候选**（agent 用 WebSearch，不靠脚本抓网络）：
   - 方向关键词 + 时间窗（"feed-forward 3D reconstruction 2026"、"monocular geometry arXiv July 2026"）。
   - 会议渠道：CVPR/ICCV/ECCV/NeurIPS 的 accepted/best paper/oral 列表。
   - 版本更新：仓库已有方法的新代次（MoGe-3、DUSt3R→MASt3R、VGGT→VGGT-Ω 这类），查作者/项目页/GitHub releases。
   - 产出候选清单 TSV：每行 `arxiv_id<TAB>标题`（无 arXiv 的会议论文用 venue-slug 占位）。
4. **差分**：`python3 scripts/frontier_scan.py diff < candidates.tsv` → 只剩新条目。
5. **价值判断**（对新条目，人工/agent 评估，不是照单全收）：
   - 是否本仓库已覆盖方向、是否 SOTA/best paper/被广泛引用、是否是已有方法的重要新代次、是否有开源。
   - 分三级：**值得深度分析**（转 `deep-analysis`）/ **仅登记待观察**（记 KB-MEMORY，不建文件）/ **忽略**（增量小/偏题）。
6. **记录本次扫描**：`python3 scripts/frontier_scan.py record --window <窗口> --note "<方向> <频率>扫" < candidates.tsv`（把这次看过的**全部**候选登记为已见，下次不再冒出来）。
7. 值得分析的 → 交给 `deep-analysis` skill 逐篇产出；产出后其 frontmatter 的 `arxiv` 会自动进入「已见」。
8. 更新 `KB-MEMORY.md`：本次新增了哪些、哪些登记待观察、下次续扫的起点窗口。

## 与其他 skill 的关系
- 捞到值得分析的新论文 → `deep-analysis`（或单篇 `paper-read`）。
- 捞到已有方法的新版本、且旧版未开源/新版开源 → 顺带触发 `open-source-tracking` 重审。
- 同方向新论文 ≥2 篇同主题 → 可能要 `paper-compare` 建/更对比。

## 验收
- `frontier_scan.py diff` 的「已见」数与预期一致（已入库论文一定被过滤）。
- 复扫时时间窗与上次衔接，无缝无重叠遗漏。
- 每次扫描都 `record` 了（否则下次会重复冒出同批候选）。
- 值得分析的条目已转 `deep-analysis` 或明确记入 KB-MEMORY 待观察。
