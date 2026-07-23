---
name: open-source-tracking
description: 扫描仓库已分析论文的开源程度（未开源/仅推理/训练开源），并对未完全开源的重新核实。适用于「哪些论文开源了」「开源状态更新」「这篇现在放代码了吗」。
---

# 开源状态跟踪 SOP

## 何时用
- 想知道仓库里哪些方法未开源、哪些只放了推理、哪些连训练都开源。
- 隔段时间重新核实：曾经未开源的现在放代码了吗（如 MoGe-3 的 v3、MAGiSt3R 的 repo）。

## 分类阶梯（从 frontmatter 推导，不新增必填字段）
- **未开源**：`open_source` 非 true，或无 `code`/`github`。
- **仅推理开源**：`open_source: true`，但 `training_open_source` 关闭/占位（`false` / `"\\"` / `n/a` / `unknown`）。
- **训练(部分)开源**：`open_source: true` 且 `training_open_source` 为 `true`/`partial`。

## 步骤
1. 先读 `KB-MEMORY.md`。
2. 生成/刷新总表：`python3 scripts/scan_open_source.py` → 写 `indices/open-source.md`（三档分桶 + 上次核实日期）。
3. 列出待重审：`python3 scripts/scan_open_source.py --stale`（未开源 + 仅推理 + partial，按 `oss_last_checked` 陈旧度排序，`never`=从未核实排最前）。
4. **重新 review**（对 --stale 里挑选的篇目，或用户点名的篇目）：
   - 打开该 paper 的 `github`/`code` 链接，用 WebFetch 看 README/Releases 现状：代码是否已发布、有无 `train*.py`/训练配置、权重（HuggingFace/checkpoint）、数据、license。
   - 判断当前档位是否变化（如「未开源」→「仅推理开源」，或「仅推理」→「训练开源」）。
5. **回填 frontmatter**（仅在状态确有变化或需要留痕时）：
   - 更新 `open_source` / `training_open_source` 为核实后的值；
   - 加/更新 `oss_last_checked: YYYY-MM-DD`（本次核实日期，用于下次 --stale 排序）；
   - 若代码新发布，顺带更新该 paper「## 7. 复现判断」与 `reproduction` 字段（可转 `planned`）。
6. 重跑 `python3 scripts/scan_open_source.py` 刷新总表。
7. 在 `KB-MEMORY.md`「各方向缺口」或决策记录里记一句本次核实结论（哪些篇目档位变了）。

## 与其他 skill 的关系
- 发现代码新发布、值得复现 → 转 `repro-plan`。
- 新论文首次入库时的开源状态由 `deep-analysis`/`paper-read` 在 frontmatter 填好，本 skill 只做**存量跟踪与重审**。

## 验收
- `indices/open-source.md` 三档桶数之和 = 论文总数（与 `build_indices` 的篇数一致）。
- 重审过的篇目 `oss_last_checked` 已回填为本次日期。
- 档位变化的篇目，frontmatter 与「复现判断」小节一致。
