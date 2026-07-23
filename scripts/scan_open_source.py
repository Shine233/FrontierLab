#!/usr/bin/env python3
"""开源状态扫描：从论文 frontmatter 归类开源程度，产出 indices/open-source.md。

分类阶梯（从 frontmatter 现有字段推导，不新增必填字段）：
  - 未开源          : open_source 非 true（false/unknown），或无 code/github
  - 仅推理开源      : open_source true，但 training_open_source 关闭/占位（false/"\\"/n/a/unknown）
  - 训练(部分)开源  : open_source true 且 training_open_source true/partial

「需重新 review」= 未开源 + 仅推理开源 + partial 的并集，按 oss_last_checked 陈旧度排序
（缺失=从未核实，排最前）。oss_last_checked 为可选 frontmatter 字段，重新核实后回填 YYYY-MM-DD。

用法：python3 scripts/scan_open_source.py [repo_root]   (默认当前目录)
     python3 scripts/scan_open_source.py --stale [repo_root]   只打印待重审清单
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_frontmatter as fm
import lib_render as render


def _norm(v):
    """把杂乱的 open/training 值归一到 {open, closed, partial, n/a, unknown}。"""
    if v is True:
        return "open"
    if v is False:
        return "closed"
    s = str(v or "").strip().lower()
    if s in ("true", "yes"):
        return "open"
    if s in ("false", "no"):
        return "closed"
    if s.startswith("partial"):
        return "partial"
    if s in ("n/a", "na", "none"):
        return "n/a"
    if s in ("", "unknown", "?"):
        return "unknown"
    if s == "\\":  # 模板约定：公开仓库仅推理/demo，未给训练代码
        return "closed"
    return "unknown"


def classify(rec):
    has_repo = bool(rec.get("code") or rec.get("github"))
    osrc = _norm(rec.get("open_source"))
    train = _norm(rec.get("training_open_source"))
    if osrc != "open" or not has_repo:
        return "未开源"
    if train in ("open", "partial"):
        return "训练(部分)开源"
    return "仅推理开源"


ORDER = ["未开源", "仅推理开源", "训练(部分)开源"]


def _repo_link(rec):
    url = rec.get("github") or rec.get("code") or ""
    return "[repo]({})".format(url) if url else "—"


def build_report(papers):
    buckets = {k: [] for k in ORDER}
    for r in papers:
        buckets[classify(r)].append(r)
    lines = ["# 开源状态跟踪", "",
             "> 由 `scripts/scan_open_source.py` 从 frontmatter 生成。",
             "> 重新核实某篇后：更新其 `open_source`/`training_open_source`，回填 `oss_last_checked: YYYY-MM-DD`，重跑本脚本。", ""]
    for k in ORDER:
        rows = sorted(buckets[k], key=lambda x: x.get("_slug", ""))
        lines.append("## {}（{}）".format(k, len(rows)))
        lines.append("")
        if not rows:
            lines.append("（无）")
            lines.append("")
            continue
        lines.append("| Paper | Year | open_source | training | 上次核实 | Repo |")
        lines.append("|---|---:|---|---|---|---|")
        for r in rows:
            lines.append("| {name} | {yr} | {os} | {tr} | {ck} | {repo} |".format(
                name=render.paper_link(r), yr=r.get("year", ""),
                os=_norm(r.get("open_source")), tr=_norm(r.get("training_open_source")),
                ck=r.get("oss_last_checked", "—"), repo=_repo_link(r)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def stale_list(papers):
    """待重审：未开源 / 仅推理开源 / partial，按 oss_last_checked 升序（缺失最前）。"""
    flagged = [r for r in papers if classify(r) != "训练(部分)开源"
               or _norm(r.get("training_open_source")) == "partial"]
    return sorted(flagged, key=lambda r: (str(r.get("oss_last_checked") or ""), r.get("_slug", "")))


def main(argv):
    stale_only = "--stale" in argv
    rest = [a for a in argv[1:] if not a.startswith("--")]
    repo_root = rest[0] if rest else "."
    papers = fm.load_papers(os.path.join(repo_root, "papers"))

    if stale_only:
        print("待重新 review（按陈旧度）：")
        for r in stale_list(papers):
            print("  {:12} {:22} 上次核实={} github={}".format(
                classify(r), r.get("_slug", ""),
                r.get("oss_last_checked", "never"),
                r.get("github") or r.get("code") or "—"))
        return

    out = os.path.join(repo_root, "indices", "open-source.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report(papers))
    n = len(papers)
    closed = sum(1 for r in papers if classify(r) == "未开源")
    infer = sum(1 for r in papers if classify(r) == "仅推理开源")
    train = sum(1 for r in papers if classify(r) == "训练(部分)开源")
    print("开源状态已生成：{} 篇（未开源 {} / 仅推理 {} / 训练开源 {}）".format(n, closed, infer, train))


if __name__ == "__main__":
    main(sys.argv)
