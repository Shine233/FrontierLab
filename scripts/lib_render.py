"""把论文记录渲染成各索引 markdown 片段。纯函数，不写文件。"""
from collections import Counter, defaultdict


def display_name(rec):
    for k in ("short_name", "title"):
        v = rec.get(k)
        if v:
            return str(v)
    return rec.get("_slug", "")


def paper_link(rec):
    path = rec.get("_path", "")
    return "[{}](../{})".format(display_name(rec), path) if path else "[{}]()".format(display_name(rec))


def _join_list(v):
    if isinstance(v, list):
        return " / ".join(str(x) for x in v)
    return str(v) if v else ""


def _cell(v):
    """表格单元：转字符串并转义竖线。"""
    return _join_list(v).replace("|", "\\|")


def render_papers_table(papers):
    lines = [
        "| Paper | Year | Direction | Method family | Status | Reproduction | Org | Key idea | Link |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for r in papers:
        link = r.get("github") or r.get("code") or ""
        link_md = "[repo]({})".format(link) if link else ""
        lines.append(
            "| {name} | {year} | {direction} | {mf} | {status} | {repro} | {org} | {key} | {link} |".format(
                name=paper_link(r),
                year=r.get("year", ""),
                direction=_cell(r.get("direction")),
                mf=_cell(r.get("method_family")),
                status=_cell(r.get("status")),
                repro=_cell(r.get("reproduction")),
                org=_cell(r.get("org")),
                key=_cell(r.get("key_idea")),
                link=link_md,
            )
        )
    return "\n".join(lines)


def _group_by_list_field(papers, field):
    groups = defaultdict(list)
    for r in papers:
        vals = r.get(field) or []
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            groups[str(v)].append(r)
    return groups


def render_directions(papers):
    groups = _group_by_list_field(papers, "direction")
    out = []
    for key in sorted(groups):
        out.append("## {}".format(key))
        out.append("")
        for r in sorted(groups[key], key=lambda x: x.get("_slug", "")):
            out.append("- {} （{}）".format(paper_link(r), r.get("year", "")))
        out.append("")
    return "\n".join(out).rstrip()


def render_methods(papers):
    groups = _group_by_list_field(papers, "method_family")
    out = []
    for key in sorted(groups):
        out.append("## {}".format(key))
        out.append("")
        for r in sorted(groups[key], key=lambda x: x.get("_slug", "")):
            out.append("- {} （{}）".format(paper_link(r), r.get("year", "")))
        out.append("")
    return "\n".join(out).rstrip()


def render_timeline(papers):
    by_year = defaultdict(list)
    for r in papers:
        by_year[r.get("year") or 0].append(r)
    out = []
    for year in sorted(by_year, reverse=True):
        out.append("## {}".format(year))
        out.append("")
        dir_counter = Counter()
        for r in sorted(by_year[year], key=lambda x: x.get("_slug", "")):
            star = "★ " if r.get("landmark") is True else ""
            out.append("- {}{} — {}".format(star, paper_link(r), _join_list(r.get("key_idea"))))
            dirs = r.get("direction") or []
            if not isinstance(dirs, list):
                dirs = [dirs]
            for d in dirs:
                dir_counter[str(d)] += 1
        out.append("")
        summary = ", ".join("{}: {}".format(d, n) for d, n in sorted(dir_counter.items()))
        out.append("方向分布：{}".format(summary))
        out.append("")
    return "\n".join(out).rstrip()


def render_landmarks(papers):
    out = []
    for r in sorted(papers, key=lambda x: (-(x.get("year") or 0), x.get("_slug", ""))):
        if r.get("landmark") is not True:
            continue
        out.append("## {} （{}）".format(display_name(r), r.get("year", "")))
        out.append("")
        out.append("- 链接：{}".format(paper_link(r)))
        if r.get("key_idea"):
            out.append("- 核心：{}".format(r["key_idea"]))
        if r.get("supersedes"):
            out.append("- 取代/改进：{}".format(_join_list(r["supersedes"])))
        if r.get("builds_on"):
            out.append("- 基于：{}".format(_join_list(r["builds_on"])))
        out.append("")
    return "\n".join(out).rstrip()
