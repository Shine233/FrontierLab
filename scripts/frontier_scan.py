#!/usr/bin/env python3
"""前沿跟踪的 search-history 管理与差分。

不自己抓网络——由 skill 里的 agent 用 WebSearch/arXiv 产出候选清单，本脚本负责：
  1) 记住「已见过/已入库」的论文，避免重复分析；
  2) 差分：给一批候选，只回没见过的（首扫可能几百篇，再扫可能几篇甚至 0）；
  3) 记录每次扫描的时间窗，支撑「上次扫到哪、这次从哪续」。

状态文件：indices/frontier-scan-history.jsonl（每行一次扫描记录，可 git 追踪）。
已入库论文的 arxiv id 由 frontmatter 自动纳入「已见」，无需手动登记。

用法：
  # 差分：从 stdin 读候选（每行 "arxiv_id\\t标题"），打印其中的新条目
  python3 scripts/frontier_scan.py diff < candidates.tsv
  # 记录一次扫描（扫完后调用，登记这次覆盖的窗口与看过的 id）
  python3 scripts/frontier_scan.py record --window 2026-06-23..2026-07-23 --note "3d-recon 月扫" < candidates.tsv
  # 看历史
  python3 scripts/frontier_scan.py log
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_frontmatter as fm

HIST = os.path.join("indices", "frontier-scan-history.jsonl")


def _norm_id(s):
    """归一 arXiv id：去 arxiv: 前缀、去版本号、去空白、小写。"""
    s = str(s or "").strip().lower()
    for p in ("arxiv:", "arxiv ", "abs/", "pdf/"):
        s = s.replace(p, "")
    s = s.strip().rstrip("/")
    if "arxiv.org" in s:
        s = s.split("/")[-1]
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def seen_ids(repo_root="."):
    """已见 id 集合 = 历史扫描记录里所有 id ∪ 已入库论文 frontmatter 的 arxiv。"""
    ids = set()
    hist = os.path.join(repo_root, HIST)
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for i in rec.get("ids", []):
                    ids.add(_norm_id(i))
    for p in fm.load_papers(os.path.join(repo_root, "papers")):
        ax = p.get("arxiv")
        if ax:
            ids.add(_norm_id(ax))
    return ids


def _read_candidates(stream):
    """stdin 每行 'id<TAB>title'（title 可省）。返回 [(id, title), ...]。"""
    out = []
    for line in stream:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        cid = _norm_id(parts[0])
        title = parts[1].strip() if len(parts) > 1 else ""
        if cid:
            out.append((cid, title))
    return out


def cmd_diff(repo_root, args):
    seen = seen_ids(repo_root)
    cands = _read_candidates(sys.stdin)
    fresh = [(i, t) for i, t in cands if i not in seen]
    sys.stderr.write("候选 {} 篇，已见 {} 篇，新 {} 篇\n".format(len(cands), len(cands) - len(fresh), len(fresh)))
    for i, t in fresh:
        print("{}\t{}".format(i, t))


def cmd_record(repo_root, args):
    window = None
    note = ""
    if "--window" in args:
        window = args[args.index("--window") + 1]
    if "--note" in args:
        note = args[args.index("--note") + 1]
    cands = _read_candidates(sys.stdin)
    rec = {
        "window": window or "",
        "note": note,
        "n": len(cands),
        "ids": [i for i, _ in cands],
    }
    hist = os.path.join(repo_root, HIST)
    os.makedirs(os.path.dirname(hist), exist_ok=True)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stderr.write("已记录扫描：window={} 覆盖 {} 篇\n".format(window, len(cands)))


def cmd_log(repo_root, args):
    hist = os.path.join(repo_root, HIST)
    if not os.path.exists(hist):
        print("（无扫描历史）")
        return
    with open(hist, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            print("window={:24} n={:<4} {}".format(r.get("window", "?"), r.get("n", 0), r.get("note", "")))


def main(argv):
    if len(argv) < 2 or argv[1] not in ("diff", "record", "log"):
        print(__doc__)
        return
    cmd = argv[1]
    rest = argv[2:]
    repo_root = "."
    # 允许末尾传 repo_root（不以 -- 开头且不是 --window/--note 的值）
    {"diff": cmd_diff, "record": cmd_record, "log": cmd_log}[cmd](repo_root, rest)


if __name__ == "__main__":
    main(sys.argv)
