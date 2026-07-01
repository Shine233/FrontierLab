"""论文 frontmatter 解析。唯一事实源读取层。"""
import os
import glob
import yaml


def parse_paper(path):
    """读单篇论文，返回 frontmatter dict（注入 _slug / _path）。

    无 frontmatter 或解析失败返回 None，不抛异常。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    data["_slug"] = os.path.splitext(os.path.basename(path))[0]
    data["_path"] = path.replace(os.sep, "/")
    return data


def load_papers(papers_dir):
    """递归扫描 papers_dir 下所有 *.md，跳过 _ 开头文件，按 _slug 排序。"""
    records = []
    pattern = os.path.join(papers_dir, "**", "*.md")
    for path in glob.glob(pattern, recursive=True):
        if os.path.basename(path).startswith("_"):
            continue
        rec = parse_paper(path)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: r.get("_slug", ""))
    return records
