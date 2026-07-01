import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lib_frontmatter as fm

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "papers")


def test_parse_paper_injects_slug_and_path():
    p = os.path.join(FIX, "image-matching", "2024-sample.md")
    rec = fm.parse_paper(p)
    assert rec is not None
    assert rec["_slug"] == "2024-sample"
    assert rec["_path"].endswith("image-matching/2024-sample.md")
    assert rec["year"] == 2024
    assert rec["landmark"] is True
    assert rec["builds_on"] == ["2023-sample-backbone"]


def test_parse_paper_tolerates_extra_and_missing_fields():
    p = os.path.join(FIX, "world-models", "2026-sample.md")
    rec = fm.parse_paper(p)
    assert rec is not None
    assert "landmark" not in rec  # 缺失字段不报错、不伪造
    assert rec["org"] == ["SampleAuto"]


def test_parse_paper_returns_none_without_frontmatter(tmp_path):
    f = tmp_path / "nofm.md"
    f.write_text("# 没有 frontmatter\n正文", encoding="utf-8")
    assert fm.parse_paper(str(f)) is None


def test_load_papers_skips_underscore_and_sorts(tmp_path):
    d = tmp_path / "papers" / "x"
    d.mkdir(parents=True)
    (tmp_path / "papers" / "_template.md").write_text(
        "---\ntitle: t\n---\n", encoding="utf-8"
    )
    (d / "2024-b.md").write_text(
        "---\ntitle: B\nyear: 2024\n---\n", encoding="utf-8"
    )
    (d / "2024-a.md").write_text(
        "---\ntitle: A\nyear: 2024\n---\n", encoding="utf-8"
    )
    recs = fm.load_papers(str(tmp_path / "papers"))
    assert [r["_slug"] for r in recs] == ["2024-a", "2024-b"]
