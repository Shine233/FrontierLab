import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lib_render as r

PAPERS = [
    {
        "_slug": "2024-roma", "_path": "papers/image-matching/2024-roma.md",
        "title": "RoMa", "short_name": "RoMa", "year": 2024,
        "direction": ["image-matching"], "method_family": ["dense-feature-matching"],
        "status": "compared", "reproduction": "planned",
        "org": ["LinkopingU"], "key_idea": "稠密匹配。",
        "landmark": True, "supersedes": ["2023-dkm"], "builds_on": ["2023-dinov2"],
    },
    {
        "_slug": "2026-sample-wm", "_path": "papers/world-models/2026-sample-wm.md",
        "title": "Sample WM", "year": 2026,
        "direction": ["world-models"], "method_family": ["driving-world-model"],
        "status": "analyzed", "reproduction": "blocked-code-unavailable",
        "org": ["SampleAuto"], "key_idea": "世界模型。",
    },
]


def test_display_name_priority():
    assert r.display_name(PAPERS[0]) == "RoMa"
    assert r.display_name({"_slug": "x", "title": "T"}) == "T"
    assert r.display_name({"_slug": "x"}) == "x"


def test_paper_link():
    assert r.paper_link(PAPERS[0]) == "[RoMa](papers/image-matching/2024-roma.md)"


def test_papers_table_has_header_and_rows():
    out = r.render_papers_table(PAPERS)
    assert "| Paper | Year |" in out
    assert "[RoMa](papers/image-matching/2024-roma.md)" in out
    assert "2026" in out


def test_timeline_sorts_desc_and_marks_landmark():
    out = r.render_timeline(PAPERS)
    i2026 = out.index("2026")
    i2024 = out.index("2024")
    assert i2026 < i2024  # 倒序
    assert "★" in out  # landmark 标记
    assert "image-matching: 1" in out  # 2024 方向小计


def test_landmarks_lists_lineage():
    out = r.render_landmarks(PAPERS)
    assert "RoMa" in out
    assert "2023-dkm" in out
    assert "2023-dinov2" in out
    assert "Sample WM" not in out  # 非 landmark 不出现


def test_directions_groups():
    out = r.render_directions(PAPERS)
    assert "## image-matching" in out
    assert "## world-models" in out


def test_methods_groups():
    out = r.render_methods(PAPERS)
    assert "dense-feature-matching" in out
    assert "driving-world-model" in out
