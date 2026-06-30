import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_indices as bi


def _make_repo(tmp_path):
    papers = tmp_path / "papers" / "image-matching"
    papers.mkdir(parents=True)
    (papers / "2024-x.md").write_text(
        "---\ntitle: X\nshort_name: X\nyear: 2024\n"
        "direction: [image-matching]\nmethod_family: [m1]\n"
        "status: compared\nreproduction: planned\nlandmark: true\n"
        "org: [Lab]\nkey_idea: K\n---\n# X\n",
        encoding="utf-8",
    )
    (tmp_path / "indices").mkdir()
    return tmp_path


def test_write_full_has_header(tmp_path):
    p = tmp_path / "out.md"
    bi.write_full(str(p), "BODY")
    text = p.read_text(encoding="utf-8")
    assert text.startswith(bi.AUTO_HEADER)
    assert "BODY" in text


def test_splice_preserves_manual_block(tmp_path):
    p = tmp_path / "papers_index.md"
    p.write_text(
        bi.AUTO_HEADER + "\n\n"
        "<!-- AUTO:start -->\nOLD AUTO\n<!-- AUTO:end -->\n\n"
        "<!-- MANUAL:start -->\n人工叙述保留我\n<!-- MANUAL:end -->\n",
        encoding="utf-8",
    )
    new = bi.splice_auto_block(str(p), "NEW AUTO")
    assert "NEW AUTO" in new
    assert "OLD AUTO" not in new
    assert "人工叙述保留我" in new


def test_splice_creates_when_missing(tmp_path):
    p = tmp_path / "new.md"
    new = bi.splice_auto_block(str(p), "BODY")
    assert "<!-- AUTO:start -->" in new
    assert "BODY" in new
    assert "<!-- MANUAL:start -->" in new


def test_main_generates_all_indices(tmp_path):
    repo = _make_repo(tmp_path)
    bi.main(str(repo))
    idx = repo / "indices"
    assert (idx / "timeline.md").exists()
    assert (idx / "landmarks.md").exists()
    assert "★" in (idx / "timeline.md").read_text(encoding="utf-8")
    papers_md = (idx / "papers.md").read_text(encoding="utf-8")
    assert "[X](" in papers_md


def test_main_is_idempotent(tmp_path):
    repo = _make_repo(tmp_path)
    bi.main(str(repo))
    first = (repo / "indices" / "papers.md").read_text(encoding="utf-8")
    bi.main(str(repo))
    second = (repo / "indices" / "papers.md").read_text(encoding="utf-8")
    assert first == second
