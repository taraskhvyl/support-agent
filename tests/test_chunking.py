from support_agent import config
from support_agent.chunking import build_chunks, chunk_text


def test_splits_on_headings_with_path():
    out = chunk_text("Sync", "intro\n# Setup\nstep one\n## Pricing\ncosts money")
    assert [h for h, _ in out] == ["Sync", "Sync > Setup", "Sync > Pricing"]


def test_strips_frontmatter():
    out = chunk_text("T", "---\ntags: x\n---\nbody text")
    assert out == [("T", "body text")]


def test_hash_comment_inside_code_fence_is_not_a_heading():
    out = chunk_text("T", "# Real\n```sh\n# just a comment\nls\n```\nafter")
    assert [h for h, _ in out] == ["T > Real"]
    assert "# just a comment" in out[0][1]


def test_long_section_is_split_with_overlap(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_CHARS", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 20)
    text = "\n\n".join(f"paragraph number {i} " * 2 for i in range(20))
    out = chunk_text("T", text)
    assert len(out) > 1
    assert all(len(b) <= 100 for _, b in out)
    # overlap: the start of chunk n+1 appears at the end of chunk n
    assert out[1][1][:10] in out[0][1]


def test_build_chunks_skips_tiny_and_sets_metadata(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "A.md").write_text("# H\n" + "x" * 80)
    (tmp_path / "B.md").write_text("tiny")
    rows = build_chunks(tmp_path)
    assert len(rows) == 1
    cid, text, meta = rows[0]
    assert cid == "sub/A.md#0" and meta == {"source": "sub/A.md", "heading": "A > H"}
    assert text.startswith("A > H\n")
