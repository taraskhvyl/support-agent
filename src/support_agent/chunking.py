"""Split markdown into heading-scoped chunks. Pure functions, no I/O except build_chunks."""

import re
from pathlib import Path

from . import config


def chunk_text(title: str, text: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)]. Splits on #..### headings (ignoring code fences)
    and cuts long sections by paragraph with overlap."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # frontmatter
    heading, buf, out, in_fence = title, [], [], False

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        while len(body) > config.CHUNK_CHARS:
            cut = body.rfind("\n\n", 0, config.CHUNK_CHARS)
            cut = cut if cut > config.CHUNK_CHARS // 2 else config.CHUNK_CHARS
            out.append((heading, body[:cut].strip()))
            body = body[max(cut - config.CHUNK_OVERLAP, 1) :].strip()
        if body:
            out.append((heading, body))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"#{1,3} (.+)", line)
        if m:
            flush()
            heading = f"{title} > {m.group(1).strip()}"
        else:
            buf.append(line)
    flush()
    return out


def build_chunks(docs_dir: Path) -> list[tuple[str, str, dict]]:
    """Return [(id, text_to_embed, metadata)] for every markdown file under docs_dir."""
    rows = []
    for p in sorted(docs_dir.rglob("*.md")):
        rel = str(p.relative_to(docs_dir))
        for i, (heading, body) in enumerate(chunk_text(p.stem, p.read_text(encoding="utf-8"))):
            if len(body) >= config.MIN_CHUNK_CHARS:
                rows.append(
                    (f"{rel}#{i}", f"{heading}\n{body}", {"source": rel, "heading": heading})
                )
    return rows
