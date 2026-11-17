"""The documentation index behind the `search_docs` and `read_doc_section` tools.

The unit is a section (one heading and the text under it), not a fixed-size chunk: an
agent searches, gets a few section ids with a short snippet each, and reads the sections
it wants, a page of text at a time. Search is BM25 over heading path + text, so the server
needs no embedding model.

`robomcp-build-index` writes one JSON line per section to artifacts/docs_sections.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from robomcp.config import load_settings
from robomcp.rst import Document, load_corpus

_TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
PAGE_CHARS = 2000  # how much of a section one read returns
SNIPPET_CHARS = 280


def tokenize(text: str) -> list[str]:
    """Lower-case words; identifiers such as `use_sim_time` stay whole and are also split."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "_" in tok:
            out.extend(tok.split("_"))
    return out


@dataclass(frozen=True)
class DocSection:
    id: str  # "<doc_id>#<n>", n = position of the section in its page
    doc_id: str
    title: str
    heading_path: tuple[str, ...]
    url: str
    text: str

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


def sections_from(docs: list[Document]) -> list[DocSection]:
    out = []
    for d in docs:
        for n, s in enumerate(d.sections):
            if s.text.strip():
                out.append(DocSection(f"{d.doc_id}#{n}", d.doc_id, d.title,
                                      tuple(s.heading_path), d.source, s.text))
    return out


def save_sections(sections: list[DocSection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    tmp.replace(path)  # a crash never leaves a half-written index behind


def load_sections(path: Path) -> list[DocSection]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["heading_path"] = tuple(row["heading_path"])
                out.append(DocSection(**row))
    return out


def snippet(text: str, query_tokens: set[str], width: int = SNIPPET_CHARS) -> str:
    """The paragraph that shares most words with the query, cut to `width` characters."""
    paras = [p for p in text.split("\n\n") if p.strip()] or [text]
    best = max(paras, key=lambda p: len(query_tokens & set(tokenize(p))))
    best = " ".join(best.split())
    return best if len(best) <= width else best[: width - 1].rstrip() + "…"


class DocsIndex:
    def __init__(self, sections: list[DocSection]):
        if not sections:
            raise ValueError("the documentation index is empty")
        self.sections = sections
        self._by_id = {s.id: s for s in sections}
        if len(self._by_id) != len(sections):
            raise ValueError("duplicate section ids in the documentation index")
        self._bm25 = BM25Okapi([tokenize(s.heading + "\n" + s.text) for s in sections])

    @classmethod
    def load(cls, path: Path) -> DocsIndex:
        return cls(load_sections(path))

    def __len__(self) -> int:
        return len(self.sections)

    def search(self, query: str, k: int = 5) -> list[tuple[DocSection, float]]:
        """Top-k sections by BM25 score; sections that share no word with the query are left out."""
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))  # stable on ties
        return [(self.sections[i], float(scores[i])) for i in order[:k] if scores[i] > 0]

    def get(self, section_id: str) -> DocSection | None:
        return self._by_id.get(section_id)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the documentation section index.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    root = cfg.data_dir / "ros2_documentation" / "source"
    if not root.exists():
        raise SystemExit(f"{root} not found; run robomcp-download first")
    docs = load_corpus(root, cfg.corpus.exclude_globs)
    sections = sections_from(docs)
    save_sections(sections, cfg.index_path)
    chars = sorted(len(s.text) for s in sections)
    print(f"{len(docs)} pages -> {len(sections)} sections in {cfg.index_path}")
    print(f"section length (characters): median {chars[len(chars) // 2]}, "
          f"p95 {chars[int(len(chars) * 0.95)]}, max {chars[-1]}")


if __name__ == "__main__":
    main()
