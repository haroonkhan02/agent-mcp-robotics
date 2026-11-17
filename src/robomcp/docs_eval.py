"""`robomcp-docs-eval`: how good is the search the `search_docs` tool exposes?

Uses the hand-written question set of robotics-docs-rag. A returned section is relevant
when it comes from a gold page and contains one of the question's answer phrases, the
same definition as in that project, with whole sections here instead of 180-word chunks.

Writes reports/docs_search.md and reports/docs_search.json.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from robomcp.config import load_settings
from robomcp.docs import PAGE_CHARS, DocSection, DocsIndex

_WS = re.compile(r"\s+")
KS = (1, 3, 5, 10)


def norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\\", "")).strip().lower()


@dataclass
class Question:
    id: str
    category: str
    question: str
    gold_docs: list[str] = field(default_factory=list)
    answer_keys: list[str] = field(default_factory=list)
    reference_answer: str = ""


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def key_offset(section: DocSection, q: Question) -> int | None:
    """Character offset of the first answer phrase in the section, or None if not relevant."""
    if section.doc_id not in q.gold_docs:
        return None
    text = norm(section.text)
    hits = [text.find(norm(k)) for k in q.answer_keys]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else None


def evaluate(index: DocsIndex, questions: list[Question], k_max: int = 10) -> dict:
    answerable = [q for q in questions if q.gold_docs]
    rows = []
    for q in answerable:
        results = [s for s, _ in index.search(q.question, k_max)]
        offsets = [key_offset(s, q) for s in results]
        rank = next((i for i, o in enumerate(offsets, start=1) if o is not None), None)
        page_rank = next(
            (i for i, s in enumerate(results, start=1) if s.doc_id in q.gold_docs), None)
        rows.append({
            "id": q.id, "category": q.category, "rank": rank, "page_rank": page_rank,
            "offset": offsets[rank - 1] if rank else None,
        })
    n = len(rows)

    def hit(key: str, k: int) -> float:
        return sum(1 for r in rows if r[key] and r[key] <= k) / n

    found = [r for r in rows if r["rank"]]
    return {
        "n_questions": n,
        "n_sections": len(index),
        "section": {f"hit@{k}": hit("rank", k) for k in KS},
        "page": {f"hit@{k}": hit("page_rank", k) for k in KS},
        "mrr@10": sum(1 / r["rank"] for r in found) / n,
        "found": len(found),
        "answer_on_first_page": sum(1 for r in found if r["offset"] < PAGE_CHARS),
        "missed": [r["id"] for r in rows if not r["rank"]],
        "rows": rows,
    }


def to_markdown(res: dict) -> str:
    lines = [
        "# Documentation search behind `search_docs`",
        "",
        f"BM25 over {res['n_sections']:,} sections of the ROS 2 docs, "
        f"{res['n_questions']} answerable questions from robotics-docs-rag.",
        "",
        "| Counted as a hit | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR@10 |",
        "|---|" + "---|" * (len(KS) + 1),
        "| Section from a gold page that contains an answer phrase | "
        + " | ".join(f"{res['section'][f'hit@{k}']:.3f}" for k in KS)
        + f" | {res['mrr@10']:.3f} |",
        "| Any section from a gold page | "
        + " | ".join(f"{res['page'][f'hit@{k}']:.3f}" for k in KS) + " | |",
        "",
        f"Of the {res['found']} questions with a relevant section in the top 10, "
        f"{res['answer_on_first_page']} have the answer phrase within the first "
        f"{PAGE_CHARS:,} characters of that section, which is what one `read_doc_section` "
        "call returns.",
        "",
        f"No relevant section in the top 10: {', '.join(res['missed']) or 'none'}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    index = DocsIndex.load(cfg.index_path)
    res = evaluate(index, load_questions(cfg.data_dir / "eval" / "questions.jsonl"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "docs_search.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "docs_search.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
