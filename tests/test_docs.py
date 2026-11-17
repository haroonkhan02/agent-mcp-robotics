import io
import zipfile

import pytest

from robomcp.docs import (
    DocsIndex,
    load_sections,
    save_sections,
    snippet,
    tokenize,
)
from robomcp.docs_eval import Question, evaluate, key_offset
from robomcp.download import extract_ros2_zip, safe_members
from robomcp.rst import load_corpus, parse_rst


def test_rst_sections_and_markup(rst_root):
    docs = load_corpus(rst_root, ["Releases/*Complete-Changelog*"])
    assert [d.doc_id for d in docs] == ["Concepts/About-Parameters", "Concepts/About-Topics"]
    topics = docs[1]
    assert topics.title == "Topics"
    assert topics.source == "https://docs.ros.org/en/jazzy/Concepts/About-Topics.html"
    assert [s.heading_path for s in topics.sections] == [
        ("Topics",), ("Topics", "Durability"), ("Topics", "Commands")]
    text = topics.text
    assert "images/topic.gif" not in text and "Old/Path" not in text and "contents" not in text
    assert "See the QoS guide for ros2 topic echo details." in text
    assert "$ ros2 topic list" in text


def test_heading_levels_follow_first_appearance():
    _, blocks = parse_rst("Aaa\n===\n\nx\n\nBbb\n---\n\ny\n\nCcc\n===\n\nz\n")
    assert [(lvl, h) for lvl, h, _ in blocks if lvl] == [(1, "Aaa"), (2, "Bbb"), (1, "Ccc")]


def test_tokenize_keeps_and_splits_identifiers():
    assert tokenize("Set use_sim_time to True!") == [
        "set", "use_sim_time", "use", "sim", "time", "to", "true"]


def test_search_ranks_the_matching_section_first(docs_index):
    hits = docs_index.search("which durability keeps samples for late-joining subscriptions", 3)
    assert hits[0][0].id == "Concepts/About-Topics#1"
    assert hits[0][0].heading == "Topics > Durability"
    assert docs_index.search("change a parameter at run time", 1)[0][0].doc_id.endswith(
        "About-Parameters")
    assert docs_index.search("use_sim_time", 1)[0][0].id == "Concepts/About-Parameters#0"


def test_search_edge_cases(docs_index):
    assert docs_index.search("", 5) == []
    assert docs_index.search("???", 5) == []
    assert docs_index.search("zebra quantum", 5) == []  # no shared word: nothing, not noise
    assert docs_index.search("topics", 0) == []
    assert len(docs_index.search("ros2", 2)) == 2
    assert docs_index.get("nope#0") is None


def test_index_round_trip_and_validation(docs_index, tmp_path):
    path = tmp_path / "idx" / "sections.jsonl"
    save_sections(docs_index.sections, path)
    assert load_sections(path) == docs_index.sections
    assert not path.with_suffix(".jsonl.tmp").exists()
    with pytest.raises(ValueError):
        DocsIndex([])
    with pytest.raises(ValueError):
        DocsIndex([docs_index.sections[0], docs_index.sections[0]])


def test_snippet_picks_the_matching_paragraph():
    text = "First paragraph about nodes.\n\nSecond one about durability settings.\n\nThird."
    assert snippet(text, {"durability"}) == "Second one about durability settings."
    assert len(snippet("word " * 200, {"word"}, width=50)) == 50


def test_docs_eval_metrics(docs_index):
    qs = [
        Question("q1", "c", "durability for late-joining subscriptions",
                 ["Concepts/About-Topics"], ["transient local"]),
        Question("q2", "c", "how do I change a parameter", ["Concepts/About-Topics"], ["nope"]),
        Question("q3", "c", "unanswerable", [], []),
    ]
    res = evaluate(docs_index, qs)
    assert res["n_questions"] == 2 and res["found"] == 1 and res["missed"] == ["q2"]
    assert res["section"]["hit@1"] == 0.5 and res["mrr@10"] == 0.5
    assert res["answer_on_first_page"] == 1
    sec = docs_index.get("Concepts/About-Parameters#0")
    assert key_offset(sec, qs[0]) is None  # right words would not matter: wrong page


def make_zip(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, "Title\n=====\n\ntext\n")
    return buf.getvalue()


def test_zip_extraction_keeps_text_only(tmp_path):
    blob = make_zip(["repo-abc/source/A.rst", "repo-abc/source/img.png", "repo-abc/LICENSE"])
    n = extract_ros2_zip(blob, tmp_path / "docs", "abc")
    assert n == 2
    assert (tmp_path / "docs" / "source" / "A.rst").exists()
    assert not (tmp_path / "docs" / "source" / "img.png").exists()
    assert (tmp_path / "docs" / "COMMIT").read_text() == "abc\n"


@pytest.mark.parametrize("name", ["../evil.rst", "/abs/evil.rst", "a/../../evil.rst"])
def test_zip_path_traversal_is_refused(name):
    with zipfile.ZipFile(io.BytesIO(make_zip([name]))) as zf, pytest.raises(ValueError):
        safe_members(zf, (".rst",))
