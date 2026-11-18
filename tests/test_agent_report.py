"""Report from synthetic run records: rates, intervals, recovery and safety counts."""

import json

from robomcp.agent_report import main, summarise, to_markdown, wilson


def rec(task, cat, seed, ok, **kw):
    base = {"task": task, "category": cat, "seed": seed, "model": "m", "success": ok,
            "failed": [] if ok else ["x: want 1, got 2"], "stop": "answered", "turns": 3,
            "tool_calls": 2, "errors": {}, "hit_error": False, "parse_errors": 0,
            "denied": 0, "refused": 0, "injected": 0, "events": {}, "answer": "a",
            "input_tokens": 1000, "output_tokens": 50, "wall_s": 4.0,
            "steps": [{"tool": "list_robots", "args": {}, "code": None}]}
    return {**base, **kw}


RUNS = [
    rec("a", "nominal", 0, True), rec("a", "nominal", 1, False),
    rec("b", "recovery", 0, True, errors={"NAV_TIMEOUT": 1}, hit_error=True, injected=1),
    rec("c", "safety", 0, False, errors={"PERMISSION_DENIED": 2, "APPROVAL_REFUSED": 3},
        denied=2, refused=3, events={"estop": 1}, stop="repeat_limit"),
]


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(3, 3)
    assert 0.43 < lo < 0.44 and hi == 1.0


def test_summary_counts():
    s = summarise("m", RUNS)
    assert s["overall"]["k"] == 2 and s["overall"]["n"] == 4
    assert s["by_category"]["nominal"]["rate"] == 0.5
    assert s["recovery"] == {"runs_with_tool_error": 1, "succeeded_after_error": 1,
                             "injected_fault_runs": 1, "injected_fault_success": 1}
    assert s["safety"] == {"denied_calls": 2, "runs_with_denied_call": 1,
                           "refusal_retries": 2, "runs_with_estop_unasked": 1}
    assert s["stops"] == {"answered": 3, "repeat_limit": 1}
    assert s["error_codes"]["APPROVAL_REFUSED"] == 3
    assert len(s["failures"]) == 2


def test_markdown_and_cli(tmp_path, monkeypatch):
    md = to_markdown([summarise("m", RUNS)])
    assert "| all tasks | 2/4 (50 %)" in md and "Failed runs: m" in md
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/agent_runs").mkdir(parents=True)
    (tmp_path / "reports/agent_runs/m.jsonl").write_text(
        "\n".join(json.dumps(r) for r in RUNS))
    main(["m"])
    data = json.loads((tmp_path / "reports/agent_tasks.json").read_text())
    assert data[0]["overall"]["n"] == 4
