from robomcp.injection_report import summarise, to_markdown


def rec(task, att, succ, ok, **kw):
    return {"task": task, "seed": 0, "model": "mlx/m", "attack_attempted": att,
            "attack_succeeded": succ, "success": ok, "denied": 0, "scope_denied": 0,
            "defence": "none", "scope": False, "prompt": "autonomous",
            "steps": [{"tool": "restart_node", "args": {"robot_id": "amr_2"}, "code": None}],
            **kw}


def test_summary_counts_and_markdown():
    runs = [rec("a", True, True, False), rec("a", True, False, True, scope_denied=1),
            rec("b", False, False, True)]
    s = summarise("x", runs)
    assert (s["attempted"]["k"], s["succeeded"]["k"], s["task_done"]["k"]) == (2, 1, 2)
    assert s["by_task"]["a"] == {"attack_attempted": 2, "attack_succeeded": 1, "success": 1,
                                 "n": 2}
    assert s["scope_denied_calls"] == 1
    md = to_markdown([s], [])
    assert "| x | m |" in md and "restart_node(amr_2)" in md
