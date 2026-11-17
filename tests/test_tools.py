from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robomcp.executor import AuditLog, ToolExecutor
from robomcp.permissions import Policy, PolicyError
from robomcp.tools import TIERS, build_tools

ROOT = Path(__file__).resolve().parents[1]
ALLOW_ALL = Policy({t: "allow" for t in TIERS})


def shipped_policy():
    return Policy.load(ROOT / "configs/policy.yaml")


def executor(world, docs_index=None, policy=ALLOW_ALL, approver=None):
    return ToolExecutor(build_tools(world, docs_index), policy, approver)


def test_every_tool_has_a_valid_closed_schema(world, docs_index):
    tools = build_tools(world, docs_index)
    assert len(tools) == 16
    for t in tools.values():
        Draft202012Validator.check_schema(t.input_schema)
        assert t.input_schema["additionalProperties"] is False
        assert t.tier in TIERS and len(t.description) > 20
    tiers = {t.name: t.tier for t in tools.values()}
    assert tiers["navigate_to"] == "motion" and tiers["emergency_stop"] == "stop"
    assert tiers["clear_emergency_stop"] == "release" and tiers["search_docs"] == "read"


def test_read_tools(world):
    ex = executor(world)
    robots = ex.call("list_robots").data["robots"]
    assert [r["status"] for r in robots] == ["idle", "charging", "fault"]
    state = ex.call("get_robot_state", {"robot_id": "amr_3"}).data
    assert state["fault"] == "LIDAR_TIMEOUT" and state["params"]["max_speed"] == 1.0
    locs = ex.call("list_locations").data
    shelf_a = next(x for x in locs["locations"] if x["name"] == "shelf_A")
    assert shelf_a["items"] == ["gripper", "wheel_kit"]
    assert {z["name"] for z in locs["zones"]} >= {"maintenance", "packing_area"}
    diag = ex.call("read_diagnostics", {"robot_id": "amr_3", "limit": 1}).data["entries"]
    assert diag[0]["level"] == "ERROR"
    plan = ex.call("plan_route", {"robot_id": "amr_1", "goal": "packing_1"}).data
    assert plan["feasible"] and "_path" not in plan
    params = ex.call("get_params", {"robot_id": "amr_1"}).data["params"]
    assert {p["name"] for p in params} == {"max_speed", "inflation_radius", "goal_tolerance"}


def test_a_delivery_end_to_end(world):
    ex = executor(world)
    for name, args in [
        ("pick_item", {"robot_id": "amr_1", "item": "gripper"}),
        ("navigate_to", {"robot_id": "amr_1", "goal": "packing_1"}),
        ("drop_item", {"robot_id": "amr_1"}),
    ]:
        assert ex.call(name, args).ok
    assert world.delivered["packing_1"] == ["gripper"]
    assert [r.tool for r in ex.audit.records] == ["pick_item", "navigate_to", "drop_item"]


def test_error_codes(world):
    ex = executor(world)
    cases = [
        ("fly_to", {}, "UNKNOWN_TOOL"),
        ("navigate_to", {"robot_id": "amr_1"}, "INVALID_ARGUMENTS"),
        ("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B", "fast": True},
         "INVALID_ARGUMENTS"),
        ("set_param", {"robot_id": "amr_1", "name": "max_speed", "value": "fast"},
         "INVALID_ARGUMENTS"),
        ("list_robots", [1, 2], "INVALID_ARGUMENTS"),
        ("navigate_to", {"robot_id": "amr_3", "goal": "shelf_B"}, "FAULT"),
        ("navigate_to", {"robot_id": "amr_1", "goal": "maintenance_bay"}, "GOAL_IN_KEEP_OUT"),
        ("set_param", {"robot_id": "amr_1", "name": "max_speed", "value": 5}, "PARAM_OUT_OF_RANGE"),
        ("search_docs", {"query": "topics"}, "DOCS_UNAVAILABLE"),
    ]
    before = world.snapshot()
    for name, args, code in cases:
        res = ex.call(name, args)
        assert not res.ok and res.error["code"] == code, (name, res.error)
        assert res.payload() == {"error": res.error}
    assert world.snapshot() == before
    assert "navigate_to" in ex.call("fly_to").error["message"]  # the error lists real tools


def test_handler_crash_is_contained(world):
    tools = build_tools(world)
    spec = tools["list_robots"]
    tools["list_robots"] = type(spec)(spec.name, spec.tier, spec.description, spec.input_schema,
                                      lambda a: 1 / 0)
    res = ToolExecutor(tools, ALLOW_ALL).call("list_robots")
    assert res.error == {"code": "INTERNAL_ERROR",
                         "message": "list_robots failed (ZeroDivisionError)"}


def test_docs_tools(world, docs_index):
    ex = executor(world, docs_index)
    out = ex.call("search_docs", {"query": "durability late-joining subscriptions", "k": 2}).data
    top = out["results"][0]
    assert top["section_id"] == "Concepts/About-Topics#1" and "Transient local" in top["snippet"]
    sec = ex.call("read_doc_section", {"section_id": top["section_id"]}).data
    assert sec["next_offset"] is None and sec["total_chars"] == len(sec["text"])
    assert ex.call("read_doc_section", {"section_id": "x#9"}).error["code"] == "UNKNOWN_SECTION"
    assert ex.call("read_doc_section", {"section_id": top["section_id"], "offset": 10**6}
                   ).error["code"] == "OFFSET_OUT_OF_RANGE"
    assert ex.call("search_docs", {"query": "x", "k": 99}).error["code"] == "INVALID_ARGUMENTS"


def test_long_sections_are_paged(world, docs_index):
    from robomcp.docs import PAGE_CHARS, DocSection, DocsIndex

    long = DocSection("Long#0", "Long", "Long", ("Long",), "u", "abcdefghij" * 450)
    ex = executor(world, DocsIndex([*docs_index.sections, long]))
    first = ex.call("read_doc_section", {"section_id": "Long#0"}).data
    assert len(first["text"]) == PAGE_CHARS and first["next_offset"] == PAGE_CHARS
    pages, offset = [first["text"]], first["next_offset"]
    while offset is not None:
        page = ex.call("read_doc_section", {"section_id": "Long#0", "offset": offset}).data
        pages.append(page["text"])
        offset = page["next_offset"]
    assert "".join(pages) == long.text


# ---------------------------------------------------------------- permissions

def test_shipped_policy_tiers(world):
    ex = executor(world, policy=shipped_policy())
    assert ex.call("list_robots").decision == "allowed"
    assert ex.call("emergency_stop", {"robot_id": "amr_1"}).decision == "allowed"
    res = ex.call("navigate_to", {"robot_id": "amr_2", "goal": "shelf_B"})
    assert res.error["code"] == "APPROVAL_REQUIRED" and res.decision == "unapproved"
    res = ex.call("clear_emergency_stop", {"robot_id": "amr_1"})
    assert res.error["code"] == "PERMISSION_DENIED" and world.robots["amr_1"].estopped


def test_approver_is_asked_only_for_ask_tiers(world):
    asked = []

    def approver(tool, args, reason):
        asked.append(tool.name)
        return tool.name != "dock"

    ex = executor(world, policy=shipped_policy(), approver=approver)
    ex.call("list_robots")
    assert ex.call("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B"}).decision == "approved"
    refused = ex.call("dock", {"robot_id": "amr_1", "dock": "dock_2"})
    assert refused.error["code"] == "APPROVAL_REFUSED" and world.robots["amr_1"].at == "shelf_B"
    # denied tiers and constraint violations never reach the approver
    ex.call("clear_emergency_stop", {"robot_id": "amr_1"})
    res = ex.call("set_param", {"robot_id": "amr_1", "name": "inflation_radius", "value": 0.3})
    assert res.error["code"] == "PERMISSION_DENIED" and "inflation_radius" in res.error["message"]
    assert asked == ["navigate_to", "dock"]
    # invalid arguments are rejected before anyone is asked
    ex.call("navigate_to", {"robot_id": "amr_1"})
    assert asked == ["navigate_to", "dock"]


def test_on_ask_allow_runs_unattended(world):
    policy = Policy({**shipped_policy().tiers}, on_ask="allow")
    res = executor(world, policy=policy).call("navigate_to", {"robot_id": "amr_1",
                                                              "goal": "shelf_B"})
    assert res.ok and res.decision == "unattended"


def test_numeric_constraints(world):
    policy = Policy({t: "allow" for t in TIERS},
                    constraints={"set_param": {"value": {"min": 0.3, "max": 0.8}}})
    ex = executor(world, policy=policy)
    args = {"robot_id": "amr_1", "name": "max_speed"}
    assert ex.call("set_param", {**args, "value": 0.5}).ok
    assert "maximum" in ex.call("set_param", {**args, "value": 1.2}).error["message"]
    assert "minimum" in ex.call("set_param", {**args, "value": 0.25}).error["message"]


@pytest.mark.parametrize("raw", [
    None,
    {"tiers": {"read": "allow"}},
    {"tiers": {t: "allow" for t in TIERS}, "extra": 1},
    {"tiers": {**{t: "allow" for t in TIERS}, "read": "maybe"}},
    {"tiers": {t: "allow" for t in TIERS}, "on_ask": "ask"},
    {"tiers": {t: "allow" for t in TIERS}, "constraints": {"dock": {"dock": {"regex": "x"}}}},
    {"tiers": {t: "allow" for t in TIERS}, "constraints": {"dock": {"dock": {"enum": "d"}}}},
])
def test_bad_policies_are_rejected(raw):
    with pytest.raises(PolicyError):
        Policy.from_dict(raw)


def test_constraint_on_unknown_tool_is_rejected(world):
    policy = Policy({t: "allow" for t in TIERS}, constraints={"set_parm": {"name": {"enum": []}}})
    with pytest.raises(PolicyError):
        executor(world, policy=policy)


def test_audit_log_records_everything(world, tmp_path):
    import json

    audit = AuditLog(tmp_path / "audit" / "calls.jsonl")
    ex = ToolExecutor(build_tools(world), shipped_policy(), audit=audit)
    ex.call("list_robots")
    ex.call("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B"})
    ex.call("nope")
    rows = [json.loads(x) for x in audit.path.read_text().splitlines()]
    assert [(r["seq"], r["tool"], r["tier"], r["decision"], r["error_code"]) for r in rows] == [
        (1, "list_robots", "read", "allowed", None),
        (2, "navigate_to", "motion", "unapproved", "APPROVAL_REQUIRED"),
        (3, "nope", "?", "n/a", "UNKNOWN_TOOL"),
    ]
    assert rows[1]["arguments"] == {"robot_id": "amr_1", "goal": "shelf_B"}
