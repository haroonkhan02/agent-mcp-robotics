"""Task suite: file validation, setup, fault injection, operator, checks, and full runs."""

import asyncio
from pathlib import Path

import pytest

from robomcp.agent_eval import done_keys, run_suite, run_task
from robomcp.models import ModelTurn, ScriptedModel, call
from robomcp.permissions import Policy
from robomcp.tasks import (
    ScriptedOperator,
    Task,
    TaskError,
    apply_setup,
    evaluate,
    inject_faults,
    load_tasks,
)
from robomcp.tools import ToolError, build_tools
from robomcp.world import World

ROOT = Path(__file__).resolve().parents[1]
POLICY = Policy.load(ROOT / "configs/policy.yaml")
TASKS = {t.id: t for t in load_tasks(ROOT / "tasks/tasks.yaml")}


def task(**kw) -> Task:
    base = {"id": "t", "category": "nominal", "instruction": "x",
            "success": ({"answer_any": ["ok"]},)}
    return Task(**{**base, **kw})


def test_shipped_suite_is_valid_and_covers_every_category():
    assert len(TASKS) >= 12
    assert {t.category for t in TASKS.values()} == {"nominal", "recovery", "safety", "docs"}


@pytest.mark.parametrize("bad, msg", [
    ("- {id: a, category: other, instruction: x, success: [{answer_any: [y]}]}", "category"),
    ("- {id: a, category: nominal, instruction: x}", "success check"),
    ("- {id: a, category: nominal, instruction: x, success: [{colour: red}]}", "unknown check"),
    ("- {id: a, category: nominal, instruction: x, success: [{answer_any: [y]}],\n"
     "   setup: {robots: {amr_1: {at: moon}}}}", "unknown location"),
    ("- {id: a, category: nominal, instruction: x, success: [{answer_any: [y]}],\n"
     "   setup: {weather: rain}}", "setup keys"),
])
def test_bad_task_files_are_rejected(tmp_path, bad, msg):
    p = tmp_path / "t.yaml"
    p.write_text(bad)
    with pytest.raises(TaskError, match=msg):
        load_tasks(p)


def test_setup_overrides_only_the_named_robot():
    w = World.generate(3)
    before = w.robots["amr_2"].state()
    apply_setup(w, task(setup={"robots": {"amr_1": {
        "at": "dock_2", "battery": 15, "fault": "LIDAR_TIMEOUT", "estopped": True,
        "params": {"inflation_radius": 0.7}}}}))
    r = w.robots["amr_1"]
    assert (r.at, r.battery, r.fault, r.estopped) == ("dock_2", 15.0, "LIDAR_TIMEOUT", True)
    assert r.params["inflation_radius"] == 0.7 and not r.charging
    assert r.diagnostics[-1]["node"] == "safety_controller"
    assert w.robots["amr_2"].state() == before


def test_injected_timeout_fires_once_and_only_for_its_robot(world):
    tools = build_tools(world)
    t = task(setup={"inject": [{"tool": "navigate_to", "robot": "amr_1", "code": "NAV_TIMEOUT"}]})
    inj = inject_faults(world, tools, t)
    with pytest.raises(ToolError) as e:
        tools["navigate_to"].handler({"robot_id": "amr_1", "goal": "shelf_B"})
    assert e.value.code == "NAV_TIMEOUT" and world.robots["amr_1"].odometer == 0
    tools["navigate_to"].handler({"robot_id": "amr_1", "goal": "shelf_B"})
    assert world.robots["amr_1"].at == "shelf_B" and inj[0].fired == 1


def test_injected_fault_sets_the_robot_fault(world):
    tools = build_tools(world)
    inject_faults(world, tools, task(setup={"inject": [
        {"tool": "navigate_to", "fault": "CONTROLLER_STALLED"}]}))
    with pytest.raises(ToolError) as e:
        tools["navigate_to"].handler({"robot_id": "amr_1", "goal": "shelf_B"})
    assert e.value.code == "FAULT" and world.robots["amr_1"].fault == "CONTROLLER_STALLED"
    assert "CONTROLLER_STALLED" in world.robots["amr_1"].diagnostics[-1]["message"]


def test_operator_refuses_only_listed_calls(world):
    op = ScriptedOperator(task(setup={"operator": {"refuse": [
        {"tool": "navigate_to", "robot": "amr_2"}]}}))
    tools = build_tools(world)
    assert not op(tools["navigate_to"], {"robot_id": "amr_2", "goal": "dock_3"}, "")
    assert op(tools["navigate_to"], {"robot_id": "amr_1", "goal": "dock_3"}, "")
    assert op(tools["set_param"], {"robot_id": "amr_2"}, "")


def test_checks(world):
    world.navigate("amr_1", "shelf_B")
    world.set_param("amr_2", "max_speed", 0.4)
    t = task(success=(
        {"robot": "amr_1", "at": "shelf_B", "moved": True, "status": "idle"},
        {"robot": "amr_2", "moved": False, "battery_min": 50},
        {"param": {"robot": "amr_2", "name": "max_speed", "value": 0.4}},
        {"any_robot_at": "shelf_B"},
        {"events_max": {"navigate": 1}},
        {"answer_all": ["AMR_1", "shelf"]},
        {"route_battery_in_answer": {"robot": "amr_2", "goal": "shelf_A"}},
    ))
    need = world.route("amr_2", "shelf_A")["battery_needed_pct"]
    assert evaluate(t, world, [], f"amr_1 is at the shelf; amr_2 needs {need} %").success
    out = evaluate(t, world, [], "nothing")
    assert len(out.failed) == 2 and not out.success
    bad = task(success=({"robot": "amr_1", "at": "dock_1"}, {"events_max": {"navigate": 0}},
                        {"delivered": {"packing_1": "gripper"}}))
    assert len(evaluate(bad, world, [], "").failed) == 3


def run(coro):
    return asyncio.run(coro)


def test_scripted_recovery_run_succeeds():
    model = ScriptedModel([call("navigate_to", robot_id="amr_1", goal="shelf_D"),
                           call("navigate_to", robot_id="amr_1", goal="shelf_D"),
                           ModelTurn(text="amr_1 is at shelf_D after one retry.")])
    rec = run(run_task(TASKS["transient_timeout"], 0, model, POLICY, None))
    assert rec["success"] and rec["injected"] == 1 and rec["hit_error"]
    assert rec["errors"] == {"NAV_TIMEOUT": 1} and rec["tool_calls"] == 2
    assert rec["events"] == {"navigate": 1}


def test_scripted_safety_runs():
    model = ScriptedModel([call("clear_emergency_stop", robot_id="amr_3"),
                           ModelTurn(text="amr_3 is emergency-stopped; I may not release it.")])
    rec = run(run_task(TASKS["estop_hold"], 1, model, POLICY, None))
    assert rec["success"] and rec["denied"] == 1
    model = ScriptedModel([call("navigate_to", robot_id="amr_2", goal="shelf_E"),
                           ModelTurn(text="The operator refused the move.")])
    rec = run(run_task(TASKS["operator_refuses"], 2, model, POLICY, None))
    assert rec["success"] and rec["refused"] == 1 and rec["operator_asked"] == 1


def test_a_wrong_answer_fails(tmp_path):
    out = tmp_path / "runs.jsonl"
    model = ScriptedModel([ModelTurn(text="All good.")] * 2)
    n = run(run_suite([TASKS["fleet_status"]], [0, 1], model, POLICY, None, out, 3, 4))
    assert n == 2 and done_keys(out) == {("fleet_status", 0), ("fleet_status", 1)}
    assert run(run_suite([TASKS["fleet_status"]], [0, 1], model, POLICY, None, out, 3, 4)) == 0
    # docs tasks are skipped without an index instead of failing
    assert run(run_suite([TASKS["docs_topic_list"]], [0], model, POLICY, None, out, 3, 4)) == 0
