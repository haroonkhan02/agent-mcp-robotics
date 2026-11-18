"""The agent loop over a real in-process MCP server, driven by scripted models."""

import asyncio
from pathlib import Path

from mcp import Client

from robomcp.agent import Agent
from robomcp.executor import ToolExecutor
from robomcp.models import ModelTurn, ScriptedModel, ToolCall, call
from robomcp.permissions import Policy
from robomcp.server import build_server
from robomcp.tools import build_tools

ROOT = Path(__file__).resolve().parents[1]


def run_agent(world, model, approver=lambda *a: True, **kw):
    ex = ToolExecutor(build_tools(world), Policy.load(ROOT / "configs/policy.yaml"),
                      approver=approver)

    async def go():
        async with Client(build_server(ex)) as client:
            return await Agent(model, **kw).run(client, "do it")

    return asyncio.run(go()), ex


def test_delivery_through_mcp(world):
    model = ScriptedModel([
        call("pick_item", robot_id="amr_1", item="gripper"),
        call("navigate_to", robot_id="amr_1", goal="packing_1"),
        call("drop_item", robot_id="amr_1"),
        ModelTurn(text="Delivered."),
    ])
    run, ex = run_agent(world, model)
    assert run.stop == "answered" and run.answer == "Delivered."
    assert [s.tool for s in run.steps] == ["pick_item", "navigate_to", "drop_item"]
    assert all(s.ok for s in run.steps) and run.turns == 4
    assert world.delivered["packing_1"] == ["gripper"]
    assert [r.decision for r in ex.audit.records] == ["approved"] * 3


def test_errors_reach_the_model_as_tool_results(world):
    seen = {}

    def inspect(messages):
        seen["last"] = messages[-1]
        return ModelTurn(text="amr_3 has a fault.")

    run, _ = run_agent(world, ScriptedModel([call("navigate_to", robot_id="amr_3",
                                                  goal="shelf_A"), inspect]))
    assert run.steps[0].error_code == "FAULT" and not run.steps[0].ok
    assert seen["last"]["role"] == "tool" and seen["last"]["is_error"]
    assert '"code": "FAULT"' in seen["last"]["content"]
    assert run.errors == {"FAULT": 1}


def test_repeat_limit_stops_a_stuck_model(world):
    stuck = [call("clear_emergency_stop", robot_id="amr_1")] * 5
    run, _ = run_agent(world, ScriptedModel(stuck), max_same_failure=3)
    assert run.stop == "repeat_limit" and len(run.steps) == 3
    assert run.errors == {"PERMISSION_DENIED": 3}


def test_refused_approval_is_reported(world):
    model = ScriptedModel([call("navigate_to", robot_id="amr_1", goal="shelf_B"),
                           ModelTurn(text="The operator refused the move.")])
    run, _ = run_agent(world, model, approver=lambda *a: False)
    assert run.steps[0].error_code == "APPROVAL_REFUSED"
    assert world.robots["amr_1"].odometer == 0


def test_unreadable_call_gets_a_reminder_and_bad_args_are_not_sent(world):
    seen = {}

    def after_reminder(messages):
        seen["reminder"] = messages[-1]
        return ModelTurn(text="", tool_calls=[ToolCall("x", "list_robots", "not a dict")])

    model = ScriptedModel([ModelTurn(text="<tool_call>{", parse_error="bad JSON"),
                           after_reminder, ModelTurn(text="ok")])
    run, ex = run_agent(world, model)
    assert run.parse_errors == 1 and "could not be read" in seen["reminder"]["content"]
    assert run.steps[0].error_code == "INVALID_ARGUMENTS"
    assert ex.audit.records == []  # never reached the server


def test_turn_budget_and_refusal(world):
    run, _ = run_agent(world, ScriptedModel([call("list_robots")] * 10), max_turns=4)
    assert run.stop == "max_steps" and run.turns == 4 and run.answer == ""
    run, _ = run_agent(world, ScriptedModel([ModelTurn(text="", stop_reason="refusal")]))
    assert run.stop == "refusal"
    d = run.to_dict()
    assert d["stop"] == "refusal" and d["errors"] == {}


def test_long_results_are_truncated_for_the_model(world):
    seen = {}

    def look(messages):
        seen["content"] = messages[-1]["content"]
        return ModelTurn(text="done")

    run_agent(world, ScriptedModel([call("list_locations"), look]), max_result_chars=100)
    assert seen["content"].endswith("[truncated]") and len(seen["content"]) < 130
