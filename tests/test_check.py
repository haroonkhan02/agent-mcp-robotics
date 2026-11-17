import asyncio
from pathlib import Path

from mcp import Client

from robomcp.check import (
    delivery_cases,
    deny_cases,
    docs_over_mcp,
    pct,
    pick_delivery,
    run_cases,
    to_markdown,
)
from robomcp.docs_eval import Question
from robomcp.executor import ToolExecutor
from robomcp.permissions import Policy
from robomcp.server import build_server
from robomcp.tools import build_tools

ROOT = Path(__file__).resolve().parents[1]


def executor(world, docs_index=None, on_ask="deny"):
    policy = Policy.load(ROOT / "configs/policy.yaml")
    return ToolExecutor(build_tools(world, docs_index),
                        Policy(policy.tiers, on_ask, policy.constraints))


def test_scripted_cases_pass_against_the_fixture_world(world):
    async def go():
        async with Client(build_server(executor(world))) as client:
            robots = [r.state() for r in world.robots.values()]
            return await run_cases(client, deny_cases(robots))

    rows = asyncio.run(go())
    assert [r for r in rows if not r["pass"]] == []
    assert {r["got"] for r in rows} >= {"ok", "APPROVAL_REQUIRED", "PERMISSION_DENIED",
                                        "INVALID_ARGUMENTS", "UNKNOWN_TOOL"}


def test_delivery_runs_when_approval_is_waived(world):
    async def go():
        async with Client(build_server(executor(world, on_ask="allow"))) as client:
            robot, shelf, item = await pick_delivery(client)
            return robot, shelf, item, await run_cases(client, delivery_cases(robot, shelf, item))

    robot, shelf, item, rows = asyncio.run(go())
    assert (robot, shelf, item) == ("amr_1", "", "gripper")
    assert all(r["pass"] for r in rows), rows
    assert world.delivered["packing_1"] == ["gripper"]


def test_a_wrong_outcome_is_reported_not_hidden(world):
    async def go():
        async with Client(build_server(executor(world, on_ask="allow"))) as client:
            robots = [r.state() for r in world.robots.values()]
            return await run_cases(client, deny_cases(robots))

    rows = asyncio.run(go())  # motion now runs, so the APPROVAL_REQUIRED expectations fail
    wrong = [r for r in rows if not r["pass"]]
    assert {r["tool"] for r in wrong} == {"navigate_to", "set_param"}
    res = {"protocol": "x", "seed": 0, "tools": [], "deny": rows, "allow": [],
           "delivery": {"robot": "r", "item": "i", "delivered_at_packing_1": [], "pass": False},
           "audit": {"records": 0, "calls_made": 0, "decisions": []},
           "latency_ms": {}, "startup_ms": 0}
    assert "(wrong)" in to_markdown(res, None)


def test_docs_questions_through_the_tool(world, docs_index):
    qs = [Question("q1", "c", "durability late-joining subscriptions",
                   ["Concepts/About-Topics"], ["transient local"]),
          Question("q2", "c", "change a parameter at run time", ["Concepts/Nope"], ["x"]),
          Question("q3", "c", "no gold page", [], [])]

    async def go():
        async with Client(build_server(executor(world, docs_index))) as client:
            return await docs_over_mcp(client, qs)

    res = asyncio.run(go())
    assert res["n"] == 2 and len(res["ms"]) == 2
    assert res["page"]["hit@1"] == 0.5 and res["page"]["hit@10"] == 0.5


def test_percentile():
    assert pct([5.0, 1.0, 3.0], 0.5) == 3.0
    assert pct(list(range(1, 101)), 0.95) == 95
    assert pct([7.0], 0.95) == 7.0
