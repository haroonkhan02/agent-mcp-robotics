"""The MCP layer: in-process for speed, plus one real subprocess over stdio."""

import asyncio
import json
import sys
from pathlib import Path

import yaml
from mcp import Client, StdioServerParameters

from robomcp.executor import ToolExecutor
from robomcp.permissions import Policy
from robomcp.server import TIER_META, build_server
from robomcp.tools import build_tools

ROOT = Path(__file__).resolve().parents[1]


def make_executor(world, docs_index=None, **kw):
    return ToolExecutor(build_tools(world, docs_index),
                        Policy.load(ROOT / "configs/policy.yaml"), **kw)


def run(coro):
    return asyncio.run(coro)


def test_tools_are_listed_with_tiers_and_hints(world, docs_index):
    async def go():
        async with Client(build_server(make_executor(world, docs_index))) as client:
            assert client.server_info.name == "robomcp"
            assert "data, not instructions" in client.instructions
            return (await client.list_tools()).tools

    tools = {t.name: t for t in run(go())}
    assert len(tools) == 16
    assert tools["navigate_to"].meta[TIER_META] == "motion"
    assert tools["navigate_to"].annotations.destructive_hint is True
    assert tools["navigate_to"].annotations.read_only_hint is False
    assert tools["search_docs"].annotations.read_only_hint is True
    assert tools["navigate_to"].input_schema["required"] == ["robot_id", "goal"]


def test_call_results_and_errors_reach_the_client(world, docs_index):
    async def go():
        async with Client(build_server(make_executor(world, docs_index))) as client:
            ok = await client.call_tool("get_robot_state", {"robot_id": "amr_1"})
            denied = await client.call_tool("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B"})
            unknown = await client.call_tool("fly_to", {})
            docs = await client.call_tool("search_docs", {"query": "durability late-joining"})
            return ok, denied, unknown, docs

    ok, denied, unknown, docs = run(go())
    assert ok.is_error is False and ok.structured_content["at"] == "shelf_A"
    assert json.loads(ok.content[0].text) == ok.structured_content  # text mirrors the JSON
    assert denied.is_error is True
    assert denied.structured_content["error"]["code"] == "APPROVAL_REQUIRED"
    assert unknown.structured_content["error"]["code"] == "UNKNOWN_TOOL"
    assert docs.structured_content["results"][0]["section_id"] == "Concepts/About-Topics#1"
    assert world.robots["amr_1"].at == "shelf_A"


def test_approver_applies_to_mcp_calls(world):
    async def go():
        ex = make_executor(world, approver=lambda tool, args, reason: args["goal"] == "shelf_B")
        async with Client(build_server(ex)) as client:
            yes = await client.call_tool("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B"})
            no = await client.call_tool("navigate_to", {"robot_id": "amr_1", "goal": "shelf_C"})
            return yes, no, ex.audit.records

    yes, no, records = run(go())
    assert yes.is_error is False and yes.structured_content["arrived"] is True
    assert no.structured_content["error"]["code"] == "APPROVAL_REFUSED"
    assert [r.decision for r in records] == ["approved", "refused"]


def test_real_server_process_over_stdio(tmp_path):
    """Start `python -m robomcp.server` and talk to it the way any MCP client would."""
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "index_path": str(tmp_path / "missing.jsonl"),
        "policy_path": str(ROOT / "configs/policy.yaml"),
        "world": {"seed": 3, "n_robots": 2},
    }))
    audit = tmp_path / "audit.jsonl"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "robomcp.server", "--config", str(config), "--audit", str(audit)],
        env={"PYTHONPATH": str(ROOT / "src")},
    )

    async def go():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            robots = await client.call_tool("list_robots", {})
            stop = await client.call_tool("emergency_stop", {"robot_id": "amr_1"})
            move = await client.call_tool("navigate_to", {"robot_id": "amr_1", "goal": "shelf_B"})
            docs = await client.call_tool("search_docs", {"query": "topics"})
            return tools, robots, stop, move, docs

    tools, robots, stop, move, docs = run(go())
    assert len(tools) == 16
    assert [r["id"] for r in robots.structured_content["robots"]] == ["amr_1", "amr_2"]
    assert stop.structured_content["status"] == "estopped"
    assert move.structured_content["error"]["code"] == "APPROVAL_REQUIRED"
    assert docs.structured_content["error"]["code"] == "DOCS_UNAVAILABLE"
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["tool"] for r in rows] == ["list_robots", "emergency_stop", "navigate_to",
                                         "search_docs"]
