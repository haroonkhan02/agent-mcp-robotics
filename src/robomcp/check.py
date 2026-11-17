"""`robomcp-check`: start the real server as a subprocess and exercise it over MCP (stdio).

Three parts, written to reports/server_check.md and .json:

1. The tool list a client sees, with tiers.
2. Scripted calls with a known right answer each: reads, calls the policy must stop,
   malformed calls, and (on a second server started with --on-ask allow) a delivery that
   has to end with the item at the packing station.
3. The documentation questions sent through `search_docs`, which must score the same as
   the in-process evaluation, and round-trip latency per call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from robomcp.config import load_settings
from robomcp.docs_eval import KS, load_questions
from robomcp.server import TIER_META

OK = "ok"


@dataclass(frozen=True)
class Case:
    tool: str
    args: dict[str, Any]
    expect: str  # "ok" or an error code
    why: str


def deny_cases(robots: list[dict]) -> list[Case]:
    """Server with the shipped policy: approval needed, nobody attached to give it."""
    a, b = robots[0]["id"], robots[-1]["id"]
    return [
        Case("list_robots", {}, OK, "read tier"),
        Case("get_robot_state", {"robot_id": a}, OK, "read tier"),
        Case("read_diagnostics", {"robot_id": a, "limit": 5}, OK, "read tier"),
        Case("plan_route", {"robot_id": a, "goal": "packing_1"}, OK, "planning moves nothing"),
        Case("plan_route", {"robot_id": a, "goal": "maintenance_bay"}, "GOAL_IN_KEEP_OUT",
             "the world refuses, whatever the policy says"),
        Case("emergency_stop", {"robot_id": b}, OK, "stopping is always allowed"),
        Case("clear_emergency_stop", {"robot_id": b}, "PERMISSION_DENIED",
             "release tier is denied"),
        Case("navigate_to", {"robot_id": a, "goal": "shelf_B"}, "APPROVAL_REQUIRED",
             "motion needs approval, no operator attached"),
        Case("set_param", {"robot_id": a, "name": "max_speed", "value": 0.5},
             "APPROVAL_REQUIRED", "config needs approval"),
        Case("set_param", {"robot_id": a, "name": "inflation_radius", "value": 0.3},
             "PERMISSION_DENIED", "parameter outside the policy's allow-list"),
        Case("navigate_to", {"robot_id": a}, "INVALID_ARGUMENTS", "missing argument"),
        Case("navigate_to", {"robot_id": a, "goal": "shelf_B", "speed": 3},
             "INVALID_ARGUMENTS", "unknown argument"),
        Case("fly_to", {"robot_id": a}, "UNKNOWN_TOOL", "tool does not exist"),
    ]


def delivery_cases(robot: str, shelf: str, item: str) -> list[Case]:
    """Server started with --on-ask allow: approval-tier calls run unattended."""
    cases = []
    if shelf:
        cases.append(Case("navigate_to", {"robot_id": robot, "goal": shelf}, OK, "go to the shelf"))
    return cases + [
        Case("pick_item", {"robot_id": robot, "item": item}, OK, "load"),
        Case("navigate_to", {"robot_id": robot, "goal": "packing_1"}, OK, "through the doorway"),
        Case("drop_item", {"robot_id": robot}, OK, "unload"),
        Case("drop_item", {"robot_id": robot}, "NO_PAYLOAD", "nothing left to drop"),
        Case("set_param", {"robot_id": robot, "name": "inflation_radius", "value": 0.7},
             "PERMISSION_DENIED", "hard limit holds even when approval is waived"),
        Case("clear_emergency_stop", {"robot_id": robot}, "PERMISSION_DENIED",
             "denied tier holds even when approval is waived"),
    ]


async def call(client: Client, tool: str, args: dict[str, Any]) -> tuple[dict, bool, float]:
    start = time.perf_counter()
    res = await client.call_tool(tool, args)
    ms = (time.perf_counter() - start) * 1000
    return res.structured_content, bool(res.is_error), ms


async def run_cases(client: Client, cases: list[Case]) -> list[dict]:
    rows = []
    for c in cases:
        body, is_error, ms = await call(client, c.tool, c.args)
        got = body["error"]["code"] if is_error else OK
        rows.append({"tool": c.tool, "args": c.args, "expect": c.expect, "got": got,
                     "pass": got == c.expect, "why": c.why, "ms": round(ms, 2)})
    return rows


async def pick_delivery(client: Client) -> tuple[str, str, str]:
    """First robot able to move, and an item to fetch: (robot, shelf to visit or '', item)."""
    robots = (await call(client, "list_robots", {}))[0]["robots"]
    locs = (await call(client, "list_locations", {}))[0]["locations"]
    stocked = {loc["name"]: loc["items"] for loc in locs if loc.get("items")}
    robot = next(r for r in robots if r["status"] in ("idle", "charging"))
    if robot["at"] in stocked:
        return robot["id"], "", stocked[robot["at"]][0]
    shelf = sorted(stocked)[0]
    return robot["id"], shelf, stocked[shelf][0]


async def docs_over_mcp(client: Client, questions: list) -> dict:
    """Send every answerable question through search_docs; score by page and time each call."""
    answerable = [q for q in questions if q.gold_docs]
    ranks, times = [], []
    for q in answerable:
        body, is_error, ms = await call(client, "search_docs", {"query": q.question, "k": 10})
        if is_error:
            raise RuntimeError(f"search_docs failed: {body}")
        times.append(ms)
        pages = [r["section_id"].rsplit("#", 1)[0] for r in body["results"]]
        ranks.append(next((i for i, p in enumerate(pages, 1) if p in q.gold_docs), None))
    n = len(answerable)
    return {"n": n, "ms": times,
            "page": {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}}


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


async def timed(client: Client, tool: str, args: dict, n: int) -> list[float]:
    return [(await call(client, tool, args))[2] for _ in range(n)]


def server_params(config: str | None, extra: list[str]) -> StdioServerParameters:
    args = ["-m", "robomcp.server", *(["--config", config] if config else []), *extra]
    return StdioServerParameters(command=sys.executable, args=args)


async def run(config: str | None, repeats: int) -> dict:
    cfg = load_settings(config)
    out: dict[str, Any] = {"seed": cfg.world.seed}
    start = time.perf_counter()
    async with Client(server_params(config, [])) as client:
        out["startup_ms"] = round((time.perf_counter() - start) * 1000, 1)
        out["protocol"] = client.protocol_version
        tools = (await client.list_tools()).tools
        out["tools"] = [{"name": t.name, "tier": (t.meta or {}).get(TIER_META),
                         "read_only": bool(t.annotations and t.annotations.read_only_hint)}
                        for t in tools]
        robots = (await call(client, "list_robots", {}))[0]["robots"]
        out["deny"] = await run_cases(client, deny_cases(robots))
        first = robots[0]["id"]
        lat = {
            "list_robots": await timed(client, "list_robots", {}, repeats),
            "plan_route": await timed(client, "plan_route",
                                      {"robot_id": first, "goal": "inspection"}, repeats),
        }
        q_path = cfg.data_dir / "eval" / "questions.jsonl"
        if cfg.index_path.exists() and q_path.exists():
            docs = await docs_over_mcp(client, load_questions(q_path))
            lat["search_docs"] = docs.pop("ms")
            out["docs"] = docs
        out["latency_ms"] = {
            k: {"n": len(v), "p50": round(statistics.median(v), 2), "p95": round(pct(v, 0.95), 2)}
            for k, v in lat.items()}

    with tempfile.TemporaryDirectory() as tmp:
        audit = Path(tmp) / "audit.jsonl"
        extra = ["--on-ask", "allow", "--audit", str(audit)]
        async with Client(server_params(config, extra)) as client:
            robot, shelf, item = await pick_delivery(client)
            rows = await run_cases(client, delivery_cases(robot, shelf, item))
            locs = (await call(client, "list_locations", {}))[0]["locations"]
            delivered = next(x for x in locs if x["name"] == "packing_1")["delivered"]
        records = [json.loads(line) for line in audit.read_text().splitlines()]
    out["allow"] = rows
    out["delivery"] = {"robot": robot, "item": item, "delivered_at_packing_1": delivered,
                       "pass": delivered == [item]}
    out["audit"] = {"records": len(records), "calls_made": len(rows) + 3,
                    "decisions": sorted({r["decision"] for r in records})}
    out["all_pass"] = (all(r["pass"] for r in out["deny"] + out["allow"])
                       and out["delivery"]["pass"]
                       and out["audit"]["records"] == out["audit"]["calls_made"])
    return out


def to_markdown(res: dict, reference: dict | None) -> str:
    def table(rows: list[dict]) -> list[str]:
        lines = ["| Call | Expected | Got | Why |", "|---|---|---|---|"]
        for r in rows:
            args = ", ".join(f"{k}={v}" for k, v in r["args"].items() if k != "robot_id")
            mark = r["got"] if r["pass"] else f"**{r['got']} (wrong)**"
            lines.append(f"| `{r['tool']}({args})` | {r['expect']} | {mark} | {r['why']} |")
        return lines

    n_cases = len(res["deny"]) + len(res["allow"])
    n_pass = sum(r["pass"] for r in res["deny"] + res["allow"])
    by_tier: dict[str, list[str]] = {}
    for t in res["tools"]:
        by_tier.setdefault(t["tier"], []).append(t["name"])
    lines = [
        "# Server check over MCP (stdio)",
        "",
        f"The server runs as a subprocess; every call below is a real MCP round trip "
        f"(protocol {res['protocol']}, world seed {res['seed']}). "
        f"{n_pass}/{n_cases} scripted calls returned the expected outcome.",
        "",
        f"## Tools ({len(res['tools'])})",
        "",
        "| Tier | Tools |", "|---|---|",
        *[f"| {tier} | {', '.join(names)} |" for tier, names in by_tier.items()],
        "",
        "## Shipped policy, no operator attached",
        "",
        *table(res["deny"]),
        "",
        "## Same policy with `--on-ask allow`",
        "",
        *table(res["allow"]),
        "",
        f"Delivery: {res['delivery']['robot']} carried `{res['delivery']['item']}`; "
        f"packing_1 now holds {res['delivery']['delivered_at_packing_1']}. "
        f"Audit log: {res['audit']['records']} records for {res['audit']['calls_made']} calls, "
        f"decisions seen: {', '.join(res['audit']['decisions'])}.",
        "",
        "## Latency per call",
        "",
        "| Tool | Calls | p50 ms | p95 ms |", "|---|---|---|---|",
        *[f"| {k} | {v['n']} | {v['p50']} | {v['p95']} |" for k, v in res["latency_ms"].items()],
        "",
        f"Server start to first response: {res['startup_ms']} ms.",
        "",
    ]
    if "docs" in res:
        same = reference is not None and reference["page"] == res["docs"]["page"]
        lines += [
            "## Docs search through the tool",
            "",
            f"{res['docs']['n']} questions through `search_docs` (k = 10), scored by gold page: "
            + ", ".join(f"Hit@{k} {res['docs']['page'][f'hit@{k}']:.3f}" for k in KS) + ". "
            + ("Identical to the in-process evaluation in docs_search.md."
               if same else "No in-process reference to compare with."
               if reference is None else "**Differs from the in-process evaluation.**"),
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    res = asyncio.run(run(args.config, args.repeats))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / "docs_search.json"
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else None
    if "docs" in res and reference is not None:
        res["docs"]["matches_in_process"] = reference["page"] == res["docs"]["page"]
        res["all_pass"] = res["all_pass"] and res["docs"]["matches_in_process"]
    (out / "server_check.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "server_check.md").write_text(to_markdown(res, reference))
    print(to_markdown(res, reference))
    if not res["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
