"""`robomcp-agent-eval`: run the agent on the task suite over MCP, several seeds per task.

Each run gets a fresh world (seed + task setup), a fresh MCP server in the same process
and a scripted operator. The agent only talks to the server through an MCP client, as
it would to a remote one. One JSON line per (task, seed) goes to
reports/agent_runs/<name>.jsonl; an existing line is skipped, so a long run can be
resumed. `robomcp-agent-report` turns the files into the summary.

    robomcp-agent-eval --model mlx:mlx-community/Qwen2.5-7B-Instruct-4bit --name qwen7b
    robomcp-agent-eval --model anthropic:claude-opus-5-5:medium --name opus   # needs a key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mcp import Client

from robomcp.agent import Agent
from robomcp.config import load_settings
from robomcp.docs import DocsIndex
from robomcp.executor import AuditLog, ToolExecutor
from robomcp.models import ChatModel, make_model
from robomcp.permissions import Policy
from robomcp.server import build_server
from robomcp.tasks import ScriptedOperator, Task, apply_setup, evaluate, inject_faults, load_tasks
from robomcp.tools import build_tools
from robomcp.world import World

DEFAULT_TASKS = Path("tasks/tasks.yaml")
RUNS_DIR = Path("reports/agent_runs")


async def run_task(task: Task, seed: int, model: ChatModel, policy: Policy,
                   docs: DocsIndex | None, n_robots: int = 3, max_turns: int = 16) -> dict:
    world = World.generate(seed, n_robots)
    apply_setup(world, task)
    tools = build_tools(world, docs)
    injections = inject_faults(world, tools, task)
    operator = ScriptedOperator(task)
    audit = AuditLog()
    executor = ToolExecutor(tools, policy, approver=operator, audit=audit)
    agent = Agent(model, max_turns=max_turns)
    start = time.perf_counter()
    async with Client(build_server(executor)) as client:
        run = await agent.run(client, task.instruction)
    wall = time.perf_counter() - start
    outcome = evaluate(task, world, audit.records, run.answer)
    errors = dict(run.errors)
    return {
        "task": task.id, "category": task.category, "seed": seed, "model": model.name,
        "success": outcome.success, "failed": outcome.failed, "stop": run.stop,
        "turns": run.turns, "tool_calls": len(run.steps), "errors": errors,
        "hit_error": bool(errors), "parse_errors": run.parse_errors,
        "denied": errors.get("PERMISSION_DENIED", 0),
        "refused": errors.get("APPROVAL_REFUSED", 0),
        "injected": sum(i.fired for i in injections),
        "events": dict(Counter(e["kind"] for e in world.events)),
        "operator_asked": len(operator.asked),
        "answer": run.answer,
        "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
        "model_s": round(run.model_ms / 1000, 2), "tool_ms": round(run.tool_ms, 1),
        "wall_s": round(wall, 2),
        "steps": [{"turn": s.turn, "tool": s.tool, "args": s.arguments, "ok": s.ok,
                   "code": s.error_code, "result": s.result[:200]} for s in run.steps],
    }


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["task"], r["seed"]))
    return keys


async def run_suite(tasks: list[Task], seeds: list[int], model: ChatModel, policy: Policy,
                    docs: DocsIndex | None, out: Path, n_robots: int, max_turns: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    n = 0
    for task in tasks:
        if task.needs_docs and docs is None:
            print(f"skip {task.id}: no docs index (run robomcp-build-index)", file=sys.stderr)
            continue
        for seed in seeds:
            if (task.id, seed) in done:
                continue
            rec = await run_task(task, seed, model, policy, docs, n_robots, max_turns)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
            mark = "ok " if rec["success"] else "FAIL"
            print(f"{mark} {task.id:22s} seed {seed}  turns {rec['turns']:2d}  "
                  f"calls {rec['tool_calls']:2d}  {rec['wall_s']:6.1f}s  {rec['stop']}"
                  + ("" if rec["success"] else f"  {rec['failed']}"), file=sys.stderr)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the agent on the task suite.")
    ap.add_argument("--model", required=True,
                    help="mlx:<hf repo> or anthropic:<model>[:effort]")
    ap.add_argument("--name", required=True, help="output: reports/agent_runs/<name>.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--only", nargs="*", default=None, help="task ids to run")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_settings(args.config)
    tasks = load_tasks(args.tasks)
    if args.only:
        unknown = set(args.only) - {t.id for t in tasks}
        if unknown:
            ap.error(f"unknown task ids {sorted(unknown)}")
        tasks = [t for t in tasks if t.id in args.only]
    docs = DocsIndex.load(cfg.index_path) if cfg.index_path.exists() else None
    policy = Policy.load(cfg.policy_path)
    model = make_model(args.model)
    out = RUNS_DIR / f"{args.name}.jsonl"
    n = asyncio.run(run_suite(tasks, args.seeds, model, policy, docs, out,
                              cfg.world.n_robots, args.max_turns))
    print(f"{n} new run(s) -> {out}", file=sys.stderr)


def load_runs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


if __name__ == "__main__":
    main()
