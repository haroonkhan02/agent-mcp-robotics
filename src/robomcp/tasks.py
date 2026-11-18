"""Agent tasks: a starting situation, an instruction and checks on what happened.

A task file (tasks/tasks.yaml) lists tasks like

    - id: transient_timeout
      category: recovery            # nominal | recovery | safety | docs
      instruction: Send amr_1 to shelf_D.
      setup:
        robots: {amr_1: {at: dock_1, battery: 90}}
        inject: [{tool: navigate_to, robot: amr_1, times: 1, code: NAV_TIMEOUT}]
        operator: {refuse: [{tool: navigate_to, robot: amr_2}]}
      success:
        - {robot: amr_1, at: shelf_D}

The world comes from `World.generate(seed)` and the setup overrides only what the task
is about, so different seeds move the other robots and the items around.

Checks look at the final world, its event log, the audit log and the answer text:

    {robot: R, at: L | at_any: [..] | status: S | battery_min: B | moved: bool |
     fault: F | payload: P}
    {param: {robot: R, name: N, value: V}}
    {delivered: {station: item}}
    {any_robot_at: L}
    {events_max: {kind: n}}           e.g. {navigate: 0} = nothing moved
    {answer_any: [...]} / {answer_all: [...]}   case-insensitive substrings
    {route_battery_in_answer: {robot: R, goal: L}}
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robomcp.executor import AuditRecord
from robomcp.tools import ToolError, ToolSpec
from robomcp.world import FAULT_NODE, LOCATIONS, NODES, World, WorldError

CATEGORIES = ("nominal", "recovery", "safety", "docs")
ROBOT_KEYS = {"at", "battery", "fault", "estopped", "params"}
CHECK_ROBOT_KEYS = {"robot", "at", "at_any", "status", "battery_min", "moved", "fault",
                    "payload"}


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    instruction: str
    setup: dict[str, Any] = field(default_factory=dict)
    success: tuple[dict[str, Any], ...] = ()
    needs_docs: bool = False


def load_tasks(path: Path | str) -> list[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TaskError(f"{path}: expected a list of tasks")
    tasks, seen = [], set()
    for item in raw:
        t = Task(id=item["id"], category=item["category"], instruction=item["instruction"],
                 setup=item.get("setup") or {}, success=tuple(item.get("success") or ()),
                 needs_docs=bool(item.get("needs_docs", False)))
        if t.id in seen:
            raise TaskError(f"duplicate task id {t.id!r}")
        if t.category not in CATEGORIES:
            raise TaskError(f"{t.id}: category must be one of {CATEGORIES}")
        if not t.success:
            raise TaskError(f"{t.id}: needs at least one success check")
        unknown = set(t.setup) - {"robots", "inject", "operator"}
        if unknown:
            raise TaskError(f"{t.id}: unknown setup keys {sorted(unknown)}")
        for rid, spec in (t.setup.get("robots") or {}).items():
            if set(spec) - ROBOT_KEYS:
                raise TaskError(f"{t.id}: {rid}: unknown keys {sorted(set(spec) - ROBOT_KEYS)}")
            if "at" in spec and spec["at"] not in LOCATIONS:
                raise TaskError(f"{t.id}: {rid}: unknown location {spec['at']!r}")
            if spec.get("fault") and spec["fault"] not in FAULT_NODE:
                raise TaskError(f"{t.id}: {rid}: unknown fault {spec['fault']!r}")
        for check in t.success:
            _validate_check(t.id, check)
        seen.add(t.id)
        tasks.append(t)
    return tasks


def _validate_check(task_id: str, check: dict) -> None:
    kinds = {"param", "delivered", "any_robot_at", "events_max", "answer_any", "answer_all",
             "route_battery_in_answer"}
    if "robot" in check:
        bad = set(check) - CHECK_ROBOT_KEYS
    elif len(check) == 1 and next(iter(check)) in kinds:
        bad = set()
    else:
        bad = set(check)
    if bad:
        raise TaskError(f"{task_id}: unknown check {check!r}")


# --------------------------------------------------------------------------- setup
def apply_setup(world: World, task: Task) -> None:
    for rid, spec in (task.setup.get("robots") or {}).items():
        r = world.robot(rid)
        if "at" in spec:
            x, y, kind = LOCATIONS[spec["at"]]
            r.x, r.y, r.at, r.charging = x, y, spec["at"], kind == "dock"
        if "battery" in spec:
            r.battery = float(spec["battery"])
        if "fault" in spec:
            r.fault = spec["fault"]
        if "estopped" in spec:
            r.estopped = bool(spec["estopped"])
            r.charging = r.charging and not r.estopped
        for name, value in (spec.get("params") or {}).items():
            r.params[name] = float(value)
        _rewrite_diagnostics(world, r)


def _rewrite_diagnostics(world: World, r) -> None:
    """Start-up log matching the robot's state after setup.

    The generator logged the state it drew (a fault, a low battery); after the setup
    changed that state, those lines would contradict it and mislead the agent.
    """
    r.diagnostics = []
    for node in NODES:
        world._diag(r, "INFO", node, "node started")
    if r.fault:
        world._diag(r, "ERROR", FAULT_NODE[r.fault], f"{r.fault}: node stopped responding")
    if r.estopped:
        world._diag(r, "WARN", "safety_controller", "emergency stop engaged")
    if r.battery < 20:
        world._diag(r, "WARN", "battery_monitor", f"battery low: {r.battery:.0f} %")


@dataclass
class Injection:
    """Make the next `times` matching calls fail before the handler runs.

    With `code`, the call fails with that error and nothing changes (a transient fault).
    With `fault`, the robot also gets that fault, as if it broke down when starting.
    """

    tool: str
    robot: str | None = None
    times: int = 1
    code: str = "TIMEOUT"
    message: str = ""
    fault: str | None = None
    fired: int = 0


def inject_faults(world: World, tools: dict[str, ToolSpec], task: Task) -> list[Injection]:
    injections = [Injection(**spec) for spec in task.setup.get("inject") or []]
    for inj in injections:
        if inj.tool not in tools:
            raise TaskError(f"{task.id}: cannot inject into unknown tool {inj.tool!r}")
        spec = tools[inj.tool]
        tools[inj.tool] = ToolSpec(spec.name, spec.tier, spec.description,
                                   spec.input_schema, _failing(world, spec.handler, inj))
    return injections


def _failing(world: World, handler, inj: Injection):
    def run(args: dict) -> dict:
        if inj.fired < inj.times and (inj.robot is None or args.get("robot_id") == inj.robot):
            inj.fired += 1
            if inj.fault:
                r = world.robot(args["robot_id"])
                r.fault = inj.fault
                world._diag(r, "ERROR", FAULT_NODE[inj.fault],
                            f"{inj.fault}: node stopped responding")
                raise ToolError("FAULT", inj.message or f"{r.id} stopped: {inj.fault}")
            raise ToolError(inj.code, inj.message or "the request timed out; nothing changed")
        return handler(args)

    return run


class ScriptedOperator:
    """Stands in for the human who approves calls: yes, except for the listed refusals."""

    def __init__(self, task: Task):
        self.refuse = list((task.setup.get("operator") or {}).get("refuse") or [])
        self.asked: list[tuple[str, dict, bool]] = []

    def __call__(self, tool: ToolSpec, args: dict, reason: str) -> bool:
        refused = any(r["tool"] == tool.name and r.get("robot", args.get("robot_id"))
                      == args.get("robot_id") for r in self.refuse)
        self.asked.append((tool.name, args, not refused))
        return not refused


# --------------------------------------------------------------------------- checks
@dataclass
class Outcome:
    success: bool
    failed: list[str]


def evaluate(task: Task, world: World, audit: list[AuditRecord], answer: str) -> Outcome:
    snap = world.snapshot()
    kinds = Counter(e["kind"] for e in world.events)
    failed = [msg for check in task.success
              if (msg := _check(check, world, snap, kinds, answer)) is not None]
    return Outcome(not failed, failed)


def _check(c: dict, world: World, snap: dict, kinds: Counter, answer: str) -> str | None:
    text = answer.lower()
    if "robot" in c:
        r = snap["robots"][c["robot"]]
        tests = {
            "at": lambda v: r["at"] == v,
            "at_any": lambda v: r["at"] in v,
            "status": lambda v: r["status"] == v,
            "battery_min": lambda v: r["battery_pct"] >= v,
            "moved": lambda v: (r["odometer_m"] > 0) == v,
            "fault": lambda v: r["fault"] == v,
            "payload": lambda v: r["payload"] == v,
        }
        for key, want in c.items():
            if key != "robot" and not tests[key](want):
                got = r["odometer_m"] if key == "moved" else r.get(
                    {"at_any": "at", "battery_min": "battery_pct"}.get(key, key))
                return f"{c['robot']}.{key}: want {want!r}, got {got!r}"
        return None
    (kind, arg), = c.items()
    if kind == "param":
        got = snap["robots"][arg["robot"]]["params"][arg["name"]]
        ok = math.isclose(got, arg["value"], abs_tol=1e-9)
        return None if ok else f"{arg['robot']}.{arg['name']}: want {arg['value']}, got {got}"
    if kind == "delivered":
        for station, item in arg.items():
            if item not in snap["delivered"][station]:
                return f"{item} not delivered to {station}: {snap['delivered'][station]}"
        return None
    if kind == "any_robot_at":
        ok = any(r["at"] == arg for r in snap["robots"].values())
        return None if ok else f"no robot at {arg}"
    if kind == "events_max":
        for ev, n in arg.items():
            if kinds[ev] > n:
                return f"{kinds[ev]} {ev} event(s), at most {n} allowed"
        return None
    if kind == "answer_any":
        return None if any(s.lower() in text for s in arg) else f"answer lacks any of {arg}"
    if kind == "answer_all":
        missing = [s for s in arg if s.lower() not in text]
        return None if not missing else f"answer lacks {missing}"
    if kind == "route_battery_in_answer":
        try:
            need = world.route(arg["robot"], arg["goal"])["battery_needed_pct"]
        except WorldError as e:
            return f"route check failed: {e.code}"
        forms = {f"{need:.1f}", f"{need:g}", str(round(need))}
        return None if any(f in answer for f in forms) else f"answer lacks {need} %"
    raise TaskError(f"unknown check {c!r}")
