"""The tools: name, permission tier, JSON Schema and handler for each one.

Defined once here and used both in-process (tests, evaluations) and by the MCP server, so
the two cannot drift apart. A handler takes validated arguments and returns a JSON-able
dict, or raises ToolError with a stable code the caller can react to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robomcp.docs import PAGE_CHARS, DocsIndex, snippet, tokenize
from robomcp.world import LOCATIONS, NODES, PARAM_RANGES, ZONES, World, WorldError

TIERS = ("read", "stop", "config", "motion", "release")
MAX_SEARCH_RESULTS = 10


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    props = props or {}
    return {
        "type": "object",
        "properties": props,
        "required": list(props) if required is None else required,
        "additionalProperties": False,
    }


ROBOT = {"type": "string", "description": "Robot id, e.g. amr_1"}
LOCATION = {"type": "string", "description": "Location name from list_locations"}


def _fleet_tools(w: World) -> list[ToolSpec]:
    def list_robots(_: dict) -> dict:
        return {"clock_s": round(w.clock, 1),
                "robots": [r.state() for r in w.robots.values()]}

    def get_robot_state(a: dict) -> dict:
        r = w.robot(a["robot_id"])
        return {**r.state(), "params": dict(r.params)}

    def list_locations(_: dict) -> dict:
        locs = []
        for name, (x, y, kind) in LOCATIONS.items():
            entry: dict[str, Any] = {"name": name, "kind": kind, "x": x, "y": y}
            if name in w.shelves:
                entry["items"] = list(w.shelves[name])
            if name in w.delivered:
                entry["delivered"] = list(w.delivered[name])
            locs.append(entry)
        zones = [{"name": n, "kind": z.kind, "rect": list(z.rect), "note": z.note,
                  **({"speed_limit": z.speed_limit} if z.speed_limit else {})}
                 for n, z in ZONES.items()]
        return {"locations": locs, "zones": zones}

    def get_params(a: dict) -> dict:
        r = w.robot(a["robot_id"])
        return {"robot": r.id, "params": [
            {"name": n, "value": r.params[n], "min": lo, "max": hi}
            for n, (lo, hi) in PARAM_RANGES.items()]}

    def read_diagnostics(a: dict) -> dict:
        r = w.robot(a["robot_id"])
        return {"robot": r.id, "entries": r.diagnostics[-a.get("limit", 10):]}

    def plan_route(a: dict) -> dict:
        plan = w.route(a["robot_id"], a["goal"])
        plan.pop("_path")
        return plan

    return [
        ToolSpec("list_robots", "read",
                 "List every robot with its status, position, battery and payload.",
                 _schema(), list_robots),
        ToolSpec("get_robot_state", "read",
                 "State and parameters of one robot.",
                 _schema({"robot_id": ROBOT}), get_robot_state),
        ToolSpec("list_locations", "read",
                 "Named locations (docks, shelves with their items, stations with what was "
                 "delivered) and zones (keep-out areas, speed limits).",
                 _schema(), list_locations),
        ToolSpec("get_params", "read",
                 "Navigation parameters of a robot with their allowed ranges.",
                 _schema({"robot_id": ROBOT}), get_params),
        ToolSpec("read_diagnostics", "read",
                 "Most recent diagnostic messages of a robot, oldest first.",
                 _schema({"robot_id": ROBOT,
                          "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                                    "default": 10}},
                         required=["robot_id"]), read_diagnostics),
        ToolSpec("plan_route", "read",
                 "Plan a trip without moving: distance, duration, battery needed and whether "
                 "the robot would arrive above the battery reserve.",
                 _schema({"robot_id": ROBOT, "goal": LOCATION}), plan_route),
        ToolSpec("emergency_stop", "stop",
                 "Stop a robot immediately. It will not move until the stop is cleared.",
                 _schema({"robot_id": ROBOT}),
                 lambda a: w.emergency_stop(a["robot_id"])),
        ToolSpec("set_param", "config",
                 "Change one navigation parameter of a robot.",
                 _schema({"robot_id": ROBOT,
                          "name": {"type": "string", "enum": sorted(PARAM_RANGES)},
                          "value": {"type": "number"}}),
                 lambda a: w.set_param(a["robot_id"], a["name"], a["value"])),
        ToolSpec("restart_node", "config",
                 "Restart one software node on a robot. Clears a fault caused by that node.",
                 _schema({"robot_id": ROBOT, "node": {"type": "string", "enum": list(NODES)}}),
                 lambda a: w.restart_node(a["robot_id"], a["node"])),
        ToolSpec("navigate_to", "motion",
                 "Drive a robot to a named location.",
                 _schema({"robot_id": ROBOT, "goal": LOCATION}),
                 lambda a: w.navigate(a["robot_id"], a["goal"])),
        ToolSpec("dock", "motion",
                 "Drive a robot to a free charging dock and charge it to 100 %.",
                 _schema({"robot_id": ROBOT, "dock": LOCATION}),
                 lambda a: w.dock(a["robot_id"], a["dock"])),
        ToolSpec("pick_item", "motion",
                 "Load an item from the shelf the robot is standing at.",
                 _schema({"robot_id": ROBOT, "item": {"type": "string"}}),
                 lambda a: w.pick_item(a["robot_id"], a["item"])),
        ToolSpec("drop_item", "motion",
                 "Unload the carried item at the station or shelf the robot is standing at.",
                 _schema({"robot_id": ROBOT}),
                 lambda a: w.drop_item(a["robot_id"])),
        ToolSpec("clear_emergency_stop", "release",
                 "Release an emergency stop so the robot can move again.",
                 _schema({"robot_id": ROBOT}),
                 lambda a: w.clear_emergency_stop(a["robot_id"])),
    ]


def _docs_tools(docs: DocsIndex | None) -> list[ToolSpec]:
    def need() -> DocsIndex:
        if docs is None:
            raise ToolError("DOCS_UNAVAILABLE",
                            "the documentation index is not built; run robomcp-build-index")
        return docs

    def search_docs(a: dict) -> dict:
        query = a["query"]
        q_tokens = set(tokenize(query))
        hits = need().search(query, a.get("k", 5))
        return {"query": query, "results": [
            {"section_id": s.id, "page": s.title, "heading": s.heading, "url": s.url,
             "score": round(score, 2), "chars": len(s.text), "snippet": snippet(s.text, q_tokens)}
            for s, score in hits]}

    def read_doc_section(a: dict) -> dict:
        s = need().get(a["section_id"])
        if s is None:
            raise ToolError("UNKNOWN_SECTION",
                            f"no section {a['section_id']!r}; use an id returned by search_docs")
        offset = a.get("offset", 0)
        if offset >= len(s.text) and offset > 0:
            raise ToolError("OFFSET_OUT_OF_RANGE",
                            f"offset {offset} is past the end ({len(s.text)} characters)")
        end = offset + PAGE_CHARS
        return {"section_id": s.id, "page": s.title, "heading": s.heading, "url": s.url,
                "text": s.text[offset:end], "total_chars": len(s.text),
                "next_offset": end if end < len(s.text) else None}

    return [
        ToolSpec("search_docs", "read",
                 "Search the ROS 2 documentation. Returns section ids with a short snippet; "
                 "read a section with read_doc_section before relying on it.",
                 _schema({"query": {"type": "string", "minLength": 1, "maxLength": 300},
                          "k": {"type": "integer", "minimum": 1,
                                "maximum": MAX_SEARCH_RESULTS, "default": 5}},
                         required=["query"]), search_docs),
        ToolSpec("read_doc_section", "read",
                 f"Read one documentation section, up to {PAGE_CHARS} characters per call; "
                 "pass next_offset to continue a long section.",
                 _schema({"section_id": {"type": "string"},
                          "offset": {"type": "integer", "minimum": 0, "default": 0}},
                         required=["section_id"]), read_doc_section),
    ]


def build_tools(world: World, docs: DocsIndex | None = None) -> dict[str, ToolSpec]:
    """All tools, bound to one world and (optionally) one documentation index."""
    specs = _fleet_tools(world) + _docs_tools(docs)
    wrapped = {}
    for spec in specs:
        assert spec.tier in TIERS, spec.tier
        wrapped[spec.name] = ToolSpec(spec.name, spec.tier, spec.description,
                                      spec.input_schema, _translate(spec.handler))
    return wrapped


def _translate(handler: Callable[[dict], dict]) -> Callable[[dict], dict]:
    def run(args: dict) -> dict:
        try:
            return handler(args)
        except WorldError as e:
            raise ToolError(e.code, e.message) from None

    return run
