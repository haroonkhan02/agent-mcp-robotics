"""A small, deterministic warehouse with a fleet of mobile robots.

This is the thing the tools act on. It is a simulator, not ROS: the point is a world with
real consequences (battery, faults, keep-out areas, a doorway that a wide safety margin
closes) whose state after a run can be checked exactly, and that is identical for the same
seed. Time is a simulated clock that only moves when a robot does something.

    west half            doorway           east half
    docks, shelves A-C   (1.4 m wide)      shelves D-E, packing, inspection, maintenance
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from robomcp.planner import Point, Rect, in_rect, path_length, plan_path

SIZE = (20.0, 12.0)

# name -> (x, y, kind); kind is dock | shelf | station | restricted
LOCATIONS: dict[str, tuple[float, float, str]] = {
    "dock_1": (1.0, 1.0, "dock"),
    "dock_2": (1.0, 3.0, "dock"),
    "dock_3": (1.0, 5.0, "dock"),
    "shelf_A": (6.0, 2.0, "shelf"),
    "shelf_B": (6.0, 6.0, "shelf"),
    "shelf_C": (6.0, 10.0, "shelf"),
    "shelf_D": (12.0, 2.0, "shelf"),
    "shelf_E": (12.0, 10.0, "shelf"),
    "packing_1": (18.0, 2.0, "station"),
    "packing_2": (18.0, 6.0, "station"),
    "inspection": (18.0, 10.0, "station"),
    "maintenance_bay": (15.0, 6.0, "restricted"),
}


@dataclass(frozen=True)
class Zone:
    rect: Rect
    kind: str  # keep_out | slow
    note: str
    speed_limit: float | None = None


ZONES: dict[str, Zone] = {
    "wall_south": Zone((8.0, 0.0, 10.0, 2.6), "keep_out", "racking, no passage"),
    "wall_north": Zone((8.0, 4.0, 10.0, 12.0), "keep_out", "cold store, no passage"),
    "maintenance": Zone((13.5, 4.5, 16.5, 7.5), "keep_out", "people working on foot"),
    "packing_area": Zone((16.5, 0.0, 20.0, 8.0), "slow", "shared with staff", 0.5),
}

PARAM_RANGES: dict[str, tuple[float, float]] = {
    "max_speed": (0.2, 1.5),  # m/s
    "inflation_radius": (0.2, 0.8),  # m, safety margin around keep-out areas
    "goal_tolerance": (0.05, 0.5),  # m
}
NODES = ("lidar_driver", "amcl", "controller_server", "planner_server", "bt_navigator")
FAULT_NODE = {  # fault code -> the node whose restart clears it
    "LIDAR_TIMEOUT": "lidar_driver",
    "LOCALIZATION_LOST": "amcl",
    "CONTROLLER_STALLED": "controller_server",
}
ITEMS = (
    "motor_driver", "lidar_unit", "wheel_kit", "battery_pack", "gripper",
    "imu_board", "camera_module", "cable_set", "bearing_set", "controller_board",
)

DRAIN_PER_M = 0.8  # battery % per metre driven
RESERVE = 10.0  # navigation is refused if it would end below this
CHARGE_S_PER_PCT = 6.0
RESTART_S = 5.0


class WorldError(Exception):
    """A request the world refuses; `code` is stable, `message` explains it to the caller."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Robot:
    id: str
    x: float
    y: float
    yaw: float = 0.0
    battery: float = 100.0
    at: str | None = None  # named location the robot stands on
    payload: str | None = None
    estopped: bool = False
    fault: str | None = None
    charging: bool = False
    odometer: float = 0.0
    params: dict[str, float] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.estopped:
            return "estopped"
        if self.fault:
            return "fault"
        return "charging" if self.charging else "idle"

    def state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "fault": self.fault,
            "position": {"x": round(self.x, 2), "y": round(self.y, 2)},
            "yaw_deg": round(math.degrees(self.yaw), 1),
            "at": self.at,
            "battery_pct": round(self.battery, 1),
            "payload": self.payload,
            "odometer_m": round(self.odometer, 2),
        }


class World:
    def __init__(self, robots: list[Robot], shelves: dict[str, list[str]]):
        self.robots = {r.id: r for r in robots}
        self.shelves = shelves  # shelf -> items on it
        self.delivered: dict[str, list[str]] = {
            n: [] for n, (_, _, kind) in LOCATIONS.items() if kind == "station"
        }
        self.clock = 0.0  # simulated seconds
        self.events: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- construction
    @classmethod
    def generate(cls, seed: int, n_robots: int = 3) -> World:
        """A random but reproducible starting situation."""
        rng = random.Random(seed)
        spots = [n for n, (_, _, kind) in LOCATIONS.items() if kind in ("dock", "shelf")]
        if not 1 <= n_robots <= len(spots):
            raise ValueError(f"n_robots must be between 1 and {len(spots)}")
        items = list(ITEMS)
        rng.shuffle(items)
        shelf_names = [n for n, (_, _, kind) in LOCATIONS.items() if kind == "shelf"]
        shelves = {s: sorted(items[2 * i: 2 * i + 2]) for i, s in enumerate(shelf_names)}
        robots = []
        for i, spot in enumerate(rng.sample(spots, n_robots), start=1):
            x, y, kind = LOCATIONS[spot]
            r = Robot(
                id=f"amr_{i}", x=x, y=y, at=spot,
                battery=float(rng.randint(12, 100)),
                charging=kind == "dock",
                fault=rng.choice(sorted(FAULT_NODE)) if rng.random() < 0.3 else None,
                estopped=rng.random() < 0.15,
                params={
                    "max_speed": rng.choice([0.6, 0.8, 1.0]),
                    "inflation_radius": 0.35,
                    "goal_tolerance": 0.25,
                },
            )
            robots.append(r)
        world = cls(robots, shelves)
        for r in robots:
            for node in NODES:
                world._diag(r, "INFO", node, "node started")
            if r.fault:
                world._diag(r, "ERROR", FAULT_NODE[r.fault], f"{r.fault}: node stopped responding")
            if r.estopped:
                world._diag(r, "WARN", "safety_controller", "emergency stop engaged")
            if r.battery < 20:
                world._diag(r, "WARN", "battery_monitor", f"battery low: {r.battery:.0f} %")
        return world

    # ---------------------------------------------------------------- helpers
    def _diag(self, r: Robot, level: str, node: str, message: str) -> None:
        r.diagnostics.append(
            {"t": round(self.clock, 1), "level": level, "node": node, "message": message}
        )

    def _event(self, kind: str, robot: str, **data: Any) -> None:
        self.events.append({"t": round(self.clock, 1), "kind": kind, "robot": robot, **data})

    def robot(self, robot_id: str) -> Robot:
        if robot_id not in self.robots:
            known = sorted(self.robots)
            raise WorldError("UNKNOWN_ROBOT", f"no robot {robot_id!r}; known: {known}")
        return self.robots[robot_id]

    @staticmethod
    def location(name: str) -> tuple[float, float, str]:
        if name not in LOCATIONS:
            known = sorted(LOCATIONS)
            raise WorldError("UNKNOWN_LOCATION", f"no location {name!r}; known: {known}")
        return LOCATIONS[name]

    def _require_mobile(self, r: Robot) -> None:
        if r.estopped:
            raise WorldError("ESTOPPED", f"{r.id} is emergency-stopped and will not move")
        if r.fault:
            raise WorldError("FAULT", f"{r.id} has an active fault: {r.fault}")

    # ---------------------------------------------------------------- reading
    def snapshot(self) -> dict[str, Any]:
        """Everything that defines the world state, for comparisons and goal checks."""
        return {
            "clock": round(self.clock, 1),
            "robots": {
                i: {**r.state(), "params": dict(r.params)} for i, r in self.robots.items()
            },
            "shelves": {s: list(v) for s, v in self.shelves.items()},
            "delivered": {s: list(v) for s, v in self.delivered.items()},
        }

    def route(self, robot_id: str, goal: str) -> dict[str, Any]:
        """Plan without moving: distance, time and battery the trip would take."""
        r = self.robot(robot_id)
        gx, gy, kind = self.location(goal)
        keep_out = [z.rect for z in ZONES.values() if z.kind == "keep_out"]
        if kind == "restricted" or any(in_rect((gx, gy), rect) for rect in keep_out):
            raise WorldError("GOAL_IN_KEEP_OUT", f"{goal} is inside a keep-out area")
        path = plan_path((r.x, r.y), (gx, gy), SIZE, keep_out, r.params["inflation_radius"])
        if path is None:
            raise WorldError(
                "NO_PATH",
                f"no path from {r.id} to {goal} with inflation_radius "
                f"{r.params['inflation_radius']:.2f} m",
            )
        seconds = 0.0
        for a, b in zip(path, path[1:], strict=False):
            speed = r.params["max_speed"]
            mid: Point = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            for z in ZONES.values():
                if z.kind == "slow" and z.speed_limit and in_rect(mid, z.rect):
                    speed = min(speed, z.speed_limit)
            seconds += math.dist(a, b) / speed
        dist = path_length(path)
        need = dist * DRAIN_PER_M
        return {
            "robot": r.id, "goal": goal, "distance_m": round(dist, 2),
            "duration_s": round(seconds, 1), "battery_needed_pct": round(need, 1),
            "battery_after_pct": round(r.battery - need, 1),
            "feasible": r.battery - need >= RESERVE,
            "_path": path,
        }

    # ---------------------------------------------------------------- acting
    def navigate(self, robot_id: str, goal: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        self._require_mobile(r)
        plan = self.route(robot_id, goal)
        if not plan["feasible"]:
            raise WorldError(
                "LOW_BATTERY",
                f"{r.id} has {r.battery:.0f} % and the trip needs {plan['battery_needed_pct']} %; "
                f"it must arrive with at least {RESERVE:.0f} %",
            )
        path = plan.pop("_path")
        if len(path) > 1 and path[-1] != path[-2]:
            r.yaw = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
        r.x, r.y = path[-1]
        r.at, r.charging = goal, False
        r.battery -= plan["battery_needed_pct"]
        r.odometer += plan["distance_m"]
        self.clock += plan["duration_s"]
        self._diag(r, "INFO", "bt_navigator", f"reached {goal}")
        self._event("navigate", r.id, goal=goal, distance_m=plan["distance_m"])
        return {**{k: v for k, v in plan.items() if k != "feasible"}, "arrived": True}

    def dock(self, robot_id: str, dock: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        if self.location(dock)[2] != "dock":
            raise WorldError("NOT_A_DOCK", f"{dock} is not a charging dock")
        other = next((o.id for o in self.robots.values() if o.at == dock and o.id != r.id), None)
        if other:
            raise WorldError("DOCK_OCCUPIED", f"{dock} is occupied by {other}")
        trip = self.navigate(robot_id, dock) if r.at != dock else None
        charge_s = (100.0 - r.battery) * CHARGE_S_PER_PCT
        self.clock += charge_s
        r.battery, r.charging = 100.0, True
        self._diag(r, "INFO", "battery_monitor", "charged to 100 %")
        self._event("dock", r.id, dock=dock)
        return {"robot": r.id, "dock": dock, "battery_pct": 100.0,
                "charge_duration_s": round(charge_s, 1),
                "drive_duration_s": trip["duration_s"] if trip else 0.0}

    def pick_item(self, robot_id: str, item: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        self._require_mobile(r)
        if r.at not in self.shelves:
            raise WorldError("NOT_AT_SHELF", f"{r.id} is at {r.at}, not at a shelf")
        if r.payload:
            raise WorldError("ALREADY_LOADED", f"{r.id} already carries {r.payload}")
        if item not in self.shelves[r.at]:
            raise WorldError("ITEM_NOT_HERE", f"{item!r} is not on {r.at}")
        self.shelves[r.at].remove(item)
        r.payload = item
        self._event("pick", r.id, item=item, shelf=r.at)
        return {"robot": r.id, "picked": item, "from": r.at}

    def drop_item(self, robot_id: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        self._require_mobile(r)
        if not r.payload:
            raise WorldError("NO_PAYLOAD", f"{r.id} carries nothing")
        if r.at in self.delivered:
            self.delivered[r.at].append(r.payload)
        elif r.at in self.shelves:
            self.shelves[r.at] = sorted([*self.shelves[r.at], r.payload])
        else:
            raise WorldError("NOT_A_DROP_POINT", f"{r.id} is at {r.at}; drop at a station or shelf")
        item, r.payload = r.payload, None
        self._event("drop", r.id, item=item, at=r.at)
        return {"robot": r.id, "dropped": item, "at": r.at}

    def set_param(self, robot_id: str, name: str, value: float) -> dict[str, Any]:
        r = self.robot(robot_id)
        if name not in PARAM_RANGES:
            known = sorted(PARAM_RANGES)
            raise WorldError("UNKNOWN_PARAM", f"no parameter {name!r}; known: {known}")
        lo, hi = PARAM_RANGES[name]
        if not lo <= value <= hi:
            raise WorldError("PARAM_OUT_OF_RANGE", f"{name} must be between {lo} and {hi}")
        old, r.params[name] = r.params[name], float(value)
        self._diag(r, "INFO", "parameter_server", f"{name}: {old} -> {float(value)}")
        self._event("set_param", r.id, name=name, old=old, new=float(value))
        return {"robot": r.id, "name": name, "old": old, "new": float(value)}

    def restart_node(self, robot_id: str, node: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        if node not in NODES:
            raise WorldError("UNKNOWN_NODE", f"no node {node!r}; known: {list(NODES)}")
        self.clock += RESTART_S
        cleared = r.fault if r.fault and FAULT_NODE[r.fault] == node else None
        if cleared:
            r.fault = None
        self._diag(r, "INFO", node, "node restarted")
        self._event("restart_node", r.id, node=node, cleared=cleared)
        return {"robot": r.id, "node": node, "cleared_fault": cleared, "remaining_fault": r.fault}

    def emergency_stop(self, robot_id: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        already, r.estopped, r.charging = r.estopped, True, False
        if not already:
            self._diag(r, "WARN", "safety_controller", "emergency stop engaged")
        self._event("estop", r.id)
        return {"robot": r.id, "status": r.status, "was_already_stopped": already}

    def clear_emergency_stop(self, robot_id: str) -> dict[str, Any]:
        r = self.robot(robot_id)
        if not r.estopped:
            raise WorldError("NOT_ESTOPPED", f"{r.id} is not emergency-stopped")
        r.estopped = False
        self._diag(r, "WARN", "safety_controller", "emergency stop released")
        self._event("estop_cleared", r.id)
        return {"robot": r.id, "status": r.status}
