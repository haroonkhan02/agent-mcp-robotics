"""A per-task grant: what this request is allowed to touch, whatever the model decides.

The permission policy says what the *deployment* allows. A grant narrows that for one
request: "this task may move amr_1 and nothing else". It comes from the trusted side (the
user's request, the front end), never from text the agent read, so an instruction hidden in
a document or a diagnostics line cannot widen it.

    grant:
      tiers: [motion]        # state-changing tiers this request may use (read is always ok)
      robots: [amr_1]        # robots a state-changing call may name

`emergency_stop` (tier stop) is always allowed: stopping is the safe direction. A call
outside the grant fails with SCOPE_DENIED before the policy or any approver sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robomcp.tools import TIERS, ToolSpec

ALWAYS = ("read", "stop")


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class Grant:
    tiers: frozenset[str] = frozenset()
    robots: frozenset[str] = frozenset()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Grant:
        raw = raw or {}
        unknown = set(raw) - {"tiers", "robots"}
        if unknown:
            raise ScopeError(f"unknown grant key(s): {sorted(unknown)}")
        tiers = frozenset(raw.get("tiers") or ())
        bad = tiers - set(TIERS)
        if bad:
            raise ScopeError(f"unknown tier(s) in grant: {sorted(bad)}")
        return cls(tiers, frozenset(raw.get("robots") or ()))

    def violation(self, tool: ToolSpec, args: dict[str, Any]) -> str | None:
        if tool.tier in ALWAYS:
            return None
        if tool.tier not in self.tiers:
            return f"this request may not use '{tool.tier}' tools"
        robot = args.get("robot_id")
        if robot is not None and robot not in self.robots:
            return f"this request may not act on {robot}"
        return None
