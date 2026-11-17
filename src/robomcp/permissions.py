"""Permission policy: which tool calls are allowed, which need approval, which never run.

Two layers, both declared in configs/policy.yaml:

1. Argument constraints (hard limits). A call outside them is denied outright; nobody is
   asked, because an approver who clicks "yes" too quickly should not be able to widen
   the envelope.
2. The tier of the tool (read / stop / config / motion / release) maps to
   allow, ask or deny.

The policy decides; the executor enforces. The model never sees or edits the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robomcp.tools import TIERS, ToolSpec

OUTCOMES = ("allow", "ask", "deny")
_CONSTRAINT_KEYS = {"enum", "min", "max"}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    outcome: str  # allow | ask | deny
    reason: str


@dataclass(frozen=True)
class Policy:
    tiers: dict[str, str]
    on_ask: str = "deny"  # what "ask" becomes when no approver is attached
    constraints: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> Policy:
        if not isinstance(raw, dict):
            raise PolicyError("policy must be a mapping")
        unknown = set(raw) - {"tiers", "on_ask", "constraints"}
        if unknown:
            raise PolicyError(f"unknown policy key(s): {sorted(unknown)}")
        tiers = raw.get("tiers")
        if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
            # a tier left out must not silently default to anything
            raise PolicyError(f"tiers must list exactly {list(TIERS)}")
        for tier, outcome in tiers.items():
            if outcome not in OUTCOMES:
                raise PolicyError(f"tier {tier}: {outcome!r} is not one of {list(OUTCOMES)}")
        on_ask = raw.get("on_ask", "deny")
        if on_ask not in ("allow", "deny"):
            raise PolicyError("on_ask must be allow or deny")
        constraints = raw.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise PolicyError("constraints must be a mapping of tool -> argument -> limits")
        for tool, by_arg in constraints.items():
            if not isinstance(by_arg, dict):
                raise PolicyError(f"constraints.{tool} must be a mapping")
            for arg, limits in by_arg.items():
                if not isinstance(limits, dict) or not limits or set(limits) - _CONSTRAINT_KEYS:
                    raise PolicyError(
                        f"constraints.{tool}.{arg}: use one or more of {sorted(_CONSTRAINT_KEYS)}")
                if "enum" in limits and not isinstance(limits["enum"], list):
                    raise PolicyError(f"constraints.{tool}.{arg}.enum must be a list")
        return cls(dict(tiers), on_ask, constraints)

    @classmethod
    def load(cls, path: Path | str) -> Policy:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def check_known_tools(self, tool_names: set[str]) -> None:
        """A constraint for a tool that does not exist is a typo that would protect nothing."""
        unknown = set(self.constraints) - tool_names
        if unknown:
            raise PolicyError(f"constraints name unknown tool(s): {sorted(unknown)}")

    def violation(self, tool: str, args: dict[str, Any]) -> str | None:
        for arg, limits in self.constraints.get(tool, {}).items():
            if arg not in args:
                continue
            value = args[arg]
            if "enum" in limits and value not in limits["enum"]:
                return f"{arg}={value!r} is not permitted (allowed: {limits['enum']})"
            if ("min" in limits or "max" in limits) and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                return f"{arg}={value!r} must be a number"
            if "min" in limits and value < limits["min"]:
                return f"{arg}={value!r} is below the permitted minimum {limits['min']}"
            if "max" in limits and value > limits["max"]:
                return f"{arg}={value!r} is above the permitted maximum {limits['max']}"
        return None

    def decide(self, tool: ToolSpec, args: dict[str, Any]) -> Decision:
        outcome = self.tiers[tool.tier]
        if outcome == "deny":
            return Decision("deny", f"tools of tier '{tool.tier}' are not permitted")
        problem = self.violation(tool.name, args)
        if problem:
            return Decision("deny", problem)
        if outcome == "ask":
            return Decision("ask", f"tools of tier '{tool.tier}' need operator approval")
        return Decision("allow", f"tier '{tool.tier}' is allowed")
