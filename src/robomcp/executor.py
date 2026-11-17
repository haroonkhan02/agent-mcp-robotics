"""The only way a tool runs: validate, check permission, run, record.

    unknown tool      -> UNKNOWN_TOOL
    bad arguments     -> INVALID_ARGUMENTS   (JSON Schema, before anything else looks at them)
    policy says deny  -> PERMISSION_DENIED
    policy says ask   -> the approver decides; refused -> APPROVAL_REFUSED,
                         nobody to ask -> APPROVAL_REQUIRED (unless on_ask is allow)
    handler refuses   -> the handler's own code, e.g. LOW_BATTERY
    handler crashes   -> INTERNAL_ERROR      (no traceback reaches the caller)

Every call, allowed or not, is appended to the audit log.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robomcp.permissions import Policy
from robomcp.tools import ToolError, ToolSpec

Approver = Callable[[ToolSpec, dict[str, Any], str], bool]


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, str] | None = None  # {"code", "message"}
    decision: str = ""  # allowed | approved | unattended | denied | refused | unapproved | n/a

    def payload(self) -> dict[str, Any]:
        """What the caller (a model) gets to see."""
        return self.data if self.ok else {"error": self.error}


@dataclass
class AuditRecord:
    seq: int
    tool: str
    tier: str
    arguments: Any
    decision: str
    ok: bool
    error_code: str | None
    ms: float


@dataclass
class AuditLog:
    path: Path | None = None
    records: list[AuditRecord] = field(default_factory=list)

    def add(self, **kw: Any) -> AuditRecord:
        rec = AuditRecord(seq=len(self.records) + 1, **kw)
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), default=str) + "\n")
        return rec


class ToolExecutor:
    def __init__(
        self,
        tools: dict[str, ToolSpec],
        policy: Policy,
        approver: Approver | None = None,
        audit: AuditLog | None = None,
    ):
        policy.check_known_tools(set(tools))
        self.tools = tools
        self.policy = policy
        self.approver = approver
        self.audit = audit or AuditLog()
        self._validators = {n: Draft202012Validator(t.input_schema) for n, t in tools.items()}

    def call(self, name: str, arguments: Any = None) -> ToolResult:
        start = time.perf_counter()
        args = {} if arguments is None else arguments
        result = self._run(name, args)
        tool = self.tools.get(name)
        self.audit.add(
            tool=name, tier=tool.tier if tool else "?", arguments=args,
            decision=result.decision, ok=result.ok,
            error_code=result.error["code"] if result.error else None,
            ms=round((time.perf_counter() - start) * 1000, 3),
        )
        return result

    def _run(self, name: str, args: Any) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return _fail("UNKNOWN_TOOL", f"no tool {name!r}; known: {sorted(self.tools)}", "n/a")
        errors = sorted(self._validators[name].iter_errors(args), key=lambda e: list(e.path))
        if errors:
            e = errors[0]
            where = ".".join(str(p) for p in e.path) or "arguments"
            return _fail("INVALID_ARGUMENTS", f"{where}: {e.message}", "n/a")

        decision = self.policy.decide(tool, args)
        if decision.outcome == "deny":
            return _fail("PERMISSION_DENIED", decision.reason, "denied")
        how = "allowed"
        if decision.outcome == "ask":
            if self.approver is not None:
                if not self.approver(tool, args, decision.reason):
                    return _fail("APPROVAL_REFUSED",
                                 f"the operator refused {name}", "refused")
                how = "approved"
            elif self.policy.on_ask == "allow":
                how = "unattended"
            else:
                return _fail("APPROVAL_REQUIRED",
                             f"{decision.reason}, and no operator is attached", "unapproved")
        try:
            return ToolResult(True, data=tool.handler(args), decision=how)
        except ToolError as e:
            return ToolResult(False, error={"code": e.code, "message": e.message}, decision=how)
        except Exception as e:  # a bug in a handler must not take the server down
            return ToolResult(
                False, decision=how,
                error={"code": "INTERNAL_ERROR", "message": f"{name} failed ({type(e).__name__})"},
            )


def _fail(code: str, message: str, decision: str) -> ToolResult:
    return ToolResult(False, error={"code": code, "message": message}, decision=decision)
