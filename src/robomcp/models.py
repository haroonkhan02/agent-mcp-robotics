"""Chat models the agent can drive: Claude, a local open-weight model on MLX, or a script.

The agent keeps the conversation in one neutral format and each backend translates it:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": provider blocks}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}

Tools are given as {"name", "description", "input_schema"}, exactly what MCP lists.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

FALLBACK_BETA = "server-side-fallback-2026-07-01"
TEMPLATE_DATE = "01 Jan 2026"  # Llama's template prints today's date; fixed for reproducibility


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any  # a dict when the model produced valid JSON; anything else is passed on


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    parse_error: str | None = None  # the model tried to call a tool but the call is unreadable
    raw: Any = None  # provider content to send back unchanged (Claude's thinking blocks)
    served_by: str = ""


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


# --------------------------------------------------------------------------- parsing
_TAGGED = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def _as_call(obj: Any, n: int) -> ToolCall | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):  # some models double-encode the arguments
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    return ToolCall(id=f"call_{n}", name=obj["name"], arguments=args)


def _json_objects(text: str) -> list[Any] | None:
    """Every top-level JSON value in `text` separated only by whitespace, ';' or ','."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i] in " \t\r\n;,":
            i += 1
            continue
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            return None
        out.append(obj)
    return out


def parse_tool_calls(text: str, start: int = 0) -> tuple[str, list[ToolCall], str | None]:
    """Split generated text into (prose, tool calls, parse error).

    Qwen wraps each call in <tool_call>...</tool_call>; Llama 3.2 answers with a bare JSON
    object {"name": ..., "parameters": ...}. Text that is neither is a final answer.
    """
    calls: list[ToolCall] = []
    # Llama 3.x marks calls after the first turn with <|python_tag|> (and may end <|eom_id|>)
    text = text.replace("<|python_tag|>", "").replace("<|eom_id|>", "")
    if "<tool_call>" in text:
        for body in _TAGGED.findall(text):
            try:
                obj = json.loads(body.strip())
            except json.JSONDecodeError as e:
                return text, [], f"tool call is not valid JSON ({e.msg})"
            call = _as_call(obj, start + len(calls))
            if call is None:
                return text, [], 'tool call needs {"name": ..., "arguments": {...}}'
            calls.append(call)
        prose = _TAGGED.sub("", text).strip()
        return prose, calls, None
    stripped = text.strip()
    if stripped.startswith("{"):
        objs = _json_objects(stripped)
        if objs and all(isinstance(o, dict) and "name" in o for o in objs):
            for obj in objs:
                call = _as_call(obj, start + len(calls))
                if call is None:
                    return text, [], 'tool call needs {"name": ..., "arguments": {...}}'
                calls.append(call)
            return "", calls, None
    return text.strip(), [], None


# --------------------------------------------------------------------------- scripted
class ScriptedModel:
    """Replays fixed turns, or a function of the conversation. For tests and dry runs."""

    def __init__(self, turns: Sequence[ModelTurn | Callable[[list[dict]], ModelTurn]],
                 name: str = "scripted"):
        self.turns, self.name, self.i = list(turns), name, 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        if self.i >= len(self.turns):
            return ModelTurn(text="(script exhausted)")
        turn = self.turns[self.i]
        self.i += 1
        return turn(messages) if callable(turn) else turn


def call(name: str, **arguments: Any) -> ModelTurn:
    """Shorthand for a scripted turn that makes one tool call."""
    return ModelTurn(text="", tool_calls=[ToolCall(f"s_{name}", name, arguments)],
                     stop_reason="tool_use")


# --------------------------------------------------------------------------- Claude
def to_anthropic(messages: list[dict]) -> list[dict]:
    """Neutral -> Messages API. Tool results of one turn go back in one user message."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant":
            if m.get("raw") is not None:  # unchanged, thinking blocks included
                out.append({"role": "assistant", "content": m["raw"]})
                continue
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name,
                        "input": c.arguments if isinstance(c.arguments, dict) else {}}
                       for c in m.get("tool_calls", [])]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m["content"], "is_error": bool(m.get("is_error"))}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


class AnthropicModel:
    """Claude through the Messages API with the agent's own tool loop.

    Opus 5.5 always thinks; `effort` sets how much. Server-side refusal fallbacks are on,
    so a classifier decline is retried on a fallback model inside the same request; a
    refusal of the whole chain comes back as stop_reason "refusal" and ends the run.
    Requires ANTHROPIC_API_KEY (or another credential the SDK resolves).
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium",
                 max_tokens: int = 16000, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=3)
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.name = f"anthropic:{model}:{effort}"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        start = time.perf_counter()
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[{"name": t["name"], "description": t["description"],
                    "input_schema": t["input_schema"]} for t in tools],
            messages=to_anthropic(messages),
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        turn = ModelTurn(text="", stop_reason=resp.stop_reason or "end_turn",
                         input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                         latency_ms=ms, raw=resp.content,
                         served_by=getattr(resp, "model", None) or self.model)
        if turn.stop_reason == "refusal":
            return turn
        turn.text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if turn.stop_reason != "max_tokens":  # a cut-off tool call may be incomplete
            turn.tool_calls = [ToolCall(b.id, b.name, b.input)
                               for b in resp.content if b.type == "tool_use"]
        return turn


# --------------------------------------------------------------------------- local (MLX)
def to_hf(messages: list[dict]) -> list[dict]:
    """Neutral -> the message format Hugging Face chat templates expect."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content", "")}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{"type": "function", "function": {
                    "name": c.name, "arguments": c.arguments}} for c in m["tool_calls"]]
            out.append(entry)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def one_call_per_turn(messages: list[dict]) -> list[dict]:
    """Rewrite turns with several tool calls as call, result, call, result.

    Some chat templates (Llama 3.x) reject an assistant message with more than one call.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") or []
        if m["role"] != "assistant" or len(calls) < 2:
            out.append(m)
            i += 1
            continue
        results = []
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            results.append(messages[j])
            j += 1
        by_id = {r.get("tool_call_id"): r for r in results}
        for k, c in enumerate(calls):
            out.append({**m, "content": m.get("content", "") if k == 0 else "",
                        "tool_calls": [c], "raw": None})
            if c.id in by_id:
                out.append(by_id[c.id])
        i = j
    return out


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class MLXModel:
    """An open-weight instruct model on Apple silicon via mlx-lm, greedy decoding.

    The tool list and system prompt are rendered by the model's own chat template. Each
    turn re-renders the whole conversation, so the KV cache is kept between calls and
    only the tokens after the longest common prefix are processed again.
    """

    def __init__(self, repo: str, max_tokens: int = 768, loader: Any = None,
                 generator: Any = None, cache_mod: Any = None):
        if loader is None or generator is None or cache_mod is None:
            import mlx_lm
            from mlx_lm.models import cache as mlx_cache

            loader, generator = loader or mlx_lm.load, generator or mlx_lm.stream_generate
            cache_mod = cache_mod or mlx_cache
        self.model, self.tokenizer = loader(repo)
        self._generate = generator
        self._cache_mod = cache_mod
        self.repo, self.max_tokens = repo, max_tokens
        self.name = f"mlx:{repo.split('/')[-1]}"
        self._cache: list[Any] | None = None
        self._cached: list[int] = []
        self._calls = 0

    def render(self, system: str, messages: list[dict], tools: list[dict]) -> str:
        hf_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}} for t in tools]
        def render(msgs: list[dict]) -> str:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, *to_hf(msgs)], tools=hf_tools,
                tokenize=False, add_generation_prompt=True, date_string=TEMPLATE_DATE)

        try:
            return render(messages)
        except Exception as e:  # jinja2 TemplateError; jinja2 is not a direct dependency
            if "single tool-call" not in str(e):
                raise
            return render(one_call_per_turn(messages))

    def _reuse(self, tokens: list[int]) -> int:
        """Trim the cache to the shared prefix; return how many prompt tokens it covers."""
        keep = min(common_prefix(tokens, self._cached), len(tokens) - 1)
        cm = self._cache_mod
        if self._cache is None or not cm.can_trim_prompt_cache(self._cache):
            self._cache, self._cached = cm.make_prompt_cache(self.model), []
            return 0
        drop = len(self._cached) - keep
        if drop and cm.trim_prompt_cache(self._cache, drop) != drop:
            self._cache, self._cached = cm.make_prompt_cache(self.model), []
            return 0
        self._cached = self._cached[:keep]
        return keep

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        start = time.perf_counter()
        prompt = self.render(system, messages, tools)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        reused = self._reuse(tokens)
        pieces, generated, last = [], [], None
        for last in self._generate(self.model, self.tokenizer, tokens[reused:],
                                   max_tokens=self.max_tokens, prompt_cache=self._cache):
            pieces.append(last.text)
            generated.append(last.token)
        # Every generated token except the last one has been fed through the model.
        self._cached = tokens + generated[:-1]
        offset = getattr(self._cache[0], "offset", None) if self._cache else None
        if offset is not None and offset != len(self._cached):
            self._cache, self._cached = None, []  # out of sync: start clean next time
        text = "".join(pieces)
        prose, calls, err = parse_tool_calls(text, start=self._calls)
        self._calls += len(calls)
        finish = getattr(last, "finish_reason", None)
        return ModelTurn(
            text=prose, tool_calls=calls, parse_error=err,
            stop_reason="max_tokens" if finish == "length" else (
                "tool_use" if calls else "end_turn"),
            input_tokens=len(tokens), output_tokens=len(generated),
            latency_ms=(time.perf_counter() - start) * 1000, served_by=self.repo,
        )


def make_model(spec: str) -> ChatModel:
    """`anthropic:<model>[:<effort>]` or `mlx:<hf repo>`."""
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        model, _, effort = (rest or "claude-opus-5-5").partition(":")
        return AnthropicModel(model, effort or "medium")
    if kind == "mlx" and rest:
        return MLXModel(rest)
    raise ValueError(f"unknown model {spec!r}; use anthropic:<model>[:effort] or mlx:<repo>")
