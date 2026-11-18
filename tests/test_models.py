"""Backends without a model: parsing, message translation, Claude mocked, MLX cache reuse."""

from types import SimpleNamespace

import pytest

from robomcp.models import (
    FALLBACK_BETA,
    AnthropicModel,
    MLXModel,
    ToolCall,
    common_prefix,
    make_model,
    parse_tool_calls,
    to_anthropic,
    to_hf,
)


def test_parse_qwen_tagged_calls():
    text = 'Checking.\n<tool_call>\n{"name": "list_robots", "arguments": {}}\n</tool_call>'
    prose, calls, err = parse_tool_calls(text)
    assert err is None and prose == "Checking."
    assert [(c.name, c.arguments) for c in calls] == [("list_robots", {})]


def test_parse_two_tagged_calls_and_unterminated_tag():
    text = ('<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": "{\\"y\\": 2}"}')
    _, calls, err = parse_tool_calls(text, start=5)
    assert err is None
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_5", "a", {"x": 1}), ("call_6", "b", {"y": 2})]


def test_parse_llama_bare_json_with_parameters():
    _, calls, err = parse_tool_calls('{"name": "get_params", "parameters": {"robot_id": "amr_1"}}')
    assert err is None and calls[0].arguments == {"robot_id": "amr_1"}
    two = '{"name": "a", "parameters": {}}; {"name": "b", "parameters": {}}'
    _, calls, _ = parse_tool_calls(two)
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_plain_answer_and_broken_call():
    assert parse_tool_calls("amr_2 has a lidar fault.") == ("amr_2 has a lidar fault.", [], None)
    assert parse_tool_calls('{"robots": 3} is the count')[1] == []  # JSON-ish prose is text
    _, calls, err = parse_tool_calls('<tool_call>{"name": "a", "arguments": {</tool_call>')
    assert calls == [] and "not valid JSON" in err
    _, _, err = parse_tool_calls('<tool_call>{"tool": "a"}</tool_call>')
    assert "needs" in err


CONVO = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
        ToolCall("t1", "list_robots", {}), ToolCall("t2", "get_params", {"robot_id": "a"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "list_robots", "content": "{}"},
    {"role": "tool", "tool_call_id": "t2", "name": "get_params", "content": "{}",
     "is_error": True},
    {"role": "assistant", "content": "done", "tool_calls": [], "raw": ["RAW"]},
]


def test_to_anthropic_groups_results_and_keeps_raw_blocks():
    out = to_anthropic(CONVO)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    results = out[2]["content"]
    assert [(b["tool_use_id"], b["is_error"]) for b in results] == [("t1", False), ("t2", True)]
    assert out[3]["content"] == ["RAW"]  # thinking blocks go back unchanged


def test_to_hf_uses_function_calls_and_tool_role():
    out = to_hf(CONVO)
    assert out[1]["tool_calls"][1]["function"] == {"name": "get_params",
                                                   "arguments": {"robot_id": "a"}}
    assert out[2] == {"role": "tool", "content": "{}"}


def _resp(content, stop="tool_use"):
    return SimpleNamespace(content=content, stop_reason=stop, model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class FakeClient:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


TOOLS = [{"name": "list_robots", "description": "d", "input_schema": {"type": "object"}}]


def test_anthropic_model_request_and_tool_calls():
    blocks = [SimpleNamespace(type="thinking", thinking=""),
              SimpleNamespace(type="text", text="Let me look."),
              SimpleNamespace(type="tool_use", id="tu_1", name="list_robots", input={})]
    client = FakeClient(_resp(blocks))
    turn = AnthropicModel(client=client, effort="low").complete("SYS", CONVO[:1], TOOLS)
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "SYS"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert "temperature" not in kw and "thinking" not in kw
    assert turn.text == "Let me look." and turn.raw is blocks
    assert [(c.id, c.name) for c in turn.tool_calls] == [("tu_1", "list_robots")]
    assert (turn.input_tokens, turn.output_tokens) == (100, 20)


def test_anthropic_refusal_and_max_tokens_drop_tool_calls():
    tool = SimpleNamespace(type="tool_use", id="x", name="list_robots", input={})
    turn = AnthropicModel(client=FakeClient(_resp([tool], "refusal"))).complete("", [], TOOLS)
    assert turn.stop_reason == "refusal" and turn.tool_calls == []
    turn = AnthropicModel(client=FakeClient(_resp([tool], "max_tokens"))).complete("", [], TOOLS)
    assert turn.tool_calls == []


# ----------------------------------------------------------------- MLX with fakes
class FakeKV:
    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


class FakeCacheMod:
    @staticmethod
    def make_prompt_cache(model):
        return [FakeKV()]

    @staticmethod
    def can_trim_prompt_cache(cache):
        return True

    @staticmethod
    def trim_prompt_cache(cache, n):
        return cache[0].trim(n)


class FakeTokenizer:
    """One token per character; the 'template' just joins contents."""

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, date_string):
        return "|".join(m["content"] for m in messages) + "|A:"

    def encode(self, text, add_special_tokens):
        return [ord(c) for c in text]


def make_fake_mlx(reply: str):
    fed: list[int] = []

    def generator(model, tok, prompt, max_tokens, prompt_cache):
        fed.append(len(prompt))
        kv = prompt_cache[0]
        kv.offset += len(prompt)
        for i, ch in enumerate(reply):
            if i > 0:
                kv.offset += 1  # the previous token is fed before the next one is sampled
            yield SimpleNamespace(text=ch, token=ord(ch),
                                  finish_reason="stop" if i == len(reply) - 1 else None)

    m = MLXModel("org/fake", loader=lambda repo: (object(), FakeTokenizer()),
                 generator=generator, cache_mod=FakeCacheMod)
    return m, fed


def test_mlx_reuses_cache_for_shared_prefix():
    m, fed = make_fake_mlx('<tool_call>{"name": "list_robots", "arguments": {}}</tool_call>')
    t1 = m.complete("SYS", [{"role": "user", "content": "go"}], TOOLS)
    assert t1.tool_calls[0].name == "list_robots" and t1.tool_calls[0].id == "call_0"
    first = fed[0]
    assert first == len("SYS|go|A:")
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "x", "tool_calls": t1.tool_calls},
            {"role": "tool", "tool_call_id": "call_0", "content": "{}"}]
    t2 = m.complete("SYS", msgs, TOOLS)
    assert fed[1] < len("SYS|go|x|{}|A:")  # only the new tail was processed
    assert t2.tool_calls[0].id == "call_1"
    assert m._cache[0].offset == len(m._cached)


def test_make_model_rejects_unknown_spec():
    with pytest.raises(ValueError):
        make_model("openai:gpt")
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
