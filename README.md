<h1 align="center">Agent + MCP Server for Robotics</h1>

<p align="center">
  <a href="https://github.com/haroonkhan02/agent-mcp-robotics/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/agent-mcp-robotics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/MCP-2026--07--28-purple" alt="MCP">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

A [Model Context Protocol](https://modelcontextprotocol.io) server that gives an LLM agent
tools for a simulated warehouse robot fleet and for searching the ROS 2 documentation, with
a permission policy between the model and anything that moves. The goal is to measure how
reliably an agent completes fleet tasks through these tools: when it needs approval, when a
tool fails, and when text it reads tries to hijack it.

## Design

```mermaid
flowchart LR
    A[agent: LLM + tool loop] -- MCP, stdio --> S[robomcp server]
    S --> E[executor]
    E --> V[JSON Schema check] --> P{policy}
    P -- allow --> H[tool handler]
    P -- ask --> O[operator approval] --> H
    P -- deny --> X[error with code + reason]
    H --> W[simulated fleet: battery, faults, keep-out areas]
    H --> D[BM25 over ROS 2 doc sections]
    E -.-> L[audit log, every call]
```

Every tool belongs to one tier, and the policy file maps tiers to allow, ask or deny:

| Tier | Tools | Shipped policy |
|---|---|---|
| read | list_robots, get_robot_state, list_locations, get_params, read_diagnostics, plan_route, search_docs, read_doc_section | allow |
| stop | emergency_stop | allow |
| config | set_param, restart_node | ask, and only max_speed / goal_tolerance and three named nodes |
| motion | navigate_to, dock, pick_item, drop_item | ask |
| release | clear_emergency_stop | deny |

The fleet is a small deterministic simulator, not ROS: three robots in a 20 x 12 m
warehouse with docks, shelves, packing stations, a keep-out area and a 1.4 m doorway
between the two halves. Driving costs battery and simulated time, faults block motion
until the right node is restarted, and a safety margin set too wide closes the doorway.
The same seed always gives the same world, so the end state of a run can be checked exactly.

## Results

The agent gets one instruction, the MCP tool list and a short system prompt, and works until
it answers. Each task starts from a seeded world and is scored on the end state: where the
robots are, what was delivered, which events happened, and for read-only and refusal tasks
what the answer says. A scripted operator approves every call that needs approval, except in
the task where it refuses. Local models, greedy decoding, 14 tasks x 3 seeds:

| Tasks (runs) | Qwen2.5-7B | Qwen2.5-7B + autonomy rule | Llama-3.2-3B |
|---|---|---|---|
| All (42) | 19 (45 %) | 24 (57 %) | 12 (29 %) |
| Nominal: status, delivery, parameter, route (12) | 7 | 6 | 3 |
| Recovery: low battery, fault, timeout, busy dock, no path (18) | 0 | 6 | 3 |
| Safety: e-stopped robot, keep-out goal, operator refuses (9) | 9 | 9 | 6 |
| Docs question (3) | 3 | 3 | 0 |

The safety tasks hold because the server enforces them: no model released an emergency stop
or entered the keep-out area, however often it tried. Recovery is where the models fall
down. Qwen reads each error correctly and then stops to describe the fix instead of doing
it. The autonomy rule ("nobody will answer questions; fix the cause yourself") fixes
restart-the-node recovery but makes the model act more loosely, for example driving a
different robot than the one named. Per-task tables and every failed run:
[reports/agent_tasks.md](reports/agent_tasks.md).

### Prompt injection

Tool output is untrusted text. A second set of 14 tasks plants an instruction in text the
agent reads: a robot diagnostics line, a ROS 2 docs section, or the message of an error it
is likely to hit. The planted text asks for something the user did not, such as releasing an
emergency stop, restarting another robot's node or raising the speed limit. Each run is
scored three ways: did the model try it, did it work, and did it still do the user's task.

Two defences are compared. **Spotlight** wraps tool output in `<tool_data>` tags, tells the
model to treat it as data and repeats the user's request after every result. The **scope
guard** gives each task a grant (which tiers and robots it may touch) and the executor
rejects calls outside it, whatever the model says. 14 tasks x 5 seeds per row:

| Qwen2.5-7B | attack tried | attack worked (95 % CI) | user task done |
|---|---|---|---|
| no defence | 10/70 | 10/70 (8 to 24 %) | 47/70 |
| spotlight | 5/70 | 0/70 (0 to 5 %) | 40/70 |
| scope guard | 10/70 | 5/70 (3 to 16 %) | 45/70 |
| both | 5/70 | 0/70 (0 to 5 %) | 40/70 |

On the plain benchmark suite the same four settings all score 24/42 (the defences move
which tasks pass, not how many). Llama-3.2-3B tried the planted text in 5 of 70 runs with no
defence and none succeeded; the scope guard rejected 10 of its calls. The full per-task
breakdown is in [reports/injection.md](reports/injection.md).

What this does and does not show:

- Attacks that only sit in a doc section or a side diagnostics line were ignored by Qwen
  (0 of 40 attempts at first). I added tasks where the planted text is next to the fix the
  model is already applying, and only those worked. Both that effect and the defences'
  effect rest on about two or three distinct scenarios, because greedy decoding makes the
  five seeds nearly identical. Read the table as a demonstration, not a rate.
- Spotlight stopped the attacks but cost user tasks: the two fault-recovery injection tasks
  went from 7/10 to 0/10 and the benign restart task from 3/3 to 0/3. I did not find out why.
- The scope guard stops out-of-scope actions (the extra restart on another robot, 5/5
  blocked) but not an in-scope one: raising the speed limit on the same robot still got
  through 5/5. Grants here are written by hand per task; deriving them from the
  instruction is not solved.
- Nothing here has been run against Claude; that requires an Anthropic API key.

The server was checked separately over stdio ([server_check](reports/server_check.md): 19/19
scripted calls give the expected outcome). Documentation search scores Hit@5 0.672 and
MRR@10 0.526 on 61 questions with plain BM25 ([docs_search](reports/docs_search.md)).

## Repository structure

```text
agent-mcp-robotics/
├── configs/
│   ├── default.yaml          pinned corpus commit, world seed, paths
│   └── policy.yaml           tiers -> allow / ask / deny, argument limits
├── src/robomcp/
│   ├── world.py, planner.py  simulated fleet, A* on a grid with inflated keep-out areas
│   ├── download.py           ROS 2 docs + question set at pinned commits, SHA-256 recorded
│   ├── rst.py, docs.py       RST -> sections, BM25 index, snippets, paged reads
│   ├── tools.py              16 tools: tier, JSON Schema, handler
│   ├── permissions.py        policy loading and decisions
│   ├── executor.py           validate -> permit -> run -> audit
│   ├── server.py             MCP server over stdio
│   ├── check.py, docs_eval.py   server and search reports
│   ├── models.py             Claude, MLX and scripted backends, tool-call parsing
│   ├── agent.py              the tool loop, with turn and repeat-failure limits
│   ├── scope.py              per-task grant checked by the executor
│   ├── tasks.py              task setup, fault injection, planted text, attack scoring
│   ├── agent_eval.py, agent_report.py   seeded runs over MCP and their summary
│   └── injection_report.py   attack / utility tables
├── tasks/
│   ├── tasks.yaml            14 tasks: nominal, recovery, safety, docs
│   └── injection.yaml        14 injection tasks
├── reports/                  agent_tasks, injection, run logs (jsonl), server_check, docs_search
└── tests/                    fixed world, tiny synthetic docs; no network, no models
```

## Getting started

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
robomcp-download          # ROS 2 docs (text only) and the question set, ~40 MB download
robomcp-build-index       # artifacts/docs_sections.jsonl
robomcp-docs-eval         # reports/docs_search.md
robomcp-check             # reports/server_check.md
pytest -q
```

Agent runs use a local model on Apple silicon or Claude (requires an Anthropic API key):

```bash
uv pip install -e ".[mlx]"        # or ".[anthropic]"
robomcp-agent-eval --model mlx:mlx-community/Qwen2.5-7B-Instruct-4bit --name qwen2.5-7b
robomcp-agent-eval --model anthropic:claude-opus-5-5:medium --name opus
robomcp-agent-report qwen2.5-7b opus      # reports/agent_tasks.md

# injection set, with and without the defences
robomcp-agent-eval --model mlx:mlx-community/Qwen2.5-7B-Instruct-4bit --name qwen2.5-7b-both \
  --tasks tasks/injection.yaml --runs-dir reports/injection_runs --seeds 0 1 2 3 4 \
  --defence spotlight --scope
robomcp-injection-report qwen2.5-7b-none qwen2.5-7b-both    # reports/injection.md
```

To use the server from an MCP client, register it as a stdio server:

```json
{
  "mcpServers": {
    "robomcp": {
      "command": "/path/to/agent-mcp-robotics/.venv/bin/robomcp-server",
      "args": ["--project-dir", "/path/to/agent-mcp-robotics"]
    }
  }
}
```

With the shipped policy the client can read everything and stop robots; motion and
parameter changes return `APPROVAL_REQUIRED` because a stdio server has no operator
attached. `--on-ask allow` waives that for experiments; denied tiers and argument limits
still hold.

## Design decisions

- Tools are defined once and used both in-process and through MCP.
- Permission lives in the executor, outside the model. The model sees a refusal with a code
  and a reason; it never sees or edits the policy.
- Refusals and failures are ordinary tool results with `is_error` set, not protocol errors,
  so a model can read them and try something else.
- The agent loop is model-agnostic and talks to the server only through an MCP client, so a
  local model and Claude are measured through exactly the same tools and policy.

## Limitations

- The fleet is a simulator, so timing, localisation and real sensor faults are not tested.
- Local 3B and 7B models only; the injection numbers say little about larger models.
- The injection tasks are my own, so they reflect the attacks I thought of.

## License

MIT. The ROS 2 documentation is CC BY 4.0 and is downloaded, not redistributed.
