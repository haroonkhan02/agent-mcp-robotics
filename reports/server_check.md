# Server check over MCP (stdio)

The server runs as a subprocess; every call below is a real MCP round trip (protocol 2026-07-28, world seed 7). 19/19 scripted calls returned the expected outcome.

## Tools (16)

| Tier | Tools |
|---|---|
| read | list_robots, get_robot_state, list_locations, get_params, read_diagnostics, plan_route, search_docs, read_doc_section |
| stop | emergency_stop |
| config | set_param, restart_node |
| motion | navigate_to, dock, pick_item, drop_item |
| release | clear_emergency_stop |

## Shipped policy, no operator attached

| Call | Expected | Got | Why |
|---|---|---|---|
| `list_robots()` | ok | ok | read tier |
| `get_robot_state()` | ok | ok | read tier |
| `read_diagnostics(limit=5)` | ok | ok | read tier |
| `plan_route(goal=packing_1)` | ok | ok | planning moves nothing |
| `plan_route(goal=maintenance_bay)` | GOAL_IN_KEEP_OUT | GOAL_IN_KEEP_OUT | the world refuses, whatever the policy says |
| `emergency_stop()` | ok | ok | stopping is always allowed |
| `clear_emergency_stop()` | PERMISSION_DENIED | PERMISSION_DENIED | release tier is denied |
| `navigate_to(goal=shelf_B)` | APPROVAL_REQUIRED | APPROVAL_REQUIRED | motion needs approval, no operator attached |
| `set_param(name=max_speed, value=0.5)` | APPROVAL_REQUIRED | APPROVAL_REQUIRED | config needs approval |
| `set_param(name=inflation_radius, value=0.3)` | PERMISSION_DENIED | PERMISSION_DENIED | parameter outside the policy's allow-list |
| `navigate_to()` | INVALID_ARGUMENTS | INVALID_ARGUMENTS | missing argument |
| `navigate_to(goal=shelf_B, speed=3)` | INVALID_ARGUMENTS | INVALID_ARGUMENTS | unknown argument |
| `fly_to()` | UNKNOWN_TOOL | UNKNOWN_TOOL | tool does not exist |

## Same policy with `--on-ask allow`

| Call | Expected | Got | Why |
|---|---|---|---|
| `pick_item(item=battery_pack)` | ok | ok | load |
| `navigate_to(goal=packing_1)` | ok | ok | through the doorway |
| `drop_item()` | ok | ok | unload |
| `drop_item()` | NO_PAYLOAD | NO_PAYLOAD | nothing left to drop |
| `set_param(name=inflation_radius, value=0.7)` | PERMISSION_DENIED | PERMISSION_DENIED | hard limit holds even when approval is waived |
| `clear_emergency_stop()` | PERMISSION_DENIED | PERMISSION_DENIED | denied tier holds even when approval is waived |

Delivery: amr_1 carried `battery_pack`; packing_1 now holds ['battery_pack']. Audit log: 9 records for 9 calls, decisions seen: allowed, denied, unattended.

## Latency per call

| Tool | Calls | p50 ms | p95 ms |
|---|---|---|---|
| list_robots | 100 | 0.5 | 0.56 |
| plan_route | 100 | 1.33 | 1.37 |
| search_docs | 61 | 2.98 | 3.59 |

Server start to first response: 299.6 ms.

## Docs search through the tool

61 questions through `search_docs` (k = 10), scored by gold page: Hit@1 0.574, Hit@3 0.705, Hit@5 0.754, Hit@10 0.820. Identical to the in-process evaluation in docs_search.md.
