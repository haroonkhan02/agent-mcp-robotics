"""Grid path planner for the simulated warehouse: A* on an 8-connected grid.

Keep-out rectangles are grown by the robot's inflation radius before cells are blocked,
so a larger radius can close a gap that a smaller one fits through. Diagonal moves are
not allowed to cut a blocked corner.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

Point = tuple[float, float]
Rect = tuple[float, float, float, float]  # x0, y0, x1, y1
Cell = tuple[int, int]

CELL = 0.5  # metres
_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def in_rect(p: Point, r: Rect, margin: float = 0.0) -> bool:
    return r[0] - margin <= p[0] <= r[2] + margin and r[1] - margin <= p[1] <= r[3] + margin


def to_cell(p: Point) -> Cell:
    return (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)))


def centre(c: Cell) -> Point:
    return ((c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL)


def path_length(path: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))


def plan_path(
    start: Point,
    goal: Point,
    size: tuple[float, float],
    keep_out: Sequence[Rect] = (),
    inflation: float = 0.0,
) -> list[Point] | None:
    """Shortest grid path from start to goal, or None when the goal cannot be reached.

    The path starts at `start`, ends at `goal` and passes through cell centres in between.
    """
    nx, ny = int(math.ceil(size[0] / CELL)), int(math.ceil(size[1] / CELL))

    def blocked(c: Cell) -> bool:
        if not (0 <= c[0] < nx and 0 <= c[1] < ny):
            return True
        p = centre(c)
        return any(in_rect(p, r, inflation) for r in keep_out)

    s, g = to_cell(start), to_cell(goal)
    if blocked(g) or any(in_rect(goal, r, inflation) for r in keep_out):
        return None
    if s == g:
        return [start, goal]

    def h(c: Cell) -> float:
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)) * CELL  # octile distance

    best: dict[Cell, float] = {s: 0.0}
    parent: dict[Cell, Cell] = {}
    heap: list[tuple[float, float, Cell]] = [(h(s), 0.0, s)]
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == g:
            cells = [cur]
            while cells[-1] in parent:
                cells.append(parent[cells[-1]])
            inner = [centre(c) for c in reversed(cells)][1:-1]
            return [start, *inner, goal]
        if cost > best.get(cur, math.inf):
            continue
        for dx, dy in _MOVES:
            nxt = (cur[0] + dx, cur[1] + dy)
            if blocked(nxt):
                continue
            if dx and dy and (blocked((cur[0] + dx, cur[1])) or blocked((cur[0], cur[1] + dy))):
                continue
            new = cost + math.hypot(dx, dy) * CELL
            if new < best.get(nxt, math.inf):
                best[nxt] = new
                parent[nxt] = cur
                heapq.heappush(heap, (new + h(nxt), new, nxt))
    return None
