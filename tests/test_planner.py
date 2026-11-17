import math

from robomcp.planner import path_length, plan_path

SIZE = (10.0, 6.0)


def test_straight_line_without_obstacles():
    path = plan_path((0.75, 0.75), (8.75, 0.75), SIZE)
    assert path[0] == (0.75, 0.75) and path[-1] == (8.75, 0.75)
    assert path_length(path) == 8.0


def test_same_cell_is_a_direct_hop():
    assert plan_path((1.1, 1.1), (1.3, 1.2), SIZE) == [(1.1, 1.1), (1.3, 1.2)]


def test_detours_around_a_keep_out_area():
    wall = (4.0, 0.0, 5.0, 4.0)  # leaves a gap at the top
    path = plan_path((1.25, 1.25), (8.25, 1.25), SIZE, [wall])
    assert path is not None
    assert path_length(path) > 7.0 + 2.0
    assert not any(4.0 <= x <= 5.0 and y <= 4.0 for x, y in path)


def test_goal_inside_keep_out_is_unreachable():
    assert plan_path((1.0, 1.0), (4.5, 2.0), SIZE, [(4.0, 0.0, 5.0, 4.0)]) is None


def test_inflation_closes_a_narrow_gap():
    walls = [(4.0, 0.0, 5.0, 2.6), (4.0, 4.0, 5.0, 6.0)]  # 1.4 m doorway
    assert plan_path((1.0, 3.25), (8.0, 3.25), SIZE, walls, inflation=0.35) is not None
    assert plan_path((1.0, 3.25), (8.0, 3.25), SIZE, walls, inflation=0.7) is None


def test_diagonals_do_not_cut_corners():
    block = (2.0, 2.0, 2.5, 2.5)  # blocks exactly cell (4, 4)
    path = plan_path((1.75, 2.25), (2.25, 2.75), SIZE, [block])
    # the direct diagonal would clip the blocked cell's corner
    assert path_length(path) > math.sqrt(2) * 0.5 + 1e-9
