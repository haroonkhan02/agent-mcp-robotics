import pytest

from robomcp.world import FAULT_NODE, LOCATIONS, RESERVE, Robot, World, WorldError


def make_world(**robot):
    defaults = dict(id="amr_1", x=1.0, y=1.0, at="dock_1", battery=80.0,
                    params={"max_speed": 1.0, "inflation_radius": 0.35, "goal_tolerance": 0.25})
    r = Robot(**{**defaults, **robot})
    return World([r], {"shelf_A": ["gripper", "wheel_kit"], "shelf_B": [], "shelf_C": [],
                       "shelf_D": [], "shelf_E": []}), r


def code(fn, *args):
    with pytest.raises(WorldError) as e:
        fn(*args)
    return e.value.code


def test_same_seed_same_world():
    assert World.generate(3).snapshot() == World.generate(3).snapshot()
    assert World.generate(3).snapshot() != World.generate(4).snapshot()


def test_generated_world_is_consistent():
    for seed in range(30):
        w = World.generate(seed, n_robots=4)
        spots = [r.at for r in w.robots.values()]
        assert len(set(spots)) == 4
        assert sorted(i for items in w.shelves.values() for i in items) == sorted(
            {i for items in w.shelves.values() for i in items})  # every item exactly once
        for r in w.robots.values():
            assert r.charging == (LOCATIONS[r.at][2] == "dock")
            if r.fault:
                assert any(d["level"] == "ERROR" and r.fault in d["message"] for d in r.diagnostics)


def test_navigate_moves_drains_and_takes_time():
    w, r = make_world()
    out = w.navigate("amr_1", "shelf_A")
    assert out["arrived"] and r.at == "shelf_A" and (r.x, r.y) == (6.0, 2.0)
    assert out["distance_m"] >= 5.09  # straight-line distance
    assert r.battery == pytest.approx(80.0 - out["battery_needed_pct"])
    assert w.clock == pytest.approx(out["duration_s"])
    assert w.events[-1]["kind"] == "navigate"


def test_slow_zone_limits_speed():
    w, _ = make_world(x=12.0, y=2.0, at="shelf_D")
    plan = w.route("amr_1", "packing_1")
    assert plan["duration_s"] > plan["distance_m"] / 1.0 + 1.0  # part of it at 0.5 m/s


def test_route_does_not_change_the_world():
    w, _ = make_world()
    before = w.snapshot()
    w.route("amr_1", "inspection")
    assert w.snapshot() == before


def test_refusals():
    w, r = make_world()
    assert code(w.navigate, "amr_9", "shelf_A") == "UNKNOWN_ROBOT"
    assert code(w.navigate, "amr_1", "nowhere") == "UNKNOWN_LOCATION"
    assert code(w.navigate, "amr_1", "maintenance_bay") == "GOAL_IN_KEEP_OUT"
    r.estopped = True
    assert code(w.navigate, "amr_1", "shelf_A") == "ESTOPPED"
    r.estopped, r.fault = False, "LIDAR_TIMEOUT"
    assert code(w.navigate, "amr_1", "shelf_A") == "FAULT"


def test_low_battery_refused_and_nothing_changes():
    w, r = make_world(battery=RESERVE + 2.0)
    before = w.snapshot()
    assert code(w.navigate, "amr_1", "inspection") == "LOW_BATTERY"
    assert w.snapshot() == before


def test_wide_inflation_closes_the_doorway():
    w, _ = make_world()
    w.set_param("amr_1", "inflation_radius", 0.7)
    assert code(w.navigate, "amr_1", "shelf_D") == "NO_PATH"
    w.set_param("amr_1", "inflation_radius", 0.35)
    assert w.navigate("amr_1", "shelf_D")["arrived"]


def test_set_param_validates():
    w, r = make_world()
    assert code(w.set_param, "amr_1", "gain", 1.0) == "UNKNOWN_PARAM"
    assert code(w.set_param, "amr_1", "max_speed", 9.0) == "PARAM_OUT_OF_RANGE"
    assert w.set_param("amr_1", "max_speed", 0.5) == {
        "robot": "amr_1", "name": "max_speed", "old": 1.0, "new": 0.5}


def test_pick_carry_drop():
    w, r = make_world()
    assert code(w.pick_item, "amr_1", "gripper") == "NOT_AT_SHELF"
    w.navigate("amr_1", "shelf_A")
    assert code(w.pick_item, "amr_1", "lidar_unit") == "ITEM_NOT_HERE"
    w.pick_item("amr_1", "gripper")
    assert code(w.pick_item, "amr_1", "wheel_kit") == "ALREADY_LOADED"
    w.drop_item("amr_1")  # putting it back on a shelf is allowed
    assert w.shelves["shelf_A"] == ["gripper", "wheel_kit"] and r.payload is None
    w.pick_item("amr_1", "gripper")
    w.navigate("amr_1", "packing_1")
    w.drop_item("amr_1")
    assert w.delivered["packing_1"] == ["gripper"] and w.shelves["shelf_A"] == ["wheel_kit"]
    assert code(w.drop_item, "amr_1") == "NO_PAYLOAD"


def test_dock_charges_and_respects_occupancy():
    w, r = make_world(x=6.0, y=2.0, at="shelf_A", battery=40.0)
    other = Robot(id="amr_2", x=1.0, y=3.0, at="dock_2", params=dict(r.params))
    w.robots["amr_2"] = other
    assert code(w.dock, "amr_1", "shelf_B") == "NOT_A_DOCK"
    assert code(w.dock, "amr_1", "dock_2") == "DOCK_OCCUPIED"
    out = w.dock("amr_1", "dock_1")
    assert r.battery == 100.0 and r.status == "charging"
    assert out["charge_duration_s"] > 0 and w.clock >= out["charge_duration_s"]


@pytest.mark.parametrize("fault,node", sorted(FAULT_NODE.items()))
def test_restart_clears_only_the_matching_fault(fault, node):
    w, r = make_world(fault=fault)
    wrong = next(n for n in FAULT_NODE.values() if n != node)
    assert w.restart_node("amr_1", wrong)["remaining_fault"] == fault
    assert w.restart_node("amr_1", node)["cleared_fault"] == fault
    assert r.status in ("idle", "charging")
    assert code(w.restart_node, "amr_1", "teleop") == "UNKNOWN_NODE"


def test_emergency_stop_and_release():
    w, r = make_world(fault="LIDAR_TIMEOUT")
    assert code(w.clear_emergency_stop, "amr_1") == "NOT_ESTOPPED"
    assert w.emergency_stop("amr_1")["status"] == "estopped"
    assert w.emergency_stop("amr_1")["was_already_stopped"] is True
    assert w.clear_emergency_stop("amr_1")["status"] == "fault"  # the fault is still there
