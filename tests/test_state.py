import json
from pathlib import Path

import pytest
from astrbot_plugin_qq_group_enhance.state import Settings, TrafficController

KEY = ("qq-1", "group-1")


@pytest.mark.parametrize(
    "count,level",
    [(0, "low"), (15, "low"), (16, "medium"), (30, "medium"), (31, "high")],
)
def test_thresholds_change_only_after_minute_settlement(count, level):
    controller = TrafficController(Settings())
    for _ in range(count or 1):
        group = controller.record(KEY, 0, 30)
    if not count:
        group.current_count = 0
    assert group.level == "low"
    assert sum(group.buckets) == 0
    controller.advance(group, 1, 60)
    assert group.level == level
    assert sum(group.buckets) == count
    assert group.current_count == 0


def test_roll_drops_exactly_oldest_minute_and_adds_newest():
    controller = TrafficController(Settings())
    for minute in range(11):
        for _ in range(minute + 1):
            group = controller.record(KEY, minute, minute * 60)
    assert list(group.buckets) == list(range(1, 11))
    assert group.current_count == 11
    controller.tick(11, 660)
    assert list(group.buckets) == list(range(2, 12))
    assert group.current_count == 0


def test_silent_minutes_roll_and_idle_groups_are_removed():
    controller = TrafficController(Settings())
    for _ in range(31):
        group = controller.record(KEY, 0, 10)
    controller.tick(1, 60)
    assert group.level == "high"
    controller.tick(10, 600)
    assert group.level == "high"
    assert list(group.buckets) == [31] + [0] * 9
    controller.tick(11, 660)
    assert KEY not in controller.groups


def test_delayed_timer_catches_up_without_unbounded_buckets():
    controller = TrafficController(Settings())
    group = controller.record(KEY, 0, 0)
    controller.advance(group, 50000, 3000000)
    assert len(group.buckets) == 10
    assert sum(group.buckets) == 0
    assert group.current_count == 0


def test_shared_wake_is_refreshed_by_another_member():
    controller = TrafficController(Settings())
    group = controller.record(KEY, 0, 0)
    controller.wake(group, "alice", 0)
    assert controller.allows(group, "bob", 179)
    controller.wake(group, "bob", 100)
    assert controller.allows(group, "carol", 279)
    assert not controller.allows(group, "carol", 280)


def test_ordinary_messages_do_not_extend_wake():
    controller = TrafficController(Settings())
    group = controller.record(KEY, 0, 0)
    controller.wake(group, "alice", 0)
    controller.record(KEY, 2, 179)
    assert controller.allows(group, "bob", 179)
    assert not controller.allows(group, "bob", 180)


def test_medium_wake_is_per_user_and_per_platform_group():
    controller = TrafficController(Settings())
    for _ in range(16):
        group = controller.record(KEY, 0, 0)
    controller.tick(1, 60)
    controller.wake(group, "alice", 60)
    assert controller.allows(group, "alice", 120)
    assert not controller.allows(group, "bob", 120)
    controller.wake(group, "bob", 200)
    assert not controller.allows(group, "alice", 240)
    assert controller.allows(group, "bob", 240)
    for key in [("qq-2", "group-1"), ("qq-1", "group-2")]:
        other = controller.record(key, 4, 240)
        assert not controller.allows(other, "alice", 240)


def test_low_to_medium_keeps_only_explicitly_awakened_members():
    controller = TrafficController(Settings())
    group = controller.record(KEY, 0, 0)
    controller.wake(group, "alice", 0)
    controller.wake(group, "bob", 20)
    for _ in range(15):
        controller.record(KEY, 0, 30)
    assert controller.allows(group, "carol", 40)
    controller.tick(1, 60)
    assert group.level == "medium"
    assert controller.allows(group, "alice", 60)
    assert controller.allows(group, "bob", 60)
    assert not controller.allows(group, "carol", 60)


def test_high_clears_windows_and_explicit_wake_never_leaves_a_window():
    controller = TrafficController(Settings(awake_window_seconds=1800))
    group = controller.record(KEY, 0, 0)
    controller.wake(group, "alice", 0)
    for _ in range(30):
        controller.record(KEY, 0, 10)
    controller.tick(1, 60)
    assert group.shared_until == 0
    assert not group.users
    controller.wake(group, "bob", 60)
    assert not controller.allows(group, "bob", 61)
    controller.advance(group, 11, 660)
    assert group.level == "low"
    assert not controller.allows(group, "alice", 660)


def test_medium_to_low_shares_remaining_explicit_window():
    controller = TrafficController(Settings(traffic_window_minutes=1))
    for _ in range(16):
        group = controller.record(KEY, 0, 0)
    controller.tick(1, 60)
    controller.wake(group, "alice", 90)
    controller.tick(2, 120)
    assert group.level == "low"
    assert controller.allows(group, "bob", 269)
    assert not controller.allows(group, "bob", 270)


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": "false"},
        {"traffic_window_minutes": 0},
        {"traffic_window_minutes": True},
        {"traffic_window_minutes": 1441},
        {"low_max_messages": -1},
        {"medium_max_messages": 15},
        {"keywords": "小爱"},
        {"keywords": [""]},
        {"keywords": [None]},
        {"awake_window_seconds": 0},
        {"awake_window_seconds": float("inf")},
        {"awake_window_seconds": float("nan")},
        {"count_bot_messages": 1},
        {"keyword_case_sensitive": "false"},
    ],
)
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        Settings.from_mapping(config)


def test_schema_defaults_and_empty_keyword_list():
    schema = json.loads(
        (Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    assert (
        Settings.from_mapping({key: item["default"] for key, item in schema.items()})
        == Settings()
    )
    assert Settings.from_mapping({"keywords": []}).keywords == ()
    assert Settings.from_mapping({"keywords": [" 小爱 ", "小爱"]}).keywords == ("小爱",)
