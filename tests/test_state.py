import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot_plugin_qq_group_enhance.state import (
    RETIRED,
    ChatMessage,
    Settings,
    WakeController,
)

KEY = ("qq", "123")


def message(mid):
    return ChatMessage(str(mid), "456", "Alice", f"message {mid}", 0)


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("field", sorted(RETIRED))
def test_retired_configuration_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        Settings.from_mapping({field: 10})


@pytest.mark.parametrize(
    "config",
    [
        {"semantic": {"jev_threshold": float("nan")}},
        {"semantic": {"jev_threshold": 2}},
        {"semantic": {"history_messages": True}},
        {"semantic": {"history_messages": 0}},
        {"semantic": {"debounce_seconds": 0}},
        {"semantic": {"debounce_seconds": True}},
        {"semantic": {"debounce_seconds": float("nan")}},
        {"semantic": {"max_wait_seconds": float("inf")}},
        {"semantic": {"max_wait_seconds": 0}},
        {"semantic": {"debounce_seconds": 6, "max_wait_seconds": 5}},
        {"semantic": {"max_messages_after_bot": 0}},
        {"semantic": {"max_messages_after_bot": True}},
        {"semantic": {"max_messages_after_bot": 1.5}},
        {"semantic": {"max_messages_after_bot": 1001}},
        {"semantic": {"jev_retries": -1}},
        {"semantic": {"jev_model": "jev-latest"}},
        {"unknown": 1},
        {"keyword_wake": {"keywords": [""]}},
        {"keyword_wake": {"keyword_ignore_case": "true"}},
        {"keyword_wake": None},
        {"keyword_wake": []},
        {"keyword_wake": {1: True}},
        {"keyword_wake": {"enabled": True}},
        {"keyword_wake": {"jev_api_key": "test"}},
        {"semantic": {"jev_api_key": None}},
        {"semantic": None},
        {"semantic": []},
        {"semantic": {1: True}},
        {"semantic": {"unknown": 1}},
        {"semantic": {"keywords": ["test"]}},
    ],
)
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        Settings.from_mapping(config)


def test_configuration_defaults_and_secret():
    settings = Settings.from_mapping({"semantic": {"jev_api_key": "secret-value"}})
    assert settings.keywords == ()
    assert settings.keyword_ignore_case is True
    assert settings.jev_threshold == 0.75
    assert (settings.history_messages, settings.awake_window_seconds) == (30, 180)
    assert (settings.debounce_seconds, settings.max_wait_seconds) == (3, 6)
    assert settings.max_messages_after_bot == 5
    assert (
        Settings.from_mapping({"semantic": {"max_wait_seconds": 15}}).max_wait_seconds
        == 15
    )
    assert (settings.jev_timeout_seconds, settings.jev_retries) == (10, 1)
    assert "secret-value" not in repr(settings)
    assert Settings.from_mapping({}).jev_api_key == ""
    assert settings.jev_api_url == "https://api.typesafe.ai/v1/systemone"
    assert not Settings.from_mapping({"semantic": {"enabled": False}}).enabled


@pytest.mark.parametrize(
    "url",
    [
        None,
        123,
        "",
        "  ",
        "api.typesafe.ai/v1/systemone",
        "ftp://example.com/jev",
        "https:///jev",
        "http://localhost:bad/jev",
        "http://localhost:65536/jev",
        "http://[::1/jev",
        "https://example.com/jev#fragment",
        "https://user:secret@example.com/jev",
        "https://example.com/bad path",
        "https://exam\nple.com/jev",
    ],
)
def test_invalid_jev_request_url_is_rejected(url):
    with pytest.raises(ValueError, match="jev_api_url"):
        Settings.from_mapping({"semantic": {"jev_api_url": url}})


@pytest.mark.parametrize(
    "url",
    [
        "https://proxy.example.com/custom/jev?route=group",
        "http://127.0.0.1:8080/v1/systemone",
        "http://[::1]:8080/judge",
    ],
)
def test_jev_request_url_preserves_custom_path_and_query(url):
    assert (
        Settings.from_mapping({"semantic": {"jev_api_url": f"  {url}  "}}).jev_api_url
        == url
    )


@pytest.mark.parametrize(
    "field",
    sorted(Settings.__dataclass_fields__),
)
def test_flat_configuration_requires_explicit_move(field):
    config = {field: getattr(Settings(), field), "semantic": {"enabled": False}}
    group = (
        "keyword_wake" if field in {"keywords", "keyword_ignore_case"} else "semantic"
    )
    with pytest.raises(ValueError, match=f"移入 {group} 配置分组：{field}"):
        Settings.from_mapping(config)


@pytest.mark.parametrize(
    "config",
    [
        {"keyword_case_sensitive": True},
        {"keyword_wake": {"keyword_case_sensitive": False}},
        {"keyword_wake": {"keyword_case_sensitive": True, "keyword_ignore_case": True}},
    ],
)
def test_removed_case_sensitive_setting_requires_explicit_replacement(config):
    with pytest.raises(
        ValueError, match="keyword_case_sensitive 已移除.*keyword_ignore_case"
    ):
        Settings.from_mapping(config)


def test_grouped_configuration_round_trip_and_visibility(tmp_path):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(schema) == ["keyword_wake", "semantic"]
    assert schema["keyword_wake"]["type"] == "object"
    keyword_items = schema["keyword_wake"]["items"]
    assert list(keyword_items) == ["keywords", "keyword_ignore_case"]
    assert all("condition" not in item for item in keyword_items.values())
    assert schema["semantic"]["type"] == "object"
    items = schema["semantic"]["items"]
    assert next(iter(items)) == "enabled"
    assert "condition" not in items["enabled"]
    assert all(
        item.get("condition") == {"enabled": True}
        for key, item in items.items()
        if key != "enabled"
    )

    path = str(tmp_path / "plugin.json")
    config = AstrBotConfig(path, schema=schema)
    assert Settings.from_mapping(config) == Settings()
    config["keyword_wake"]["keywords"] = ["test"]
    config["keyword_wake"]["keyword_ignore_case"] = False
    config["semantic"].update(
        enabled=False,
        jev_api_key="test-secret",
        jev_threshold=0.9,
        history_messages=45,
        jev_api_url="https://proxy.example.com/custom/jev",
    )
    config.save_config()
    reloaded = AstrBotConfig(path, schema=schema)
    settings = Settings.from_mapping(reloaded)
    assert settings == replace(
        Settings(),
        keywords=("test",),
        keyword_ignore_case=False,
        enabled=False,
        jev_api_key="test-secret",
        jev_api_url="https://proxy.example.com/custom/jev",
        jev_threshold=0.9,
        history_messages=45,
    )
    reloaded["semantic"]["enabled"] = True
    reloaded.save_config()
    assert Settings.from_mapping(AstrBotConfig(path, schema=schema)) == replace(
        settings, enabled=True
    )


async def test_debounce_window_awake_scope_and_reply_renewal():
    now = [100.0]
    c = WakeController(Settings(), AsyncMock(), AsyncMock(), lambda: now[0])
    try:
        c.add(KEY, replace(message("bot"), bot=True), False)
        c.add(KEY, message(1), True)
        group = c.groups[KEY]
        assert group.deadline == 103
        now[0] = 102
        c.add(KEY, message(2), True)
        c.awakened(KEY)
        assert group.deadline == 105
        assert group.awake_until == 282
        now[0] = 105
        c.awakened(KEY)
        assert group.awake_until == 282
        c.covered(KEY, {"1", "2"})
        c.add(KEY, message(3), True)
        assert group.deadline == 108
        now[0] = 108
        c.add(KEY, message(4), True)
        assert group.deadline == 111
        c.replied(KEY)
        assert group.awake_until == 288
        assert not c.is_awake(c.group(("other-instance", "123")))
        assert not c.is_awake(c.group(("qq", "other-group")))
        now[0] = 289
        c.replied(KEY)
        assert group.awake_until == 469
    finally:
        await c.close()


@pytest.mark.parametrize(
    "arrivals,deadline",
    [([100, 101], 104), ([100, 101, 102, 103, 104], 106)],
)
async def test_debounce_waits_for_silence_but_has_a_hard_aggregation_limit(
    arrivals, deadline
):
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(Settings(), judge, AsyncMock(), lambda: now[0])
    try:
        c.awakened(KEY)
        for i, arrived in enumerate(arrivals):
            now[0] = arrived
            c.add(KEY, message(i), True)
        group = c.groups[KEY]
        assert group.deadline == deadline
        now[0] = deadline - 0.1
        group.changed.set()
        await asyncio.sleep(0.01)
        judge.assert_not_called()
        now[0] = deadline
        group.changed.set()
        await eventually(lambda: group.worker is None)
        assert judge.await_count == 1
        assert [m["message_id"] for m in judge.call_args.args[2]] == [
            str(i) for i in range(len(arrivals))
        ]
    finally:
        await c.close()


async def test_coverage_recalculates_remaining_deadline_without_restarting_wait():
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(Settings(), judge, AsyncMock(), lambda: now[0])
    try:
        c.awakened(KEY)
        for mid, arrived in ((1, 100), (2, 101), (3, 104)):
            now[0] = arrived
            c.add(KEY, message(mid), True)
        group = c.groups[KEY]
        assert group.deadline == 106
        c.covered(KEY, {"1"})
        assert group.deadline == 107
        c.covered(KEY, {"3"})
        assert group.deadline == 104
        c.add(KEY, message("background"), False)
        assert not c.add(KEY, message(2), True)
        assert group.deadline == 104
        await eventually(lambda: group.worker is None)
        assert [m["message_id"] for m in judge.call_args.args[2]] == ["2"]
        now[0] = 110
        c.add(KEY, message(4), True)
        assert group.deadline == 113
        c.covered(KEY, {"4"})
        assert group.deadline is None
        await eventually(lambda: group.worker is None)
        assert judge.await_count == 1
    finally:
        await c.close()


async def test_slow_preparation_is_deferred_without_raw_history_or_reordering():
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(Settings(), judge, AsyncMock(), lambda: now[0])
    try:
        background = replace(message("history"), ready=False)
        c.add(KEY, background, False)
        c.awakened(KEY)
        slow = replace(message("slow"), text="", ready=False)
        c.add(KEY, slow, True)
        group = c.groups[KEY]
        now[0] = 105
        group.changed.set()
        await asyncio.sleep(0.01)
        judge.assert_not_called()
        assert not group.worker.done()
        fast = message("fast")
        c.add(KEY, fast, True)
        assert group.deadline == 108
        await asyncio.sleep(0.01)
        judge.assert_not_called()
        now[0] = 108
        group.changed.set()
        await eventually(lambda: judge.await_count == 1 and not group.inflight)
        assert judge.call_args.args[1] == []
        assert [m["message_id"] for m in judge.call_args.args[2]] == ["fast"]
        assert list(group.pending) == ["slow"]
        now[0] = 300  # Neither awake expiry nor completion order revokes eligibility.
        slow.text = "transcribed voice"
        c.prepared(KEY, slow)
        await eventually(lambda: group.worker is None)
        assert judge.await_count == 2
        assert judge.call_args.args[2][0]["text"] == "transcribed voice"
        assert [m["message_id"] for m in judge.call_args.args[1]] == ["fast"]
        assert list(group.history) == ["history", "slow", "fast"]
        assert slow.arrived_at == 100
        assert slow.sequence < fast.sequence
    finally:
        await c.close()


async def test_preparation_of_expired_batch_does_not_flush_new_debounce_batch():
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(Settings(), judge, AsyncMock(), lambda: now[0])
    try:
        c.awakened(KEY)
        slow = replace(message("slow"), ready=False)
        c.add(KEY, slow, True)
        group = c.groups[KEY]
        now[0] = 105
        group.changed.set()
        await eventually(lambda: slow.debounce_elapsed)
        assert group.deadline is None
        c.add(KEY, message("new"), True)
        assert group.deadline == 108
        c.prepared(KEY, slow)
        await eventually(lambda: judge.await_count == 1 and not group.inflight)
        assert [m["message_id"] for m in judge.call_args.args[2]] == ["slow"]
        assert list(group.pending) == ["new"]
        assert group.deadline == 108
        now[0] = 108
        group.changed.set()
        await eventually(lambda: group.worker is None)
        assert [m["message_id"] for m in judge.call_args.args[2]] == ["new"]
    finally:
        await c.close()


async def test_late_preparation_does_not_restore_covered_or_reset_messages():
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(Settings(), judge, AsyncMock(), lambda: now[0])
    try:
        c.awakened(KEY)
        covered = replace(message(1), ready=False)
        c.add(KEY, covered, True)
        group = c.groups[KEY]
        c.covered(KEY, {"1"})
        now[0] = 110
        c.prepared(KEY, covered)
        assert covered.ready and covered.status == "covered"
        assert group.deadline is None
        await eventually(lambda: group.worker is None)
        stale = replace(message(2), ready=False)
        c.add(KEY, stale, True)
        c.reset(KEY)
        c.prepared(KEY, stale)
        assert not c.groups and not stale.ready
        judge.assert_not_called()
    finally:
        await c.close()


async def test_prepared_content_enforces_pending_byte_budget(webui_logs):
    c = WakeController(Settings(), AsyncMock(), AsyncMock())
    c.MAX_PENDING_BYTES = 8
    try:
        c.awakened(KEY)
        item = replace(message(1), text="", ready=False)
        c.add(KEY, item, True)
        item.text = "long transcript"
        c.prepared(KEY, item)
        group = c.groups[KEY]
        assert not group.pending and group.deadline is None
        assert item.status == "capacity_failed"
        assert any("capacity exceeded" in entry["data"] for entry in webui_logs)
    finally:
        await c.close()


async def test_separate_history_snapshot_and_concurrent_coverage():
    started, finish = asyncio.Event(), asyncio.Event()
    calls, wakes = [], []

    async def judge(key, history, candidates, valid):
        calls.append((history, candidates))
        started.set()
        await finish.wait()
        return {m["message_id"]: 1.0 for m in candidates}

    async def wake(key, group, positive):
        wakes.append([m.id for m in positive])
        return True

    c = WakeController(replace(Settings(), debounce_seconds=0.01), judge, wake)
    try:
        c.awakened(KEY)
        for mid in range(30):
            c.add(KEY, message(mid), False)
        for mid in range(30, 70):
            c.add(KEY, message(mid), True)
        await started.wait()
        group = c.groups[KEY]
        assert len(calls[0][0]) == 30
        assert len(calls[0][1]) == 40
        assert len(group.history) == 30
        assert len(group.inflight) == 40
        for mid in range(70, 78):
            c.add(KEY, message(mid), True)
        c.covered(KEY, {str(mid) for mid in range(30, 77)})
        c.covered(KEY, {"70", "76"})
        assert list(group.pending) == ["77"]
        finish.set()
        await eventually(lambda: bool(wakes))
        assert wakes == [["77"]]
        assert len(calls) == 2
        assert [m["message_id"] for m in calls[1][1]] == ["77"]
        assert len(group.history) == 30
        assert list(group.history) == [str(i) for i in range(48, 78)]
    finally:
        await c.close()


async def test_partial_inflight_coverage_and_single_wake():
    started, finish = asyncio.Event(), asyncio.Event()
    wakes = AsyncMock(return_value=True)

    async def judge(key, history, candidates, valid):
        started.set()
        await finish.wait()
        assert not valid("1")
        return {"1": 1, "2": 0.9, "3": 1}

    c = WakeController(replace(Settings(), debounce_seconds=0.01), judge, wakes)
    try:
        c.awakened(KEY)
        for mid in (1, 2, 3):
            c.add(KEY, message(mid), True)
        await started.wait()
        c.covered(KEY, {"1"})
        finish.set()
        await eventually(lambda: c.groups[KEY].worker is None)
        assert wakes.await_count == 1
        assert [m.id for m in wakes.call_args.args[2]] == ["2", "3"]
        assert c.groups[KEY].history["1"].status == "covered"
    finally:
        await c.close()


async def test_failure_archives_without_repeating_or_blocking_other_groups():
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def judge(key, history, candidates, valid):
        calls.append((key, candidates))
        if key == KEY:
            started.set()
            await finish.wait()
            raise RuntimeError("failed")
        return {m["message_id"]: 0 for m in candidates}

    wakes = AsyncMock()
    c = WakeController(replace(Settings(), debounce_seconds=0.01), judge, wakes)
    try:
        c.awakened(KEY)
        c.add(KEY, message(1), True)
        await started.wait()
        c.add(KEY, message(2), True)
        other = ("qq", "other")
        c.awakened(other)
        c.add(other, message(3), True)
        await eventually(lambda: "3" in c.groups[other].history)
        assert len(calls) == 2
        finish.set()
        await eventually(lambda: c.groups[KEY].worker is None)
        assert len(calls) == 3
        assert [m.status for m in c.groups[KEY].history.values()] == ["failed"] * 2
        wakes.assert_not_called()
    finally:
        await c.close()


async def test_reset_cancels_old_result_and_unload_cleans_tasks():
    started = asyncio.Event()

    async def judge(*args):
        started.set()
        await asyncio.Future()

    c = WakeController(replace(Settings(), debounce_seconds=0.01), judge, AsyncMock())
    c.awakened(KEY)
    c.add(KEY, message(1), True)
    await started.wait()
    worker = c.groups[KEY].worker
    c.reset(KEY)
    c.awakened(KEY)
    c.add(KEY, message(2), True)
    await c.close()
    assert worker.cancelled()
    assert not c.groups
    assert not c.tasks


async def test_capacity_failure_is_explicit_and_preserves_normal_processing(webui_logs):
    c = WakeController(Settings(), AsyncMock(), AsyncMock())
    c.MAX_PENDING = 2
    try:
        c.awakened(KEY)
        for mid in (1, 2, 3):
            c.add(KEY, message(mid), True)
        assert list(c.groups[KEY].pending) == ["2", "3"]
        assert c.groups[KEY].history["1"].status == "capacity_failed"
        assert any("capacity exceeded" in entry["data"] for entry in webui_logs)
        assert not c.add(KEY, message(3), True)
    finally:
        await c.close()


async def test_asleep_without_bot_only_keeps_history_and_reset_removes_anchor():
    judge = AsyncMock()
    c = WakeController(Settings(), judge, AsyncMock())
    try:
        for mid in range(40):
            c.add(KEY, message(mid), True)
        group = c.groups[KEY]
        assert len(group.history) == 30
        assert not group.pending
        assert group.deadline is None
        assert group.worker is None
        assert not c.tasks
        judge.assert_not_called()
        c.add(KEY, replace(message("bot"), bot=True), False)
        for other in (("qq", "other-group"), ("other-instance", "123")):
            c.add(other, message("followup"), True)
            assert not c.groups[other].pending
        c.reset(KEY)
        c.add(KEY, message("after-reset"), True)
        assert c.groups[KEY].last_bot_sequence is None
        assert not c.groups[KEY].pending
    finally:
        await c.close()


@pytest.mark.parametrize("limit", [2, 5])
async def test_asleep_distance_boundary_dedup_history_eviction_and_batch(limit):
    now = [100.0]
    judge = AsyncMock(return_value={})
    c = WakeController(
        replace(Settings(), max_messages_after_bot=limit, history_messages=1),
        judge,
        AsyncMock(),
        lambda: now[0],
    )
    try:
        c.add(KEY, replace(message("bot"), bot=True), False)
        group = c.groups[KEY]
        c.add(KEY, message(1), True)
        assert not c.add(KEY, message(1), True)
        assert not c.add(KEY, replace(message("bot"), bot=True), False)
        for mid in range(2, limit + 3):
            c.add(KEY, message(mid), True)
        assert list(group.pending) == [str(i) for i in range(1, limit + 1)]
        assert group.sequence == limit + 3
        assert "bot" not in group.history
        assert group.deadline == 103
        assert not c.is_awake(group)
        # Exceeding the distance does not revoke candidates already admitted.
        now[0] = 105
        group.changed.set()
        await eventually(lambda: group.worker is None)
        assert judge.await_count == 1
        assert [m["message_id"] for m in judge.call_args.args[2]] == [
            str(i) for i in range(1, limit + 1)
        ]
        c.add(KEY, message("late"), True)
        assert group.worker is None
        c.add(KEY, replace(message("new-bot"), bot=True), False)
        c.add(KEY, message("new-followup"), True)
        assert list(group.pending) == ["new-followup"]
    finally:
        await c.close()


async def test_awake_bypasses_distance_but_expiry_does_not_reset_count():
    now = [100.0]
    c = WakeController(Settings(), AsyncMock(), AsyncMock(), lambda: now[0])
    try:
        c.awakened(KEY)
        c.add(KEY, message("before-bot"), True)
        assert "before-bot" in c.groups[KEY].pending
        c.add(KEY, replace(message("bot"), bot=True), False)
        for mid in range(1, 8):
            c.add(KEY, message(mid), True)
        group = c.groups[KEY]
        assert len(group.pending) == 8
        now[0] = 280
        c.add(KEY, message(8), True)
        assert "8" not in group.pending
        assert len(group.pending) == 8
        c.add(KEY, replace(message("new-bot"), bot=True), False)
        c.add(KEY, message(9), True)
        assert "9" in group.pending
        assert not c.is_awake(group)
    finally:
        await c.close()


async def test_background_commands_and_covered_messages_consume_distance():
    c = WakeController(Settings(), AsyncMock(), AsyncMock())
    try:
        c.add(KEY, replace(message("bot"), bot=True), False)
        c.add(KEY, message("background"), False)
        c.record_arrival(KEY, "command", False)
        assert c.record_arrival(KEY, "command", False) is None
        c.add(KEY, message("covered"), True)
        c.covered(KEY, {"covered"})
        c.add(KEY, replace(message("image"), text="[图片]"), True)
        c.add(KEY, message("fifth"), True)
        c.add(KEY, message("sixth"), True)
        assert list(c.groups[KEY].pending) == ["image", "fifth"]
    finally:
        await c.close()
