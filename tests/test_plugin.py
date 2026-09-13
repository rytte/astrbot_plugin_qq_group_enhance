import asyncio
from copy import copy, deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot.api.message_components import At, AtAll, Image, Plain, Reply
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.stage import ProcessStage, StarRequestSubStage
from astrbot.core.pipeline.session_status_check.stage import SessionStatusCheckStage
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.pipeline.whitelist_check.stage import WhitelistCheckStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.session_plugin_manager import SessionPluginManager
from astrbot.core.star.star_handler import star_handlers_registry
from astrbot_plugin_qq_group_enhance import main
from astrbot_plugin_qq_group_enhance.main import GroupWakeFilter, QQGroupEnhancePlugin
from astrbot_plugin_qq_group_enhance.state import Settings


class LocalEvent(AstrMessageEvent):
    async def send(self, message):
        self._has_send_oper = True


def make_event(
    text="普通消息",
    sender="alice",
    group="group-1",
    platform="qq-1",
    parts=None,
    private=False,
    post_type="message",
):
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE if private else MessageType.GROUP_MESSAGE
    message.self_id = "bot-1"
    message.group_id = "" if private else group
    message.session_id = sender if private else group
    message.sender = MessageMember(user_id=sender, nickname=sender)
    message.message = parts if parts is not None else [Plain(text=text)]
    message.message_str = text
    message.message_id = "test-message"
    message.raw_message = {"post_type": post_type}
    return LocalEvent(
        text,
        message,
        PlatformMetadata("aiocqhttp", "test", id=platform),
        message.session_id,
    )


@pytest.fixture
async def harness(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        main,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now),
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"]["ignore_bot_self_message"] = False
    plugin_context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conv")
        ),
    )
    plugin = QQGroupEnhancePlugin(plugin_context)
    ctx = SimpleNamespace(
        astrbot_config=config,
        astrbot_config_id="test",
        db_helper=None,
        plugin_manager=SimpleNamespace(context=plugin_context),
    )
    handlers = []
    for registered in star_handlers_registry.get_handlers_by_module_name(main.__name__):
        handler = copy(registered)
        handler.handler = getattr(plugin, handler.handler_name)
        handlers.append(handler)
    monkeypatch.setattr(
        star_handlers_registry,
        "get_handlers_by_event_type",
        lambda *args, **kwargs: handlers,
    )
    monkeypatch.setattr(
        SessionPluginManager,
        "filter_handlers_by_session",
        AsyncMock(side_effect=lambda event, handlers: handlers),
    )
    monkeypatch.setattr(
        SessionServiceManager, "is_session_enabled", AsyncMock(return_value=True)
    )
    waking = WakingCheckStage()
    await waking.initialize(ctx)
    waking._umo_auto_name_recorder = MagicMock()
    whitelist = WhitelistCheckStage()
    await whitelist.initialize(ctx)
    session = SessionStatusCheckStage()
    await session.initialize(ctx)
    requests = []

    async def request(event):
        requests.append(event)
        yield

    process = ProcessStage()
    process.ctx = ctx
    process.agent_sub_stage = SimpleNamespace(process=request)
    process.star_request_sub_stage = StarRequestSubStage()
    await process.star_request_sub_stage.initialize(ctx)

    async def run(event):
        await waking.process(event)
        for stage in (whitelist, session):
            if not event.is_stopped():
                await stage.process(event)
        if not event.is_stopped():
            async for _ in process.process(event):
                pass
        return event

    fixture = SimpleNamespace(
        plugin=plugin,
        clock=clock,
        config=config,
        handlers=handlers,
        waking=waking,
        whitelist=whitelist,
        requests=requests,
        run=run,
    )
    yield fixture
    await plugin.terminate()


async def test_keyword_in_middle_keeps_full_text_and_reaches_default_llm(harness):
    event = make_event("有人知道小爱今天在不在吗？")
    original_parts = list(event.get_messages())
    await harness.run(event)
    assert harness.requests == [event]
    assert event.message_str == "有人知道小爱今天在不在吗？"
    assert event.message_obj.message_str == event.message_str
    assert event.get_messages() == original_parts
    assert harness.plugin.controller.groups[("qq-1", "group-1")].shared_until == 180


@pytest.mark.parametrize("case_sensitive", [True, False])
@pytest.mark.parametrize(
    "keyword,text,sensitive_match,insensitive_match",
    [
        ("Airi", "你好 Airi", True, True),
        ("Airi", "你好 airi", False, True),
        ("airi", "你好 AIRI", False, True),
        ("AIRI", "你好 AiRi", False, True),
        ("Straße", "你好 STRASSE", False, True),
        ("小爱", "小爱，你好", True, True),
        ("Airi", "你好 Airo", False, False),
    ],
)
async def test_keyword_case_switch_preserves_original_input(
    harness, case_sensitive, keyword, text, sensitive_match, insensitive_match
):
    settings = Settings.from_mapping(
        {"keywords": [keyword], "keyword_case_sensitive": case_sensitive}
    )
    harness.plugin.settings = settings
    harness.plugin.controller.settings = settings
    event = make_event(text)
    await harness.run(event)
    expected = sensitive_match if case_sensitive else insensitive_match
    assert harness.requests == ([event] if expected else [])
    assert event.message_str == text
    assert event.message_obj.message_str == text
    assert event.get_messages()[0].text == text
    assert harness.plugin.settings.keywords == (keyword,)


async def test_low_shared_window_refresh_and_expiry(harness):
    await harness.run(make_event("小爱你好"))
    harness.clock.now = 100
    await harness.run(make_event("小爱", sender="bob"))
    harness.clock.now = 279
    await harness.run(make_event(sender="carol"))
    assert len(harness.requests) == 3
    harness.clock.now = 280
    await harness.run(make_event(sender="carol"))
    assert len(harness.requests) == 3


async def test_non_waking_messages_are_counted_and_transition_on_minute(harness):
    for _ in range(16):
        await harness.run(make_event())
    assert not harness.requests
    group = harness.plugin.controller.groups[("qq-1", "group-1")]
    assert group.current_count == 16
    assert group.level == "low"
    harness.clock.now = 60
    await harness.run(make_event("小爱"))
    assert group.level == "medium"
    await harness.run(make_event(sender="bob"))
    await harness.run(make_event(sender="alice"))
    assert [event.get_sender_id() for event in harness.requests] == ["alice", "alice"]


async def test_unique_session_does_not_split_group_traffic_or_shared_window(harness):
    harness.waking.unique_session = True
    await harness.run(make_event("小爱"))
    await harness.run(make_event(sender="bob"))
    assert len(harness.requests) == 2
    assert (
        harness.requests[0].unified_msg_origin != harness.requests[1].unified_msg_origin
    )
    assert len(harness.plugin.controller.groups) == 1
    await harness.run(make_event(group="group-2"))
    await harness.run(make_event(platform="qq-2"))
    assert len(harness.requests) == 2


async def test_high_requires_explicit_keyword_or_actual_bot_mention(harness):
    for _ in range(31):
        await harness.run(make_event())
    harness.clock.now = 60
    await harness.run(make_event("小爱"))
    await harness.run(make_event())
    await harness.run(make_event("你好", parts=[At(qq="bot-1"), Plain("你好")]))
    await harness.run(make_event())
    assert len(harness.requests) == 2
    group = harness.plugin.controller.groups[("qq-1", "group-1")]
    assert group.level == "high"
    assert not group.users
    assert group.shared_until == 0


@pytest.mark.parametrize(
    "parts",
    [
        [AtAll(), Plain("你好")],
        [Reply(id="quote", sender_id="bot-1", chain=[Plain("小爱")]), Plain("你好")],
        [At(qq="someone-else", name="小爱"), Plain("你好")],
    ],
)
async def test_quotes_and_other_mentions_do_not_implicitly_wake(harness, parts):
    event = make_event("你好", parts=parts)
    await harness.run(event)
    assert not harness.requests
    assert not event.is_at_or_wake_command


async def test_bot_mention_opens_window_even_without_text(harness):
    await harness.run(make_event("", parts=[At(qq="bot-1")]))
    await harness.run(make_event(sender="bob"))
    assert len(harness.requests) == 2


async def test_image_can_continue_an_awake_conversation(harness):
    await harness.run(make_event("小爱"))
    image = make_event("", sender="bob", parts=[Image(file="test.png")])
    await harness.run(image)
    assert harness.requests[-1] is image


async def test_self_messages_neither_count_nor_wake_by_default(harness):
    event = make_event("小爱", sender="bot-1")
    await harness.run(event)
    assert not harness.requests
    assert not harness.plugin.controller.groups


async def test_self_count_option_never_enables_self_replies(harness):
    from astrbot_plugin_qq_group_enhance.state import Settings, TrafficController

    harness.plugin.settings = Settings(count_bot_messages=True)
    harness.plugin.controller = TrafficController(harness.plugin.settings)
    event = make_event("/小爱", sender="bot-1", post_type="message_sent")
    await harness.run(event)
    assert not harness.requests
    assert not event.is_at_or_wake_command
    assert harness.plugin.controller.groups[("qq-1", "group-1")].current_count == 1


@pytest.mark.parametrize(
    "kwargs", [{"private": True}, {"post_type": "notice"}, {"post_type": "request"}]
)
async def test_private_and_non_message_events_are_not_managed(harness, kwargs):
    event = make_event("小爱", **kwargs)
    await harness.run(event)
    assert not harness.plugin.controller.groups
    assert event.get_extra(main.DECISION_KEY) is None


async def test_filter_is_idempotent_without_promoting_rejected_event(harness):
    event = make_event()
    wake_filter = GroupWakeFilter()
    assert wake_filter.filter(event, harness.config) is False
    assert wake_filter.filter(event, harness.config) is False
    assert harness.plugin.controller.groups[("qq-1", "group-1")].current_count == 1


async def test_whitelist_blocks_llm_and_window_creation(harness):
    harness.whitelist.enable_whitelist_check = True
    harness.whitelist.whitelist = ["group-2"]
    await harness.run(make_event("小爱"))
    assert not harness.requests
    group = harness.plugin.controller.groups[("qq-1", "group-1")]
    assert not group.shared_until
    assert not group.users


async def test_disabled_session_does_not_create_wake_window(harness, monkeypatch):
    monkeypatch.setattr(
        SessionServiceManager, "is_session_enabled", AsyncMock(return_value=False)
    )
    await harness.run(make_event("小爱"))
    assert not harness.requests
    assert not harness.plugin.controller.groups[("qq-1", "group-1")].shared_until


async def test_session_plugin_disable_prevents_wake_side_effects(harness, monkeypatch):
    monkeypatch.setattr(
        SessionPluginManager, "filter_handlers_by_session", AsyncMock(return_value=[])
    )
    await harness.run(make_event("小爱"))
    assert not harness.requests
    assert not harness.plugin.controller.groups[("qq-1", "group-1")].shared_until


async def test_recognized_commands_still_execute_without_opening_window(harness):
    command_calls = []

    async def command_handler(event):
        command_calls.append(event.message_str)
        await event.send(event.plain_result("指令完成"))

    handler = copy(harness.handlers[0])
    handler.handler = command_handler
    handler.event_filters = [
        CommandFilter(
            "help", handler_md=SimpleNamespace(handler=lambda self, event: None)
        )
    ]
    harness.handlers.append(handler)
    await harness.run(make_event("/help"))
    assert command_calls == ["help"]
    assert not harness.requests
    assert not harness.plugin.controller.groups[("qq-1", "group-1")].shared_until


async def test_plugin_disable_and_terminate_restore_core_behavior(harness):
    from astrbot_plugin_qq_group_enhance.state import Settings

    harness.plugin.settings = Settings(enabled=False)
    await harness.run(make_event("/你好"))
    assert len(harness.requests) == 1
    assert not harness.plugin.controller.groups
    await harness.plugin.terminate()
    assert GroupWakeFilter.plugin is None


async def test_minute_task_rolls_silent_group_and_stops_on_unload(harness, monkeypatch):
    await harness.run(make_event("小爱"))
    calls = []
    entered = asyncio.Event()

    async def sleep(delay):
        calls.append(delay)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(
        main,
        "asyncio",
        SimpleNamespace(
            create_task=asyncio.create_task,
            sleep=sleep,
            CancelledError=asyncio.CancelledError,
        ),
    )
    harness.clock.now = 60
    await harness.plugin.initialize()
    await entered.wait()
    assert calls == [60]
    assert sum(harness.plugin.controller.groups[("qq-1", "group-1")].buckets) == 1
    task = harness.plugin.timer
    await harness.plugin.terminate()
    assert task.cancelled()
    assert not harness.plugin.controller.groups


async def test_actual_qq_debouncer_accepts_new_wake_flag(harness):
    from astrbot_plugin_qq_enhance.debounce import (
        ARRIVAL_KEY,
        ArrivalFilter,
        MessageDebouncer,
    )

    qq = SimpleNamespace(
        config={
            "platform": {"platform_id": ""},
            "debounce": {
                "enabled": True,
                "shared_group": True,
                "initial_window_seconds": 0,
                "max_wait_seconds": 5,
                "ignore_prefixes": ["/", "!"],
            },
        },
        context=SimpleNamespace(
            get_config=lambda **kw: harness.config, get_all_stars=lambda: []
        ),
    )
    debouncer = MessageDebouncer(qq)
    seen = []

    async def debounce_handler(event):
        await debouncer.capture(event)
        seen.append(event.get_extra(ARRIVAL_KEY).batch)

    handler = copy(harness.handlers[0])
    handler.handler = debounce_handler
    handler.event_filters = [ArrivalFilter()]
    harness.handlers.append(handler)
    event = make_event("小爱，你好")
    await asyncio.create_task(harness.run(event))
    assert seen == [[event.get_extra(ARRIVAL_KEY)]]
    assert harness.requests == [event]
