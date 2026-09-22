import asyncio
import itertools
import json
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot.api.message_components import At, Face, Plain, Record, Reply
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.core.agent.message import Message
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.event_bus import EventBus
from astrbot.core.message.message_event_result import MessageChain, ResultContentType
from astrbot.core.pipeline.context import call_event_hook
from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage
from astrbot.core.pipeline.process_stage.stage import (
    AgentRequestSubStage,
    ProcessStage,
    StarRequestSubStage,
)
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.pipeline.session_status_check.stage import SessionStatusCheckStage
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.pipeline.whitelist_check.stage import WhitelistCheckStage
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.session_plugin_manager import SessionPluginManager
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from astrbot.core.utils.active_event_registry import (
    ActiveEventRegistry,
    active_event_registry,
)
from astrbot_plugin_qq_group_enhance import main
from astrbot_plugin_qq_group_enhance.bridge import ACTIVE_RUN, fingerprint
from astrbot_plugin_qq_group_enhance.main import (
    BOT_MENTION_KEY,
    RECORD_KEY,
    GroupWakeFilter,
    QQGroupEnhancePlugin,
)
from astrbot_plugin_qq_group_enhance.reply import REPLY_KEY

IDS = itertools.count(1)
KEY = ("qq-1", "100")


class Bot:
    def __init__(self):
        self.calls = []
        self.fail_at = None

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated OneBot failure")
        return {"message_id": 10000 + len(self.calls)}

    async def send_group_msg(self, **params):
        return await self.call_action("send_group_msg", **params)


def make_event(
    text="普通消息",
    *,
    sender="200",
    group="100",
    platform="qq-1",
    parts=None,
    mid=None,
    private=False,
    post_type="message",
    bot=None,
):
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE if private else MessageType.GROUP_MESSAGE
    message.self_id = "300"
    message.group_id = "" if private else group
    message.session_id = sender if private else group
    message.sender = MessageMember(user_id=sender, nickname=f"用户{sender}")
    message.message = parts if parts is not None else [Plain(text)]
    message.message_str = text
    message.message_id = str(mid or next(IDS))
    message.raw_message = {"post_type": post_type}
    return AiocqhttpMessageEvent(
        text,
        message,
        PlatformMetadata("aiocqhttp", "test", id=platform),
        message.session_id,
        bot or Bot(),
    )


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.fixture
async def harness(monkeypatch, request):
    monkeypatch.setattr(active_event_registry, "_events", defaultdict(set))
    monkeypatch.setattr(active_event_registry, "_agent_stop_callbacks", {})
    config = deepcopy(DEFAULT_CONFIG)
    config["platform_settings"]["ignore_bot_self_message"] = False
    config["platform_settings"]["unique_session"] = False
    config["platform_settings"]["segmented_reply"]["enable"] = False
    config["provider_ltm_settings"]["group_icl_enable"] = True
    config["provider_ltm_settings"]["image_caption"] = False
    config["provider_settings"]["wake_prefix"] = ""
    queue = asyncio.Queue()
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation")
        ),
        get_config=lambda **kwargs: config,
        get_event_queue=lambda: queue,
        get_using_tts_provider_async=AsyncMock(return_value=None),
    )
    plugin = QQGroupEnhancePlugin(
        context,
        getattr(
            request,
            "param",
            {
                "keyword_wake": {"keywords": ["小爱"]},
                "semantic": {"jev_api_key": "test"},
            },
        ),
    )
    if plugin.client is not None:
        plugin.client._send = AsyncMock(
            side_effect=AssertionError("unmocked Jev request")
        )
    clock = SimpleNamespace(now=100.0)
    if plugin.controller is not None:
        plugin.controller.clock = lambda: clock.now
    ctx = SimpleNamespace(
        astrbot_config=config,
        astrbot_config_id="test",
        db_helper=None,
        plugin_manager=SimpleNamespace(context=context),
    )
    native = GroupChatContext(None, context)
    handlers = []
    for registered in star_handlers_registry.get_handlers_by_module_name(main.__name__):
        handler = copy(registered)
        handler.handler = getattr(plugin, handler.handler_name)
        handlers.append(handler)
    adapter = next(h for h in handlers if h.handler_name == "observe_message")
    native_handler = copy(adapter)
    native_handler.handler_name = "native_record"

    async def native_record(event):
        await native.handle_message(event)

    native_handler.handler = native_record
    native_handler.extras_configs = {"priority": 0}
    handlers.append(native_handler)

    def get_handlers(event_type, **kwargs):
        selected = handlers
        names = kwargs.get("plugins_name")
        if names is not None and "*" not in names and main.PLUGIN_NAME not in names:
            selected = []
        return sorted(
            [h for h in selected if h.event_type == event_type],
            key=lambda h: -h.extras_configs.get("priority", 0),
        )

    monkeypatch.setattr(
        star_handlers_registry, "get_handlers_by_event_type", get_handlers
    )
    monkeypatch.setattr(
        SessionPluginManager,
        "filter_handlers_by_session",
        AsyncMock(side_effect=lambda event, handlers: handlers),
    )
    monkeypatch.setattr(
        SessionPluginManager,
        "is_plugin_enabled_for_session",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        SessionServiceManager, "is_session_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        SessionServiceManager,
        "should_process_llm_request",
        AsyncMock(return_value=True),
    )
    stages = [
        WakingCheckStage(),
        WhitelistCheckStage(),
        SessionStatusCheckStage(),
        PreProcessStage(),
    ]
    for stage in stages:
        await stage.initialize(ctx)
    stages[0]._umo_auto_name_recorder = MagicMock()
    respond = RespondStage()
    await respond.initialize(ctx)
    decorate = ResultDecorateStage()
    await decorate.initialize(ctx)
    fixture = SimpleNamespace(
        plugin=plugin,
        config=config,
        context=context,
        queue=queue,
        clock=clock,
        native=native,
        handlers=handlers,
        adapter=adapter,
        waking=stages[0],
        whitelist=stages[1],
        respond=respond,
        decorate=decorate,
        requests=[],
        payloads=[],
        before_submit=None,
        auto_send=False,
        bot=Bot(),
    )

    async def chat(**payload):
        fixture.payloads.append(payload)
        return LLMResponse(role="assistant", completion_text="模型回复")

    async def chat_stream(**payload):
        yield await chat(**payload)

    provider = SimpleNamespace(
        text_chat=chat, text_chat_stream=chat_stream, provider_config={}
    )
    fixture.provider = provider

    async def request(event):
        fixture.requests.append(event)
        req = ProviderRequest(
            prompt=event.message_str, session_id=event.unified_msg_origin
        )
        await native.on_req_llm(event, req)
        if await call_event_hook(event, EventType.OnLLMRequestEvent, req):
            return
        runner = object.__new__(ToolLoopAgentRunner)
        runner.provider = provider
        runner.req = req
        runner.run_context = SimpleNamespace(
            context=SimpleNamespace(event=event),
            messages=[Message.model_validate(await req.assemble_context())],
        )
        runner._abort_signal = asyncio.Event()
        runner.request_max_retries = 0
        runner.streaming = False
        await call_event_hook(event, EventType.OnAgentBeginEvent, runner.run_context)
        if fixture.before_submit:
            await fixture.before_submit(event, runner)
        async for response in runner._iter_llm_responses():
            await call_event_hook(event, EventType.OnLLMResponseEvent, response)
            if fixture.auto_send:
                result = event.plain_result(response.completion_text)
                result.result_content_type = ResultContentType.LLM_RESULT
                event.set_result(result)
        yield

    process = ProcessStage()
    process.ctx = ctx
    process.agent_sub_stage = AgentRequestSubStage()
    await process.agent_sub_stage.initialize(ctx)
    process.agent_sub_stage.process = request
    process.star_request_sub_stage = StarRequestSubStage()
    await process.star_request_sub_stage.initialize(ctx)

    async def run(event):
        event.bot = fixture.bot
        await asyncio.create_task(scheduler.execute(event))
        return event

    scheduler = object.__new__(PipelineScheduler)
    scheduler.stages = [*stages, process, decorate, respond]
    scheduler.ctx = ctx
    fixture.scheduler = scheduler
    fixture.process = process
    fixture.run = run
    fixture.request = request

    async def settle():
        await asyncio.gather(*plugin.replies.tasks, return_exceptions=True)
        await eventually(lambda: not plugin.replies.tasks)

    fixture.settle = settle
    await plugin.initialize()
    try:
        yield fixture
    finally:
        await plugin.terminate()


@pytest.fixture
def awake_group(harness):
    """Reply/coverage tests start with an active conversation."""
    harness.plugin.controller.awakened(KEY)


@pytest.mark.parametrize(
    "harness,allowed",
    [
        (
            {
                "group_whitelist": whitelist,
                "keyword_wake": {"keywords": ["小爱"]},
                "semantic": {"jev_api_key": "test"},
            },
            allowed,
        )
        for whitelist, allowed in (
            ([], True),
            (["100"], True),
            (["999", "100"], True),
            (["999"], False),
            (["1000"], False),
        )
    ],
    indirect=["harness"],
)
async def test_group_whitelist_scopes_observation_and_preserves_core_wake(
    harness, allowed
):
    plugin = harness.plugin
    observer_filter = GroupWakeFilter()
    keyword = make_event("小爱")
    assert observer_filter.filter(keyword, harness.config) is allowed
    assert (
        observer_filter.filter(make_event(platform="qq-2"), harness.config) is allowed
    )
    assert not observer_filter.filter(make_event(private=True), harness.config)

    bot = await harness.run(
        make_event("机器人发言", sender="300", post_type="message_sent")
    )
    await harness.run(keyword)
    assert keyword.is_at_or_wake_command is allowed
    assert harness.requests == ([keyword] if allowed else [])

    mention = await harness.run(
        make_event("你好", parts=[At(qq="300", name="Airi"), Plain("你好")])
    )
    assert harness.requests[-1] is mention
    assert not mention.is_stopped()
    assert mention.message_str == ("@Airi 你好" if allowed else "你好")
    assert bool(mention.get_extra(BOT_MENTION_KEY)) is allowed
    assert bool(plugin.controller.groups) is allowed
    assert bool(plugin.session_groups) is allowed
    for event in (bot, keyword, mention):
        assert (event.get_extra(RECORD_KEY) is not None) is allowed
        assert plugin.admitted(event) is allowed
    harness.plugin.client._send.assert_not_called()


@pytest.mark.parametrize("harness", [{"group_whitelist": ["999"]}], indirect=True)
async def test_group_whitelist_cannot_be_bypassed_by_activated_handlers(harness):
    event = make_event("你好", parts=[At(qq="300", name="Airi"), Plain("你好")])
    event.set_extra("activated_handlers", [harness.adapter])
    assert not harness.plugin.admitted(event)
    await harness.plugin.observe(event)
    await harness.plugin.collect_message(event)
    await harness.plugin.observe_request(event, ProviderRequest())
    assert event.message_str == "你好"
    assert event.get_extra(RECORD_KEY) is None
    assert not event.is_at_or_wake_command
    assert not harness.plugin.controller.groups
    assert not harness.plugin.session_groups


@pytest.mark.parametrize(
    "harness",
    [{"group_whitelist": ["100"], "semantic": {"jev_api_key": "test"}}],
    indirect=True,
)
async def test_group_whitelist_isolates_semantic_requests_and_bot_history(harness):
    harness.plugin.client._send = AsyncMock(
        return_value={"answers": {"q0": {"type": "noul", "noul": 0.99}}}
    )
    for group_id in ("100", "999"):
        await harness.run(
            make_event(
                "机器人发言", sender="300", group=group_id, post_type="message_sent"
            )
        )
        await harness.bot.send_group_msg(
            group_id=int(group_id), message="机器人主动发言"
        )
    allowed = await harness.run(make_event("那你更推荐哪个？"))
    excluded = await harness.run(make_event("那你更推荐哪个？", group="999"))
    group = harness.plugin.controller.groups[KEY]
    assert set(harness.plugin.controller.groups) == {KEY}
    assert any(m.text == "机器人主动发言" for m in group.history.values())
    assert excluded.get_extra(RECORD_KEY) is None
    assert set(group.pending) == {allowed.message_obj.message_id}

    harness.clock.now = 110
    group.changed.set()
    await eventually(lambda: bool(harness.requests))
    await harness.settle()
    harness.plugin.client._send.assert_awaited_once()
    candidates = harness.plugin.client._send.call_args.args[0]["state"]["candidates"]
    assert [m["message_id"] for m in candidates] == [allowed.message_obj.message_id]
    assert len(harness.requests) == 1
    assert harness.requests[0].get_group_id() == "100"
    assert excluded.unified_msg_origin not in harness.plugin.session_groups


@pytest.mark.parametrize(
    "parts",
    [
        None,
        [At(qq="300"), Plain("你好")],
        [Reply(id="42", sender_id="300", chain=[Plain("旧消息")]), Plain("你好")],
    ],
)
async def test_explicit_wake_preserves_text_and_never_calls_jev(harness, parts):
    event = make_event("你好小爱", parts=parts)
    await harness.run(event)
    assert harness.requests == [event]
    if parts and isinstance(parts[0], At) and str(parts[0].qq) == "300":
        assert event.message_str == "@机器人 你好小爱"
        assert event.message_obj.message_str == event.message_str
        assert event.get_extra(BOT_MENTION_KEY) is True
    else:
        assert event.message_str == "你好小爱"
    assert harness.plugin.controller.groups[KEY].awake_until == 280
    assert event.get_extra(RECORD_KEY).status == "covered"
    harness.plugin.client._send.assert_not_called()


async def test_bot_mention_is_preserved_without_rewriting_message_chain(harness):
    parts = [At(qq="300", name="Airi"), Plain("你在吗")]
    event = make_event("你在吗", parts=parts)

    await harness.run(event)

    assert harness.requests == [event]
    assert event.message_str == "@Airi 你在吗"
    assert event.message_obj.message_str == "@Airi 你在吗"
    assert event.get_extra(BOT_MENTION_KEY) is True
    assert event.get_messages() == parts
    assert not any("@" in getattr(part, "text", "") for part in parts)


async def test_only_bot_mention_enters_normal_reply_without_waiting(harness):
    event = make_event("", parts=[At(qq="300", name="Airi")])

    await harness.run(event)

    assert harness.requests == [event]
    assert event.message_str == "@Airi"
    assert event.message_obj.message_str == "@Airi"
    assert not event.is_stopped()
    assert event.get_extra(BOT_MENTION_KEY) is True


@pytest.fixture
def builtin_collector(harness):
    """Exercise the real core collector instead of recording every component."""
    from astrbot.builtin_stars.astrbot.main import Main

    builtin = Main.__new__(Main)
    builtin.context = harness.context
    builtin.group_chat_context = harness.native
    harness.context.get_registered_star = MagicMock(
        return_value=SimpleNamespace(star_cls=builtin)
    )
    collector = next(h for h in harness.handlers if h.handler_name == "native_record")
    collector.handler = builtin.on_message
    return builtin


@pytest.mark.usefixtures("builtin_collector")
@pytest.mark.parametrize("text", ["", "please respond"])
async def test_bot_mention_injects_history_through_builtin_collector(harness, text):
    previous = await harness.run(make_event("Show me my scheduled tasks"))
    parts = [At(qq="300", name="Rin")]
    if text:
        parts.append(Plain(text))
    mention = await harness.run(make_event(text, parts=parts))

    assert harness.requests == [mention]
    assert mention.get_messages() == parts
    payload = harness.payloads[0]["contexts"][-1].model_dump_json()
    assert payload.count(previous.message_str) == 1
    assert "@Rin" in payload
    assert "[At: Rin]" not in payload
    assert previous.get_extra(RECORD_KEY).status == "covered"
    assert mention.get_extra(RECORD_KEY).native_record_id
    assert not harness.native.raw_records[mention.unified_msg_origin]
    harness.plugin.client._send.assert_not_called()


@pytest.mark.usefixtures("builtin_collector")
async def test_only_bot_mention_preserves_messages_arriving_during_debounce(harness):
    previous = await harness.run(make_event("Show me my scheduled tasks"))
    mention = make_event("", parts=[At(qq="300", name="Rin")])
    waiting, release = asyncio.Event(), asyncio.Event()

    async def debounce(event):
        if event is mention:
            waiting.set()
            await release.wait()

    add_inbound(harness, debounce, priority=-20000)
    task = asyncio.create_task(harness.run(mention))
    try:
        async with asyncio.timeout(2):
            await waiting.wait()
        later = await harness.run(make_event("A later unrelated message"))
    finally:
        release.set()
        await task

    payload = harness.payloads[0]["contexts"][-1].model_dump_json()
    assert previous.message_str in payload
    assert later.message_str not in payload
    assert any(
        later.message_str in entry
        for entry in harness.native.raw_records[mention.unified_msg_origin]
    )


@pytest.mark.usefixtures("builtin_collector")
@pytest.mark.parametrize("gate", ["context", "mention", "group"])
async def test_mention_context_repair_respects_feature_scope(harness, gate):
    if gate == "context":
        harness.config["provider_ltm_settings"]["group_icl_enable"] = False
    elif gate == "mention":
        harness.plugin.settings = replace(
            harness.plugin.settings, preserve_bot_mention=False
        )
    else:
        harness.plugin.settings = replace(
            harness.plugin.settings, group_whitelist=("999",)
        )

    mention = await harness.run(make_event("", parts=[At(qq="300", name="Rin")]))

    assert mention.get_extra("_group_context_record_id") is None
    harness.context.get_registered_star.assert_not_called()
    assert not harness.native.raw_records.get(mention.unified_msg_origin)


@pytest.mark.usefixtures("builtin_collector")
@pytest.mark.parametrize("text", ["", "please respond"])
async def test_injected_group_context_is_saved_and_available_next_turn(
    harness, tmp_path, text
):
    """Verify native persistence keeps injected group content across turns."""
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )

    previous = await harness.run(make_event("Show me my scheduled tasks"))
    parts = [At(qq="300", name="Rin")]
    if text:
        parts.append(Plain(text))
    mention = await harness.run(make_event(text, parts=parts))
    db = SQLiteDatabase(str(tmp_path / "group-context.db"))
    try:
        await db.initialize()
        manager = ConversationManager(db)
        umo = mention.unified_msg_origin
        cid = await manager.new_conversation(umo, platform_id=mention.get_platform_id())
        conversation = await manager.get_conversation(umo, cid)
        stage = InternalAgentSubStage()
        stage.conv_manager = manager
        response = LLMResponse(role="assistant", completion_text="Here are your tasks")
        await stage._save_to_history(
            mention,
            ProviderRequest(conversation=conversation),
            response,
            [
                *harness.payloads[0]["contexts"],
                Message(role="assistant", content=response.completion_text),
            ],
            runner_stats=None,
        )

        restored = await ConversationManager(db).get_conversation(umo, cid)
        request = ProviderRequest(
            prompt="What did I ask before?", contexts=json.loads(restored.history)
        )
        await harness.provider.text_chat(
            contexts=[*request.contexts, await request.assemble_context()]
        )
        payload = json.dumps(harness.payloads[-1]["contexts"])
        assert payload.count(previous.message_str) == 1
        assert request.prompt in payload
    finally:
        await db.engine.dispose()


async def test_empty_mention_wrapper_skips_only_preserved_bot_mentions(harness):
    calls = []

    async def original(owner, event):
        assert owner is harness.plugin
        calls.append(event)
        yield "delegated"

    wrapped = harness.plugin.observer._empty_mention(partial(original, harness.plugin))
    marked = make_event("", parts=[At(qq="300", name="Airi")])
    marked.set_extra(BOT_MENTION_KEY, True)
    ordinary = make_event("普通消息")

    assert [item async for item in wrapped(marked)] == []
    assert [item async for item in wrapped(ordinary)] == ["delegated"]
    assert calls == [ordinary]


@pytest.mark.parametrize(
    "harness",
    [
        {"preserve_bot_mention": preserve, "semantic": {"enabled": semantic}}
        for preserve in (True, False)
        for semantic in (True, False)
    ],
    indirect=True,
)
async def test_bot_mention_switch_is_independent_of_jev(harness):
    preserve = harness.plugin.settings.preserve_bot_mention
    parts = [At(qq="300", name="Airi"), Plain("你在吗")]
    event = await harness.run(make_event("你在吗", parts=parts))
    assert event.message_str == ("@Airi 你在吗" if preserve else "你在吗")
    assert event.message_obj.message_str == event.message_str
    assert bool(event.get_extra(BOT_MENTION_KEY)) is preserve
    assert event.get_messages() == parts
    assert event.is_at_or_wake_command
    assert harness.requests == [event]
    harness.plugin.client._send.assert_not_called()


@pytest.mark.parametrize(
    "harness",
    [{"preserve_bot_mention": True}, {"preserve_bot_mention": False}],
    indirect=True,
)
async def test_bot_mention_switch_controls_registered_empty_mention(
    harness, monkeypatch
):
    from astrbot.builtin_stars.astrbot import main as builtin

    # Use the real handler, ending its waiter immediately to avoid a 60s test.
    waiter = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr(builtin, "session_waiter", lambda timeout: lambda fn: waiter)
    harness.config["platform_settings"].update(
        empty_mention_waiting=True, empty_mention_waiting_need_reply=True
    )
    harness.context.conversation_manager.get_conversation = AsyncMock(return_value=None)
    handler = next(
        h
        for h in star_handlers_registry.get_handlers_by_module_name(
            builtin.Main.__module__
        )
        if h.handler_name == "handle_empty_mention"
    )
    event = make_event("", parts=[At(qq="300", name="Airi")])
    event.set_extra("activated_handlers", [harness.adapter])
    await harness.plugin.observe(event)
    owner = SimpleNamespace(context=harness.context)
    results = [result async for result in handler.handler(owner, event)]
    if harness.plugin.settings.preserve_bot_mention:
        assert event.message_str == "@Airi"
        assert results == []
        assert not event.is_stopped()
        waiter.assert_not_called()
    else:
        assert event.message_str == ""
        assert event.message_obj.message_str == ""
        assert event.get_extra(BOT_MENTION_KEY) is None
        assert len(results) == 1  # AstrBot's preset request is restored.
        assert event.is_stopped()
        waiter.assert_awaited_once_with(event)
        assert all(
            target is not handler for target, *_ in harness.plugin.observer.patches
        )


async def test_bot_mention_does_not_rewrite_other_mentions(harness):
    parts = [
        At(qq="300", name="Airi"),
        Plain("你怎么看"),
        At(qq="123", name="小明"),
    ]
    event = make_event("你怎么看 @小明", parts=parts)

    await harness.run(event)

    assert harness.requests == [event]
    assert event.message_str == "@Airi 你怎么看 @小明"
    assert event.message_obj.message_str == event.message_str
    assert event.get_messages() == parts
    assert "@小明" in event.message_str
    assert "(123)" not in event.message_str


@pytest.mark.parametrize(
    "harness,text,expected",
    [
        (
            {"keyword_wake": {"keywords": ["Airi"], "keyword_ignore_case": False}},
            "你好 airi",
            False,
        ),
        (
            {"keyword_wake": {"keywords": ["Airi"], "keyword_ignore_case": True}},
            "你好 airi",
            True,
        ),
        ({"keyword_wake": {"keywords": ["Airi"]}}, "你好 AIRI", True),
        (
            {"keyword_wake": {"keywords": ["Airi"], "keyword_ignore_case": False}},
            "你好 Airi",
            True,
        ),
        (
            {"keyword_wake": {"keywords": [], "keyword_ignore_case": True}},
            "你好 airi",
            False,
        ),
    ],
    indirect=["harness"],
)
async def test_keyword_group_controls_matching_without_a_switch(
    harness, text, expected
):
    event = make_event(text)
    await harness.run(event)
    assert event.message_str == text
    assert harness.requests == ([event] if expected else [])
    for event in (
        make_event("你好", parts=[At(qq="300"), Plain("你好")]),
        make_event("追问", parts=[Reply(id="42", sender_id="300"), Plain("追问")]),
    ):
        await harness.run(event)
        assert harness.requests[-1] is event
    assert len(harness.requests) == int(expected) + 2
    harness.plugin.client._send.assert_not_called()


async def test_observer_and_awake_group_do_not_automatically_wake(harness):
    await harness.run(make_event())
    group = harness.plugin.controller.groups[KEY]
    assert group.awake_until == 0
    assert group.deadline is None
    assert group.worker is None
    assert not group.pending
    await harness.run(make_event("小爱"))
    harness.clock.now = 110
    await harness.run(make_event("仍然需要语义判断"))
    assert len(harness.requests) == 1
    assert group.awake_until == 280
    assert group.deadline == 113


@pytest.mark.usefixtures("awake_group")
async def test_semantic_wake_injects_arrivals_and_cleans_only_actual_input(harness):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def judge(key, history, candidates, valid):
        entered.set()
        await finish.wait()
        return {m["message_id"]: 0.99 for m in candidates}

    harness.plugin.controller.judge = judge
    events = [make_event(f"消息{i}", mid=i) for i in range(1, 9)]
    for event in events[:3]:
        await harness.run(event)
    group = harness.plugin.controller.groups[KEY]
    harness.clock.now = 110
    group.changed.set()
    await entered.wait()
    for event in events[3:5]:
        await harness.run(event)
    finish.set()
    await eventually(lambda: len(harness.requests) == 1)
    await harness.settle()
    replay = harness.requests[0]
    assert replay.get_extra(REPLY_KEY)
    assert replay.message_obj.message_id == "5"
    assert harness.queue.empty()
    assert len(harness.requests) == 1
    assert all(events[i].get_extra(RECORD_KEY).status == "covered" for i in range(5))
    assert not group.pending
    assert not harness.native.raw_records[replay.unified_msg_origin]
    for event in events[5:]:
        await harness.run(event)
    # Simulate a later debounced input including only 6/7: preparation is not coverage.
    text = "\n".join(e.get_extra(RECORD_KEY).native_text for e in events[5:7])
    payload = {"contexts": [{"role": "user", "content": text}]}
    assert set(group.pending) == {"6", "7", "8"}
    token = ACTIVE_RUN.set((harness.plugin.observer, replay))
    try:
        await harness.provider.text_chat(**payload)
    finally:
        ACTIVE_RUN.reset(token)
    assert set(group.pending) == {"8"}
    assert events[5].get_extra(RECORD_KEY).status == "covered"


@pytest.mark.usefixtures("awake_group")
async def test_trimmed_input_is_not_marked_covered(harness):
    old = make_event("未被模型实际看到")
    await harness.run(old)

    async def trim(event, runner):
        assert old.get_extra(RECORD_KEY).status == "pending"
        runner.run_context.messages = [Message(role="user", content="只有裁剪后的摘要")]

    harness.before_submit = trim
    await harness.run(make_event("小爱"))
    assert old.get_extra(RECORD_KEY).status == "pending"


@pytest.mark.parametrize("gate", ["whitelist", "session", "plugin", "profile"])
async def test_admission_checks_run_before_caching(harness, monkeypatch, gate):
    if gate == "whitelist":
        harness.whitelist.enable_whitelist_check = True
        harness.whitelist.whitelist = ["999"]
    elif gate == "session":
        monkeypatch.setattr(
            SessionServiceManager, "is_session_enabled", AsyncMock(return_value=False)
        )
    elif gate == "plugin":
        monkeypatch.setattr(
            SessionPluginManager,
            "filter_handlers_by_session",
            AsyncMock(return_value=[]),
        )
    else:
        harness.config["plugin_set"] = []
    await harness.run(make_event("小爱"))
    assert not harness.plugin.controller.groups


@pytest.mark.parametrize(
    "change", ["reset", "conversation", "disabled", "covered", "plugin_whitelist"]
)
@pytest.mark.usefixtures("awake_group")
async def test_queued_wake_cannot_survive_invalidated_state(
    harness, monkeypatch, change
):
    event = make_event("你觉得呢")
    await harness.run(event)
    group = harness.plugin.controller.groups[KEY]
    record = event.get_extra(RECORD_KEY)
    assert await harness.plugin._semantic_wake(KEY, group, [record])
    replay = next(iter(harness.plugin.replies.tasks.values())).event
    if change == "reset":
        ActiveEventRegistry().stop_all(event.unified_msg_origin)
    elif change == "conversation":
        harness.context.conversation_manager.get_curr_conversation_id.return_value = (
            "new"
        )
    elif change == "disabled":
        monkeypatch.setattr(
            SessionPluginManager,
            "is_plugin_enabled_for_session",
            AsyncMock(return_value=False),
        )
    elif change == "plugin_whitelist":
        monkeypatch.setattr(
            harness.plugin,
            "settings",
            replace(harness.plugin.settings, group_whitelist=("999",)),
        )
    else:
        harness.plugin.controller.covered(KEY, {record.id})
    await harness.settle()
    assert not harness.requests
    assert replay.is_stopped()


@pytest.mark.usefixtures("awake_group")
async def test_reset_during_awaited_wake_eligibility(harness, monkeypatch):
    event = make_event()
    await harness.run(event)
    record = event.get_extra(RECORD_KEY)
    group = harness.plugin.controller.groups[KEY]

    async def reset(*args):
        harness.plugin.controller.reset(KEY)
        return True

    monkeypatch.setattr(harness.plugin, "_session_eligible", reset)
    assert not await harness.plugin._semantic_wake(KEY, group, [record])
    assert harness.queue.empty()


async def test_shared_awake_state_across_isolated_sessions(harness):
    harness.waking.unique_session = True
    first, second = make_event("小爱"), make_event("群员互聊", sender="201")
    await harness.run(first)
    await harness.run(second)
    assert first.unified_msg_origin != second.unified_msg_origin
    assert len(harness.plugin.controller.groups) == 1
    assert harness.plugin.controller.groups[KEY].awake_until == 280
    assert len(harness.requests) == 1


async def test_command_is_executed_once_and_never_a_candidate(harness):
    calls = []

    async def command(event):
        calls.append(event.message_str)
        await event.send(event.plain_result("指令完成"))

    handler = copy(harness.adapter)
    handler.handler_name = "command"
    handler.handler = command
    handler.extras_configs = {"priority": 1}
    handler.event_filters = [
        CommandFilter(
            "help", handler_md=SimpleNamespace(handler=lambda self, event: None)
        )
    ]
    harness.handlers.append(handler)
    await harness.run(make_event("/help"))
    assert calls == ["help"]
    assert not harness.requests
    assert not harness.plugin.controller.groups[KEY].pending


async def test_send_success_renews_but_rewake_and_model_response_do_not(harness):
    await harness.run(make_event("小爱"))
    group = harness.plugin.controller.groups[KEY]
    harness.clock.now = 110
    event = make_event("小爱，再问一次")
    await harness.run(event)
    assert group.awake_until == 280
    harness.clock.now = 120
    result = event.plain_result("实际发送")
    result.result_content_type = ResultContentType.LLM_RESULT
    event.set_result(result)
    await harness.respond.process(event)
    assert group.awake_until == 300
    assert any(m.bot and m.text == "实际发送" for m in group.history.values())
    # Duplicate respond notifications for this response cannot renew again.
    harness.clock.now = 130
    event.set_result(result)
    await harness.respond.process(event)
    assert group.awake_until == 300


@pytest.mark.parametrize("fail_at", [1, 2])
async def test_partial_or_failed_segmented_reply_does_not_renew(harness, fail_at):
    event = make_event("小爱")
    await harness.run(event)
    harness.respond.enable_seg = True
    harness.respond.interval_method = "random"
    harness.respond.interval = [0, 0]
    harness.bot.fail_at = fail_at
    harness.clock.now = 120
    result = event.chain_result([Plain("第一段"), Plain("第二段")])
    result.result_content_type = ResultContentType.LLM_RESULT
    event.set_result(result)
    await harness.respond.process(event)
    assert len(harness.bot.calls) == 2
    assert harness.plugin.controller.groups[KEY].awake_until == 280


@pytest.mark.parametrize("fail", [False, True])
async def test_streaming_completion_renews_only_after_delivery(harness, fail):
    event = make_event("小爱")
    await harness.run(event)

    async def stream():
        yield MessageChain([Plain("第一段")])
        assert harness.plugin.controller.groups[KEY].awake_until == 280
        yield MessageChain([Plain("第二段")])

    harness.config["provider_settings"]["unsupported_streaming_strategy"] = "collect"
    if fail:
        harness.bot.fail_at = 1
    harness.clock.now = 120
    result = event.plain_result("")
    result.result_content_type = ResultContentType.STREAMING_RESULT
    result.async_stream = stream()
    event.set_result(result)
    if fail:
        with pytest.raises(RuntimeError, match="OneBot"):
            await harness.respond.process(event)
    else:
        await harness.respond.process(event)
    assert harness.plugin.controller.groups[KEY].awake_until == (280 if fail else 300)


async def test_tool_delivery_is_history_but_does_not_renew(harness):
    event = make_event("小爱")
    await harness.run(event)
    harness.clock.now = 120
    await event.send(event.plain_result("工具内部消息"))
    assert harness.plugin.controller.groups[KEY].awake_until == 280


@pytest.mark.usefixtures("awake_group")
async def test_provider_failure_does_not_restore_covered_candidates(harness):
    pending = make_event("未直接唤醒")
    await harness.run(pending)

    async def fail(**kwargs):
        raise RuntimeError("provider failure")

    harness.provider.text_chat = fail
    with pytest.raises(RuntimeError, match="provider failure"):
        await harness.run(make_event("小爱"))
    assert pending.get_extra(RECORD_KEY).status == "covered"
    assert not harness.plugin.controller.groups[KEY].pending


async def test_observer_failure_preserves_provider_result_and_send(
    harness, monkeypatch
):
    monkeypatch.setattr(
        harness.plugin.observer,
        "_submitted",
        MagicMock(side_effect=RuntimeError("observer")),
    )
    harness.auto_send = True
    await harness.run(make_event("小爱"))
    assert len(harness.payloads) == 1
    assert len(harness.bot.calls) == 1


@pytest.mark.usefixtures("awake_group")
async def test_ambiguous_equal_fingerprints_do_not_clear_unrelated_inputs(harness):
    a, b = make_event("相同文本"), make_event("相同文本")
    await harness.run(a)
    await harness.run(b)
    for event in (a, b):
        record = event.get_extra(RECORD_KEY)
        record.input_fingerprints.add(fingerprint("相同文本"))
    token = ACTIVE_RUN.set((harness.plugin.observer, a))
    try:
        harness.plugin.observer._submitted(
            {"contexts": [{"role": "user", "content": "相同文本"}]}
        )
    finally:
        ACTIVE_RUN.reset(token)
    assert all(e.get_extra(RECORD_KEY).status == "pending" for e in (a, b))


async def test_unload_restores_hooks_and_provider_methods(harness):
    await harness.run(make_event("小爱"))
    assert "text_chat" in vars(harness.provider)
    patches = list(harness.plugin.observer.patches)
    await harness.plugin.terminate()
    assert GroupWakeFilter.plugin is None
    for target, name, existed, raw, _ in patches:
        assert (name in vars(target)) == existed
        if existed:
            assert vars(target)[name] is raw


async def test_real_qq_debouncer_accepts_explicit_but_not_semantic_wakes(harness):
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
                "followup_window_seconds": 0,
                "max_wait_seconds": 5,
                "ignore_prefixes": ["/", "!"],
            },
        },
        context=SimpleNamespace(
            get_config=lambda **kw: harness.config, get_all_stars=lambda: []
        ),
    )
    debouncer = MessageDebouncer(qq)
    batches = []

    async def capture(event):
        await debouncer.capture(event)
        batches.append(event.get_extra(ARRIVAL_KEY).batch)

    handler = copy(harness.adapter)
    handler.handler_name = "qq_debounce"
    handler.handler = capture
    handler.event_filters = [ArrivalFilter()]
    handler.extras_configs = {"priority": -20000}
    harness.handlers.append(handler)
    try:
        explicit = make_event("小爱，你好")
        await harness.run(explicit)
        assert batches[-1] == [explicit.get_extra(ARRIVAL_KEY)]
        ordinary = make_event("你再解释下")
        await harness.run(ordinary)
        assert batches[-1] == []
        assert len(harness.requests) == 1
        record = ordinary.get_extra(RECORD_KEY)
        group = harness.plugin.controller.groups[KEY]
        assert await harness.plugin._semantic_wake(KEY, group, [record])
        await harness.settle()
        assert len(batches) == 2
        assert harness.requests[-1].get_extra(ARRIVAL_KEY) is None
        assert len(harness.requests) == 2
        assert record.status == "covered"
    finally:
        await debouncer.close()


@pytest.mark.parametrize(
    "unsupported",
    [
        "remote",
        "no_injection",
        "disabled",
        "session_disabled",
        "whitelist",
        "plugin_whitelist",
    ],
)
@pytest.mark.usefixtures("awake_group")
async def test_semantic_prerequisites_checked_before_network(
    harness, monkeypatch, unsupported
):
    event = make_event()
    await harness.run(event)
    if unsupported == "remote":
        harness.process.agent_sub_stage.agent_sub_stage = object()
    elif unsupported == "no_injection":
        harness.config["provider_ltm_settings"]["group_icl_enable"] = False
    elif unsupported == "session_disabled":
        monkeypatch.setattr(
            SessionServiceManager, "is_session_enabled", AsyncMock(return_value=False)
        )
    elif unsupported == "whitelist":
        harness.config["platform_settings"]["enable_id_white_list"] = True
        harness.config["platform_settings"]["id_whitelist"] = ["999"]
    elif unsupported == "plugin_whitelist":
        monkeypatch.setattr(
            harness.plugin,
            "settings",
            replace(harness.plugin.settings, group_whitelist=("999",)),
        )
    else:
        monkeypatch.setattr(
            SessionPluginManager,
            "is_plugin_enabled_for_session",
            AsyncMock(return_value=False),
        )
    group = harness.plugin.controller.groups[KEY]
    harness.clock.now = 110
    group.changed.set()
    await eventually(lambda: group.worker is None)
    harness.plugin.client._send.assert_not_called()
    assert not harness.requests


@pytest.mark.usefixtures("awake_group")
async def test_queued_identity_survives_history_eviction_and_is_released(harness):
    from astrbot_plugin_qq_group_enhance.state import ChatMessage

    event = make_event("这是对你的追问")
    await harness.run(event)
    record = event.get_extra(RECORD_KEY)
    group = harness.plugin.controller.groups[KEY]
    assert await harness.plugin._semantic_wake(KEY, group, [record])
    for i in range(40):
        harness.plugin.controller.add(
            KEY,
            ChatMessage(f"bot-{i}", "300", "机器人", "背景消息", 0, bot=True),
            False,
        )
    assert record.id not in group.history
    assert group.reserved[record.id] is record
    # A normal request can still invalidate this queued wake after history rolls.
    harness.plugin.controller.covered(KEY, {record.id})
    await harness.settle()
    assert not harness.requests
    assert not group.reserved


@pytest.mark.usefixtures("awake_group")
async def test_rejected_reply_releases_reserved_tracking(harness):
    event = make_event()
    await harness.run(event)
    group = harness.plugin.controller.groups[KEY]
    assert await harness.plugin._semantic_wake(
        KEY, group, [event.get_extra(RECORD_KEY)]
    )
    assert group.reserved
    harness.config["platform_settings"]["enable_id_white_list"] = True
    harness.config["platform_settings"]["id_whitelist"] = ["999"]
    await harness.settle()
    assert not group.reserved
    assert not harness.requests


async def test_stream_tool_messages_alone_cannot_prove_reply_delivery(harness):
    event = make_event("小爱")
    await harness.run(event)
    harness.clock.now = 120

    async def stream():
        await event.send(MessageChain([Plain("工具输出")], type="tool_direct_result"))
        if False:
            yield

    result = event.plain_result("")
    result.result_content_type = ResultContentType.STREAMING_RESULT
    result.async_stream = stream()
    event.set_result(result)
    await harness.respond.process(event)
    assert len(harness.bot.calls) == 1
    assert harness.plugin.controller.groups[KEY].awake_until == 280


@pytest.mark.usefixtures("awake_group")
async def test_other_plugin_stop_cannot_be_reawakened_later(harness):
    async def stop(event):
        event.stop_event()

    handler = copy(harness.adapter)
    handler.handler_name = "other_plugin_stop"
    handler.handler = stop
    handler.extras_configs = {"priority": -100}
    harness.handlers.append(handler)
    event = make_event()
    await harness.run(event)
    assert event.get_extra(RECORD_KEY).status == "ignored"
    group = harness.plugin.controller.groups[KEY]
    await eventually(lambda: group.worker is None)
    assert not group.pending
    harness.plugin.client._send.assert_not_called()


@pytest.mark.parametrize(
    "harness",
    [
        {"keyword_wake": {"keywords": ["小爱"]}},
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"jev_api_key": ""},
        },
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"jev_api_key": " \n "},
        },
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"enabled": False},
        },
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"enabled": False, "jev_api_key": "test"},
        },
    ],
    indirect=True,
)
async def test_disabled_semantics_keeps_direct_wakes_and_history(harness):
    assert harness.plugin.observer.installed
    assert harness.plugin.active
    assert not harness.plugin.semantic_enabled
    harness.plugin.controller.awakened(KEY)
    for i in range(35):
        await harness.run(make_event(f"普通消息{i}"))
    group = harness.plugin.controller.groups[KEY]
    assert len(group.history) == 30
    assert not group.pending
    assert not group.inflight
    assert group.deadline is None
    assert group.worker is None
    assert not harness.plugin.controller.tasks
    assert not harness.requests

    for event in (
        make_event("小爱，请回答"),
        make_event("你好", parts=[At(qq="300"), Plain("你好")]),
        make_event("追问", parts=[Reply(id="42", sender_id="300"), Plain("追问")]),
    ):
        await harness.run(event)
        assert harness.requests[-1] is event
    assert len(harness.requests) == 3
    assert group.awake_until == 280
    await harness.run(make_event("普通追问"))
    assert not group.pending
    assert group.worker is None
    assert len(harness.requests) == 3
    assert await harness.plugin._judge(KEY, [], [], lambda _: True) == {}
    harness.plugin.client._send.assert_not_called()
    assert harness.plugin.client.session is None


@pytest.mark.parametrize("enabled", [True, False])
async def test_missing_key_notice_only_when_semantics_enabled(monkeypatch, enabled):
    warning = MagicMock()
    monkeypatch.setattr(main.logger, "warning", warning)
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), {"semantic": {"enabled": enabled}})
    try:
        await plugin.initialize()
        assert plugin.active
        assert plugin.observer.installed
        assert warning.call_count == int(enabled)
        if enabled:
            assert "jev_api_key" in warning.call_args.args[0]
            assert "插件已加载" in warning.call_args.args[0]
    finally:
        await plugin.terminate()


@pytest.mark.parametrize(
    "harness",
    [
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"enabled": False},
        },
        {
            "keyword_wake": {"keywords": ["小爱"]},
            "semantic": {"enabled": False, "jev_api_key": "test"},
        },
    ],
    indirect=True,
)
async def test_disabled_semantics_retains_bot_history_and_distance_without_window(
    harness,
):
    # Direct wakes and caching also work without native group context injection.
    harness.config["provider_ltm_settings"]["group_icl_enable"] = False
    seed = await harness.run(make_event("背景消息"))
    await seed.send(seed.plain_result("机器人最近发言"))
    group = harness.plugin.controller.groups[KEY]
    assert group.awake_until == 0
    assert group.last_bot_sequence == 2
    for i in range(6):
        event = await harness.run(make_event(f"普通消息{i}"))
        assert event.get_extra(RECORD_KEY).status == "background"
    assert group.sequence == 8
    assert any(m.bot and m.text == "机器人最近发言" for m in group.history.values())
    assert len(group.history) == 8
    assert not group.pending
    assert not group.inflight
    assert group.deadline is None
    assert group.worker is None
    assert not harness.plugin.controller.tasks
    assert not await harness.plugin._session_eligible(
        event, event.get_extra(RECORD_KEY)
    )
    assert not await harness.plugin._semantic_wake(
        KEY, group, [event.get_extra(RECORD_KEY)]
    )
    assert not harness.plugin.replies.tasks
    harness.auto_send = True
    for event in (
        make_event("小爱，请回答"),
        make_event("你好", parts=[At(qq="300"), Plain("你好")]),
        make_event("追问", parts=[Reply(id="42", sender_id="300"), Plain("追问")]),
    ):
        assert harness.plugin.explicit(event)
        await harness.run(event)
        assert harness.requests[-1] is event
    assert len(harness.requests) == 3
    assert len(harness.bot.calls) == 4
    assert group.awake_until == 280
    assert any(m.bot and m.text == "模型回复" for m in group.history.values())
    harness.plugin.client._send.assert_not_called()
    assert harness.plugin.client.session is None


@pytest.mark.usefixtures("awake_group")
@pytest.mark.parametrize("phase", ["window", "retry", "result"])
async def test_disabled_semantics_blocks_pending_judgment_and_wake(
    harness, monkeypatch, phase
):
    event = await harness.run(make_event("普通追问"))
    group = harness.plugin.controller.groups[KEY]

    def disable():
        monkeypatch.setattr(
            harness.plugin, "settings", replace(harness.plugin.settings, enabled=False)
        )

    async def send(payload):
        disable()
        if phase == "retry":
            raise TimeoutError()
        return {
            "answers": {
                key: {"type": "noul", "noul": 0.99} for key in payload["questions"]
            }
        }

    harness.plugin.client._send = AsyncMock(side_effect=send)
    if phase == "window":
        disable()
    harness.clock.now = 105
    group.changed.set()
    await eventually(lambda: group.worker is None)
    assert harness.plugin.client._send.await_count == (0 if phase == "window" else 1)
    assert not group.pending
    assert not group.inflight
    assert not group.reserved
    assert not harness.plugin.replies.tasks
    assert not harness.requests
    assert event.get_extra(RECORD_KEY).id in group.history
    await harness.run(make_event("小爱，直接唤醒"))
    assert len(harness.requests) == 1


@pytest.mark.usefixtures("awake_group")
@pytest.mark.parametrize("phase", ["queued", "submit"])
async def test_disabled_semantics_blocks_queued_reply_before_provider(
    harness, monkeypatch, phase
):
    event = await harness.run(make_event("普通追问"))

    def disable():
        monkeypatch.setattr(
            harness.plugin, "settings", replace(harness.plugin.settings, enabled=False)
        )

    if phase == "submit":

        async def before_submit(event, runner):
            disable()

        harness.before_submit = before_submit
    request = await schedule_reply(harness, event)
    if phase == "queued":
        disable()
    await harness.settle()
    assert request.event.is_stopped()
    assert not harness.payloads
    assert not harness.bot.calls
    assert not request.group.reserved


@pytest.mark.parametrize(
    "harness",
    [
        {"group_whitelist": "100"},
        {"group_whitelist": [""]},
        {"preserve_bot_mention": "false"},
        {"semantic": {"jev_threshold": 0}},
        {"semantic": {"jev_threshold": "0.75"}},
        {"semantic": {"jev_threshold": 2}},
        {"semantic": {"debounce_seconds": 0}},
        {"semantic": {"awake_observe_seconds": 5}},
        {"semantic": {"max_messages_after_bot": 0}},
        {"asleep_observe_seconds": 10},
        {"semantic": {"history_messages": 0}},
        {"keyword_wake": {"keywords": [""]}},
        {"keyword_wake": {"keyword_ignore_case": "true"}},
        {"keyword_wake": {"keyword_case_sensitive": True}},
        {"keyword_wake": []},
        {"keyword_wake": {"enabled": True}},
        {"keywords": ["test"]},
        {"semantic": {"enabled": "true"}},
        {"semantic": {"jev_model": "jev-latest"}},
        {"semantic": {"jev_api_key": None}},
        {"semantic": {"jev_api_url": "not-a-url"}},
        {"semantic": []},
        {"semantic": {"keywords": ["test"]}},
        {"enabled": False},
        {"jev_api_key": "test-secret"},
        {"traffic_window_minutes": 10},
        {"unknown_field": 1},
    ],
    indirect=True,
)
async def test_invalid_config_keeps_plugin_loaded_and_core_messages_working(harness):
    plugin = harness.plugin
    assert plugin.config_error
    assert GroupWakeFilter.plugin is plugin
    assert not plugin.active
    assert plugin.settings is None
    assert plugin.client is None
    assert plugin.controller is None
    assert not plugin.observer.installed
    assert not plugin.observer.patches
    event = make_event("小爱")
    assert not GroupWakeFilter().filter(event, harness.config)
    await harness.run(event)
    assert not harness.requests
    assert event.get_extra(RECORD_KEY) is None
    assert await plugin._judge(KEY, [], [], lambda _: True) == {}

    # Core wake/reply remains usable while this plugin's settings are invalid.
    mention = make_event("你好", parts=[At(qq="300"), Plain("你好")])
    await harness.run(mention)
    assert harness.requests == [mention]
    assert not plugin.session_groups
    await plugin.terminate()
    assert GroupWakeFilter.plugin is None


async def test_invalid_config_logs_reason_preserves_values_and_recovers_on_reload(
    monkeypatch,
):
    error_log = MagicMock()
    monkeypatch.setattr(main.logger, "error", error_log)
    config = {"semantic": {"jev_api_key": "test-secret", "jev_threshold": 0}}
    original = deepcopy(config)
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), config)
    try:
        await plugin.initialize()
        assert "jev_threshold" in plugin.config_error
        assert config == original
        error_log.assert_called_once()
        rendered = error_log.call_args.args[0] % error_log.call_args.args[1:]
        assert "插件已加载" in rendered
        assert "WebUI" in rendered
        assert "jev_threshold" in rendered
        assert "test-secret" not in rendered
    finally:
        await plugin.terminate()
    config["semantic"]["jev_threshold"] = 0.9
    reloaded = QQGroupEnhancePlugin(SimpleNamespace(), config)
    try:
        await reloaded.initialize()
        assert reloaded.active
        assert reloaded.config_error is None
        assert reloaded.settings.jev_threshold == 0.9
        assert reloaded.observer.installed
        assert reloaded.client is not None
        assert reloaded.controller is not None
    finally:
        await reloaded.terminate()


@pytest.mark.parametrize("config", [[], "invalid", {1: True}])
async def test_malformed_config_object_is_reported_without_failing_load(config):
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), config)
    try:
        await plugin.initialize()
        assert plugin.config_error
        assert not plugin.active
        assert plugin.settings is None
        assert not plugin.observer.installed
    finally:
        await plugin.terminate()


@pytest.mark.usefixtures("awake_group")
async def test_event_bus_only_dispatches_real_messages_and_logs_reply_request(
    harness, webui_logs
):
    bus = EventBus(
        harness.queue,
        {"test": SimpleNamespace(execute=harness.run)},
        SimpleNamespace(get_conf_info=lambda _: {"id": "test", "name": "default"}),
    )
    dispatch = asyncio.create_task(bus.dispatch())
    event = make_event("你在干嘛", mid="501")
    try:
        await harness.queue.put(event)
        await eventually(
            lambda: bool(event.get_extra(RECORD_KEY)) and not bus._pending_tasks
        )
        harness.plugin.client._send = AsyncMock(
            return_value={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 123},
                "answers": {"q0": {"type": "noul", "noul": 0.96}},
            }
        )
        group = harness.plugin.controller.groups[KEY]
        harness.clock.now = 110
        group.changed.set()
        await eventually(lambda: len(harness.payloads) == 1 and not bus._pending_tasks)

        # A real second message, even with identical text, still logs normally.
        duplicate_text = make_event("你在干嘛", mid="502")
        await harness.queue.put(duplicate_text)
        await eventually(
            lambda: (
                bool(duplicate_text.get_extra(RECORD_KEY)) and not bus._pending_tasks
            )
        )
        entries = list(webui_logs)
        incoming = [e for e in entries if "你在干嘛" in e["data"]]
        assert len(incoming) == 2
        assert all(e["category"] == "user_chat" for e in incoming)
        replays = [e for e in entries if "Jev 请求回复" in e["data"]]
        assert len(replays) == 1
        assert "message_id=501" in replays[0]["data"]
        assert "group=100" in replays[0]["data"]
        assert replays[0]["category"] == "system"
        assert any("Jev 判断完成" in e["data"] for e in entries)
        assert any(
            "Jev 唤醒汇总" in e["data"] and "新增唤醒=是" in e["data"] for e in entries
        )
        assert len(harness.requests) == 1
        assert event.get_extra(RECORD_KEY).status == "covered"
        assert duplicate_text.get_extra(RECORD_KEY).status == "pending"
        assert len(harness.native.raw_records[event.unified_msg_origin]) == 1
    finally:
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
        await asyncio.gather(*bus._pending_tasks, return_exceptions=True)


def add_inbound(harness, handler, priority=10):
    registered = copy(harness.adapter)
    registered.handler_name = handler.__name__
    registered.handler = handler
    registered.extras_configs = {"priority": priority}
    harness.handlers.append(registered)


@pytest.fixture
def qq_enrichment(harness):
    from astrbot_plugin_qq_enhance.main import QQEnhancePlugin

    qq = object.__new__(QQEnhancePlugin)
    qq.config = {
        "platform": {"platform_id": ""},
        "inbound": {
            "component_spoof_protection": {"enabled": False},
            "semanticize_components": True,
            "enhance_voice_messages": True,
            "respond_to_red_packet": False,
            "respond_to_poke": False,
            "max_semantic_chars": 2000,
        },
        "limits": {"max_components": 50},
    }
    qq.runtime = SimpleNamespace(
        verify_platform=AsyncMock(),
        call_action=AsyncMock(return_value={"text": "转写结果"}),
    )
    add_inbound(harness, qq.enrich_inbound_qq_components, priority=0)
    return qq


@pytest.mark.parametrize(
    "harness",
    [{"semantic": {"jev_api_key": "test"}}, {"semantic": {"enabled": False}}],
    indirect=True,
)
@pytest.mark.usefixtures("awake_group")
async def test_collects_real_qq_enrichment_before_debounce_without_outline_overwrite(
    harness, qq_enrichment
):
    event = make_event("你怎么看", parts=[Plain("你怎么看"), Face(id=14)])
    event.message_obj.raw_message["message"] = [
        {"type": "face", "data": {"id": "14"}},
        {"type": "share", "data": {"title": "新闻标题", "url": "https://example.com"}},
    ]
    seen = []

    async def debounce(event):
        record = event.get_extra(RECORD_KEY)
        seen.append((record.ready, record.text))

    add_inbound(harness, debounce, priority=-20000)
    await harness.run(event)
    record = event.get_extra(RECORD_KEY)
    assert "QQ表情：微笑" in record.text
    assert "QQ链接分享：新闻标题" in record.text
    assert record.text.count("你怎么看") == 1
    assert "QQ表情：微笑" not in event.get_message_outline()
    assert seen == [(True, event.message_str)]
    harness.plugin.refresh(event)
    assert record.text == event.message_str
    if harness.plugin.semantic_enabled:
        judge = AsyncMock(return_value={})
        harness.plugin.controller.judge = judge
        harness.clock.now = 103
        group = harness.plugin.controller.groups[KEY]
        group.changed.set()
        await eventually(lambda: group.worker is None)
        assert judge.call_args.args[2][0]["text"] == event.message_str
    else:
        group = harness.plugin.controller.groups[KEY]
        assert group.history[record.id] is record
        assert not group.pending and group.worker is None


@pytest.mark.usefixtures("awake_group")
async def test_collects_real_qq_referenced_voice_and_mentions(
    harness, qq_enrichment, monkeypatch
):
    monkeypatch.setattr(
        Record, "convert_to_file_path", AsyncMock(side_effect=FileNotFoundError())
    )
    quote = Reply(id="42", sender_id="456", chain=[Record(file="voice.silk")])
    event = make_event("说得对", parts=[quote, At(qq="456"), Plain("说得对")])
    event.message_obj.raw_message["message"] = [
        {"type": "reply", "data": {"id": "42"}},
        {"type": "text", "data": {"text": "说得对"}},
    ]
    await harness.run(event)
    record = event.get_extra(RECORD_KEY)
    assert record.ready
    assert record.mentions == ("456",)
    assert record.reply == {
        "message_id": "42",
        "sender_id": "456",
        "text": "[QQ component|QQ语音消息：转写结果]",
    }
    assert record.text == "说得对"
    assert record.reply["text"] == quote.chain[0].text


async def test_slow_qq_voice_preserves_arrival_eligibility_without_delaying_text(
    harness, qq_enrichment, monkeypatch
):
    monkeypatch.setattr(
        Record, "convert_to_file_path", AsyncMock(side_effect=FileNotFoundError())
    )
    entered, finish = asyncio.Event(), asyncio.Event()

    async def transcribe(*args, **kwargs):
        entered.set()
        await finish.wait()
        return {"text": "我也想问这个"}

    qq_enrichment.runtime.call_action.side_effect = transcribe
    controller = harness.plugin.controller
    judge = AsyncMock(return_value={})
    controller.judge = judge
    await harness.run(make_event("机器人发言", sender="300", mid="anchor"))
    voice = make_event("", parts=[Record(file="voice.silk")], mid="101")
    voice.message_obj.raw_message.update(
        message_id=101, message=[{"type": "record", "data": {"file": "voice.silk"}}]
    )
    task = asyncio.create_task(harness.run(voice))
    try:
        await eventually(entered.is_set)
        group = controller.groups[KEY]
        record = voice.get_extra(RECORD_KEY)
        assert not record.ready and record.text == ""
        harness.clock.now = 101
        events = [await harness.run(make_event(f"消息{i}")) for i in range(5)]
        assert events[-1].get_extra(RECORD_KEY).status == "background"
        harness.clock.now = 105
        group.changed.set()
        await eventually(lambda: judge.await_count == 1 and not group.inflight)
        assert [m["message_id"] for m in judge.call_args.args[2]] == [
            e.message_obj.message_id for e in events[:4]
        ]
        assert list(group.pending) == ["101"]
        assert not task.done()
        finish.set()
        await task
        await eventually(lambda: group.worker is None)
        assert judge.await_count == 2
        assert judge.call_args.args[2][0]["text"] == (
            "[QQ component|QQ语音消息：我也想问这个]"
        )
        assert record.sequence == 2 and record.arrived_at == 100
        assert not controller.is_awake(group)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.usefixtures("awake_group")
async def test_semantic_reply_does_not_wait_for_a_new_unprepared_input(harness):
    first = await harness.run(make_event("已完成语义化的追问"))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def enrich(event):
        entered.set()
        await finish.wait()
        event.message_str = "稍后完成的增强文本"

    add_inbound(harness, enrich, priority=0)
    later = make_event("仍在处理的消息")
    task = asyncio.create_task(harness.run(later))
    try:
        await eventually(entered.is_set)
        request = await asyncio.wait_for(schedule_reply(harness, first), 1)
        await harness.settle()
        assert request.source is first.get_extra(RECORD_KEY)
        assert not task.done()
        assert later.get_extra(RECORD_KEY).status == "pending"
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


async def schedule_reply(harness, event):
    group = harness.plugin.controller.groups[KEY]
    assert await harness.plugin._semantic_wake(
        KEY, group, [event.get_extra(RECORD_KEY)]
    )
    return next(reversed(harness.plugin.replies.tasks.values()))


@pytest.mark.usefixtures("awake_group")
async def test_reply_reuses_prepared_message_without_repeating_inbound(
    harness, monkeypatch
):
    stage = next(s for s in harness.scheduler.stages if isinstance(s, PreProcessStage))
    preprocess = AsyncMock(wraps=stage.process)
    monkeypatch.setattr(stage, "process", preprocess)
    handled = []

    async def enrich(event):
        handled.append(event.message_obj.message_id)
        event.message_str += " [prepared]"
        event.message_obj.message.append(Plain("[prepared]"))
        event.set_extra("selected_model", "test-model")

    add_inbound(harness, enrich)
    event = make_event("hello")
    await harness.run(event)
    ids_before = list(harness.native._record_ids[event.unified_msg_origin])
    request = await schedule_reply(harness, event)
    await harness.settle()
    assert handled == [event.message_obj.message_id]
    assert preprocess.await_count == 1
    assert harness.queue.empty()
    assert len(harness.payloads) == 1
    assert request.event.message_str == "hello [prepared]"
    assert request.event.get_extra("selected_model") == "test-model"
    assert request.event.get_extra("activated_handlers") is None
    assert request.source.native_record_id == ids_before[0]
    assert not harness.native.raw_records[event.unified_msg_origin]


@pytest.mark.usefixtures("awake_group")
async def test_reply_runs_decoration_send_hooks_and_renews_awake(harness):
    event = make_event()
    await harness.run(event)
    harness.auto_send = True
    harness.decorate.reply_prefix = "PREFIX: "
    sent = []

    async def after_send(event):
        sent.append(event.message_obj.message_id)

    handler = copy(harness.adapter)
    handler.event_type = EventType.OnAfterMessageSentEvent
    handler.handler_name = "after_send"
    handler.handler = after_send
    harness.handlers.append(handler)

    async def before_submit(event, runner):
        assert harness.plugin.controller.groups[KEY].awake_until == 280
        harness.clock.now = 125

    harness.before_submit = before_submit
    await schedule_reply(harness, event)
    await harness.settle()
    assert len(harness.bot.calls) == 1
    assert "PREFIX:" in str(harness.bot.calls)
    assert sent == [event.message_obj.message_id]
    assert harness.plugin.controller.groups[KEY].awake_until == 305


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.usefixtures("awake_group")
async def test_covered_during_preparation_never_calls_provider(harness, streaming):
    event = make_event()
    await harness.run(event)

    async def cover(event, runner):
        runner.streaming = streaming
        harness.plugin.controller.covered(KEY, {event.get_extra(RECORD_KEY).id})

    harness.before_submit = cover
    request = await schedule_reply(harness, event)
    await harness.settle()
    assert not harness.payloads
    assert not harness.bot.calls
    assert request.event.is_stopped()
    assert not harness.plugin.controller.groups[KEY].reserved


@pytest.mark.usefixtures("awake_group")
async def test_real_agent_rechecks_after_session_lock_before_build(harness):
    from astrbot.core.pipeline.process_stage.method.agent_request import (
        AgentRequestSubStage,
    )
    from astrbot.core.utils.session_lock import session_lock_manager

    event = make_event()
    await harness.run(event)
    stage = AgentRequestSubStage()
    await stage.initialize(harness.scheduler.ctx)
    harness.process.agent_sub_stage = stage
    entered = asyncio.Event()

    async def typing():
        entered.set()

    event.send_typing = typing
    async with session_lock_manager.acquire_lock(event.unified_msg_origin):
        request = await schedule_reply(harness, event)
        await asyncio.wait_for(entered.wait(), 2)
        assert not request.event.is_stopped()
        harness.plugin.controller.covered(KEY, {event.get_extra(RECORD_KEY).id})
    await harness.settle()
    assert request.event.is_stopped()
    assert not harness.bot.calls
    assert not harness.payloads
    # The obsolete reply must not consume native context while preparing input.
    assert len(harness.native.raw_records[event.unified_msg_origin]) == 1


@pytest.mark.usefixtures("awake_group")
async def test_missing_native_id_does_not_consume_later_messages(harness):
    event = make_event("old")
    await harness.run(event)
    request = await schedule_reply(harness, event)
    umo = event.unified_msg_origin
    harness.native.raw_records[umo].clear()
    harness.native._record_ids[umo].clear()
    harness.native.raw_records[umo].append("new unrelated input")
    harness.native._record_ids[umo].append("new-id")
    await harness.settle()
    assert request.event.is_stopped()
    assert not harness.payloads
    assert list(harness.native.raw_records[umo]) == ["new unrelated input"]


@pytest.mark.parametrize("ending", ["reply", "negative", "failed", "reset", "unload"])
@pytest.mark.usefixtures("awake_group")
async def test_prepared_media_lives_until_reply_or_candidate_finishes(
    harness, tmp_path, ending
):
    from astrbot.api.message_components import Image

    path = tmp_path / "prepared.png"
    path.write_bytes(b"temporary image fixture")

    async def prepare(event):
        event.message_obj.message.append(Image(file=str(path)))
        event.track_temporary_local_file(str(path))

    add_inbound(harness, prepare)
    event = make_event("image")
    await harness.run(event)
    record = event.get_extra(RECORD_KEY)
    assert path.exists()
    assert record.media_owner is not None
    assert event._temporary_local_files == []
    if ending == "reply":

        async def verify(event, runner):
            assert path.exists()
            assert event._temporary_local_files == [str(path)]
            assert event.get_messages()[-1].file == str(path)

        harness.before_submit = verify
        await schedule_reply(harness, event)
        await harness.settle()
        assert len(harness.payloads) == 1
    elif ending in ("negative", "failed"):
        harness.plugin.controller.judge = AsyncMock(
            return_value={record.id: 0.1} if ending == "negative" else {}
        )
        group = harness.plugin.controller.groups[KEY]
        harness.clock.now = 110
        group.changed.set()
        await eventually(lambda: group.worker is None)
    elif ending == "reset":
        ActiveEventRegistry().stop_all(event.unified_msg_origin)
    else:
        await harness.plugin.terminate()
    assert not path.exists()
    assert record.media_owner is None


@pytest.mark.parametrize("ending", ["reset", "unload", "error"])
@pytest.mark.usefixtures("awake_group")
async def test_running_reply_cancellation_and_failure_release_resources(
    harness, tmp_path, ending
):
    from astrbot.core.utils.active_event_registry import active_event_registry

    path = tmp_path / "reply-resource.tmp"
    path.write_bytes(b"resource")
    event = make_event()
    event.track_temporary_local_file(str(path))
    await harness.run(event)
    entered = asyncio.Event()

    async def pause(event, runner):
        entered.set()
        if ending == "error":
            raise RuntimeError("preparation failed")
        await asyncio.Event().wait()

    harness.before_submit = pause
    request = await schedule_reply(harness, event)
    await asyncio.wait_for(entered.wait(), 2)
    if ending == "reset":
        active_event_registry.stop_all(event.unified_msg_origin)
    elif ending == "unload":
        await harness.plugin.terminate()
    await harness.settle()
    assert not path.exists()
    assert not request.group.reserved
    assert not harness.payloads
    assert event.unified_msg_origin not in active_event_registry._events


@pytest.mark.usefixtures("awake_group")
async def test_reply_can_continue_tool_rounds_after_its_own_coverage(harness):
    event = make_event()
    await harness.run(event)
    ordinary_request = harness.process.agent_sub_stage.process

    async def two_rounds(event):
        async for _ in ordinary_request(event):
            yield
        token = ACTIVE_RUN.set((harness.plugin.observer, event))
        try:
            await harness.provider.text_chat(contexts=[])
        finally:
            ACTIVE_RUN.reset(token)

    harness.process.agent_sub_stage.process = two_rounds
    await schedule_reply(harness, event)
    await harness.settle()
    assert len(harness.payloads) == 2
    assert event.get_extra(RECORD_KEY).status == "covered"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.usefixtures("awake_group")
async def test_real_agent_reply_sends_then_saves_history(
    harness, monkeypatch, streaming
):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

    event = make_event()
    await harness.run(event)
    stage = AgentRequestSubStage()
    await stage.initialize(harness.scheduler.ctx)
    agent = stage.agent_sub_stage
    agent.streaming_response = streaming
    agent.unsupported_streaming_strategy = "collect"
    saved = AsyncMock()
    agent._save_to_history = saved
    harness.process.agent_sub_stage = stage
    harness.config["provider_settings"]["unsupported_streaming_strategy"] = "collect"
    harness.provider.get_model = lambda: "test-model"
    harness.provider.meta = lambda: SimpleNamespace(type="test")
    response = LLMResponse(role="assistant", completion_text="normal reply")
    runner = SimpleNamespace(
        done=lambda: True,
        was_aborted=lambda: False,
        get_final_llm_resp=lambda: response,
        request_stop=lambda: None,
        stats=SimpleNamespace(to_dict=lambda: {}),
        provider=harness.provider,
    )

    async def build(*, event, **kwargs):
        req = ProviderRequest(prompt=event.message_str, conversation=None)
        await harness.native.on_req_llm(event, req)
        runner.run_context = SimpleNamespace(
            context=SimpleNamespace(event=event),
            messages=[Message.model_validate(await req.assemble_context())],
        )
        return SimpleNamespace(
            agent_runner=runner,
            provider_request=req,
            provider=harness.provider,
            reset_coro=None,
        )

    async def generate(runner, *args, **kwargs):
        reply_event = runner.run_context.context.event
        await call_event_hook(
            reply_event, EventType.OnAgentBeginEvent, runner.run_context
        )
        token = ACTIVE_RUN.set((harness.plugin.observer, reply_event))
        try:
            harness.plugin.observer.watch_provider(harness.provider)
            await harness.provider.text_chat(contexts=runner.run_context.messages)
        finally:
            ACTIVE_RUN.reset(token)
        await call_event_hook(reply_event, EventType.OnLLMResponseEvent, response)
        harness.clock.now = 130
        chain = MessageChain([Plain(response.completion_text)])
        if streaming:
            yield chain
        else:
            result = reply_event.chain_result(chain.chain)
            result.result_content_type = ResultContentType.LLM_RESULT
            reply_event.set_result(result)
            yield
        if not streaming:
            assert harness.bot.calls
        reply_event.clear_result()

    harness.plugin.observer.patch(
        internal,
        "build_main_agent",
        lambda original: harness.plugin.observer._build_agent(build),
    )
    monkeypatch.setattr(internal, "run_agent", generate)
    monkeypatch.setattr(internal, "_record_internal_agent_stats", AsyncMock())
    request = await schedule_reply(harness, event)
    await harness.settle()
    assert not request.event.is_stopped()
    assert len(harness.payloads) == 1
    assert len(harness.bot.calls) == 1
    assert saved.await_count >= 1
    assert harness.plugin.controller.groups[KEY].awake_until == 310


@pytest.mark.usefixtures("awake_group")
async def test_first_semantic_reply_keeps_new_conversation_identity(harness):
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
    from astrbot_plugin_qq_group_enhance.main import CID_KEY

    manager = harness.context.conversation_manager
    manager.get_curr_conversation_id.return_value = None

    async def create(*args):
        manager.get_curr_conversation_id.return_value = "created-by-reply"
        return "created-by-reply"

    manager.new_conversation = AsyncMock(side_effect=create)
    event = make_event()
    await harness.run(event)
    stage = AgentRequestSubStage()
    await stage.initialize(harness.scheduler.ctx)
    harness.process.agent_sub_stage = stage
    seen = []

    async def build(*, event, **kwargs):
        seen.append(event.get_extra(CID_KEY))
        assert await harness.plugin.replies.eligible(event)
        return None

    harness.plugin.observer.patch(
        internal,
        "build_main_agent",
        lambda original: harness.plugin.observer._build_agent(build),
    )
    await schedule_reply(harness, event)
    await harness.settle()
    assert seen == ["created-by-reply"]
    assert manager.new_conversation.await_count == 1
    assert event.get_extra(CID_KEY) is None


async def test_asleep_recent_bot_send_limits_candidates_and_semantic_wake(harness):
    seed = await harness.run(make_event("背景消息", mid="seed"))
    await seed.send(seed.plain_result("机器人发言"))
    group = harness.plugin.controller.groups[KEY]
    assert group.awake_until == 0
    assert group.last_bot_sequence == 2
    events = [make_event(f"追问{i}", mid=str(i)) for i in range(1, 7)]
    for event in events[:5]:
        await harness.run(event)
    # OneBot's later echo of a successful send must not reset the distance.
    await harness.run(
        make_event("机器人发言", sender="300", mid="10001", post_type="message_sent")
    )
    await harness.run(events[5])
    assert group.last_bot_sequence == 2
    assert group.sequence == 8
    assert list(group.pending) == [str(i) for i in range(1, 6)]
    assert events[5].get_extra(RECORD_KEY).status == "background"
    assert group.deadline == 103
    assert not harness.requests

    async def send(payload):
        assert [m["message_id"] for m in payload["state"]["candidates"]] == [
            str(i) for i in range(1, 6)
        ]
        return {
            "answers": {
                k: {"type": "noul", "noul": 0.99 if k == "q0" else 0.1}
                for k in payload["questions"]
            }
        }

    harness.plugin.client._send = AsyncMock(side_effect=send)
    harness.clock.now = 105
    group.changed.set()
    await eventually(lambda: group.worker is None)
    await harness.settle()
    assert harness.plugin.client._send.await_count == 1
    assert len(harness.requests) == 1
    assert group.awake_until == 285
    assert not group.pending
    assert all(e.get_extra(RECORD_KEY).status == "covered" for e in events[:5])


async def test_failed_send_does_not_create_distance_anchor(harness):
    seed = await harness.run(make_event("背景消息", mid="seed"))
    group = harness.plugin.controller.groups[KEY]
    harness.bot.fail_at = 1
    with pytest.raises(RuntimeError, match="OneBot"):
        await seed.send(seed.plain_result("发送失败"))
    event = await harness.run(make_event("追问"))
    assert group.last_bot_sequence is None
    assert event.get_extra(RECORD_KEY).status == "background"
    assert group.worker is None
    await seed.send(seed.plain_result("发送成功"))
    assert group.last_bot_sequence == 3
    assert group.awake_until == 0
    await harness.run(make_event("再次追问"))
    assert len(group.pending) == 1


async def test_command_arrival_counts_once_without_becoming_candidate(harness):
    await harness.run(make_event("机器人发言", sender="300", mid="anchor"))
    calls = []

    async def command(event):
        calls.append(event.message_str)

    handler = copy(harness.adapter)
    handler.handler_name = "command"
    handler.handler = command
    handler.extras_configs = {"priority": 1}
    handler.event_filters = [
        CommandFilter(
            "help", handler_md=SimpleNamespace(handler=lambda self, event: None)
        )
    ]
    harness.handlers.append(handler)
    await harness.run(make_event("/help", mid="command"))
    group = harness.plugin.controller.groups[KEY]
    assert group.sequence == 2
    assert group.last_bot_sequence == 1
    assert calls == ["help"]
    assert "command" not in group.records()
    # Commands can wake the core. Let that awake interval expire; its message
    # still consumed one of the five positions after the bot's last send.
    harness.clock.now = 281
    for i in range(1, 6):
        await harness.run(make_event(f"群聊{i}", mid=str(i)))
    assert list(group.pending) == ["1", "2", "3", "4"]
    assert group.history["5"].status == "background"
