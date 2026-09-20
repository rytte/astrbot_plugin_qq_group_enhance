"""Reversible observers at actual provider and OneBot submission boundaries.

No incoming message is held for Jev. Wrappers preserve the original arguments,
return values, exceptions and cancellation, and are removed on plugin unload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from functools import wraps

from astrbot.api import logger

from .state import ChatMessage

ACTIVE_RUN: ContextVar = ContextVar("qq_group_jev_run", default=None)
ACTIVE_PIPELINE: ContextVar = ContextVar("qq_group_jev_pipeline", default=None)
DELIVERY: ContextVar = ContextVar("qq_group_jev_delivery", default=None)
MODEL_SEND: ContextVar = ContextVar("qq_group_jev_model_send", default=None)


def content_value(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {
            k: content_value(v)
            for k, v in value.items()
            if not k.startswith("_") and v is not None
        }
    if isinstance(value, list):
        return [content_value(v) for v in value]
    return value


def fingerprint(content) -> str:
    return hashlib.sha256(
        json.dumps(content_value(content), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def text_parts(content) -> list[str]:
    value = content_value(content)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            p["text"] for p in value if isinstance(p, dict) and p.get("type") == "text"
        ]
    return []


@dataclass
class Delivery:
    event: object
    successes: int = 0
    failures: int = 0


class RuntimeObserver:
    def __init__(self, plugin):
        self.plugin = plugin
        self.patches = []
        self.bots = set()
        self.providers = set()
        self.installed = False

    @contextmanager
    def guard(self):
        """An observer failure must never replace a provider/send outcome."""
        try:
            yield
        except Exception:
            logger.exception("QQ Jev observation failed; ordinary pipeline continues")

    def patch(self, target, name, factory):
        exists = name in vars(target)
        raw = vars(target).get(name)
        original = getattr(target, name)
        replacement = factory(original)
        setattr(target, name, replacement)
        self.patches.append((target, name, exists, raw, replacement))

    def install(self):
        if self.installed:
            return
        from astrbot.api.event import AstrMessageEvent
        from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
        from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
        from astrbot.core.pipeline.respond.stage import RespondStage
        from astrbot.core.pipeline.scheduler import PipelineScheduler
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
            AiocqhttpMessageEvent,
        )
        from astrbot.core.utils.active_event_registry import ActiveEventRegistry

        for target, name in (
            (PreProcessStage, "process"),
            (RespondStage, "process"),
            (GroupChatContext, "handle_message"),
            (GroupChatContext, "on_req_llm"),
            (GroupChatContext, "_format_message"),
            (GroupChatContext, "remove_session"),
            (ToolLoopAgentRunner, "_iter_llm_responses"),
            (ActiveEventRegistry, "stop_all"),
            (AiocqhttpMessageEvent, "send"),
            (PipelineScheduler, "execute"),
            (AstrMessageEvent, "cleanup_temporary_local_files"),
            (internal, "build_main_agent"),
        ):
            if not callable(getattr(target, name, None)):
                raise RuntimeError(
                    f"Unsupported AstrBot integration: missing {target.__name__}.{name}"
                )
        self.patch(PreProcessStage, "process", self._preprocess)
        self.patch(PipelineScheduler, "execute", self._pipeline)
        self.patch(
            AstrMessageEvent, "cleanup_temporary_local_files", self._cleanup_media
        )
        self.patch(internal, "build_main_agent", self._build_agent)
        self.patch(GroupChatContext, "handle_message", self._group_record)
        self.patch(GroupChatContext, "on_req_llm", self._group_injection)
        self.patch(GroupChatContext, "_format_message", self._format_record)
        self.patch(GroupChatContext, "remove_session", self._reset_history)
        self.patch(ToolLoopAgentRunner, "_iter_llm_responses", self._runner)
        self.patch(RespondStage, "process", self._respond)
        self.patch(AiocqhttpMessageEvent, "send", self._event_send)
        self.patch(ActiveEventRegistry, "stop_all", self._stop_all)
        self.installed = True

    def _cleanup_media(self, original):
        @wraps(original)
        def wrapped(event):
            from .main import RECORD_KEY

            with self.guard():
                record = event.get_extra(RECORD_KEY)
                if (
                    self.plugin.admitted(event)
                    and record
                    and record.event is event
                    and not event.call_llm
                    and not event._has_send_oper
                    and record.status in ("pending", "judging", "woken")
                    and event._temporary_local_files
                    and self.plugin.controller.groups.get(self.plugin.key(event))
                ):
                    owner = copy(event)
                    owner._extras = {}
                    owner._temporary_local_files = list(event._temporary_local_files)
                    record.release_media()
                    record.media_owner = owner
                    event._temporary_local_files.clear()
            return original(event)

        return wrapped

    def _build_agent(self, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            event = kwargs["event"]
            # Called by the local Agent after acquiring its session lock, before
            # request assembly can consume the native group context buffer.
            if not await self.plugin.replies.eligible(event):
                event.stop_event()
                return None
            from .main import CID_KEY
            from .reply import REPLY_KEY

            if event.get_extra(REPLY_KEY) and event.get_extra(CID_KEY) is None:
                # This is the first reply in the session. Establish its identity
                # under the Agent's lock so later checks distinguish creation
                # by this request from /new or reset while it is preparing.
                manager = self.plugin.context.conversation_manager
                cid = await manager.new_conversation(
                    event.unified_msg_origin, event.get_platform_id()
                )
                event.set_extra(CID_KEY, cid)
                if not await self.plugin.replies.eligible(event):
                    event.stop_event()
                    return None
            return await original(*args, **kwargs)

        return wrapped

    def _pipeline(self, original):
        @wraps(original)
        async def wrapped(scheduler, event):
            token = ACTIVE_PIPELINE.set(scheduler)
            try:
                return await original(scheduler, event)
            except BaseException:
                with self.guard():
                    self._ignore_input(event)
                raise
            finally:
                ACTIVE_PIPELINE.reset(token)
                with self.guard():
                    from .main import RECORD_KEY

                    record = event.get_extra(RECORD_KEY)
                    if record and record.task is asyncio.current_task():
                        record.task = None
                    if event.is_stopped() or event.call_llm or event._has_send_oper:
                        self._ignore_input(event)

        return wrapped

    def _ignore_input(self, event):
        from .main import RECORD_KEY

        record = event.get_extra(RECORD_KEY)
        if record:
            key = self.plugin.key(event)
            group = self.plugin.controller.groups.get(key)
            if group and self.plugin.controller.valid(key, group, record):
                self.plugin.controller.archive(group, record, "ignored")

    def _preprocess(self, original):
        @wraps(original)
        async def wrapped(stage, event):
            with self.guard():
                if self.plugin.admitted(event):
                    await self.plugin.observe(event)
            if event.is_stopped():
                return
            result = await original(stage, event)
            with self.guard():
                if self.plugin.admitted(event):
                    self.plugin.refresh(event)
            return result

        return wrapped

    def _group_record(self, original):
        @wraps(original)
        async def wrapped(context, event):
            from .main import RECORD_KEY

            record = (
                event.get_extra(RECORD_KEY) if self.plugin.admitted(event) else None
            )
            result = await original(context, event)
            with self.guard():
                if record:
                    record.native_record_id = event.get_extra(
                        "_group_context_record_id", ""
                    )
            return result

        return wrapped

    def _group_injection(self, original):
        @wraps(original)
        async def wrapped(context, event, request):
            from .reply import REPLY_KEY

            if reply := event.get_extra(REPLY_KEY):
                if not await self.plugin.replies.eligible(event):
                    event.stop_event()
                    return
                # An old numeric index can now refer to a different message.
                # Require the original ID and disable the core's index fallback.
                from astrbot.builtin_stars.astrbot.group_chat_context import (
                    TextPart,
                    _format_group_history_block,
                )

                umo = event.unified_msg_origin
                async with context._get_lock(umo):
                    ids = context._record_ids.get(umo, ())
                    if reply.source.native_record_id not in ids:
                        logger.info(
                            "Jev 取消回复：原生上下文记录已失效 message_id=%s",
                            reply.source.id,
                        )
                        event.stop_event()
                        return
                    index = list(ids).index(reply.source.native_record_id)
                    records = context.raw_records[umo]
                    history = list(records)[:index]
                    for _ in range(index + 1):
                        records.popleft()
                        ids.popleft()
                if history:
                    request.extra_user_content_parts.append(
                        TextPart(text=_format_group_history_block(history))
                    )
                return
            return await original(context, event, request)

        return wrapped

    def _format_record(self, original):
        @wraps(original)
        async def wrapped(context, event, cfg):
            from .main import RECORD_KEY

            result = await original(context, event, cfg)
            with self.guard():
                record = (
                    event.get_extra(RECORD_KEY) if self.plugin.admitted(event) else None
                )
                if record:
                    record.native_text = result
            return result

        return wrapped

    def _reset_history(self, original):
        @wraps(original)
        async def wrapped(context, event):
            result = await original(context, event)
            with self.guard():
                if key := self.plugin.session_groups.get(event.unified_msg_origin):
                    self.plugin.replies.reset(key)
                    self.plugin.controller.reset(key)
            return result

        return wrapped

    def _stop_all(self, original):
        @wraps(original)
        def wrapped(registry, umo, *args, **kwargs):
            result = original(registry, umo, *args, **kwargs)
            with self.guard():
                if key := self.plugin.session_groups.get(umo):
                    self.plugin.replies.reset(key)
                    self.plugin.controller.reset(key)
            return result

        return wrapped

    def _runner(self, original):
        @wraps(original)
        async def wrapped(runner, **kwargs):
            event = getattr(getattr(runner.run_context, "context", None), "event", None)
            active = event is not None and self.plugin.admitted(event)
            if active:
                with self.guard():
                    self.watch_provider(runner.provider)
            token = ACTIVE_RUN.set((self, event) if active else None)
            try:
                async for response in original(runner, **kwargs):
                    yield response
            finally:
                ACTIVE_RUN.reset(token)

        return wrapped

    def watch_provider(self, provider):
        if id(provider) in self.providers:
            return
        self.providers.add(id(provider))
        self.patch(provider, "text_chat", self._provider_chat)
        self.patch(provider, "text_chat_stream", self._provider_stream)

    def _provider_chat(self, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            active = ACTIVE_RUN.get()
            if active and active[0] is self:
                await self.plugin.replies.before_submit(active[1])
            with self.guard():
                self._submitted(kwargs)
            return await original(*args, **kwargs)

        return wrapped

    def _provider_stream(self, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            active = ACTIVE_RUN.get()
            if active and active[0] is self:
                await self.plugin.replies.before_submit(active[1])
            with self.guard():
                self._submitted(kwargs)
            stream = original(*args, **kwargs)
            try:
                async for response in stream:
                    yield response
            finally:
                await stream.aclose()

        return wrapped

    def capture_current_input(self, event, messages):
        from .main import RECORD_KEY

        record = event.get_extra(RECORD_KEY)
        if not record:
            return
        for message in reversed(messages):
            value = content_value(message)
            if value.get("role") == "user":
                record.input_fingerprints.add(fingerprint(value.get("content")))
                break

    def _submitted(self, payload):
        active = ACTIVE_RUN.get()
        if not active or active[0] is not self or self.plugin.closed:
            return
        if signal := payload.get("abort_signal"):
            if signal.is_set():
                return
        event = active[1]
        key = self.plugin.key(event)
        group = self.plugin.controller.groups.get(key)
        if not group:
            return
        contents = [
            content_value(m).get("content")
            for m in payload.get("contexts", [])
            if content_value(m).get("role") == "user"
        ]
        hashes = Counter(fingerprint(c) for c in contents)
        texts = [text for content in contents for text in text_parts(content)]
        records = list(group.records().values())
        covered = set()
        for record in records:
            if record.bot:
                continue
            # Ambiguous identical inputs are not attributed to every message.
            for digest in record.input_fingerprints:
                matches = sum(
                    digest in m.input_fingerprints for m in records if not m.bot
                )
                if hashes[digest] >= matches:
                    covered.add(record.id)
            if record.native_text:
                matches = sum(
                    m.native_text == record.native_text for m in records if not m.bot
                )
                if sum(t.count(record.native_text) for t in texts) >= matches:
                    covered.add(record.id)
        self.plugin.controller.covered(key, covered)
        self.plugin.controller.awakened(key)

    def watch_bot(self, event):
        bot = getattr(event, "bot", None)
        if bot is None or id(bot) in self.bots:
            return
        self.bots.add(id(bot))
        platform_id, self_id = str(event.get_platform_id()), str(event.get_self_id())

        def factory(original):
            @wraps(original)
            async def wrapped(action, **params):
                is_group_send = action in (
                    "send_group_msg",
                    "send_group_forward_msg",
                ) or (action == "send_msg" and params.get("message_type") == "group")
                group_id = str(params.get("group_id", ""))
                delivery = DELIVERY.get()
                related = (
                    delivery
                    and self.plugin.key(delivery.event)
                    == (
                        platform_id,
                        group_id,
                    )
                    and MODEL_SEND.get() is delivery.event
                )
                try:
                    result = await original(action, **params)
                except BaseException:
                    if is_group_send and related:
                        delivery.failures += 1
                    raise
                with self.guard():
                    if is_group_send and group_id and not self.plugin.closed:
                        if related:
                            delivery.successes += 1
                        key = (platform_id, group_id)
                        if key in self.plugin.controller.groups:
                            raw = params.get("message", params.get("messages", []))
                            text = self._outgoing_text(raw)
                            mid = (
                                str(result.get("message_id", ""))
                                if isinstance(result, dict)
                                else ""
                            )
                            if mid and text:
                                self.plugin.controller.add(
                                    key,
                                    ChatMessage(
                                        mid,
                                        self_id,
                                        "机器人",
                                        text,
                                        time.time(),
                                        bot=True,
                                        bot_id=self_id,
                                    ),
                                    candidate=False,
                                )
                return result

            return wrapped

        self.patch(bot, "call_action", factory)

    def _event_send(self, original):
        @wraps(original)
        async def wrapped(event, message):
            delivery = DELIVERY.get()
            # Streaming execution may send tool status/results inside RespondStage.
            # Those sends contribute history, but cannot prove model reply delivery.
            model = delivery and delivery.event is event and message.type in (None, "")
            token = MODEL_SEND.set(event if model else None)
            try:
                return await original(event, message)
            finally:
                MODEL_SEND.reset(token)

        return wrapped

    @staticmethod
    def _outgoing_text(message):
        if isinstance(message, str):
            return message
        parts = []
        for part in message if isinstance(message, list) else []:
            if not isinstance(part, dict):
                continue
            kind, data = part.get("type", "message"), part.get("data", {})
            parts.append(str(data.get("text", "")) if kind == "text" else f"[{kind}]")
        return "".join(parts)

    def _respond(self, original):
        @wraps(original)
        async def wrapped(stage, event):
            from astrbot.core.message.message_event_result import ResultContentType

            from .main import RESPONSE_KEY

            result = event.get_result()
            if not self.plugin.admitted(event) or result is None:
                return await original(stage, event)
            model = result.result_content_type in (
                ResultContentType.LLM_RESULT,
                ResultContentType.STREAMING_RESULT,
            )
            delivery = Delivery(event)
            token = DELIVERY.set(delivery)
            try:
                returned = await original(stage, event)
            finally:
                DELIVERY.reset(token)
            response = event.get_extra(RESPONSE_KEY)
            if (
                model
                and response
                and delivery.successes
                and not delivery.failures
                and not event.is_stopped()
                and event.get_extra("_qq_group_jev_sent") is not response
            ):
                with self.guard():
                    event.set_extra("_qq_group_jev_sent", response)
                    self.plugin.controller.replied(self.plugin.key(event))
            return returned

        return wrapped

    def close(self):
        for target, name, existed, raw, replacement in reversed(self.patches):
            if vars(target).get(name) is replacement:
                if existed:
                    setattr(target, name, raw)
                else:
                    delattr(target, name)
            else:
                logger.warning(
                    "QQ Jev observer could not restore wrapped method %s", name
                )
        self.patches.clear()
        self.providers.clear()
        self.bots.clear()
        self.installed = False
