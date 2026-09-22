from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from sys import maxsize
from typing import ClassVar

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.session_plugin_manager import SessionPluginManager

from .bridge import ACTIVE_PIPELINE, RuntimeObserver
from .jev import JevClient
from .reply import REPLY_KEY, ReplyDispatcher
from .state import ChatMessage, GroupKey, GroupState, Settings, WakeController

RECORD_KEY = "_qq_group_jev_record"
CID_KEY = "_qq_group_jev_conversation"
RESPONSE_KEY = "_qq_group_jev_response"
BOT_MENTION_KEY = "_qq_group_jev_bot_mention_preserved"
PLUGIN_NAME = "astrbot_plugin_qq_group_enhance"


class GroupWakeFilter(filter.CustomFilter):
    """Select observers; this filter never changes wake flags or caches messages."""

    plugin: ClassVar[QQGroupEnhancePlugin | None] = None

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        plugin = self.plugin
        raw = getattr(event.message_obj, "raw_message", None)
        return bool(
            plugin
            and event.get_platform_name() == "aiocqhttp"
            and plugin.serves_group(event.get_group_id())
            and event.get_sender_id()
            and isinstance(raw, Mapping)
            and raw.get("post_type") in ("message", "message_sent")
        )


class QQGroupEnhancePlugin(Star):
    """Observe group conversations and wake the ordinary reply pipeline via Jev."""

    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.settings: Settings | None = None
        self.client: JevClient | None = None
        self.controller: WakeController | None = None
        self.config_error: str | None = None
        self.observer = RuntimeObserver(self)
        self.replies = ReplyDispatcher(self)
        self.closed = False
        self.session_groups: dict[str, GroupKey] = {}
        self.unsupported_profiles: set[tuple[str, str]] = set()
        GroupWakeFilter.plugin = self
        try:
            self.settings = Settings.from_mapping({} if config is None else config)
        except ValueError as error:
            # Keep the Star instance loadable so WebUI can repair its original
            # config. Invalid settings must never become runtime defaults.
            self.config_error = str(error)
            return
        self.client = JevClient(self.settings)
        self.controller = WakeController(
            self.settings, self._judge, self._semantic_wake
        )

    @property
    def active(self) -> bool:
        return not self.closed and self.settings is not None

    @property
    def semantic_enabled(self) -> bool:
        return self.active and self.settings.enabled and bool(self.settings.jev_api_key)

    def serves_group(self, group_id: str) -> bool:
        return bool(
            self.active
            and group_id
            and (
                not self.settings.group_whitelist
                or str(group_id) in self.settings.group_whitelist
            )
        )

    async def initialize(self) -> None:
        if self.config_error is not None:
            logger.error(
                "群聊增强插件已加载，但配置无效，功能已暂停：%s。"
                "请在 WebUI 中修正配置、保存并重载插件。",
                self.config_error,
            )
            return
        if self.active:
            try:
                self.observer.install()
            except Exception:
                self.observer.close()
                raise
            if not self.settings.enabled:
                logger.info(
                    "Jev 语义判断已关闭：保留关键词、@和引用的直接唤醒及内存历史；"
                    "不建立待判窗口，不请求 Jev。"
                )
            elif not self.settings.jev_api_key:
                logger.warning(
                    "未配置 jev_api_key：插件已加载，直接唤醒正常运行；"
                    "Jev 语义判断暂停。请在插件配置中填写密钥并重载插件。"
                )

    async def _judge(self, key, history, candidates, valid):
        if not self.semantic_enabled:
            return {}
        group = self.controller.groups[key]

        async def eligible(ids):
            if not self.semantic_enabled:
                return set()
            allowed = set()
            checks = {}
            for mid in ids:
                record = group.inflight.get(mid)
                if not record or not record.event:
                    continue
                session = (
                    record.event.unified_msg_origin,
                    record.event.get_extra(CID_KEY),
                )
                if session not in checks:
                    checks[session] = await self._session_eligible(record.event, record)
                if checks[session]:
                    allowed.add(mid)
            return allowed if self.semantic_enabled else set()

        def failed(ids):
            for mid in ids:
                record = group.inflight.get(mid)
                if record and valid(mid):
                    self.controller.archive(group, record, "failed")

        return await self.client.judge(history, candidates, valid, eligible, failed)

    def admitted(self, event: AstrMessageEvent) -> bool:
        return bool(
            self.active
            and not event.is_stopped()
            and event.get_platform_name() == "aiocqhttp"
            and self.serves_group(event.get_group_id())
            and (
                event.get_extra(REPLY_KEY) is not None
                or any(
                    handler.handler_module_path == __name__
                    and handler.handler_name == "observe_message"
                    for handler in event.get_extra("activated_handlers", [])
                )
            )
        )

    @staticmethod
    def key(event: AstrMessageEvent) -> GroupKey:
        return str(event.get_platform_id()), str(event.get_group_id())

    def explicit(self, event: AstrMessageEvent) -> bool:
        if not self.serves_group(event.get_group_id()):
            return False
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return False
        parts = event.get_messages()
        text = "".join(p.text for p in parts if isinstance(p, Plain))
        words = self.settings.keywords
        if self.settings.keyword_ignore_case:
            text, words = text.casefold(), tuple(w.casefold() for w in words)
        return any(w in text for w in words) or any(
            (isinstance(p, At) and str(p.qq) == str(event.get_self_id()))
            or (isinstance(p, Reply) and str(p.sender_id) == str(event.get_self_id()))
            for p in parts
        )

    @staticmethod
    def _preserve_bot_mention(event: AstrMessageEvent) -> None:
        """Restore the bot mention text without changing the message chain."""
        if event.get_extra(BOT_MENTION_KEY) or str(event.get_sender_id()) == str(
            event.get_self_id()
        ):
            return
        parts = event.get_messages()
        mention = next(
            (
                part
                for part in parts
                if isinstance(part, At) and str(part.qq) == str(event.get_self_id())
            ),
            None,
        )
        if mention is None:
            return
        event.set_extra(BOT_MENTION_KEY, True)
        current = str(event.message_str or "")
        nickname = str(getattr(mention, "name", "") or "").strip()
        if not nickname or nickname == str(event.get_self_id()):
            nickname = "机器人"
        prefix = f"@{nickname}"
        visible = current.lstrip()
        if visible == prefix or (
            visible.startswith(prefix)
            and visible[len(prefix) : len(prefix) + 1].isspace()
        ):
            return
        text = current.strip()
        restored = prefix if not text else f"{prefix} {text}"
        event.message_str = restored
        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            message_obj.message_str = restored

    async def observe(self, event: AstrMessageEvent) -> None:
        """Register arrival and wake state before asynchronous media processing."""
        if not self.admitted(event):
            return
        if self.settings.preserve_bot_mention:
            self._preserve_bot_mention(event)
        key = self.key(event)
        self.session_groups[event.unified_msg_origin] = key
        self.observer.watch_bot(event)
        if event.get_extra(RECORD_KEY):
            self.refresh(event)
            return
        bot = str(event.get_sender_id()) == str(event.get_self_id())
        mid = str(event.message_obj.message_id or "")
        command = any(
            isinstance(f, (CommandFilter, CommandGroupFilter))
            for h in event.get_extra("activated_handlers", [])
            for f in h.event_filters
        )
        if command:
            if mid:
                self.controller.record_arrival(key, mid, bot)
            if event.is_at_or_wake_command:
                self.controller.awakened(key)
            return
        if not bot and self.explicit(event):
            event.is_at_or_wake_command = True
            event.is_wake = True
        if not bot and event.is_at_or_wake_command:
            self.controller.awakened(key)
        if not mid:
            logger.error("Jev cannot track a QQ message without a message ID")
            return
        record = ChatMessage(
            id=mid,
            sender_id=str(event.get_sender_id()),
            sender_name=event.get_sender_name() or "",
            text="",
            timestamp=float(event.message_obj.timestamp or time.time()),
            ready=False,
            bot=bot,
            bot_id=str(event.get_self_id()),
            event=event,
            scheduler=ACTIVE_PIPELINE.get(),
            task=asyncio.current_task(),
        )
        event.set_extra(RECORD_KEY, record)
        added = self.controller.add(
            key,
            record,
            candidate=self.semantic_enabled
            and not bot
            and not event.is_at_or_wake_command,
        )
        if not added:
            existing = self.controller.group(key).records().get(mid)
            if existing:
                event.set_extra(RECORD_KEY, existing)
        cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        event.set_extra(CID_KEY, cid)

    def refresh(self, event: AstrMessageEvent) -> None:
        if not self.admitted(event):
            return
        if event.is_at_or_wake_command and (
            str(event.get_sender_id()) != str(event.get_self_id())
        ):
            self.controller.awakened(self.key(event))

    @filter.custom_filter(GroupWakeFilter, priority=maxsize)
    async def observe_message(self, event: AstrMessageEvent) -> None:
        await self.observe(event)

    @filter.custom_filter(GroupWakeFilter, priority=-10000)
    async def collect_message(self, event: AstrMessageEvent) -> None:
        """Capture after QQ enrichment (0), before its reply debounce (-20000)."""
        if not self.admitted(event) or event.get_extra(REPLY_KEY):
            return
        record = event.get_extra(RECORD_KEY)
        if record is None or record.event is not event or record.ready:
            return
        parts = event.get_messages()
        if (
            self.settings.preserve_bot_mention
            and event.get_extra(BOT_MENTION_KEY)
            and len(parts) == 1
            and isinstance(parts[0], At)
            and str(parts[0].qq) == str(event.get_self_id())
            and not event.get_extra("_group_context_record_id")
            and not event.get_extra("handlers_parsed_params", {})
            and self.context.get_config(umo=event.unified_msg_origin)[
                "provider_ltm_settings"
            ]["group_icl_enable"]
        ):
            # The native collector skips At-only chains. Record the boundary before
            # reply debounce so later arrivals remain available to their own turns.
            builtin = self.context.get_registered_star("astrbot")
            if (
                builtin is None
                or builtin.star_cls is None
                or builtin.star_cls.group_chat_context is None
            ):
                raise RuntimeError(
                    "Native group chat context is unavailable for a mention-only reply"
                )
            await builtin.star_cls.group_chat_context.handle_message(event)
        quote = next((p for p in parts if isinstance(p, Reply)), None)
        record.text = (
            str(event.message_str or "").strip() or event.get_message_outline()
        )
        record.mentions = tuple(str(p.qq) for p in parts if isinstance(p, At))
        record.reply = (
            {
                "message_id": str(quote.id),
                "sender_id": str(quote.sender_id),
                "text": str(quote.message_str or ""),
            }
            if quote
            else None
        )
        self.controller.prepared(self.key(event), record)
        self.refresh(event)

    @filter.custom_filter(GroupWakeFilter, priority=-maxsize)
    async def observe_later_wake(self, event: AstrMessageEvent) -> None:
        self.refresh(event)

    @filter.on_llm_request(priority=-maxsize)
    async def observe_request(self, event, request) -> None:
        if self.admitted(event):
            self.controller.awakened(self.key(event))
            self.refresh(event)

    @filter.on_agent_begin(priority=-maxsize)
    async def observe_agent(self, event, run_context) -> None:
        if self.admitted(event):
            self.observer.capture_current_input(event, run_context.messages)

    @filter.on_llm_response(priority=-maxsize)
    async def observe_response(self, event, response) -> None:
        if not self.admitted(event):
            return
        if (
            response
            and response.role == "assistant"
            and not response.tools_call_name
            and (response.completion_text or response.result_chain)
        ):
            event.set_extra(RESPONSE_KEY, response)

    async def _session_eligible(self, event, source: ChatMessage) -> bool:
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

        if (
            not self.semantic_enabled
            or not self.serves_group(event.get_group_id())
            or event.is_stopped()
            or source.event.is_stopped()
            or source.event.call_llm
            or source.event._has_send_oper
        ):
            return False
        if not await SessionServiceManager.is_session_enabled(event.unified_msg_origin):
            return False
        if not await SessionPluginManager.is_plugin_enabled_for_session(
            event.unified_msg_origin, PLUGIN_NAME
        ):
            return False
        if not await SessionServiceManager.should_process_llm_request(event):
            return False
        cfg = self.context.get_config(umo=event.unified_msg_origin)
        if source.scheduler is None or cfg is not source.scheduler.ctx.astrbot_config:
            return False
        platform = cfg.get("platform_settings", {})
        whitelist = {
            str(i).strip() for i in platform.get("id_whitelist", []) if str(i).strip()
        }
        admin_exempt = platform.get("wl_ignore_admin_on_group", False) and str(
            event.get_sender_id()
        ) in cfg.get("admins_id", [])
        if (
            platform.get("enable_id_white_list", False)
            and whitelist
            and not admin_exempt
            and event.unified_msg_origin not in whitelist
            and str(event.get_group_id()).strip() not in whitelist
        ):
            return False
        if PLUGIN_NAME not in cfg.get("plugin_set", ["*"]) and "*" not in cfg.get(
            "plugin_set", ["*"]
        ):
            return False
        if not cfg.get("provider_settings", {}).get("enable", True):
            return False
        reason = ""
        process = (
            next(
                (s for s in source.scheduler.stages if isinstance(s, ProcessStage)),
                None,
            )
            if source.scheduler is not None
            else None
        )
        if process is None or not isinstance(
            process.agent_sub_stage.agent_sub_stage, internal.InternalAgentSubStage
        ):
            reason = "需要使用 AstrBot 本地 Agent 执行器"
        elif not cfg.get("provider_ltm_settings", {}).get("group_icl_enable", False):
            reason = "需要开启 AstrBot 群聊消息记录注入上下文"
        if reason:
            report = (event.unified_msg_origin, reason)
            if report not in self.unsupported_profiles:
                self.unsupported_profiles.add(report)
                logger.error(
                    "Jev 语义唤醒未运行：%s（%s）", reason, event.unified_msg_origin
                )
            return False
        cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        return self.semantic_enabled and cid == event.get_extra(CID_KEY)

    async def _semantic_wake(
        self, key: GroupKey, group: GroupState, positive: list[ChatMessage]
    ) -> bool:
        if not self.semantic_enabled:
            return False
        positive = [m for m in positive if self.controller.valid(key, group, m)]
        if not positive:
            return False
        # Pick the newest observed message in this session, allowing native group
        # injection to include arrivals made while Jev was running.
        origin = max(positive, key=lambda m: m.sequence).event.unified_msg_origin
        source = max(
            (
                m
                for m in group.records().values()
                if m.event
                and m.ready
                and not m.bot
                and m.event.unified_msg_origin == origin
                and self.controller.valid(key, group, m)
            ),
            key=lambda m: m.sequence,
        )
        if (
            source.task
            and source.task is not asyncio.current_task()
            and not source.task.done()
        ):
            await asyncio.wait({source.task})
        if not any(self.controller.valid(key, group, m) for m in positive):
            return False
        if not await self._session_eligible(source.event, source):
            return False
        if not any(self.controller.valid(key, group, m) for m in positive):
            return False
        positive = [m for m in positive if self.controller.valid(key, group, m)]
        self.replies.submit(key, group, positive, source)
        return True

    async def terminate(self) -> None:
        self.closed = True
        if GroupWakeFilter.plugin is self:
            GroupWakeFilter.plugin = None
        if self.controller is not None:
            await self.controller.close()
        await self.replies.close()
        self.observer.close()
        if self.client is not None:
            await self.client.close()
        self.session_groups.clear()
        self.unsupported_profiles.clear()
