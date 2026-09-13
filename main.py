from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from sys import maxsize
from typing import ClassVar

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter

from .state import GroupKey, Settings, TrafficController

DECISION_KEY = "_qq_group_enhance_decision"


@dataclass(frozen=True)
class WakeDecision:
    group_key: GroupKey
    sender: str
    explicit: bool
    self_message: bool
    eligible: bool


class GroupWakeFilter(filter.CustomFilter):
    """Observe traffic before plugin handlers without changing command matching."""

    plugin: ClassVar[QQGroupEnhancePlugin | None] = None

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        """Select wake candidates and already-woken events needing policy checks.

        Args:
            event: AstrBot adapter event.
            cfg: Session configuration supplied by the waking stage.

        Returns:
            Whether the plugin's later handler should evaluate this event.
        """
        plugin = self.plugin
        if plugin is None or not plugin.settings.enabled or plugin.closed:
            return False
        if event.get_platform_name() != "aiocqhttp" or event.is_private_chat():
            return False
        raw = getattr(event.message_obj, "raw_message", None)
        if not isinstance(raw, Mapping) or raw.get("post_type") not in (
            "message",
            "message_sent",
        ):
            return False
        group_id, sender = event.get_group_id(), event.get_sender_id()
        if not group_id or not sender:
            return False
        existing = event.get_extra(DECISION_KEY)
        if existing is not None:
            return existing.eligible
        self_message = sender == event.get_self_id()
        key = (event.get_platform_id(), group_id)
        now = time.monotonic()
        minute = plugin.minute(now)
        group = None
        if not self_message or plugin.settings.count_bot_messages:
            group = plugin.controller.record(key, minute, now)
        # Only top-level user text counts as a keyword, not quotes or card data.
        messages = event.get_messages()
        text = "".join(part.text for part in messages if isinstance(part, Plain))
        keywords = plugin.settings.keywords
        if not plugin.settings.keyword_case_sensitive:
            text = text.casefold()
            keywords = tuple(keyword.casefold() for keyword in keywords)
        explicit = not self_message and (
            any(keyword in text for keyword in keywords)
            or any(
                isinstance(part, At) and str(part.qq) == str(event.get_self_id())
                for part in messages
            )
        )
        # Core @all/quote/prefix wake must reach the handler to be checked too.
        eligible = bool(
            event.is_at_or_wake_command
            or explicit
            or (
                group
                and not self_message
                and plugin.controller.allows(group, sender, now)
            )
        )
        event.set_extra(
            DECISION_KEY, WakeDecision(key, sender, explicit, self_message, eligible)
        )
        return eligible


class QQGroupEnhancePlugin(Star):
    """Add traffic-dependent QQ wake scopes without model tools or extra queues."""

    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.settings = Settings.from_mapping(config or {})
        self.controller = TrafficController(self.settings)
        # Align to wall-clock minutes once, then use monotonic time for stability.
        self.clock_offset = time.time() - time.monotonic()
        self.timer: asyncio.Task | None = None
        self.closed = False
        GroupWakeFilter.plugin = self

    def minute(self, now: float) -> int:
        return int((now + self.clock_offset) // 60)

    async def initialize(self) -> None:
        if self.settings.enabled and self.timer is None:
            self.timer = asyncio.create_task(self._minute_loop())

    async def _minute_loop(self) -> None:
        while True:
            now = time.monotonic()
            self.controller.tick(self.minute(now), now)
            await asyncio.sleep(60 - (time.monotonic() + self.clock_offset) % 60)

    @filter.custom_filter(GroupWakeFilter, priority=maxsize)
    async def apply_wake_policy(self, event: AstrMessageEvent) -> None:
        """Apply policy before empty-mention waiting, QQ enrichment and debounce.

        Args:
            event: Event admitted by the normal whitelist and session checks.
        """
        decision = event.get_extra(DECISION_KEY)
        if (
            not self.settings.enabled
            or self.closed
            or decision is None
            or event.is_stopped()
        ):
            return
        if decision.self_message:
            event.is_at_or_wake_command = False
            return
        # Recognized commands keep their ordinary AstrBot behavior and do not
        # accidentally create wake windows (e.g. a keyword in a command argument).
        if any(
            isinstance(item, (CommandFilter, CommandGroupFilter))
            for handler in event.get_extra("activated_handlers", [])
            for item in handler.event_filters
        ):
            return
        now = time.monotonic()
        group = self.controller.groups.get(decision.group_key)
        if group is None:
            return
        self.controller.advance(group, self.minute(now), now)
        if decision.explicit:
            self.controller.wake(group, decision.sender, now)
        allowed = decision.explicit or self.controller.allows(
            group, decision.sender, now
        )
        event.is_at_or_wake_command = allowed
        if allowed:
            event.is_wake = True

    async def terminate(self) -> None:
        self.closed = True
        if GroupWakeFilter.plugin is self:
            GroupWakeFilter.plugin = None
        if self.timer is not None:
            self.timer.cancel()
            with suppress(asyncio.CancelledError):
                await self.timer
            self.timer = None
        self.controller.groups.clear()
