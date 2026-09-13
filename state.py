"""Minute-bucket traffic accounting and expiring wake windows."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

TrafficLevel = Literal["low", "medium", "high"]
GroupKey = tuple[str, str]


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    keywords: tuple[str, ...] = ("小爱",)
    keyword_case_sensitive: bool = True
    traffic_window_minutes: int = 10
    low_max_messages: int = 15
    medium_max_messages: int = 30
    awake_window_seconds: float = 180
    count_bot_messages: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping) -> Settings:
        """Validate configuration before activating any event handlers.

        Args:
            config: Plugin configuration supplied by AstrBot.

        Returns:
            Validated settings with duplicate keywords removed.

        Raises:
            ValueError: A setting has an invalid type or range.
        """
        defaults = cls()
        values = {
            name: config.get(name, getattr(defaults, name))
            for name in cls.__dataclass_fields__
        }
        for name in ("enabled", "count_bot_messages", "keyword_case_sensitive"):
            if type(values[name]) is not bool:
                raise ValueError(f"{name} 必须为布尔值")
        for name in (
            "traffic_window_minutes",
            "low_max_messages",
            "medium_max_messages",
        ):
            if type(values[name]) is not int or values[name] < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if not 1 <= values["traffic_window_minutes"] <= 1440:
            raise ValueError("traffic_window_minutes 必须在 1～1440 之间")
        if values["medium_max_messages"] <= values["low_max_messages"]:
            raise ValueError("medium_max_messages 必须大于 low_max_messages")
        seconds = values["awake_window_seconds"]
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("awake_window_seconds 必须为有限正数")
        keywords = values["keywords"]
        if not isinstance(keywords, (list, tuple)) or any(
            not isinstance(word, str) or not word.strip() for word in keywords
        ):
            raise ValueError(
                "keywords 必须为非空字符串组成的列表，可用空列表仅保留 @ 唤醒"
            )
        values["keywords"] = tuple(dict.fromkeys(word.strip() for word in keywords))
        return cls(**values)


@dataclass
class GroupState:
    minute: int
    buckets: deque[int]
    current_count: int = 0
    level: TrafficLevel = "low"
    shared_until: float = 0
    users: dict[str, float] = field(default_factory=dict)


class TrafficController:
    """Keep traffic per physical QQ group, independently of UMO isolation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.groups: dict[GroupKey, GroupState] = {}

    def advance(self, group: GroupState, minute: int, now: float) -> None:
        """Settle completed minutes and expire wake windows without new messages.

        Args:
            group: Mutable group state.
            minute: Current minute index on the scheduler's clock.
            now: Monotonic time used for wake expiry.
        """
        elapsed = max(0, minute - group.minute)
        if elapsed:
            size = self.settings.traffic_window_minutes
            if elapsed > size:
                group.buckets = deque([0] * size, maxlen=size)
            else:
                group.buckets.append(group.current_count)
                group.buckets.extend([0] * (elapsed - 1))
            group.current_count = 0
            group.minute = minute
            total = sum(group.buckets)
            previous = group.level
            if total <= self.settings.low_max_messages:
                group.level = "low"
            elif total <= self.settings.medium_max_messages:
                group.level = "medium"
            else:
                group.level = "high"
            if group.level == "high":
                group.shared_until = 0
                group.users.clear()
            elif group.level != previous:
                # Only explicitly awakened users survive low -> medium.
                group.shared_until = (
                    max(group.users.values(), default=0) if group.level == "low" else 0
                )
        if group.shared_until <= now:
            group.shared_until = 0
        group.users = {
            sender: expiry for sender, expiry in group.users.items() if expiry > now
        }

    def record(self, key: GroupKey, minute: int, now: float) -> GroupState:
        """Count one incoming message without changing this minute's traffic level.

        Args:
            key: Platform instance ID and physical QQ group ID.
            minute: Current minute index.
            now: Monotonic time.

        Returns:
            The group's state after accounting for the arrival.
        """
        if key not in self.groups:
            size = self.settings.traffic_window_minutes
            self.groups[key] = GroupState(minute, deque([0] * size, maxlen=size))
        group = self.groups[key]
        self.advance(group, minute, now)
        group.current_count += 1
        return group

    def tick(self, minute: int, now: float) -> None:
        """Roll every group, including silent groups, and reclaim idle state.

        Args:
            minute: Current minute index.
            now: Monotonic time.
        """
        for key, group in list(self.groups.items()):
            self.advance(group, minute, now)
            if (
                not group.current_count
                and not any(group.buckets)
                and not group.shared_until
                and not group.users
            ):
                del self.groups[key]

    def allows(self, group: GroupState, sender: str, now: float) -> bool:
        """Check an ordinary message against the currently active wake scope.

        Args:
            group: Current group state.
            sender: QQ sender ID.
            now: Monotonic time.

        Returns:
            Whether the sender may continue without an explicit wake.
        """
        if group.level == "low":
            return group.shared_until > now
        if group.level == "medium":
            return group.users.get(sender, 0) > now
        return False

    def wake(self, group: GroupState, sender: str, now: float) -> None:
        """Refresh the relevant window for an explicit keyword or bot mention.

        Args:
            group: Current group state.
            sender: QQ sender ID.
            now: Monotonic time.
        """
        if group.level == "high":
            return
        expiry = now + self.settings.awake_window_seconds
        group.users[sender] = expiry
        if group.level == "low":
            group.shared_until = expiry
