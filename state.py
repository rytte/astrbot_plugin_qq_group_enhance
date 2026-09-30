"""Group-scoped semantic wake state and independent, bounded message debouncing."""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from astrbot.api import logger as log

from .laya_models import DEFAULT_LAYA_MODEL, LAYA_MODEL_KEYS
from .semif_models import CUSTOM_SEMIF_MODEL, SEMIF_MODEL_KEYS

GroupKey = tuple[str, str]
RETIRED = {
    "traffic_window_minutes",
    "low_max_messages",
    "medium_max_messages",
    "count_bot_messages",
    "asleep_observe_seconds",
    "awake_observe_seconds",
}


def validate_custom_semif_model(source: str, revision: str) -> None:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("semif_model 必须填写本地模型目录或 Hugging Face 模型 ID")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("semif_revision 必须填写固定模型版本")
    source, revision = source.strip(), revision.strip()
    if not Path(source).is_dir() and (
        not re.fullmatch(r"[^\\/:]+/[^\\/:]+", source)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError(
            "远程 semif_model 需要 owner/repo 和 40 位提交 revision；"
            "本地模型须为现有目录"
        )


@dataclass(frozen=True)
class Settings:
    group_whitelist: tuple[str, ...] = ()
    enabled: bool = True
    keywords: tuple[str, ...] = ()
    keyword_ignore_case: bool = True
    preserve_bot_mention: bool = True
    awake_window_seconds: float = 180
    debounce_seconds: float = 3
    max_wait_seconds: float = 6
    max_messages_after_bot: int = 5
    history_messages: int = 30
    semantic_backend: str = "jev"
    semantic_threshold: float = 0.7
    jev_api_key: str = field(default="", repr=False)
    jev_api_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-1.13.0"
    jev_timeout_seconds: float = 10
    jev_retries: int = 1
    semif_preset: str = CUSTOM_SEMIF_MODEL
    semif_model: str = ""
    semif_revision: str = ""
    semif_device: str = "cuda"
    semif_dtype: str = "float16"
    semif_max_tokens: int = 4096
    semif_batch_rows: int = 8
    semif_timeout_seconds: float = 30
    semif_startup_seconds: float = 180
    laya_model: str = DEFAULT_LAYA_MODEL
    laya_device: str = "cuda"
    laya_batch_rows: int = 8
    laya_timeout_seconds: float = 30
    laya_startup_seconds: float = 180

    @classmethod
    def from_mapping(cls, config: Mapping) -> Settings:
        if not isinstance(config, Mapping) or any(
            not isinstance(k, str) for k in config
        ):
            raise ValueError("插件配置必须为字段名是字符串的配置对象")
        keyword_wake = config.get("keyword_wake", {})
        semantic = config.get("semantic", {})
        for name, group in (("keyword_wake", keyword_wake), ("semantic", semantic)):
            if not isinstance(group, Mapping) or any(
                not isinstance(k, str) for k in group
            ):
                raise ValueError(f"{name} 必须为字段名是字符串的配置对象")
        configured_fields = config.keys() | keyword_wake.keys() | semantic.keys()
        old_thresholds = {"jev_threshold", "semif_threshold", "laya_threshold"}
        removed_thresholds = old_thresholds.intersection(configured_fields)
        if removed_thresholds:
            raise ValueError(
                "旧阈值配置已移除："
                + ", ".join(sorted(removed_thresholds))
                + "；请改用 semantic.semantic_threshold"
            )
        if "semif_source" in configured_fields:
            raise ValueError(
                "semif_source 已移除；SemIf 源码随插件提供，请删除该配置字段"
            )
        removed_pythons = {"semif_python", "laya_python"}.intersection(
            configured_fields
        )
        if removed_pythons:
            raise ValueError(
                "独立 Python 配置已移除："
                + ", ".join(sorted(removed_pythons))
                + "；本地模型统一使用 AstrBot 主程序的 Python，请删除这些字段，"
                "并在 AstrBot 的 Python 环境中安装模型依赖"
            )
        if "keyword_case_sensitive" in configured_fields:
            raise ValueError(
                "keyword_case_sensitive 已移除，请改用 keyword_wake.keyword_ignore_case"
                "（开启表示不区分大小写，关闭表示区分大小写）"
            )
        retired = RETIRED.intersection(configured_fields)
        if "awake_observe_seconds" in retired:
            raise ValueError(
                "awake_observe_seconds 已移除，请配置 semantic.debounce_seconds "
                "和 semantic.max_wait_seconds"
            )
        if retired:
            raise ValueError("请删除废弃配置：" + ", ".join(sorted(retired)))
        direct_fields = {"keywords", "keyword_ignore_case"}
        root_fields = {"group_whitelist", "preserve_bot_mention"}
        semantic_fields = cls.__dataclass_fields__.keys() - direct_fields - root_fields
        for name, fields in (
            ("keyword_wake", direct_fields),
            ("semantic", semantic_fields),
        ):
            misplaced = fields.intersection(config)
            if misplaced:
                raise ValueError(
                    f"请将以下字段移入 {name} 配置分组：" + ", ".join(sorted(misplaced))
                )
        unknown = set(config) - {"keyword_wake", "semantic"} - root_fields
        unknown.update(f"keyword_wake.{k}" for k in set(keyword_wake) - direct_fields)
        unknown.update(f"semantic.{k}" for k in set(semantic) - semantic_fields)
        if unknown:
            raise ValueError("未知配置字段：" + ", ".join(sorted(unknown)))
        defaults = cls()
        supplied = {
            **{k: config[k] for k in root_fields if k in config},
            **keyword_wake,
            **semantic,
        }
        values = {
            k: supplied.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__
        }
        for name in ("enabled", "keyword_ignore_case", "preserve_bot_mention"):
            if type(values[name]) is not bool:
                raise ValueError(f"{name} 必须为布尔值")
        whitelist = values["group_whitelist"]
        if not isinstance(whitelist, (list, tuple)) or any(
            not isinstance(group_id, str)
            or not re.fullmatch(r"[1-9][0-9]*", group_id.strip())
            for group_id in whitelist
        ):
            raise ValueError(
                "group_whitelist 必须为 QQ 群号字符串列表（正整数字符串），"
                "可设为空列表表示不限制群聊"
            )
        values["group_whitelist"] = tuple(
            dict.fromkeys(group_id.strip() for group_id in whitelist)
        )
        for name in (
            "awake_window_seconds",
            "debounce_seconds",
            "max_wait_seconds",
            "semantic_threshold",
            "jev_timeout_seconds",
            "semif_timeout_seconds",
            "semif_startup_seconds",
            "laya_timeout_seconds",
            "laya_startup_seconds",
        ):
            value = values[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} 必须为有限正数")
        if values["semantic_threshold"] > 1:
            raise ValueError("semantic_threshold 必须在 (0, 1] 内")
        if values["max_wait_seconds"] < values["debounce_seconds"]:
            raise ValueError("max_wait_seconds 不能小于 debounce_seconds")
        for name, minimum, maximum in (
            ("max_messages_after_bot", 1, 1000),
            ("history_messages", 1, 1000),
            ("jev_retries", 0, 3),
            ("semif_max_tokens", 256, 32768),
            ("semif_batch_rows", 2, 32),
            ("laya_batch_rows", 2, 32),
        ):
            if type(values[name]) is not int or not minimum <= values[name] <= maximum:
                raise ValueError(f"{name} 必须是 {minimum}～{maximum} 内的整数")
        keywords = values["keywords"]
        if not isinstance(keywords, (list, tuple)) or any(
            not isinstance(k, str) or not k.strip() for k in keywords
        ):
            raise ValueError("keywords 必须为非空字符串列表，可设为空列表")
        values["keywords"] = tuple(dict.fromkeys(k.strip() for k in keywords))
        for name in ("jev_api_key", "jev_api_url", "jev_model"):
            if not isinstance(values[name], str):
                raise ValueError(f"{name} 必须为字符串")
            values[name] = values[name].strip()
        try:
            url = urlsplit(values["jev_api_url"])
            valid_url = (
                url.scheme in ("http", "https")
                and bool(url.hostname)
                and (url.port is None or url.port > 0)
                and url.username is None
                and url.password is None
                and "#" not in values["jev_api_url"]
                and not any(c.isspace() for c in values["jev_api_url"])
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(
                "jev_api_url 必须为完整的 HTTP(S) 请求地址，"
                "且不能包含空白、用户名密码或片段（#）"
            )
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", values["jev_model"]):
            raise ValueError("jev_model 必须填写固定版本，例如 jev-1.13.0")
        backend = values["semantic_backend"]
        if backend not in ("jev", "semif", "laya"):
            raise ValueError("semantic_backend 只能是 jev、semif 或 laya")
        for name in (
            "semif_preset",
            "semif_model",
            "semif_revision",
            "semif_device",
            "semif_dtype",
        ):
            if not isinstance(values[name], str):
                raise ValueError(f"{name} 必须为字符串")
            values[name] = values[name].strip()
        if values["semif_device"] not in ("cuda", "cpu"):
            raise ValueError("semif_device 只能是 cuda 或 cpu")
        if values["semif_dtype"] not in ("float16", "bfloat16", "float32"):
            raise ValueError("semif_dtype 无效")
        if values["semif_device"] == "cpu" and values["semif_dtype"] != "float32":
            raise ValueError("semif_device 为 cpu 时 semif_dtype 必须为 float32")
        if backend == "semif":
            if values["semif_preset"] not in SEMIF_MODEL_KEYS | {CUSTOM_SEMIF_MODEL}:
                raise ValueError(
                    "semif_preset 只能是 custom 或 "
                    + ", ".join(sorted(SEMIF_MODEL_KEYS))
                )
            if values["semif_preset"] == CUSTOM_SEMIF_MODEL:
                validate_custom_semif_model(
                    values["semif_model"], values["semif_revision"]
                )
        for name in ("laya_model", "laya_device"):
            if not isinstance(values[name], str):
                raise ValueError(f"{name} 必须为字符串")
            values[name] = values[name].strip()
        if values["laya_device"] not in ("cuda", "cpu"):
            raise ValueError("laya_device 只能是 cuda 或 cpu")
        if backend == "laya":
            if values["laya_model"] not in LAYA_MODEL_KEYS:
                choices = ", ".join(sorted(LAYA_MODEL_KEYS))
                raise ValueError(f"laya_model \u53ea\u80fd\u662f {choices}")
        return cls(**values)


@dataclass(eq=False)
class ChatMessage:
    id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: float
    sequence: int = 0
    arrived_at: float = 0
    ready: bool = True
    debounce_elapsed: bool = False
    bot: bool = False
    bot_id: str = ""
    mentions: tuple[str, ...] = ()
    reply: dict | None = None
    status: str = "pending"
    event: Any = field(default=None, repr=False)
    scheduler: Any = field(default=None, repr=False)
    media_owner: Any = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    native_text: str = ""
    native_record_id: str = ""
    input_fingerprints: set[str] = field(default_factory=set, repr=False)

    def release_media(self) -> None:
        if self.media_owner is not None:
            owner, self.media_owner = self.media_owner, None
            owner.cleanup_temporary_local_files()

    def payload(self) -> dict:
        return {
            "message_id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "text": self.text,
            "time": self.timestamp,
            "is_bot": self.bot,
            "bot_id": self.bot_id,
            "mentions": list(self.mentions),
            "reply": self.reply,
            "rendered_context": self.native_text,
        }


@dataclass(eq=False)
class GroupState:
    history: OrderedDict[str, ChatMessage] = field(default_factory=OrderedDict)
    pending: OrderedDict[str, ChatMessage] = field(default_factory=OrderedDict)
    inflight: dict[str, ChatMessage] = field(default_factory=dict)
    reserved: dict[str, ChatMessage] = field(default_factory=dict)
    seen: OrderedDict[str, None] = field(default_factory=OrderedDict)
    awake_until: float = 0
    deadline: float | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task | None = None
    sequence: int = 0
    last_bot_sequence: int | None = None

    def records(self) -> dict[str, ChatMessage]:
        return {**self.history, **self.reserved, **self.inflight, **self.pending}


class WakeController:
    """Own timers without awaiting them in AstrBot's message processing chain.

    Args:
        settings: Validated plugin configuration.
        judge: Semantic evaluator accepting a candidate-validity callback.
        wake: Callback scheduling an independent AstrBot reply task.
        clock: Monotonic clock, injectable for timing verification.
    """

    MAX_PENDING = 2000
    MAX_PENDING_BYTES = 8 * 1024 * 1024
    MAX_GROUPS = 512

    def __init__(
        self,
        settings: Settings,
        judge: Callable[..., Awaitable[dict[str, float | None]]],
        wake: Callable[..., Awaitable[bool]],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings, self.judge, self.wake, self.clock = settings, judge, wake, clock
        self.backend_name = {"jev": "Jev", "semif": "SemIf", "laya": "Laya"}[
            settings.semantic_backend
        ]
        self.groups: dict[GroupKey, GroupState] = {}
        self.closed = False
        self.tasks: set[asyncio.Task] = set()

    def group(self, key: GroupKey) -> GroupState:
        if key not in self.groups:
            if len(self.groups) >= self.MAX_GROUPS:
                idle = next(
                    (
                        k
                        for k, g in self.groups.items()
                        if not g.worker
                        and not g.pending
                        and not g.inflight
                        and not g.reserved
                        and g.awake_until <= self.clock()
                    ),
                    None,
                )
                if idle is None:
                    raise RuntimeError("语义判断群组内存容量已满")
                del self.groups[idle]
                log.warning("语义判断已清理容量上限处的空闲群历史：%s", idle)
            self.groups[key] = GroupState()
        return self.groups[key]

    def is_awake(self, group: GroupState) -> bool:
        return group.awake_until > self.clock()

    def awakened(self, key: GroupKey) -> None:
        group = self.group(key)
        if not self.is_awake(group):
            group.awake_until = self.clock() + self.settings.awake_window_seconds

    def replied(self, key: GroupKey) -> None:
        self.group(key).awake_until = self.clock() + self.settings.awake_window_seconds

    def record_arrival(self, key: GroupKey, mid: str, bot: bool) -> int | None:
        """Count each observed message once, independently of history retention."""
        if self.closed:
            return None
        group = self.group(key)
        if mid in group.seen:
            return None
        group.seen[mid] = None
        while len(group.seen) > 8192:
            group.seen.popitem(last=False)
        group.sequence += 1
        if bot:
            group.last_bot_sequence = group.sequence
        return group.sequence

    def add(self, key: GroupKey, message: ChatMessage, candidate: bool) -> bool:
        sequence = self.record_arrival(key, message.id, message.bot)
        if sequence is None:
            return False
        group = self.groups[key]
        message.sequence = sequence
        message.arrived_at = self.clock()
        near_bot = (
            group.last_bot_sequence is not None
            and 0
            < sequence - group.last_bot_sequence
            <= self.settings.max_messages_after_bot
        )
        # Eligibility is fixed at arrival; later messages or expiry do not
        # revoke an admitted candidate. Actual input coverage can still do so.
        if not candidate or message.bot or not (self.is_awake(group) or near_bot):
            self.archive(group, message, "background")
            return True
        group.pending[message.id] = message
        self._limit_pending(key, group)
        self._reschedule(group)
        if group.pending and group.worker is None:
            group.worker = asyncio.create_task(self._run(key, group))
            self.tasks.add(group.worker)
            group.worker.add_done_callback(self.tasks.discard)
        return True

    def prepared(self, key: GroupKey, message: ChatMessage) -> None:
        """Publish enriched content without changing its arrival or eligibility."""
        group = self.groups.get(key)
        if (
            self.closed
            or group is None
            or group.records().get(message.id) is not message
        ):
            return
        message.ready = True
        self._limit_pending(key, group)
        group.changed.set()

    def _limit_pending(self, key: GroupKey, group: GroupState) -> None:
        # Overload is explicit terminal failure, never a silent maxlen eviction.
        size = sum(len(m.text.encode("utf-8")) for m in group.pending.values())
        while len(group.pending) > self.MAX_PENDING or size > self.MAX_PENDING_BYTES:
            oldest = next(iter(group.pending.values()))
            size -= len(oldest.text.encode("utf-8"))
            self.archive(group, oldest, "capacity_failed")
            log.error(
                "%s pending capacity exceeded; archived message %s in %s",
                self.backend_name,
                oldest.id,
                key,
            )

    def _reschedule(self, group: GroupState) -> None:
        arrivals = [
            m.arrived_at for m in group.pending.values() if not m.debounce_elapsed
        ]
        group.deadline = (
            min(
                max(arrivals) + self.settings.debounce_seconds,
                min(arrivals) + self.settings.max_wait_seconds,
            )
            if arrivals
            else None
        )
        group.changed.set()

    def archive(self, group: GroupState, message: ChatMessage, status: str) -> None:
        group.pending.pop(message.id, None)
        if message.status != "covered":
            message.status = status
        if message.status == "woken":
            # Queued/in-progress wakes need identity evidence even after the
            # 30-entry background history rolls over.
            group.reserved[message.id] = message
        elif message.status == "covered":
            group.reserved.pop(message.id, None)
        if message.status != "woken":
            message.release_media()
        group.history[message.id] = message
        group.history = OrderedDict(
            sorted(group.history.items(), key=lambda item: item[1].sequence)
        )
        while len(group.history) > self.settings.history_messages:
            group.history.popitem(last=False)
        self._reschedule(group)

    def covered(self, key: GroupKey, ids: set[str]) -> None:
        group = self.groups.get(key)
        if group is None:
            return
        records = group.records()
        for mid in ids:
            if message := records.get(mid):
                self.archive(group, message, "covered")

    def valid(self, key: GroupKey, group: GroupState, message: ChatMessage) -> bool:
        return (
            not self.closed
            and self.groups.get(key) is group
            and message.status in ("pending", "judging")
        )

    async def _run(self, key: GroupKey, group: GroupState) -> None:
        try:
            while not self.closed and self.groups.get(key) is group and group.pending:
                group.changed.clear()
                if group.deadline is not None and group.deadline <= self.clock():
                    # Close the arrival batch even if some media is still being
                    # prepared. Those messages cannot expire a newer batch.
                    for message in group.pending.values():
                        message.debounce_elapsed = True
                    self._reschedule(group)
                candidates = [
                    m for m in group.pending.values() if m.ready and m.debounce_elapsed
                ]
                if not candidates:
                    delay = (
                        max(0, group.deadline - self.clock())
                        if group.deadline is not None
                        else None
                    )
                    try:
                        await asyncio.wait_for(group.changed.wait(), delay)
                    except TimeoutError:
                        pass
                    continue
                for message in candidates:
                    group.pending.pop(message.id)
                self._reschedule(group)
                group.inflight = {m.id: m for m in candidates}
                for message in candidates:
                    message.status = "judging"
                history = [m.payload() for m in group.history.values() if m.ready]
                payloads = [m.payload() for m in candidates]
                try:
                    results = await self.judge(
                        key,
                        history,
                        payloads,
                        lambda mid: (
                            mid in group.inflight
                            and self.valid(key, group, group.inflight[mid])
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("%s batch failed for %s", self.backend_name, key)
                    results = {}
                if self.groups.get(key) is not group or self.closed:
                    return
                log.debug(
                    "%s group=%s probabilities=%s", self.backend_name, key, results
                )
                positive = [
                    m
                    for m in candidates
                    if self.valid(key, group, m)
                    and results.get(m.id) is not None
                    and results[m.id] >= self.settings.semantic_threshold
                ]
                woke = False
                if positive:
                    try:
                        woke = await self.wake(key, group, positive)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception("Could not schedule semantic wake for %s", key)
                log.info(
                    "%s 唤醒汇总 | 平台=%s | 群=%s | 候选=%d条"
                    " | 有效通过=%d条 | 新增唤醒=%s",
                    self.backend_name,
                    key[0],
                    key[1],
                    len(candidates),
                    len(positive),
                    "是" if woke else "否",
                )
                for message in candidates:
                    if self.valid(key, group, message):
                        status = (
                            "failed"
                            if results.get(message.id) is None
                            else (
                                "woken"
                                if woke and message in positive
                                else "background"
                            )
                        )
                        self.archive(group, message, status)
                group.inflight.clear()
        except asyncio.CancelledError:
            raise
        finally:
            group.worker = None

    def reset(self, key: GroupKey) -> None:
        group = self.groups.pop(key, None)
        if group and group.worker:
            group.worker.cancel()
        if group:
            for message in group.records().values():
                message.release_media()

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for key in list(self.groups):
            self.reset(key)
