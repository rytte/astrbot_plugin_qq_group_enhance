"""Internal reply requests that never enter AstrBot's incoming message queue."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from copy import copy, deepcopy
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.utils.active_event_registry import active_event_registry

from .state import ChatMessage, GroupKey, GroupState

REPLY_KEY = "_qq_group_jev_reply"


@dataclass(eq=False)
class ReplyRequest:
    key: GroupKey
    group: GroupState
    positive: tuple[ChatMessage, ...]
    source: ChatMessage
    event: AstrMessageEvent
    submitted: bool = False


class ReplyDispatcher:
    def __init__(self, plugin):
        self.plugin = plugin
        self.tasks: dict[asyncio.Task, ReplyRequest] = {}

    def submit(self, key, group, positive, source) -> None:
        """Publish one internal request, retaining prepared input and its media."""
        from .main import CID_KEY, RECORD_KEY, RESPONSE_KEY

        for task, request in self.tasks.items():
            if not task.done() and request.group is group and request.source is source:
                # A newer window can judge the same source while its reply is
                # still waiting. Reuse that request and its media ownership.
                for message in positive:
                    self.plugin.controller.archive(group, message, "woken")
                request.positive = tuple(dict.fromkeys((*request.positive, *positive)))
                return
        scheduler = source.scheduler
        if scheduler is None or not any(
            isinstance(stage, ProcessStage) for stage in scheduler.stages
        ):
            raise RuntimeError("Jev 无法找到原消息对应的 AstrBot 模型回复阶段")
        original = source.event
        event = copy(original)
        AstrMessageEvent.__init__(
            event,
            original.message_str,
            deepcopy(original.message_obj),
            original.platform_meta,
            original.session_id,
        )
        event.role = original.role
        event.plugins_name = original.plugins_name
        # Retain enrichment metadata, never the old input task's debounce state
        # or handler dispatch list. Model hooks run on the new reply task.
        event._extras = dict(original.get_extra())
        for name in (
            "activated_handlers",
            "handlers_parsed_params",
            "provider_request",
            "agent_stop_requested",
            "_qq_enhance_debounce_arrival",
            RESPONSE_KEY,
        ):
            event._extras.pop(name, None)
        event.set_extra(RECORD_KEY, source)
        event.set_extra(CID_KEY, original.get_extra(CID_KEY))
        request = ReplyRequest(key, group, tuple(positive), source, event)
        event.set_extra(REPLY_KEY, request)
        event.is_at_or_wake_command = True
        event.is_wake = True
        if source.media_owner is not None:
            event._temporary_local_files = source.media_owner._temporary_local_files
            source.media_owner._temporary_local_files = []
            source.media_owner = None
        for message in positive:
            self.plugin.controller.archive(group, message, "woken")
        group.reserved[source.id] = source
        task = asyncio.create_task(self._run(request), name="qq-group-jev-reply")
        self.tasks[task] = request
        task.add_done_callback(self._done)
        self.plugin.controller.awakened(key)
        logger.info(
            "Jev 请求回复 platform=%s group=%s message_id=%s debounce=False",
            *key,
            source.id,
        )

    async def eligible(self, event) -> bool:
        request = event.get_extra(REPLY_KEY)
        if request is None:
            return True

        def current():
            return (
                self.plugin.semantic_enabled
                and self.plugin.controller.groups.get(request.key) is request.group
                and not event.is_stopped()
                and (
                    request.submitted
                    or any(m.status == "woken" for m in request.positive)
                )
            )

        if not current() or not await self.plugin._session_eligible(
            event, request.source
        ):
            return False
        return current()

    async def before_submit(self, event) -> None:
        request = event.get_extra(REPLY_KEY)
        if request is None:
            return
        if not await self.eligible(event):
            event.stop_event()
            # Cancel only our own obsolete reply. This must not become a model
            # error message or be retried by the Agent's provider retry loop.
            raise asyncio.CancelledError("Jev reply no longer needed")
        request.submitted = True

    async def _run(self, request):
        event = request.event
        active_event_registry.register(event)
        try:
            if not await self.eligible(event):
                event.stop_event()
                return
            cfg = self.plugin.context.get_config(umo=event.unified_msg_origin)
            plugins = cfg.get("plugin_set", ["*"])
            event.plugins_name = None if plugins == ["*"] else list(plugins)
            scheduler = request.source.scheduler
            index, process = next(
                (i, stage)
                for i, stage in enumerate(scheduler.stages)
                if isinstance(stage, ProcessStage)
            )
            # Preserve the scheduler's generator/yield contract, including
            # streaming consumption, result decoration and post-send hooks.
            async with aclosing(process.agent_sub_stage.process(event)) as responses:
                async for _ in responses:
                    if event.is_stopped():
                        break
                    await scheduler._process_stages(event, index + 1)
                    if event.is_stopped():
                        break
        finally:
            active_event_registry.unregister(event)

    def _done(self, task):
        request = self.tasks.pop(task)
        source = request.source
        if (
            self.plugin.semantic_enabled
            and self.plugin.controller.valid(request.key, request.group, source)
            and request.event._temporary_local_files
        ):
            # A trimmed/aborted request may leave its source in the next Jev
            # window. Return file ownership until that candidate finishes.
            owner = copy(request.event)
            owner._extras = {}
            owner._temporary_local_files = request.event._temporary_local_files
            request.event._temporary_local_files = []
            source.media_owner = owner
        request.event.cleanup_temporary_local_files()
        for message in (*request.positive, request.source):
            if not any(
                other.group is request.group
                and (message is other.source or message in other.positive)
                for other in self.tasks.values()
            ):
                request.group.reserved.pop(message.id, None)
                if message.status not in ("pending", "judging"):
                    message.release_media()
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(
                "Jev 回复任务失败 group=%s: %s",
                request.key,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    def reset(self, key):
        for task, request in list(self.tasks.items()):
            if request.key == key:
                request.event.stop_event()
                task.cancel()

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            self.tasks[task].event.stop_event()
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
