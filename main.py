from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from sys import maxsize
from typing import ClassVar
from uuid import uuid4

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.session_plugin_manager import SessionPluginManager

from .bridge import ACTIVE_PIPELINE, RuntimeObserver
from .jev import JevClient
from .laya_client import LayaClient
from .laya_models import (
    LAYA_MODELS,
    get_laya_model_dir,
    get_laya_model_spec,
    get_laya_models_dir,
    is_laya_model_ready,
    missing_model_files,
)
from .model_test import ModelTester
from .reply import REPLY_KEY, ReplyDispatcher
from .semif import SemIfClient
from .semif_models import (
    CUSTOM_SEMIF_MODEL,
    SEMIF_MODELS,
    get_semif_model_dir,
    get_semif_model_spec,
    get_semif_models_dir,
    is_semif_model_ready,
    missing_semif_model_files,
)
from .state import (
    ChatMessage,
    GroupKey,
    GroupState,
    Settings,
    WakeController,
    validate_custom_semif_model,
)

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
    """Observe group conversations and wake the reply pipeline via a chosen scorer."""

    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.settings: Settings | None = None
        self.client: JevClient | SemIfClient | LayaClient | None = None
        self.semantic_failure: str | None = None
        self.controller: WakeController | None = None
        self.config_error: str | None = None
        self.observer = RuntimeObserver(self)
        self.replies = ReplyDispatcher(self)
        self.closed = False
        self.session_groups: dict[str, GroupKey] = {}
        self.unsupported_profiles: set[tuple[str, str]] = set()
        self.plugin_config = config if isinstance(config, Mapping) else None
        self._download_lock = asyncio.Lock()
        self._download_tasks: dict[str, asyncio.Task] = {}
        self._download_errors: dict[str, str] = {}
        self._semif_download_lock = asyncio.Lock()
        self._semif_download_tasks: dict[str, asyncio.Task] = {}
        self._semif_download_errors: dict[str, str] = {}
        self.model_tester = ModelTester()
        GroupWakeFilter.plugin = self
        register_web_api = getattr(context, "register_web_api", None)
        if callable(register_web_api):
            register_web_api(
                f"/{PLUGIN_NAME}/laya/models",
                self.page_laya_models,
                ["GET"],
                "List Laya models and download status",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/laya/models/download",
                self.page_laya_model_download,
                ["POST"],
                "Download a Laya model",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/laya/models/select",
                self.page_laya_model_select,
                ["POST"],
                "Select the Laya model",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/semif/models",
                self.page_semif_models,
                ["GET"],
                "List SemIf presets and download status",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/semif/models/download",
                self.page_semif_model_download,
                ["POST"],
                "Download a pinned SemIf preset",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/semif/models/select",
                self.page_semif_model_select,
                ["POST"],
                "Select a SemIf preset or custom model",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/semantic/test/settings",
                self.page_semantic_test_settings,
                ["GET"],
                "Get model test settings",
            )
            register_web_api(
                f"/{PLUGIN_NAME}/semantic/test",
                self.page_semantic_test,
                ["POST"],
                "Test a semantic model",
            )
        try:
            self.settings = Settings.from_mapping({} if config is None else config)
        except ValueError as error:
            # Keep the Star instance loadable so WebUI can repair its original
            # config. Invalid settings must never become runtime defaults.
            self.config_error = str(error)
            return
        client_type = {
            "jev": JevClient,
            "semif": SemIfClient,
            "laya": LayaClient,
        }[self.settings.semantic_backend]
        self.client = client_type(self.settings)
        self.controller = WakeController(
            self.settings, self._judge, self._semantic_wake
        )

    @property
    def active(self) -> bool:
        return not self.closed and self.settings is not None

    def _configured_laya_model(self) -> str:
        config = self.plugin_config
        if isinstance(config, Mapping):
            semantic = config.get("semantic")
            if isinstance(semantic, Mapping) and isinstance(
                semantic.get("laya_model"), str
            ):
                return semantic["laya_model"]
        if self.settings is not None:
            return self.settings.laya_model
        return "multilingual"

    def _configured_semif_preset(self) -> str:
        config = self.plugin_config
        if isinstance(config, Mapping):
            semantic = config.get("semantic")
            if isinstance(semantic, Mapping) and isinstance(
                semantic.get("semif_preset"), str
            ):
                return semantic["semif_preset"]
        return (
            self.settings.semif_preset
            if self.settings is not None
            else CUSTOM_SEMIF_MODEL
        )

    def _semif_model_payload(self, spec) -> dict:
        task = self._semif_download_tasks.get(spec.key)
        if task is not None and not task.done():
            state = "downloading"
        elif is_semif_model_ready(spec.key):
            state = "ready"
        elif spec.key in self._semif_download_errors:
            state = "error"
        else:
            state = "idle"
        return {
            "key": spec.key,
            "label": spec.label,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "description": spec.description,
            "state": state,
            "error": self._semif_download_errors.get(spec.key),
            "selected": spec.key == self._configured_semif_preset(),
        }

    async def page_semif_models(self):
        return json_response(
            {
                "selected": self._configured_semif_preset(),
                "models": [self._semif_model_payload(spec) for spec in SEMIF_MODELS],
            }
        )

    async def page_semif_model_select(self):
        payload = await request.json(default={})
        key = payload.get("model") if isinstance(payload, Mapping) else None
        if key != CUSTOM_SEMIF_MODEL:
            try:
                get_semif_model_spec(key)
            except (TypeError, ValueError) as error:
                return error_response(str(error), status_code=400)
        config = self.plugin_config
        if not isinstance(config, Mapping) or not hasattr(config, "save_config"):
            return error_response(
                "插件配置不可写，请在 AstrBot 插件配置中选择模型", status_code=409
            )
        semantic = config.get("semantic")
        if not isinstance(semantic, dict):
            semantic = {}
        if key == CUSTOM_SEMIF_MODEL:
            try:
                validate_custom_semif_model(
                    semantic.get("semif_model"), semantic.get("semif_revision")
                )
            except ValueError as error:
                return error_response(str(error), status_code=400)
        config["semantic"] = semantic
        semantic["semif_preset"] = key
        config.save_config()
        return json_response({"selected": key, "requires_reload": True})

    async def page_semif_model_download(self):
        payload = await request.json(default={})
        key = payload.get("model") if isinstance(payload, Mapping) else None
        try:
            spec = get_semif_model_spec(key)
        except (TypeError, ValueError) as error:
            return error_response(str(error), status_code=400)
        async with self._semif_download_lock:
            task = self._semif_download_tasks.get(spec.key)
            if task is not None and not task.done():
                return json_response(self._semif_model_payload(spec))
            if is_semif_model_ready(spec.key):
                self._semif_download_errors.pop(spec.key, None)
                return json_response(self._semif_model_payload(spec))
            self._semif_download_errors.pop(spec.key, None)
            self._semif_download_tasks[spec.key] = asyncio.create_task(
                self._download_semif_model(spec)
            )
        return json_response(self._semif_model_payload(spec))

    async def _download_semif_model(self, spec) -> None:
        models_dir = get_semif_models_dir()
        temp_dir = models_dir / ".downloads" / f"{spec.key}-{uuid4().hex}"
        target_dir = get_semif_model_dir(spec.key)
        process: asyncio.subprocess.Process | None = None
        try:
            temp_dir.parent.mkdir(parents=True, exist_ok=True)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                str(Path(__file__).with_name("semif_download_worker.py")),
                "--repo",
                spec.repo_id,
                "--revision",
                spec.revision,
                "--target",
                str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TQDM_DISABLE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=3600
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("模型下载超过 1 小时")
            if process.returncode != 0:
                details = (stdout or stderr).decode("utf-8", errors="replace").strip()
                raise RuntimeError(details[-1000:] or "模型下载失败")
            missing = missing_semif_model_files(temp_dir, spec)
            if missing:
                raise RuntimeError("模型下载不完整，缺少文件：" + ", ".join(missing))
            backup = models_dir / ".downloads" / f"old-{spec.key}-{uuid4().hex}"
            if target_dir.exists():
                target_dir.rename(backup)
            try:
                temp_dir.rename(target_dir)
            except Exception:
                if backup.exists() and not target_dir.exists():
                    backup.rename(target_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except Exception as error:
            self._semif_download_errors[spec.key] = str(error)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _laya_model_payload(self, spec) -> dict:
        task = self._download_tasks.get(spec.key)
        if task is not None and not task.done():
            state = "downloading"
        elif is_laya_model_ready(spec.key):
            state = "ready"
        elif spec.key in self._download_errors:
            state = "error"
        else:
            state = "idle"
        return {
            "key": spec.key,
            "label": spec.label,
            "repo_id": spec.repo_id,
            "description": spec.description,
            "state": state,
            "downloaded": state == "ready",
            "error": self._download_errors.get(spec.key),
            "selected": spec.key == self._configured_laya_model(),
        }

    async def page_laya_models(self):
        return json_response(
            {
                "selected": self._configured_laya_model(),
                "models": [self._laya_model_payload(spec) for spec in LAYA_MODELS],
            }
        )

    async def page_laya_model_select(self):
        payload = await request.json(default={})
        model_key = payload.get("model") if isinstance(payload, Mapping) else None
        try:
            spec = get_laya_model_spec(model_key)
        except (TypeError, ValueError) as error:
            return error_response(str(error), status_code=400)
        config = self.plugin_config
        if not isinstance(config, Mapping) or not hasattr(config, "save_config"):
            return error_response(
                "\u63d2\u4ef6\u914d\u7f6e\u4e0d\u53ef\u5199\uff0c"
                "\u8bf7\u5728 AstrBot "
                "\u63d2\u4ef6\u8bbe\u7f6e\u4e2d\u9009\u62e9\u6a21\u578b",
                status_code=409,
            )
        semantic = config.get("semantic")
        if not isinstance(semantic, dict):
            semantic = {}
            config["semantic"] = semantic
        semantic["laya_model"] = spec.key
        config.save_config()
        return json_response({"selected": spec.key, "requires_reload": True})

    async def page_laya_model_download(self):
        payload = await request.json(default={})
        model_key = payload.get("model") if isinstance(payload, Mapping) else None
        try:
            spec = get_laya_model_spec(model_key)
        except (TypeError, ValueError) as error:
            return error_response(str(error), status_code=400)
        async with self._download_lock:
            task = self._download_tasks.get(spec.key)
            if task is not None and not task.done():
                return json_response(self._laya_model_payload(spec))
            if is_laya_model_ready(spec.key):
                self._download_errors.pop(spec.key, None)
                return json_response(self._laya_model_payload(spec))
            self._download_errors.pop(spec.key, None)
            task = asyncio.create_task(self._download_model(spec))
            self._download_tasks[spec.key] = task
        return json_response(self._laya_model_payload(spec))

    async def page_semantic_test_settings(self):
        if self.settings is None:
            return error_response(self.config_error or "插件配置无效", status_code=400)
        settings = self.settings
        return json_response(
            {
                "backend": settings.semantic_backend,
                "threshold": settings.semantic_threshold,
                "keywords": list(settings.keywords),
                "jev": {
                    "model": settings.jev_model,
                    "configured": bool(settings.jev_api_key),
                },
                "laya": {"model": self._configured_laya_model()},
                "semif": {"model": self._configured_semif_preset()},
            }
        )

    async def page_semantic_test(self):
        if self.closed:
            return error_response("插件已卸载", status_code=409)
        if self.settings is None:
            return error_response(self.config_error or "插件配置无效", status_code=400)
        if self.model_tester.busy:
            return error_response("已有模型测试正在运行，请稍后再试", status_code=409)
        try:
            payload = await request.json(default={})
            config = self.plugin_config if self.plugin_config is not None else {}
            return json_response(await self.model_tester.run(config, payload))
        except ValueError as error:
            return error_response(str(error), status_code=400)
        except TimeoutError:
            return error_response(
                "模型测试超时，请检查模型设备和超时配置", status_code=504
            )
        except Exception as error:
            secret = self.settings.jev_api_key
            message = str(error) or type(error).__name__
            if secret:
                message = message.replace(secret, "[已隐藏密钥]")
            return error_response(message, status_code=502)

    async def _download_model(self, spec) -> None:
        models_dir = get_laya_models_dir()
        temp_dir = models_dir / ".downloads" / f"{spec.key}-{uuid4().hex}"
        target_dir = get_laya_model_dir(spec.key)
        process: asyncio.subprocess.Process | None = None
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.parent.mkdir(parents=True, exist_ok=True)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                str(Path(__file__).with_name("laya_download_worker.py")),
                "--repo",
                spec.repo_id,
                "--target",
                str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TQDM_DISABLE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=3600
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("模型下载超过 1 小时")
            if process.returncode != 0:
                details = (stdout or stderr).decode("utf-8", errors="replace").strip()
                raise RuntimeError(details[-1000:] or "模型下载失败")
            missing = missing_model_files(temp_dir)
            if missing:
                raise RuntimeError("模型下载不完整，缺少文件：" + ", ".join(missing))
            backup = models_dir / ".downloads" / f"old-{spec.key}-{uuid4().hex}"
            if target_dir.exists():
                target_dir.rename(backup)
            try:
                temp_dir.rename(target_dir)
            except Exception:
                if backup.exists() and not target_dir.exists():
                    backup.rename(target_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except Exception as error:
            self._download_errors[spec.key] = str(error)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @property
    def semantic_enabled(self) -> bool:
        return bool(
            self.active
            and self.settings.enabled
            and self.semantic_failure is None
            and (self.settings.semantic_backend != "jev" or self.settings.jev_api_key)
        )

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
            if self.settings.enabled and self.settings.semantic_backend in (
                "semif",
                "laya",
            ):
                try:
                    await self.client.start()
                except Exception as error:
                    self.semantic_failure = str(error)
                    logger.exception(
                        "%s 启动失败，语义判断已暂停；关键词、@和引用唤醒仍可用。"
                        "请检查 AstrBot 的 Python 环境中的模型依赖、"
                        "模型路径和显存后重载插件。",
                        self.controller.backend_name,
                    )
            if not self.settings.enabled:
                logger.info(
                    "语义判断已关闭：保留关键词、@和引用的直接唤醒及内存历史；"
                    "不建立待判窗口。"
                )
            elif (
                self.settings.semantic_backend == "jev"
                and not self.settings.jev_api_key
            ):
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
            logger.error("语义判断无法追踪缺少消息 ID 的 QQ 消息")
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
                    "%s 语义唤醒未运行：%s（%s）",
                    self.controller.backend_name,
                    reason,
                    event.unified_msg_origin,
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
        await self.model_tester.close()
        downloads = list(self._download_tasks.values()) + list(
            self._semif_download_tasks.values()
        )
        for task in downloads:
            if not task.done():
                task.cancel()
        if downloads:
            await asyncio.gather(*downloads, return_exceptions=True)
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
