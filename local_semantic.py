"""Shared persistent worker transport and choice-result handling for local models."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable

from astrbot.api import logger as log

from .semantic import choice_rows, evidence, wake_probabilities
from .state import Settings


class LocalChoiceClient:
    name: str

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: asyncio.subprocess.Process | None = None
        self.stderr_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.request_id = 0
        self.history_omitted = 0

    def worker_args(self) -> list[str]:
        raise NotImplementedError

    @property
    def startup_seconds(self) -> float:
        raise NotImplementedError

    @property
    def timeout_seconds(self) -> float:
        raise NotImplementedError

    @property
    def threshold(self) -> float:
        return self.settings.semantic_threshold

    async def _stderr(self, process: asyncio.subprocess.Process) -> None:
        while line := await process.stderr.readline():
            message = line.decode("utf-8", errors="replace").rstrip()
            if message:
                log.warning("%s worker: %s", self.name, message[:500])

    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        environment = {
            **os.environ,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        self.process = await asyncio.create_subprocess_exec(
            *self.worker_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        self.stderr_task = asyncio.create_task(self._stderr(self.process))
        try:
            async with asyncio.timeout(self.startup_seconds):
                line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} worker 在模型加载期间退出")
            response = json.loads(line)
            if response.get("ready") is not True:
                reason = response.get("error", "无有效就绪消息")
                raise RuntimeError(f"{self.name} worker 启动失败：{reason}")
            log.info("%s 模型已加载：%s", self.name, response.get("model"))
        except BaseException:
            await self.close()
            raise

    async def _score(self, state: str | dict | list, rows: list[dict]) -> list[dict]:
        async with self.lock:
            await self.start()
            self.request_id += 1
            request_id = self.request_id
            message = json.dumps(
                {"request_id": request_id, "state": state, "rows": rows},
                ensure_ascii=False,
                allow_nan=False,
            )
            try:
                self.process.stdin.write((message + "\n").encode("utf-8"))
                async with asyncio.timeout(self.timeout_seconds):
                    await self.process.stdin.drain()
                    line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"{self.name} worker 在判断期间退出")
                response = json.loads(line)
                if response.get("request_id") != request_id:
                    raise RuntimeError(f"{self.name} worker 响应 ID 不匹配")
                if (
                    not isinstance(response.get("scores"), list)
                    and "error" not in response
                ):
                    raise RuntimeError(f"{self.name} worker 未返回评分列表")
            except BaseException:
                # A timed-out/cancelled reply must never be read as the next reply.
                await self.close()
                raise
            if "error" in response:
                raise RuntimeError(f"{self.name} 判断失败：{response['error']}")
            self.history_omitted = response.get("history_omitted", 0)
            if response.get("history_omitted"):
                log.info(
                    "%s 因模型输入上限省略了 %d 条较早历史",
                    self.name,
                    response["history_omitted"],
                )
            if response.get("direct_rows"):
                log.warning(
                    "%s 共享前缀不匹配，%d 道题改用逐题评分",
                    self.name,
                    response["direct_rows"],
                )
            return response["scores"]

    async def judge(
        self,
        history: list[dict],
        candidates: list[dict],
        valid: Callable[[str], bool],
        eligible: Callable[[set[str]], Awaitable[set[str]]] | None = None,
        on_failed: Callable[[set[str]], None] | None = None,
    ) -> dict[str, float | None]:
        active = [message for message in candidates if valid(message["message_id"])]
        if eligible and active:
            allowed = await eligible({m["message_id"] for m in active})
            active = [
                m
                for m in active
                if m["message_id"] in allowed and valid(m["message_id"])
            ]
        if not active:
            return {}
        try:
            state = evidence(history, active, self.settings.keywords)
            scores = await self._score(state, choice_rows(active))
            probabilities = wake_probabilities(active, scores, source=self.name)
            result = {mid: p for mid, p in probabilities.items() if valid(mid)}
            for index, message in enumerate(active):
                mid = message["message_id"]
                if mid not in result:
                    continue
                normal, reversed_order = scores[2 * index : 2 * index + 2]
                p_normal = normal["probabilities"][normal["option_ids"].index("wake")]
                p_reversed = reversed_order["probabilities"][
                    reversed_order["option_ids"].index("wake")
                ]
                log.info(
                    "%s 判断完成 | 消息=%s | 正序=%.2f | 反序=%.2f"
                    " | 平均=%.2f | 阈值=%.2f",
                    self.name,
                    mid,
                    p_normal,
                    p_reversed,
                    result[mid],
                    self.threshold,
                )
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s 批量判断失败", self.name)
            failed = {m["message_id"] for m in active if valid(m["message_id"])}
            if on_failed and failed:
                on_failed(failed)
            return dict.fromkeys(failed)

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        if self.stderr_task is not None:
            self.stderr_task.cancel()
            await asyncio.gather(self.stderr_task, return_exceptions=True)
            self.stderr_task = None
