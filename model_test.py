"""Isolated, reusable model playground for wake, choice, score, and Noul tests."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy

from .jev import JevClient, json_bytes
from .laya_client import LayaClient
from .laya_models import get_laya_model_spec, is_laya_model_ready
from .semantic import choice_rows, evidence, wake_probabilities
from .semif import SemIfClient
from .semif_models import CUSTOM_SEMIF_MODEL, get_semif_model_spec, is_semif_model_ready
from .state import Settings

LETTERS = "ABCDEFGHIJKLMNOP"


def text(value, name: str, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} 必须是 1～{limit} 个字符的非空字符串")
    return value


def messages(value, name: str, bot_id: str, *, required: bool) -> list[dict]:
    if not isinstance(value, list) or len(value) > 100 or (required and not value):
        raise ValueError(f"{name} 必须是消息列表，最多 100 条；待判断消息不能为空")
    result = []
    for index, original in enumerate(value):
        if not isinstance(original, dict):
            raise ValueError(f"{name}[{index}] 必须是包含 text 和 sender_id 的对象")
        item = dict(original)
        item["text"] = text(item.get("text"), f"{name}[{index}].text")
        item["sender_id"] = text(
            item.get("sender_id"), f"{name}[{index}].sender_id", 100
        )
        item["message_id"] = text(
            item.get("message_id", f"{name}-{index + 1}"),
            f"{name}[{index}].message_id",
            100,
        )
        item["bot_id"] = bot_id
        item["is_bot"] = item["sender_id"] == bot_id
        if "mentions" in item and (
            not isinstance(item["mentions"], list)
            or any(not isinstance(mid, str) for mid in item["mentions"])
        ):
            raise ValueError(f"{name}[{index}].mentions 必须是 ID 字符串列表")
        if (
            "reply" in item
            and item["reply"] is not None
            and not isinstance(item["reply"], dict)
        ):
            raise ValueError(f"{name}[{index}].reply 必须是对象或 null")
        result.append(item)
    if len({item["message_id"] for item in result}) != len(result):
        raise ValueError(f"{name} 中的 message_id 必须唯一")
    return result


def validate_scores(rows: list[dict], scores) -> None:
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError("模型返回的题目数量不匹配")
    for row, score in zip(rows, scores, strict=True):
        ids = [option["id"] for option in row["options"]]
        if (
            not isinstance(score, dict)
            or score.get("id") != row["id"]
            or score.get("option_ids") != ids
        ):
            raise RuntimeError("模型返回的题目 ID 或选项顺序不匹配")
        probabilities = score.get("probabilities")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != len(ids)
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities
            )
            or abs(sum(probabilities) - 1) > 1e-4
        ):
            raise RuntimeError("模型返回了无效的选项概率")


class ModelTester:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.client: JevClient | LayaClient | SemIfClient | None = None
        self.task: asyncio.Task | None = None
        self.closed = False

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    @staticmethod
    def prepare(config: Mapping, payload) -> tuple[Settings, dict]:
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        unknown = set(payload) - {
            "backend",
            "model",
            "mode",
            "threshold",
            "bot_id",
            "keywords",
            "history",
            "candidates",
            "context",
            "question",
            "options",
        }
        if unknown:
            raise ValueError("未知测试字段：" + ", ".join(sorted(unknown)))
        try:
            size = len(
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
        except (ValueError, TypeError) as error:
            raise ValueError("测试输入必须是有限 JSON 数据") from error
        if size > 64000:
            raise ValueError("测试输入不能超过 64,000 UTF-8 字节")
        backend = payload.get("backend")
        if backend not in ("jev", "laya", "semif"):
            raise ValueError("backend 只能是 jev、laya 或 semif")
        mode = payload.get("mode")
        if mode not in ("wake", "choice", "score", "noul"):
            raise ValueError("mode 只能是 wake、choice、score 或 noul")
        config = deepcopy(dict(config))
        semantic = config.setdefault("semantic", {})
        if not isinstance(semantic, dict):
            raise ValueError("semantic 必须是配置对象")
        semantic["semantic_backend"] = backend
        if "model" in payload:
            semantic[
                {"jev": "jev_model", "laya": "laya_model", "semif": "semif_preset"}[
                    backend
                ]
            ] = payload["model"]
        if "threshold" in payload:
            semantic["semantic_threshold"] = payload["threshold"]
        if "keywords" in payload:
            keyword_wake = config.setdefault("keyword_wake", {})
            if not isinstance(keyword_wake, dict):
                raise ValueError("keyword_wake 必须是配置对象")
            keyword_wake["keywords"] = payload["keywords"]
        settings = Settings.from_mapping(config)
        if backend == "jev" and not settings.jev_api_key:
            raise ValueError("请先在插件配置中填写 jev_api_key 并重载插件")
        if backend == "laya":
            spec = get_laya_model_spec(settings.laya_model)
            if not is_laya_model_ready(spec.key):
                raise ValueError(f"Laya 模型 {spec.label} 尚未下载完整，请先下载")
        if backend == "semif" and settings.semif_preset != CUSTOM_SEMIF_MODEL:
            spec = get_semif_model_spec(settings.semif_preset)
            if not is_semif_model_ready(spec.key):
                raise ValueError(f"SemIf 模型 {spec.label} 尚未下载完整，请先下载")
        if mode == "wake":
            bot_id = text(payload.get("bot_id", "bot"), "bot_id", 100)
            history = messages(
                payload.get("history", []), "history", bot_id, required=False
            )
            candidates = messages(
                payload.get("candidates"), "candidates", bot_id, required=True
            )
            if {m["message_id"] for m in history} & {
                m["message_id"] for m in candidates
            }:
                raise ValueError("历史与待判断消息的 message_id 不能重复")
            state = evidence(history, candidates, settings.keywords, bot_id)
            rows = choice_rows(candidates)
            return settings, {
                "mode": mode,
                "state": state,
                "rows": rows,
                "history": history,
                "candidates": candidates,
            }
        context = payload.get("context")
        if (
            not isinstance(context, (str, dict, list))
            or not context
            or (isinstance(context, str) and not context.strip())
        ):
            raise ValueError("context 必须是非空背景文本、JSON 对象或数组")
        question = text(payload.get("question"), "question", 2000)
        if mode == "noul":
            options = ["是", "否"]
        else:
            options = payload.get("options")
            if not isinstance(options, list) or not 2 <= len(options) <= 16:
                raise ValueError("options 必须包含 2～16 个选项")
            options = [text(option, "选项", 500) for option in options]
            if len(set(options)) != len(options):
                raise ValueError("选项内容不能重复")
        row_question = question
        if mode == "noul":
            row_question = "判断以下陈述是否成立，只在‘是’与‘否’中选择：\n" + question
        rows = [
            {
                "id": "q0",
                "question": row_question,
                "options": [
                    {"id": letter, "description": option}
                    for letter, option in zip(LETTERS, options)
                ],
            }
        ]
        return settings, {"mode": mode, "state": context, "rows": rows}

    async def run(self, config: Mapping, payload) -> dict:
        if self.closed:
            raise RuntimeError("模型测试已关闭")
        settings, plan = self.prepare(config, payload)
        async with self.lock:
            self.task = asyncio.current_task()
            start = time.monotonic()
            try:
                if self.client is None or self.client.settings != settings:
                    if self.client is not None:
                        await self.client.close()
                    client_type = {
                        "jev": JevClient,
                        "laya": LayaClient,
                        "semif": SemIfClient,
                    }[settings.semantic_backend]
                    self.client = client_type(settings)
                if settings.semantic_backend == "jev":
                    results, details = await self._jev(settings, plan)
                else:
                    scores = await self.client._score(
                        deepcopy(plan["state"]), plan["rows"]
                    )
                    validate_scores(plan["rows"], scores)
                    details = {
                        "scores": scores,
                        "history_omitted": self.client.history_omitted,
                    }
                    if plan["mode"] == "wake":
                        probabilities = wake_probabilities(
                            plan["candidates"], scores, source=self.client.name
                        )
                        results = self._wake_results(
                            settings, plan["candidates"], probabilities
                        )
                    elif plan["mode"] == "noul":
                        results = self._noul_results(scores[0]["probabilities"][0])
                    elif plan["mode"] == "choice":
                        results = self._choice_results(
                            plan["rows"][0], scores[0]["probabilities"]
                        )
                    else:
                        results = self._score_results(
                            plan["rows"][0], scores[0]["probabilities"]
                        )
                model = {
                    "jev": settings.jev_model,
                    "laya": settings.laya_model,
                    "semif": settings.semif_preset,
                }[settings.semantic_backend]
                return {
                    "backend": settings.semantic_backend,
                    "model": model,
                    "mode": plan["mode"],
                    "threshold": settings.semantic_threshold,
                    "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
                    "results": results,
                    **details,
                }
            except BaseException:
                if self.client is not None:
                    await self.client.close()
                    self.client = None
                raise
            finally:
                self.task = None

    @staticmethod
    def _wake_results(settings, candidates, probabilities):
        if set(probabilities) != {m["message_id"] for m in candidates} or any(
            p is None for p in probabilities.values()
        ):
            raise RuntimeError("Jev 判断失败，请查看 AstrBot 日志中的请求错误")
        return [
            {
                "id": m["message_id"],
                "text": m["text"],
                "probability": probabilities[m["message_id"]],
                "passed": probabilities[m["message_id"]] >= settings.semantic_threshold,
            }
            for m in candidates
        ]

    @staticmethod
    def _choice_results(row, probabilities):
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return [
            {
                "id": row["id"],
                "selected": row["options"][winner]["id"],
                "options": [
                    {
                        "id": option["id"],
                        "label": option["description"],
                        "probability": probability,
                    }
                    for option, probability in zip(
                        row["options"], probabilities, strict=True
                    )
                ],
            }
        ]

    @staticmethod
    def _score_results(row, probabilities):
        return [
            {
                "id": row["id"],
                "options": [
                    {
                        "id": option["id"],
                        "label": option["description"],
                        "probability": probability,
                    }
                    for option, probability in zip(
                        row["options"], probabilities, strict=True
                    )
                ],
            }
        ]

    @staticmethod
    def _noul_results(probability):
        return [{"id": "q0", "probability": probability}]

    async def _jev(self, settings, plan):
        if plan["mode"] == "wake":
            probabilities = await self.client.judge(
                plan["history"], plan["candidates"], lambda _: True
            )
            return self._wake_results(settings, plan["candidates"], probabilities), {}
        row = plan["rows"][0]
        question = {
            "type": "noul"
            if plan["mode"] == "noul"
            else "choice",
            "instructions": row["question"],
        }
        if plan["mode"] != "noul":
            question["criteria"] = {
                option["id"]: option["description"] for option in row["options"]
            }
        payload = {
            "model": settings.jev_model,
            "state": plan["state"],
            "questions": {
                row["id"]: question
            },
        }
        state_size = json_bytes(payload["state"]) + json_bytes(
            payload["questions"][row["id"]]
        )
        if (
            state_size > self.client.STATE_BUDGET
            or json_bytes(payload) > self.client.TOTAL_BUDGET
        ):
            raise ValueError("Jev 测试输入超过接口预算，请缩短背景、问题或选项")
        response = await self.client._send(payload)
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict) or set(answers) != {row["id"]}:
            raise RuntimeError("Jev 返回的题目 ID 不匹配")
        answer = answers[row["id"]]
        ids = [option["id"] for option in row["options"]]
        if plan["mode"] == "noul":
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "noul"
                or type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise RuntimeError("Jev 返回了无效的 noul 结果")
            probabilities = [probability, 1 - probability]
            scores = [
                {"id": row["id"], "option_ids": ids, "probabilities": probabilities}
            ]
            validate_scores([row], scores)
            results = self._noul_results(probability)
        else:
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "choice"
                or answer.get("choice") not in ids
                or not isinstance(answer.get("probabilities"), dict)
                or set(answer["probabilities"]) != set(ids)
            ):
                raise RuntimeError("Jev 返回了无效的 choice 结果")
            probabilities = [answer["probabilities"][option_id] for option_id in ids]
            scores = [
                {"id": row["id"], "option_ids": ids, "probabilities": probabilities}
            ]
            validate_scores([row], scores)
            if answer["probabilities"][answer["choice"]] != max(probabilities):
                raise RuntimeError("Jev 返回的 choice 与最高概率不匹配")
            results = (
                self._choice_results(row, probabilities)
                if plan["mode"] == "choice"
                else self._score_results(row, probabilities)
            )
        usage = response.get("usage")
        return results, {
            "scores": scores,
            "usage": usage if isinstance(usage, dict) else {},
        }

    async def close(self):
        self.closed = True
        if self.task is not None and self.task is not asyncio.current_task():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.client is not None:
            await self.client.close()
            self.client = None
