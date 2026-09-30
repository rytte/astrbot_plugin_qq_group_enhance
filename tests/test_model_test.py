import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_qq_group_enhance import main, model_test
from astrbot_plugin_qq_group_enhance.laya_worker import (
    questions_from_rows,
    score_request,
)
from astrbot_plugin_qq_group_enhance.main import QQGroupEnhancePlugin
from astrbot_plugin_qq_group_enhance.model_test import ModelTester
from astrbot_plugin_qq_group_enhance.semif_models import SEMIF_MODELS


def wake_payload(backend="jev", **updates):
    return {
        "backend": backend,
        "mode": "wake",
        "bot_id": "bot",
        "history": [{"sender_id": "bot", "text": "想看哪种科幻？"}],
        "candidates": [{"sender_id": "alice", "text": "太空探险类的呢？"}],
        **updates,
    }


def choice_payload(backend="jev", **updates):
    return {
        "backend": backend,
        "mode": "choice",
        "context": {"ticket": "物流没有更新"},
        "question": "应该交给哪个团队？",
        "options": ["物流", "售后", "销售"],
        **updates,
    }


def mock_local(monkeypatch, backend):
    monkeypatch.setattr(model_test, "is_laya_model_ready", lambda _: True)
    monkeypatch.setattr(model_test, "is_semif_model_ready", lambda _: True)
    client_type = model_test.LayaClient if backend == "laya" else model_test.SemIfClient

    async def score(self, state, rows):
        return [
            {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": [0.8, 0.2]
                if len(row["options"]) == 2
                else [0.1, 0.7, 0.2],
            }
            for row in rows
        ]

    monkeypatch.setattr(client_type, "_score", score)
    return client_type


@pytest.mark.parametrize("backend", ["laya", "semif"])
async def test_local_choices_use_custom_context_and_preserve_plugin_config(
    monkeypatch, backend
):
    mock_local(monkeypatch, backend)
    config = {"semantic": {"semantic_backend": "jev"}}
    original = deepcopy(config)
    tester = ModelTester()
    payload = choice_payload(
        backend, model="multilingual" if backend == "laya" else SEMIF_MODELS[0].key
    )
    try:
        result = await tester.run(config, payload)
        assert config == original
        assert result["backend"] == backend
        assert result["results"][0]["selected"] == "B"
        assert [o["probability"] for o in result["results"][0]["options"]] == [
            0.1,
            0.7,
            0.2,
        ]
        client = tester.client
        await tester.run(config, payload)
        assert tester.client is client
    finally:
        await tester.close()


async def test_jev_native_choice_contract_and_model_override(monkeypatch):
    send = AsyncMock(
        return_value={
            "answers": {
                "q0": {
                    "type": "choice",
                    "choice": "B",
                    "probabilities": {"A": 0.2, "B": 0.6, "C": 0.2},
                }
            },
            "usage": {"input_tokens": 12},
        }
    )
    monkeypatch.setattr(model_test.JevClient, "_send", send)
    tester = ModelTester()
    try:
        result = await tester.run(
            {"semantic": {"jev_api_key": "secret"}}, choice_payload(model="jev-1.14.0")
        )
        request = send.call_args.args[0]
        assert request["model"] == "jev-1.14.0"
        assert request["state"] == {"ticket": "物流没有更新"}
        assert request["questions"]["q0"] == {
            "type": "choice",
            "instructions": "应该交给哪个团队？",
            "criteria": {"A": "物流", "B": "售后", "C": "销售"},
        }
        assert result["results"][0]["selected"] == "B"
        assert result["usage"] == {"input_tokens": 12}
        assert "secret" not in json.dumps(result)
    finally:
        await tester.close()


async def test_jev_native_noul_contract(monkeypatch):
    send = AsyncMock(
        return_value={
            "answers": {"q0": {"type": "noul", "noul": 0.82}},
            "usage": {"input_tokens": 8},
        }
    )
    monkeypatch.setattr(model_test.JevClient, "_send", send)
    tester = ModelTester()
    try:
        result = await tester.run(
            {"semantic": {"jev_api_key": "secret"}},
            choice_payload(mode="noul"),
        )
        request = send.call_args.args[0]
        assert request["questions"]["q0"] == {
            "type": "noul",
            "instructions": (
                "判断以下陈述是否成立，只在‘是’与‘否’中选择：\n"
                "应该交给哪个团队？"
            ),
        }
        assert result["results"] == [{"id": "q0", "probability": 0.82}]
        assert result["scores"][0]["probabilities"] == [0.82, 0.18]
    finally:
        await tester.close()


@pytest.mark.parametrize("backend", ["laya", "semif"])
async def test_local_score_mode_returns_probabilities_without_winner(
    monkeypatch, backend
):
    mock_local(monkeypatch, backend)
    tester = ModelTester()
    payload = choice_payload(
        backend,
        mode="score",
        model="multilingual" if backend == "laya" else SEMIF_MODELS[0].key,
    )
    try:
        result = await tester.run({}, payload)
        assert result["results"][0].keys() == {"id", "options"}
        assert "selected" not in result["results"][0]
        assert [item["probability"] for item in result["results"][0]["options"]] == [
            0.1,
            0.7,
            0.2,
        ]
    finally:
        await tester.close()


@pytest.mark.parametrize("backend", ["jev", "laya", "semif"])
async def test_wake_test_uses_real_rules_and_temporary_threshold(monkeypatch, backend):
    mock_local(monkeypatch, backend)
    send = AsyncMock(return_value={"answers": {"q0": {"type": "noul", "noul": 0.8}}})
    monkeypatch.setattr(model_test.JevClient, "_send", send)
    config = {"semantic": {"jev_api_key": "secret"}}
    payload = wake_payload(backend, threshold=0.9)
    if backend == "semif":
        payload["model"] = SEMIF_MODELS[0].key
    tester = ModelTester()
    try:
        result = await tester.run(config, payload)
        assert result["results"][0]["passed"] is False
        assert result["results"][0]["probability"] == pytest.approx(
            0.8 if backend == "jev" else 0.5
        )
        assert config["semantic"] == {"jev_api_key": "secret"}
        if backend == "jev":
            state = send.call_args.args[0]["state"]
            assert state["bot"]["id"] == "bot"
            assert "目标消息" in state["rules"]
    finally:
        await tester.close()


@pytest.mark.parametrize(
    "payload,match",
    [
        (wake_payload(candidates=[]), "不能为空"),
        (wake_payload(threshold=True), "有限正数"),
        (wake_payload(threshold=0), "有限正数"),
        (wake_payload(model="jev-latest"), "固定版本"),
        (
            wake_payload(
                candidates=[
                    {"sender_id": "u", "text": "a", "message_id": "x"},
                    {"sender_id": "u", "text": "b", "message_id": "x"},
                ]
            ),
            "必须唯一",
        ),
        (choice_payload(options=["a"]), "2～16"),
        (choice_payload(options=["a", "a"]), "不能重复"),
        (choice_payload(context=""), "非空"),
        (choice_payload(context=float("nan")), "有限 JSON"),
        (choice_payload(context="a" * 65000), "64,000"),
        (choice_payload(question=""), "question"),
        (choice_payload(legacy_model="old"), "未知测试字段"),
    ],
)
def test_bad_inputs_fail_before_inference(payload, match):
    with pytest.raises(ValueError, match=match):
        ModelTester.prepare({"semantic": {"jev_api_key": "secret"}}, payload)


def test_missing_credentials_and_undownloaded_models_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="jev_api_key"):
        ModelTester.prepare({}, choice_payload())
    monkeypatch.setattr(model_test, "is_laya_model_ready", lambda _: False)
    with pytest.raises(ValueError, match="尚未下载完整"):
        ModelTester.prepare({}, choice_payload("laya"))
    with pytest.raises(ValueError, match="旧阈值配置已移除"):
        ModelTester.prepare(
            {"semantic": {"jev_threshold": 0.7, "jev_api_key": "secret"}},
            choice_payload(),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        {"A": True, "B": 0, "C": 0},
        {"A": 0.2, "B": 0.2, "C": 0.2},
        {"A": float("nan"), "B": 0, "C": 0},
        {"A": 0.2, "B": 0.8},
    ],
)
async def test_invalid_jev_choice_does_not_become_negative(monkeypatch, probabilities):
    monkeypatch.setattr(
        model_test.JevClient,
        "_send",
        AsyncMock(
            return_value={
                "answers": {
                    "q0": {
                        "type": "choice",
                        "choice": "A",
                        "probabilities": probabilities,
                    }
                }
            }
        ),
    )
    tester = ModelTester()
    try:
        with pytest.raises(RuntimeError, match="无效"):
            await tester.run({"semantic": {"jev_api_key": "secret"}}, choice_payload())
        assert tester.client is None
    finally:
        await tester.close()


async def test_jev_wake_failure_is_an_error(monkeypatch):
    monkeypatch.setattr(
        model_test.JevClient,
        "_send",
        AsyncMock(side_effect=RuntimeError("Jev HTTP 401")),
    )
    tester = ModelTester()
    try:
        with pytest.raises(RuntimeError, match="判断失败"):
            await tester.run({"semantic": {"jev_api_key": "secret"}}, wake_payload())
    finally:
        await tester.close()


async def test_page_metadata_redacts_key_and_rejects_concurrent_test(monkeypatch):
    context = SimpleNamespace(register_web_api=lambda *args: None)
    plugin = QQGroupEnhancePlugin(context, {"semantic": {"jev_api_key": "secret"}})
    try:
        body = json.loads((await plugin.page_semantic_test_settings()).body)
        assert body["jev"]["configured"] is True
        assert "secret" not in json.dumps(body)
        async with plugin.model_tester.lock:
            assert (await plugin.page_semantic_test()).status_code == 409
        monkeypatch.setattr(
            main,
            "request",
            SimpleNamespace(json=AsyncMock(return_value=choice_payload())),
        )
        monkeypatch.setattr(
            plugin.model_tester,
            "run",
            AsyncMock(side_effect=RuntimeError("secret failed")),
        )
        response = await plugin.page_semantic_test()
        assert response.status_code == 502
        assert "secret" not in response.body.decode()
    finally:
        await plugin.terminate()


async def test_terminate_cancels_active_test_and_releases_client(monkeypatch):
    started = asyncio.Event()

    async def pending(self, payload):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(model_test.JevClient, "_send", pending)
    plugin = QQGroupEnhancePlugin(
        SimpleNamespace(), {"semantic": {"jev_api_key": "secret"}}
    )
    task = asyncio.create_task(
        plugin.model_tester.run(plugin.plugin_config, choice_payload())
    )
    await started.wait()
    await plugin.terminate()
    assert task.cancelled()
    assert plugin.model_tester.client is None
    assert (await plugin.page_semantic_test()).status_code == 409


def test_laya_worker_accepts_multiple_options_and_maps_labels(monkeypatch):
    rows = [
        {
            "id": "q0",
            "question": "选择哪一项？",
            "options": [
                {"id": "first", "description": "第一项"},
                {"id": "second", "description": "第二项"},
                {"id": "third", "description": "第三项"},
            ],
        }
    ]
    assert list(questions_from_rows(rows)["q0"]["criteria"]) == ["A", "B", "C"]
    monkeypatch.setattr(
        "astrbot_plugin_qq_group_enhance.laya_worker.prepare_state",
        lambda request, agent, questions: (request["state"], 0),
    )
    agent = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        predict=lambda state, questions: {
            "answers": {
                "q0": {
                    "type": "choice",
                    "choice": "B",
                    "probabilities": {"A": 0.1, "B": 0.7, "C": 0.2},
                }
            }
        },
    )
    scores = score_request({"state": "自由文本", "rows": rows}, agent)["scores"]
    assert scores[0]["option_ids"] == ["first", "second", "third"]
    assert scores[0]["probabilities"] == [0.1, 0.7, 0.2]
