import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot_plugin_qq_group_enhance import laya_models, main
from astrbot_plugin_qq_group_enhance.laya_client import LayaClient
from astrbot_plugin_qq_group_enhance.laya_worker import (
    questions_from_rows,
    score_request,
)
from astrbot_plugin_qq_group_enhance.main import QQGroupEnhancePlugin
from astrbot_plugin_qq_group_enhance.semantic import choice_rows, wake_probabilities
from astrbot_plugin_qq_group_enhance.state import ChatMessage, Settings


def local_settings(tmp_path):
    return Settings.from_mapping(
        {
            "semantic": {
                "semantic_backend": "laya",
                "laya_model": "multilingual",
            }
        }
    )


def test_laya_configuration_uses_astrbot_python_and_validates_model(tmp_path):
    defaults = Settings.from_mapping({"semantic": {"semantic_backend": "laya"}})
    assert defaults.laya_model == "multilingual"
    settings = local_settings(tmp_path)
    assert LayaClient(settings).worker_args()[0] == sys.executable
    assert LayaClient(replace(settings, semantic_threshold=0.83)).threshold == 0.83
    assert settings.laya_device == "cuda"
    with pytest.raises(ValueError, match="laya_device"):
        Settings.from_mapping(
            {"semantic": {"semantic_backend": "laya", "laya_device": "mps"}}
        )
    with pytest.raises(ValueError, match="laya_model"):
        Settings.from_mapping(
            {"semantic": {"semantic_backend": "laya", "laya_model": "unknown"}}
        )


def test_laya_choice_uses_neutral_labels_and_restores_semantic_order(monkeypatch):
    message = ChatMessage("m1", "u1", "甲", "继续说", 1, bot_id="bot").payload()
    rows = choice_rows([message])
    questions = questions_from_rows(rows)
    assert list(questions["0:normal"]["criteria"]) == ["A", "B"]
    assert list(questions["0:reversed"]["criteria"]) == ["A", "B"]
    assert (
        questions["0:normal"]["criteria"]["A"]
        == questions["0:reversed"]["criteria"]["B"]
    )
    monkeypatch.setattr(
        "astrbot_plugin_qq_group_enhance.laya_worker.prepare_state",
        lambda request, agent, questions: (request["state"], 0),
    )
    agent = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        predict=lambda state, questions: {
            "answers": {
                "0:normal": {
                    "type": "choice",
                    "choice": "A",
                    "probabilities": {"A": 0.8, "B": 0.2},
                },
                "0:reversed": {
                    "type": "choice",
                    "choice": "B",
                    "probabilities": {"A": 0.4, "B": 0.6},
                },
            }
        },
    )
    scores = score_request(
        {"state": {"history": [], "candidates": [message]}, "rows": rows}, agent
    )["scores"]
    assert wake_probabilities([message], scores, source="Laya") == {
        "m1": pytest.approx(0.7)
    }


async def test_laya_worker_failure_does_not_fallback(tmp_path):
    client = LayaClient(local_settings(tmp_path))
    client._score = AsyncMock(side_effect=RuntimeError("model unavailable"))
    message = ChatMessage("m1", "u1", "甲", "追问", 1, bot_id="bot").payload()
    failed = []
    assert await client.judge(
        [], [message], lambda _: True, on_failed=failed.append
    ) == {"m1": None}
    assert failed == [{"m1"}]


@pytest.mark.parametrize("startup_fails", [False, True])
async def test_plugin_selects_laya_and_pauses_on_startup_failure(
    tmp_path, monkeypatch, startup_fails
):
    settings = local_settings(tmp_path)
    config = {
        "semantic": {
            "semantic_backend": "laya",
            "laya_model": settings.laya_model,
        }
    }
    start = AsyncMock(
        side_effect=RuntimeError("model unavailable") if startup_fails else None
    )
    monkeypatch.setattr(LayaClient, "start", start)
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), config)
    plugin.observer.install = MagicMock()
    plugin.observer.close = MagicMock()
    try:
        await plugin.initialize()
        assert isinstance(plugin.client, LayaClient)
        start.assert_awaited_once()
        assert plugin.semantic_enabled is not startup_fails
        assert (plugin.semantic_failure is not None) is startup_fails
    finally:
        await plugin.terminate()


async def test_laya_download_uses_astrbot_python(tmp_path, monkeypatch):
    monkeypatch.setattr(
        laya_models, "get_astrbot_plugin_data_path", lambda: str(tmp_path)
    )
    spec = laya_models.LAYA_MODELS[1]
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), {})
    calls = []

    async def fake_subprocess(*args, **kwargs):
        calls.append(args)
        assert args[0] == sys.executable
        assert Path(args[2]).name == "laya_download_worker.py"
        assert args[args.index("--repo") + 1] == spec.repo_id
        staging = Path(args[args.index("--target") + 1])
        for name in laya_models.REQUIRED_MODEL_FILES:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"model fixture")

        async def communicate():
            return b'{"ok": true}', b""

        return SimpleNamespace(communicate=communicate, returncode=0)

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fake_subprocess)
    try:
        await plugin._download_model(spec)
        assert len(calls) == 1
        assert laya_models.is_laya_model_ready(spec.key)
        assert plugin._laya_model_payload(spec)["state"] == "ready"
    finally:
        await plugin.terminate()
