import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot_plugin_qq_group_enhance.main import QQGroupEnhancePlugin
from astrbot_plugin_qq_group_enhance.semantic import (
    choice_rows,
    evidence,
    wake_probabilities,
)
from astrbot_plugin_qq_group_enhance.semif import SemIfClient
from astrbot_plugin_qq_group_enhance.semif_worker import score_request
from astrbot_plugin_qq_group_enhance.state import ChatMessage, Settings


def local_settings(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    return Settings.from_mapping(
        {
            "semantic": {
                "semantic_backend": "semif",
                "semif_model": str(model),
                "semif_revision": "pinned-local-model",
            }
        }
    )


def test_semif_configuration_uses_astrbot_python_and_requires_model(tmp_path):
    with pytest.raises(ValueError, match="semif_model"):
        Settings.from_mapping({"semantic": {"semantic_backend": "semif"}})
    settings = local_settings(tmp_path)
    assert SemIfClient(settings).worker_args()[0] == sys.executable
    assert SemIfClient(replace(settings, semantic_threshold=0.82)).threshold == 0.82
    with pytest.raises(ValueError, match="semif_device"):
        Settings.from_mapping(
            {"semantic": {"semantic_backend": "semif", "semif_device": "mps"}}
        )


def test_choice_scores_balance_order_and_reject_mismatched_results():
    messages = [
        ChatMessage("m1", "u1", "甲", "刚才那题呢？", 1, bot_id="bot").payload()
    ]
    state = evidence([], messages, ())
    assert state["bot"]["id"] == "bot"
    rows = choice_rows(messages)
    assert [[option["id"] for option in row["options"]] for row in rows] == [
        ["wake", "ignore"],
        ["ignore", "wake"],
    ]
    scores = [
        {
            "id": "0:normal",
            "option_ids": ["wake", "ignore"],
            "probabilities": [0.9, 0.1],
        },
        {
            "id": "0:reversed",
            "option_ids": ["ignore", "wake"],
            "probabilities": [0.7, 0.3],
        },
    ]
    assert wake_probabilities(messages, scores) == {"m1": pytest.approx(0.6)}
    scores[1]["option_ids"] = ["wake", "ignore"]
    with pytest.raises(ValueError, match="顺序"):
        wake_probabilities(messages, scores)


def test_shared_prefix_mismatch_uses_same_model_direct_scoring(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "vendor"))
    from semif_phase1 import direct, shared

    rows = [{"id": "0:normal"}, {"id": "0:reversed"}]
    monkeypatch.setattr(
        "astrbot_plugin_qq_group_enhance.semif_worker.prepare_rows",
        lambda request, tokenizer, max_tokens: (rows, 0),
    )

    def mismatched_prefix(*args):
        raise ValueError("The fixed state prefix does not match every full prompt")

    monkeypatch.setattr(shared, "score_shared", mismatched_prefix)
    monkeypatch.setattr(
        direct,
        "score",
        lambda model, tokenizer, row, metadata, max_tokens: {
            "id": row["id"],
            "option_ids": ["wake", "ignore"],
            "probabilities": [0.8, 0.2],
            "input_tokens": 100,
        },
    )
    result = score_request({"rows": rows}, object(), object(), {}, 4096, 2)
    assert len(result["scores"]) == 2
    assert result["direct_rows"] == 2


async def test_semif_worker_is_persistent_and_eligible_messages_only(tmp_path):
    settings = local_settings(tmp_path)
    stub = tmp_path / "worker.py"
    stub.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True, 'model': 'stub'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " if request['request_id']==2:\n"
        "  print(json.dumps({'request_id': 2,\n"
        "                    'error': 'input too long'}), flush=True)\n"
        "  continue\n"
        " rows=request['rows']\n"
        " scores=[{'id': row['id'],\n"
        "          'option_ids': [o['id'] for o in row['options']],\n"
        "          'probabilities': [0.8 if o['id']=='wake' else 0.2\n"
        "                            for o in row['options']]}\n"
        "         for row in rows]\n"
        " print(json.dumps({'request_id': request['request_id'],\n"
        "                   'scores': scores, 'history_omitted': 0}), flush=True)\n",
        encoding="utf-8",
    )
    client = SemIfClient(settings)
    client.WORKER_PATH = stub
    first = ChatMessage("m1", "u1", "甲", "继续说", 1, bot_id="bot").payload()
    second = ChatMessage("m2", "u2", "乙", "互聊", 2, bot_id="bot").payload()
    try:
        await client.start()
        process = client.process
        eligible = AsyncMock(return_value={"m1"})
        assert await client.judge([], [first, second], lambda _: True, eligible) == {
            "m1": pytest.approx(0.8)
        }
        assert await client.judge([], [first], lambda _: True) == {"m1": None}
        assert await client.judge([], [first], lambda _: True) == {
            "m1": pytest.approx(0.8)
        }
        assert client.process is process
        assert client.request_id == 3
    finally:
        await client.close()
    assert process.returncode is not None


async def test_semif_worker_failure_does_not_wake_or_fallback(tmp_path):
    settings = local_settings(tmp_path)
    client = SemIfClient(settings)
    client._score = AsyncMock(side_effect=RuntimeError("model unavailable"))
    message = ChatMessage("m1", "u1", "甲", "追问", 1, bot_id="bot").payload()
    failed = []
    result = await client.judge([], [message], lambda _: True, on_failed=failed.append)
    assert result == {"m1": None}
    assert failed == [{"m1"}]


@pytest.mark.parametrize("startup_fails", [False, True])
async def test_plugin_selects_semif_and_pauses_on_startup_failure(
    tmp_path, monkeypatch, startup_fails
):
    settings = local_settings(tmp_path)
    fields = ("semif_model", "semif_revision")
    config = {
        "semantic": {
            "semantic_backend": "semif",
            **{name: getattr(settings, name) for name in fields},
        }
    }
    start = AsyncMock(
        side_effect=RuntimeError("model unavailable") if startup_fails else None
    )
    monkeypatch.setattr(SemIfClient, "start", start)
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), config)
    plugin.observer.install = MagicMock()
    plugin.observer.close = MagicMock()
    try:
        await plugin.initialize()
        assert isinstance(plugin.client, SemIfClient)
        start.assert_awaited_once()
        assert plugin.semantic_enabled is not startup_fails
        assert (plugin.semantic_failure is not None) is startup_fails
    finally:
        await plugin.terminate()
