import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from astrbot_plugin_qq_group_enhance import jev
from astrbot_plugin_qq_group_enhance.jev import JevClient
from astrbot_plugin_qq_group_enhance.state import ChatMessage, Settings


def candidate(mid, text="你好", **kwargs):
    return ChatMessage(
        str(mid), "456", "Alice", text, 0, bot_id="789", **kwargs
    ).payload()


def answer(payload, probability=0.1):
    return {
        "model": "jev-1.13.0",
        "answers": {
            k: {"type": "noul", "noul": probability} for k in payload["questions"]
        },
        "usage": {"input_tokens": 42},
    }


def test_shared_state_and_explicit_per_message_targets():
    client = JevClient(Settings())
    history = [candidate(i) for i in range(30)]
    candidates = [candidate(i) for i in range(30, 70)]
    payloads = client.requests(history, candidates)
    assert len(payloads) == 1
    payload = payloads[0]
    assert len(payload["state"]["history"]) == 30
    assert len(payload["state"]["candidates"]) == 40
    assert payload["state"]["bot"]["id"] == "789"
    assert payload["state"]["rules"] == jev.INSTRUCTIONS
    assert json.dumps(payload, ensure_ascii=False).count(jev.INSTRUCTIONS) == 1
    for question in payload["questions"].values():
        assert "state.rules" in question["instructions"]["rule"]
        assert "target_fragment" not in question["instructions"]
    assert {
        q["instructions"]["target_message_id"] for q in payload["questions"].values()
    } == {str(i) for i in range(30, 70)}


def test_ordinary_messages_keep_structured_content_and_only_top_level_bot():
    client = JevClient(Settings())
    history = [
        ChatMessage(
            "1", "789", "机器人", "我在读书", 10, bot=True, bot_id="789"
        ).payload(),
        candidate(2),
    ]
    candidates = [
        candidate(3, reply={"message_id": "1", "sender_id": "789", "text": "我在读书"}),
        candidate(4, mentions=("888",)),
    ]
    candidates[0]["rendered_context"] = "[Alice/12:00:00]: 你好 [Image: 一本书]"
    snapshot = deepcopy((history, candidates))
    payload = client.requests(history, candidates)[0]
    for field, originals in (("history", history), ("candidates", candidates)):
        assert payload["state"][field] == [
            {k: v for k, v in m.items() if k not in ("bot_id", "is_bot")}
            for m in originals
        ]
    assert payload["state"]["history"][0]["sender_id"] == payload["state"]["bot"]["id"]
    assert "content_fragment" not in json.dumps(payload)
    assert '"fragment"' not in json.dumps(payload)
    assert '"fragments"' not in json.dumps(payload)
    assert (history, candidates) == snapshot


@pytest.mark.parametrize("overflow", [False, True])
def test_fragment_fields_only_appear_above_content_budget(overflow):
    client = JevClient(Settings())
    record = candidate(1, '中文\\"😀' * 5)
    serialized = json.dumps(
        {k: record[k] for k in ("text", "rendered_context", "reply")},
        ensure_ascii=False,
    )
    client.FRAGMENT_BYTES = jev.json_bytes(serialized) - 2 - int(overflow)
    payload = client.requests([], [record])[0]
    messages = payload["state"]["candidates"]
    if overflow:
        assert len(messages) == 2
        assert [m["fragment"] for m in messages] == [0, 1]
        assert all(m["fragments"] == 2 for m in messages)
        assert "".join(m["content_fragment"] for m in messages) == serialized
    else:
        assert len(messages) == 1
        assert messages[0]["text"] == record["text"]
        assert "fragment" not in messages[0]


def test_long_text_caption_and_reply_are_split_without_loss():
    client = JevClient(Settings())
    record = candidate(1, '正文\\"😀' * 9000, reply={"text": "引用" * 9000})
    record["rendered_context"] = "图片描述" * 9000
    history = [candidate(i, "旧历史" * 1000) for i in range(10, 40)]
    payloads = client.requests(history, [record, candidate(2)])
    assert len(payloads) > 1
    assert all(client._fits(p) for p in payloads)
    for payload in payloads:
        assert payload["state"]["bot"]["id"] == "789"
        assert payload["state"]["rules"] == jev.INSTRUCTIONS
        for message in (*payload["state"]["history"], *payload["state"]["candidates"]):
            assert "bot_id" not in message
            assert "is_bot" not in message
        for message, question in zip(
            payload["state"]["candidates"], payload["questions"].values(), strict=True
        ):
            assert (
                question["instructions"]["target_message_id"] == message["message_id"]
            )
            if message["message_id"] == "1":
                assert (
                    question["instructions"]["target_fragment"] == message["fragment"]
                )
            else:
                assert "target_fragment" not in question["instructions"]
    fragments = [
        m for p in payloads for m in p["state"]["candidates"] if m["message_id"] == "1"
    ]
    rebuilt = json.loads("".join(m["content_fragment"] for m in fragments))
    assert rebuilt == {k: record[k] for k in ("text", "rendered_context", "reply")}
    assert len(history) == 30
    assert any(
        m["message_id"] == "2" for p in payloads for m in p["state"]["candidates"]
    )


@pytest.mark.parametrize("failure", [False, True])
async def test_fragment_results_aggregate_and_logs_keep_fragment_ids(
    webui_logs, failure
):
    client = JevClient(Settings())
    client.FRAGMENT_BYTES = 500
    client.STATE_BUDGET = 1900
    candidate_message = candidate(1, "正文" * 1500)
    assert len(client.requests([], [candidate_message])) > 1

    async def send(payload):
        if failure and any(m["fragment"] == 1 for m in payload["state"]["candidates"]):
            raise RuntimeError("test fragment failure")
        return {
            "answers": {
                key: {
                    "type": "noul",
                    "noul": 0.98
                    if question["instructions"]["target_fragment"] == 0
                    else 0.1,
                }
                for key, question in payload["questions"].items()
            }
        }

    client._send = AsyncMock(side_effect=send)
    result = await client.judge([], [candidate_message], lambda _: True)
    assert result == {"1": None if failure else 0.98}
    entries = [e for e in webui_logs if "Jev 判断完成" in e["data"]]
    assert entries
    assert all("片段=" in e["data"] for e in entries)
    assert "片段=1/" in entries[0]["data"]
    assert all("Token=输入 未返回 / 输出 未返回" in e["data"] for e in entries)


async def test_retry_rechecks_candidate_validity_and_does_not_retry_negative():
    client = JevClient(Settings())
    active = {"1", "2"}

    async def send(payload):
        assert payload["state"]["bot"]["id"] == "789"
        assert payload["state"]["rules"] == jev.INSTRUCTIONS
        assert all("bot_id" not in m for m in payload["state"]["candidates"])
        if len(active) == 2:
            active.remove("1")
            raise TimeoutError()
        assert len(payload["questions"]) == 1
        assert payload["questions"]["q0"]["instructions"]["target_message_id"] == "2"
        return answer(payload)

    client._send = AsyncMock(side_effect=send)
    result = await client.judge([], [candidate(1), candidate(2)], active.__contains__)
    assert result == {"2": 0.1}
    assert client._send.await_count == 2


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"answers": []},
        {"answers": {"q0": []}},
        {"answers": {"q0": {"type": "noul", "noul": True}}},
        {"answers": {"q0": {"type": "noul", "noul": 1.1}}},
        {"answers": {"q0": {"type": "noul", "noul": float("nan")}}},
    ],
)
async def test_invalid_responses_exhaust_retry_and_report_failure(bad):
    client = JevClient(Settings())
    client._send = AsyncMock(return_value=bad)
    failed = []
    result = await client.judge(
        [], [candidate(1)], lambda _: True, on_failed=failed.append
    )
    assert result == {"1": None}
    assert failed == [{"1"}]
    assert client._send.await_count == 2


async def test_timeout_is_per_attempt_and_cancellation_is_propagated():
    client = JevClient(replace(Settings(), jev_timeout_seconds=0.01))

    async def hung(payload):
        await asyncio.Future()

    client._send = AsyncMock(side_effect=hung)
    assert await client.judge([], [candidate(1)], lambda _: True) == {"1": None}
    assert client._send.await_count == 2
    client._send.reset_mock()
    task = asyncio.create_task(client.judge([], [candidate(1)], lambda _: True))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client._send.await_count == 1


async def test_disable_before_submission_makes_no_network_request():
    client = JevClient(Settings())
    client._send = AsyncMock()
    eligible = AsyncMock(return_value=set())
    result = await client.judge([], [candidate(1)], lambda _: True, eligible)
    assert result == {}
    client._send.assert_not_called()


async def test_failed_subbatch_is_reported_before_next_request():
    client = JevClient(Settings())
    failed = set()
    calls = 0

    async def send(payload):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("HTTP error")
        assert "1" in failed
        return answer(payload, 0.99)

    client._send = send
    candidates = [candidate(i, "正文" * 1500) for i in range(1, 6)]
    result = await client.judge(
        [], candidates, lambda mid: mid not in failed, on_failed=failed.update
    )
    assert result["1"] is None
    assert result["5"] == 0.99


@pytest.mark.parametrize("path", ["/v1/systemone", "/proxy/jev/judge?route=test"])
async def test_real_async_http_transport_custom_url_headers_and_reuse(path):
    received = []

    async def handle(request):
        assert request.path_qs == path
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = await request.json()
        assert payload["state"]["bot"]["id"] == "789"
        assert payload["state"]["rules"] == jev.INSTRUCTIONS
        assert payload["state"]["candidates"][0]["text"] == "你好"
        assert "content_fragment" not in payload["state"]["candidates"][0]
        received.append(payload)
        return web.json_response(answer(payload, 0.9))

    app = web.Application()
    app.router.add_post(path.split("?", 1)[0], handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = JevClient(
        Settings.from_mapping(
            {
                "semantic": {
                    "jev_api_key": "test-key",
                    "jev_api_url": f"http://127.0.0.1:{port}{path}",
                }
            }
        )
    )
    try:
        assert await client.judge([], [candidate(1)], lambda _: True) == {"1": 0.9}
        session = client.session
        await client.judge([], [candidate(2)], lambda _: True)
        assert client.session is session
        assert len(received) == 2
    finally:
        await client.close()
        await runner.cleanup()
    assert session.closed


async def test_result_summary_reaches_webui_with_message_mapping(webui_logs):
    active = {"101", "102"}
    client = JevClient(replace(Settings(), jev_api_key="test-secret"))

    async def send(payload):
        active.remove("101")
        return {
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 123},
            "answers": {
                "q0": {"type": "noul", "noul": 0.96},
                "q1": {"type": "noul", "noul": 0.12},
            },
        }

    client._send = send
    await client.judge(
        [], [candidate(101, "private-body"), candidate(102)], active.__contains__
    )
    entries = [e for e in webui_logs if "Jev 判断完成" in e["data"]]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["level"] == "INFO"
    assert "[astrbot_plugin_qq_group_enhance]" in entry["data"]
    data = entry["data"]
    assert "模型=jev-1.13.0" in data
    assert "Token=输入 123 / 输出 未返回" in data
    assert "判定项=2 | 尝试=1/2 | 阈值=75.00%" in data
    assert "\n  消息=101 | 概率=96.00% | 达到阈值 | 已失效，忽略" in data
    assert "\n  消息=102 | 概率=12.00% | 未达阈值 | 仍有效" in data
    assert "片段=" not in data
    assert "private-body" not in data
    assert "test-secret" not in data


async def test_request_failure_reaches_webui(webui_logs):
    client = JevClient(Settings())
    client._send = AsyncMock(side_effect=RuntimeError("Jev HTTP 503"))
    assert await client.judge([], [candidate(1)], lambda _: True) == {"1": None}
    entries = [e for e in webui_logs if "Jev request failed" in e["data"]]
    assert len(entries) == 2
    assert all(e["level"] == "WARNING" and "HTTP 503" in e["data"] for e in entries)
