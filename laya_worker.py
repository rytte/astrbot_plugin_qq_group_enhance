"""Persistent JSON-line Laya scorer using the plugin's bundled source."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"


def emit(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def questions_from_rows(rows: list[dict]) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Laya rows 必须为非空列表")
    questions = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Laya 题目缺少 ID")
        if row["id"] in questions:
            raise ValueError(f"Laya 题目 ID 重复：{row['id']}")
        options = row.get("options")
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= len(LETTERS)
            or any(
                not isinstance(option, dict)
                or not isinstance(option.get("id"), str)
                or not option["id"]
                or not isinstance(option.get("description"), str)
                or not option["description"]
                for option in options
            )
        ):
            raise ValueError(f"Laya 题目 {row['id']} 必须包含 2～16 个有效选项")
        if len({option["id"] for option in options}) != len(options):
            raise ValueError(f"Laya 题目 {row['id']} 的选项 ID 必须唯一")
        if not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError(f"Laya 题目 {row['id']} 缺少问题描述")
        questions[row["id"]] = {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {
                label: option["description"] for label, option in zip(LETTERS, options)
            },
        }
    return questions


def prepare_state(
    request: dict, agent, questions: dict
) -> tuple[str | dict | list, int]:
    from laya.common import build_sequence, render_options

    state = request.get("state")
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("Laya state 必须为非空文本、对象或数组")
    if isinstance(state, dict) and isinstance(state.get("history"), list):
        state = {**state, "history": list(state["history"])}
    max_len = agent.cfg["max_len"]
    head_max_len = agent.cfg["head_max_len"]
    internals = {}
    for qid, question in questions.items():
        internal = agent._to_internal(question)
        for option in render_options(internal):
            tokens = agent.tok(
                " " + option.replace(agent.tok.mask_token, " "),
                add_special_tokens=False,
            )["input_ids"]
            if len(tokens) > 48:
                raise ValueError(f"Laya 题目 {qid} 的选项超过模型 48 Token 上限")
        full_head = build_sequence(agent.tok, "", internal, 10**9, 10**9)
        bounded_head = build_sequence(agent.tok, "", internal, max_len, head_max_len)
        if full_head != bounded_head:
            raise ValueError(f"Laya 题目 {qid} 超过模型题头 Token 上限")
        internals[qid] = internal

    removed = 0
    while True:
        fits = True
        for internal in internals.values():
            full = build_sequence(agent.tok, state, internal, 10**9, 10**9)
            bounded = build_sequence(agent.tok, state, internal, max_len, head_max_len)
            if full != bounded:
                fits = False
                break
        if fits:
            return state, removed
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("history"), list)
            or not state["history"]
        ):
            raise ValueError("Laya 输入超过模型 Token 上限，无法完整保留测试内容")
        state["history"].pop(0)
        removed += 1


def score_request(request: dict, agent, batch_rows: int = 8) -> dict:
    rows = request["rows"]
    questions = questions_from_rows(rows)
    state, removed = prepare_state(request, agent, questions)
    answers = {}
    ids = list(questions)
    for start in range(0, len(ids), batch_rows):
        batch_ids = ids[start : start + batch_rows]
        result = agent.predict(state, {qid: questions[qid] for qid in batch_ids})
        if agent.device.type != request.get("device", agent.device.type):
            raise RuntimeError("Laya 推理期间切换了计算设备")
        batch_answers = result.get("answers")
        if not isinstance(batch_answers, dict) or set(batch_answers) != set(batch_ids):
            raise ValueError("Laya 返回的题目 ID 不匹配")
        answers.update(batch_answers)
    scores = []
    for row in rows:
        answer = answers[row["id"]]
        labels = LETTERS[: len(row["options"])]
        if (
            not isinstance(answer, dict)
            or answer.get("type") != "choice"
            or answer.get("choice") not in labels
            or not isinstance(answer.get("probabilities"), dict)
            or set(answer["probabilities"]) != set(labels)
        ):
            raise ValueError(f"Laya 题目 {row['id']} 返回无效 choice 结果")
        scores.append(
            {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": [answer["probabilities"][label] for label in labels],
            }
        )
    return {"scores": scores, "history_omitted": removed}


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--batch-rows", type=int, required=True)
    args = parser.parse_args()
    if not 2 <= args.batch_rows <= 32:
        parser.error("batch-rows must be between 2 and 32")
    try:
        source = Path(__file__).resolve().parent / "vendor"
        if not (source / "laya" / "agent.py").is_file():
            raise ValueError("插件内缺少 vendor/laya/agent.py")
        sys.path.insert(0, str(source))
        model = Path(args.model)
        required_files = (
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        )
        if any(not (model / relative).is_file() for relative in required_files):
            raise ValueError(
                "\u6a21\u578b\u5c1a\u672a\u4e0b\u8f7d\u5b8c\u6574\uff0c"
                "\u8bf7\u5148\u5728\u63d2\u4ef6\u9875\u9762\u70b9\u51fb\u4e0b\u8f7d"
            )
        with contextlib.redirect_stdout(sys.stderr):
            import laya

            agent = laya.load(str(model), device=args.device)
        if agent.device.type != args.device:
            raise RuntimeError(f"Laya 未能在所选 {args.device} 设备上加载模型")
        emit({"ready": True, "model": str(model), "source": str(source)})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"ready": False, "error": str(error)})
        return
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            request_id = request["request_id"]
            with contextlib.redirect_stdout(sys.stderr):
                result = score_request(
                    {**request, "device": args.device}, agent, args.batch_rows
                )
            emit({"request_id": request_id, **result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(
                {
                    "request_id": request.get("request_id")
                    if isinstance(request, dict)
                    else None,
                    "error": str(error),
                }
            )


if __name__ == "__main__":
    main()
