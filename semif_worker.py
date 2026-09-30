"""Persistent, newline-delimited JSON scorer process for the SemIf backend.

Run this file with a separate Python environment containing the model dependencies.
The semif_phase1 source ships with this plugin.
Stdout is reserved for protocol messages.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path


def emit(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def prepare_rows(request: dict, tokenizer, max_tokens: int):
    from semif_phase1.direct import encode_prompt

    state = request["state"]
    definitions = request["rows"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("SemIf state 必须为非空文本、对象或数组")
    if isinstance(state, dict) and isinstance(state.get("history"), list):
        state = {**state, "history": list(state["history"])}
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("SemIf rows 必须为非空列表")
    rows = [{**definition, "state": state} for definition in definitions]
    removed = 0
    while True:
        try:
            for row in rows:
                encode_prompt(tokenizer, row, max_tokens)
            return rows, removed
        except ValueError as error:
            if (
                "input tokens exceed limit" not in str(error)
                or not isinstance(state, dict)
                or not isinstance(state.get("history"), list)
                or not state["history"]
            ):
                raise
            state["history"].pop(0)
            removed += 1


def score_request(
    request: dict, model, tokenizer, metadata: dict, max_tokens: int, batch_rows: int
):
    from semif_phase1.direct import score as score_direct
    from semif_phase1.shared import score_shared

    rows, removed = prepare_rows(request, tokenizer, max_tokens)
    scores = []
    direct_rows = 0
    for start in range(0, len(rows), batch_rows):
        batch = rows[start : start + batch_rows]
        try:
            result, _ = score_shared(model, tokenizer, batch, metadata, max_tokens)
        except ValueError as error:
            if "fixed state prefix does not match every full prompt" not in str(error):
                raise
            result = [
                score_direct(model, tokenizer, row, metadata, max_tokens)
                for row in batch
            ]
            direct_rows += len(batch)
        scores.extend(
            {
                key: row[key]
                for key in ("id", "option_ids", "probabilities", "input_tokens")
            }
            for row in result
        )
    return {"scores": scores, "history_omitted": removed, "direct_rows": direct_rows}


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), required=True
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--batch-rows", type=int, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.batch_rows < 1:
        parser.error("max-tokens and batch-rows must be positive")
    try:
        # Some model libraries print during import/loading. Keep protocol clean.
        source = Path(__file__).resolve().parent / "vendor"
        if not (source / "semif_phase1" / "core.py").is_file():
            raise ValueError("插件内缺少 vendor/semif_phase1/core.py")
        sys.path.insert(0, str(source))
        with contextlib.redirect_stdout(sys.stderr):
            from semif_phase1.core import load_causal_model

            model, tokenizer, metadata = load_causal_model(
                args.model, args.revision, args.device, args.dtype
            )
        emit({"ready": True, "model": metadata, "source": str(source)})
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
                    request,
                    model,
                    tokenizer,
                    metadata,
                    args.max_tokens,
                    args.batch_rows,
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
