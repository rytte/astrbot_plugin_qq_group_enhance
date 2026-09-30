"""Download one Laya checkpoint into a temporary directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOWNLOAD_PATTERNS = (
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/*",
    "tokenizer/*",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download

        target = Path(args.target)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(target),
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
        print(json.dumps({"ok": True}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
