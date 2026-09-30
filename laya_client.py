"""Laya choice scorer using the shared persistent local worker protocol."""

from __future__ import annotations

import sys
from pathlib import Path

from .laya_models import get_laya_model_dir
from .local_semantic import LocalChoiceClient


class LayaClient(LocalChoiceClient):
    name = "Laya"
    WORKER_PATH = Path(__file__).with_name("laya_worker.py")

    def worker_args(self) -> list[str]:
        settings = self.settings
        return [
            sys.executable,
            "-u",
            str(self.WORKER_PATH),
            "--model",
            str(get_laya_model_dir(settings.laya_model)),
            "--device",
            settings.laya_device,
            "--batch-rows",
            str(settings.laya_batch_rows),
        ]

    @property
    def startup_seconds(self) -> float:
        return self.settings.laya_startup_seconds

    @property
    def timeout_seconds(self) -> float:
        return self.settings.laya_timeout_seconds
