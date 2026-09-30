"""SemIf choice scorer using the shared persistent local worker protocol."""

from __future__ import annotations

import sys
from pathlib import Path

from .local_semantic import LocalChoiceClient
from .semif_models import (
    CUSTOM_SEMIF_MODEL,
    get_semif_model_dir,
    get_semif_model_spec,
    is_semif_model_ready,
)


class SemIfClient(LocalChoiceClient):
    name = "SemIf"
    WORKER_PATH = Path(__file__).with_name("semif_worker.py")

    def worker_args(self) -> list[str]:
        settings = self.settings
        if settings.semif_preset == CUSTOM_SEMIF_MODEL:
            model, revision = settings.semif_model, settings.semif_revision
        else:
            spec = get_semif_model_spec(settings.semif_preset)
            if not is_semif_model_ready(spec.key):
                raise RuntimeError(
                    f"SemIf 预设模型 {spec.label} 尚未下载完整，请先在插件页面点击下载"
                )
            model, revision = str(get_semif_model_dir(spec.key)), spec.revision
        return [
            sys.executable,
            "-u",
            str(self.WORKER_PATH),
            "--model",
            model,
            "--revision",
            revision,
            "--device",
            settings.semif_device,
            "--dtype",
            settings.semif_dtype,
            "--max-tokens",
            str(settings.semif_max_tokens),
            "--batch-rows",
            str(settings.semif_batch_rows),
        ]

    @property
    def startup_seconds(self) -> float:
        return self.settings.semif_startup_seconds

    @property
    def timeout_seconds(self) -> float:
        return self.settings.semif_timeout_seconds
