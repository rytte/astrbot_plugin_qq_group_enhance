"""Pinned SemIf presets stored under AstrBot's plugin data directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

PLUGIN_NAME = "astrbot_plugin_qq_group_enhance"
CUSTOM_SEMIF_MODEL = "custom"
MODEL_MANIFEST = "semif_preset.json"


@dataclass(frozen=True)
class SemIfModelSpec:
    key: str
    label: str
    repo_id: str
    revision: str
    description: str


SEMIF_MODELS = (
    SemIfModelSpec(
        "qwen3-0.6b",
        "Qwen3 0.6B",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "较小的基础模型；适合资源有限的设备，判断质量需自行验证。",
    ),
    SemIfModelSpec(
        "qwen3.5-0.8b",
        "Qwen3.5 0.8B",
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "体积较小；已有评测显示选项顺序偏置。",
    ),
    SemIfModelSpec(
        "minicpm5-2b",
        "MiniCPM5 2B",
        "openbmb/MiniCPM5-2B",
        "12a3808a956f869c767195e9266b59c4d21d92e2",
        "中等规模模型；需要更多显存。",
    ),
    SemIfModelSpec(
        "qwen3.5-4b",
        "Qwen3.5 4B",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "评测表现较稳；显存占用较高。",
    ),
)
SEMIF_MODEL_KEYS = frozenset(spec.key for spec in SEMIF_MODELS)


def get_semif_model_spec(key: str) -> SemIfModelSpec:
    for spec in SEMIF_MODELS:
        if spec.key == key:
            return spec
    raise ValueError(
        "semif_preset 只能是 custom 或 " + ", ".join(sorted(SEMIF_MODEL_KEYS))
    )


def get_semif_models_dir() -> Path:
    return Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME / "models" / "semif"


def get_semif_model_dir(key: str) -> Path:
    return get_semif_models_dir() / get_semif_model_spec(key).key


def missing_semif_model_files(directory: Path, spec: SemIfModelSpec) -> tuple[str, ...]:
    """Check the pinned identity and every shard named by the weight index."""

    missing = []
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        path = directory / name
        try:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    manifest = directory / MODEL_MANIFEST
    try:
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        if identity != {"repo_id": spec.repo_id, "revision": spec.revision}:
            missing.append(MODEL_MANIFEST)
    except (OSError, ValueError):
        missing.append(MODEL_MANIFEST)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            shards = set(data["weight_map"].values())
            if not shards:
                raise ValueError("empty weight map")
            for name in shards:
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.suffix != ".safetensors"
                ):
                    raise ValueError("invalid shard name")
                weight = directory.joinpath(*path.parts)
                if not weight.is_file() or weight.stat().st_size == 0:
                    missing.append(name)
        except (OSError, KeyError, TypeError, ValueError):
            missing.append(index.name)
    elif not any(
        path.is_file() and path.stat().st_size > 0
        for path in directory.glob("*.safetensors")
    ):
        missing.append("*.safetensors")
    return tuple(missing)


def is_semif_model_ready(key: str) -> bool:
    spec = get_semif_model_spec(key)
    return not missing_semif_model_files(get_semif_model_dir(key), spec)
