"""Laya checkpoint metadata and plugin data directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

PLUGIN_NAME = "astrbot_plugin_qq_group_enhance"


@dataclass(frozen=True)
class LayaModelSpec:
    key: str
    label: str
    repo_id: str
    description: str


LAYA_MODELS = (
    LayaModelSpec(
        key="english",
        label="English",
        repo_id="convaiinnovations/laya",
        description="适合英文文本，ModernBERT-large。",
    ),
    LayaModelSpec(
        key="multilingual",
        label="Multilingual",
        repo_id="convaiinnovations/laya-multilingual",
        description="支持 100 多种语言，适合中文群聊（默认）。",
    ),
    LayaModelSpec(
        key="typed-decisions",
        label="Typed Decisions",
        repo_id="convaiinnovations/laya-typed-decisions",
        description="针对 typed-decisions 工作流的 ModernBERT-large 模型。",
    ),
)

LAYA_MODEL_KEYS = frozenset(model.key for model in LAYA_MODELS)
DEFAULT_LAYA_MODEL = "multilingual"
REQUIRED_MODEL_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def get_laya_models_dir() -> Path:
    """Return the persistent directory for all Laya checkpoints."""

    return Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME / "models"


def get_laya_model_spec(model_key: str) -> LayaModelSpec:
    for spec in LAYA_MODELS:
        if spec.key == model_key:
            return spec
    choices = ", ".join(sorted(LAYA_MODEL_KEYS))
    raise ValueError(f"laya_model 只能是 {choices}")


def get_laya_model_dir(model_key: str) -> Path:
    """Resolve a validated model key below the plugin data directory."""

    return get_laya_models_dir() / get_laya_model_spec(model_key).key


def missing_model_files(model_dir: Path) -> tuple[str, ...]:
    return tuple(
        relative
        for relative in REQUIRED_MODEL_FILES
        if not (model_dir / relative).is_file()
    )


def is_laya_model_ready(model_key: str) -> bool:
    return not missing_model_files(get_laya_model_dir(model_key))
