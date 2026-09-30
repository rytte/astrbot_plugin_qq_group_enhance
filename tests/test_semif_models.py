import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot.core.star.star_manager import PluginManager
from astrbot.dashboard.services.plugin_page_service import PluginPageService
from astrbot_plugin_qq_group_enhance import main, semif_models
from astrbot_plugin_qq_group_enhance.main import QQGroupEnhancePlugin
from astrbot_plugin_qq_group_enhance.semif import SemIfClient
from astrbot_plugin_qq_group_enhance.semif_models import (
    MODEL_MANIFEST,
    SEMIF_MODELS,
    get_semif_model_dir,
    is_semif_model_ready,
    missing_semif_model_files,
)
from astrbot_plugin_qq_group_enhance.state import Settings


def make_preset(directory: Path, spec):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / MODEL_MANIFEST).write_text(
        json.dumps({"repo_id": spec.repo_id, "revision": spec.revision}),
        encoding="utf-8",
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (directory / "model-00001.safetensors").write_bytes(b"weights")


async def test_astrbot_discovers_semif_model_page():
    plugin_root = Path(__file__).resolve().parents[1]
    plugin = SimpleNamespace(
        name=plugin_root.name, root_dir_name=plugin_root.name, reserved=False
    )
    manager = SimpleNamespace(
        plugin_store_path=str(plugin_root.parent),
        reserved_plugin_path=str(plugin_root.parent),
    )
    service = PluginPageService(manager)
    pages = await service.serialize_plugin_pages(plugin)
    assert {page["name"] for page in pages} >= {"laya-models", "semif-models"}
    assert pages[0]["name"] == "laya-models"
    sidebar_page = (plugin_root / "pages" / pages[0]["name"] / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'data-backend="laya"' in sidebar_page
    assert 'data-backend="semif"' in sidebar_page
    assert 'data-backend="jev"' in sidebar_page
    assert 'id="test-form"' in sidebar_page
    titles = PluginManager._load_plugin_i18n(str(plugin_root))["zh-CN"]["pages"]
    assert titles["laya-models"]["title"] == "模型管理与测试"
    assert titles["semif-models"]["title"] == "SemIf 模型"


def test_semif_preset_and_custom_model_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        semif_models, "get_astrbot_plugin_data_path", lambda: str(tmp_path)
    )
    custom = tmp_path / "custom"
    custom.mkdir()
    custom_settings = Settings.from_mapping(
        {
            "semantic": {
                "semantic_backend": "semif",
                "semif_model": str(custom),
                "semif_revision": "local-revision",
            }
        }
    )
    custom_args = SemIfClient(custom_settings).worker_args()
    assert custom_settings.semif_preset == "custom"
    assert custom_args[custom_args.index("--model") + 1] == str(custom)
    assert custom_args[custom_args.index("--revision") + 1] == "local-revision"

    spec = SEMIF_MODELS[1]
    preset_settings = Settings.from_mapping(
        {
            "semantic": {
                "semantic_backend": "semif",
                "semif_preset": spec.key,
            }
        }
    )
    with pytest.raises(RuntimeError, match="尚未下载完整"):
        SemIfClient(preset_settings).worker_args()
    directory = get_semif_model_dir(spec.key)
    make_preset(directory, spec)
    assert is_semif_model_ready(spec.key)
    args = SemIfClient(preset_settings).worker_args()
    assert args[args.index("--model") + 1] == str(directory)
    assert args[args.index("--revision") + 1] == spec.revision
    with pytest.raises(ValueError, match="semif_preset"):
        Settings.from_mapping(
            {
                "semantic": {
                    "semantic_backend": "semif",
                    "semif_preset": "unknown",
                }
            }
        )


def test_semif_preset_rejects_wrong_revision_and_missing_shard(tmp_path):
    spec = SEMIF_MODELS[0]
    make_preset(tmp_path, spec)
    assert missing_semif_model_files(tmp_path, spec) == ()
    (tmp_path / "model-00001.safetensors").unlink()
    assert "model-00001.safetensors" in missing_semif_model_files(tmp_path, spec)
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    (tmp_path / MODEL_MANIFEST).write_text(
        json.dumps({"repo_id": spec.repo_id, "revision": "wrong"}),
        encoding="utf-8",
    )
    assert MODEL_MANIFEST in missing_semif_model_files(tmp_path, spec)
    (tmp_path / "tokenizer.json").write_bytes(b"")
    assert "tokenizer.json" in missing_semif_model_files(tmp_path, spec)


async def test_semif_page_selects_and_downloads_pinned_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        semif_models, "get_astrbot_plugin_data_path", lambda: str(tmp_path)
    )

    class Config(dict):
        saves = 0

        def save_config(self):
            self.saves += 1

    class Context:
        def __init__(self):
            self.routes = {}

        def register_web_api(self, route, handler, methods, description):
            self.routes[route] = (handler, methods)

    config = Config(semantic={})
    context = Context()
    plugin = QQGroupEnhancePlugin(context, config)
    spec = SEMIF_MODELS[2]
    monkeypatch.setattr(
        main,
        "request",
        SimpleNamespace(json=lambda default=None: request_model(spec.key)),
    )

    async def fake_subprocess(*args, **kwargs):
        assert args[0] == sys.executable
        assert Path(args[2]).name == "semif_download_worker.py"
        assert args[args.index("--repo") + 1] == spec.repo_id
        assert args[args.index("--revision") + 1] == spec.revision
        make_preset(Path(args[args.index("--target") + 1]), spec)

        async def communicate():
            return b'{"ok": true}', b""

        return SimpleNamespace(communicate=communicate, returncode=0)

    async def request_model(key):
        return {"model": key}

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", fake_subprocess)
    try:
        prefix = "/astrbot_plugin_qq_group_enhance/semif/models"
        assert set(context.routes) >= {prefix, prefix + "/select", prefix + "/download"}
        monkeypatch.setattr(
            main,
            "request",
            SimpleNamespace(json=lambda default=None: request_model("custom")),
        )
        invalid_custom = await plugin.page_semif_model_select()
        assert invalid_custom.status_code == 400
        assert "semif_preset" not in config["semantic"]
        custom = tmp_path / "custom"
        custom.mkdir()
        config["semantic"].update(
            semif_model=str(custom), semif_revision="local-revision"
        )
        accepted_custom = await plugin.page_semif_model_select()
        assert accepted_custom.status_code == 200
        assert config["semantic"]["semif_preset"] == "custom"
        monkeypatch.setattr(
            main,
            "request",
            SimpleNamespace(json=lambda default=None: request_model(spec.key)),
        )
        selected = await plugin.page_semif_model_select()
        assert json.loads(selected.body)["selected"] == spec.key
        assert config["semantic"]["semif_preset"] == spec.key
        assert config.saves == 2

        started = await plugin.page_semif_model_download()
        assert json.loads(started.body)["state"] == "downloading"
        again = await plugin.page_semif_model_download()
        assert json.loads(again.body)["state"] == "downloading"
        await plugin._semif_download_tasks[spec.key]
        assert is_semif_model_ready(spec.key)
        listing = json.loads((await plugin.page_semif_models()).body)
        assert listing["selected"] == spec.key
        assert (
            next(m for m in listing["models"] if m["key"] == spec.key)["state"]
            == "ready"
        )
    finally:
        await plugin.terminate()


async def test_incomplete_semif_download_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(
        semif_models, "get_astrbot_plugin_data_path", lambda: str(tmp_path)
    )
    spec = SEMIF_MODELS[0]
    plugin = QQGroupEnhancePlugin(SimpleNamespace(), {})

    async def incomplete_subprocess(*args, **kwargs):
        staging = Path(args[args.index("--target") + 1])
        make_preset(staging, spec)
        (staging / "model-00001.safetensors").unlink()

        async def communicate():
            return b'{"ok": true}', b""

        return SimpleNamespace(communicate=communicate, returncode=0)

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", incomplete_subprocess)
    try:
        await plugin._download_semif_model(spec)
        assert not get_semif_model_dir(spec.key).exists()
        assert not list((get_semif_model_dir(spec.key).parent / ".downloads").iterdir())
        payload = plugin._semif_model_payload(spec)
        assert payload["state"] == "error"
        assert "model-00001.safetensors" in payload["error"]
    finally:
        await plugin.terminate()
