from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

# AstrBot initializes storage on import. Keep integration tests off live data.
_runtime = tempfile.TemporaryDirectory(prefix="qq-group-enhance-test-")
_old_root = os.environ.get("ASTRBOT_ROOT")
os.environ["ASTRBOT_ROOT"] = _runtime.name
_workspace = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_workspace))
_source_override = os.environ.get("ASTRBOT_TEST_SOURCE")
_source = Path(_source_override) if _source_override else _workspace / "AstrBot"
if _source_override and not (_source / "astrbot" / "__init__.py").is_file():
    raise ValueError("ASTRBOT_TEST_SOURCE must contain the astrbot source package")
if _source.is_dir():
    sys.path.insert(0, str(_source))


def pytest_configure(config):
    # Keep media fixtures inside the same isolated runtime as AstrBot storage,
    # avoiding a shared pytest-of-user directory across Windows sandbox users.
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(_runtime.name) / "pytest")


@pytest.fixture(autouse=True)
def no_external_metrics(monkeypatch):
    # Real event.send schedules telemetry; it is unrelated to these tests and
    # must neither contact telemetry services nor leave cross-loop DB tasks.
    metrics = sys.modules.get("astrbot.core.utils.metrics")
    if metrics:

        async def discard(**kwargs):
            pass

        monkeypatch.setattr(metrics.Metric, "upload", discard)


@pytest.fixture
def webui_logs(monkeypatch):
    """Exercise the real WebUI broker, rather than pytest's root log capture."""
    import astrbot.api as api
    from astrbot.core import logger
    from astrbot.core.log import LogBroker, LogManager, LogQueueHandler
    from astrbot.core.star.star import star_map
    from astrbot_plugin_qq_group_enhance import main

    monkeypatch.setattr(star_map[main.__name__], "name", main.PLUGIN_NAME)
    monkeypatch.setattr(api, "_logger_cache", {})
    monkeypatch.setattr(LogManager, "_log_broker", LogManager._log_broker)
    broker = LogBroker()
    LogManager.set_queue_handler(logger, broker)
    try:
        yield broker.log_cache
    finally:
        for target in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(target, logging.Logger):
                for handler in target.handlers[:]:
                    if (
                        isinstance(handler, LogQueueHandler)
                        and handler.log_broker is broker
                    ):
                        target.removeHandler(handler)
                        handler.close()


def pytest_unconfigure(config):
    # Dispose AstrBot storage before removing the temporary runtime on Windows.
    core = sys.modules.get("astrbot.core")
    if core:

        async def close_storage():
            await core.sp.close()
            await core.db_helper.engine.dispose()

        asyncio.run(close_storage())
    if _old_root is None:
        os.environ.pop("ASTRBOT_ROOT", None)
    else:
        os.environ["ASTRBOT_ROOT"] = _old_root
    _runtime.cleanup()
