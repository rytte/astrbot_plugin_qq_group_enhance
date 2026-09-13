from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# AstrBot initializes storage on import. Keep integration tests off live data.
_runtime = tempfile.TemporaryDirectory(prefix="qq-group-enhance-test-")
_old_root = os.environ.get("ASTRBOT_ROOT")
os.environ["ASTRBOT_ROOT"] = _runtime.name
_workspace = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_workspace))
if (_workspace / "AstrBot").is_dir():
    sys.path.insert(0, str(_workspace / "AstrBot"))


def pytest_unconfigure(config):
    if _old_root is None:
        os.environ.pop("ASTRBOT_ROOT", None)
    else:
        os.environ["ASTRBOT_ROOT"] = _old_root
    _runtime.cleanup()
