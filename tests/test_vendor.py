"""The plugin release must contain importable local scorer sources."""

import subprocess
import sys
from pathlib import Path


def test_bundled_packages_are_importable_in_an_isolated_python():
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    script = (
        "import importlib.util, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "print(importlib.util.find_spec(sys.argv[2]).origin)"
    )
    for package in ("semif_phase1", "laya"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(vendor), package],
            check=True,
            capture_output=True,
            text=True,
        )
        assert (
            Path(result.stdout.strip()).resolve()
            == (vendor / package / "__init__.py").resolve()
        )
