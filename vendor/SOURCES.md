# Bundled inference sources

This directory contains Python source code used by the two local semantic backends. The plugin loads these copies first in its isolated worker processes. Model weights and Python dependencies remain external.

| Package | Source snapshot | License |
| --- | --- | --- |
| `semif_phase1` | `D:\mm\SemIf-OpenJev\src\semif_phase1`, copied from the working tree based on commit `23cf1f39fc9534fe81437200959b6dfc7106e45a`; that working tree had uncommitted changes | MIT; see [SEMIF_LICENSE](SEMIF_LICENSE) |
| `laya` | Installed `laya==0.3.11` package from `D:\SemIf-OpenJev\.venv\Lib\site-packages\laya` | Apache-2.0; see [LAYA_LICENSE](LAYA_LICENSE) |

Only `.py` files were copied from each Python package. The exact copied files are listed in [SHA256SUMS](SHA256SUMS). Update a package by reviewing its upstream changes, replacing its source files and license, regenerating the checksums, and running both worker smoke tests and the plugin test suite.
