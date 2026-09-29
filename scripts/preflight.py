#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib
import platform

REQUIRED = ["numpy", "pandas", "scipy", "yaml", "matplotlib", "tqdm", "pandapower"]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    missing = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
        except Exception as exc:
            missing.append(name)
            print(f"[MISSING] {name}: {exc}")
    if missing:
        print("\nInstall the project first: pip install -e '.[dev]'")
        return 1
    print("\nEnvironment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
