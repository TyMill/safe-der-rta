from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "environment"
out.mkdir(exist_ok=True)

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True).stdout
(out / "requirements-lock.txt").write_text(freeze, encoding="utf-8")
meta = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}
(out / "runtime.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(out / "requirements-lock.txt")
print(out / "runtime.json")
