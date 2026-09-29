# Release environment

The experimental manifests record Python 3.13.9 on macOS arm64. Exact installed Python package versions were not embedded in the historical run manifests.

Before creating the permanent Zenodo archive, run on the **same environment used for the experiments**:

```bash
python3 scripts/capture_environment.py
```

This creates `environment/requirements-lock.txt` and `environment/runtime.json`. Commit both files and include them in the Zenodo deposit. Do not substitute versions from another machine.
