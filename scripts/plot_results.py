#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="results/latest")
    args = p.parse_args()
    d = Path(args.run_dir)
    df = pd.read_csv(d / "episode_metrics.csv")
    figdir = d / "figures"
    figdir.mkdir(exist_ok=True)

    g = df.groupby("controller")["grid_safe_rate"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    g.plot(kind="bar")
    plt.ylabel("Mean grid-safe rate")
    plt.tight_layout()
    plt.savefig(figdir / "grid_safe_rate.png", dpi=220)
    plt.close()

    g = df.groupby("controller")["mean_grid_import_mw"].mean().sort_values()
    plt.figure(figsize=(8, 4.5))
    g.plot(kind="bar")
    plt.ylabel("Mean grid import [MW]")
    plt.tight_layout()
    plt.savefig(figdir / "grid_import.png", dpi=220)
    plt.close()

if __name__ == "__main__":
    main()
