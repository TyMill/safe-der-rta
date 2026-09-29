.PHONY: install preflight test smoke run summarize plots

install:
	python3 -m pip install -U pip
	python3 -m pip install -e ".[dev]"

preflight:
	python3 scripts/preflight.py

test:
	pytest -q

smoke:
	python3 scripts/run_experiments.py --config configs/smoke.yaml

run:
	python3 scripts/run_experiments.py --config configs/experiments.yaml

summarize:
	python3 scripts/summarize_results.py --run-dir results/latest

plots:
	python3 scripts/plot_results.py --run-dir results/latest
