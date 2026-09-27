# Same gates as CI. Usage: make install | make lint | make type | make test | make all
PY ?= python

.PHONY: install lint format type test all clean results gate

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy

test:
	pytest

all: lint type test

# Harness outcomes are fixed regressions; the research comparison may honestly return HOLD.
gate:
	mcpeval bench run --arch single --model scripted --out runs/ci-single
	mcpeval bench run --arch supervisor --model scripted --out runs/ci-supervisor
	$(PY) -m mcpeval.bench.regression runs/ci-single --expected gates/scripted-single.json
	$(PY) -m mcpeval.bench.regression runs/ci-supervisor --expected gates/scripted-supervisor.json
	mcpeval bench compare runs/ci-single runs/ci-supervisor --margin 0.05

# Regenerates the inputs to docs/RESULTS.md. The HF stages need a GPU and take hours.
results:
	bash scripts/run_experiments.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
