# Same gates as CI. Usage: make install | make lint | make type | make test | make all
PY ?= python

.PHONY: install lint format type test all clean results

install:
	$(PY) -m pip install -e ".[dev,serve]"

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

# Regenerates docs/RESULTS.md inputs: load tests, evaluation of baseline vs candidate through
# the gateway, and the promotion decision (the HF stages want a GPU).
results:
	bash scripts/run_experiments.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
