# Same gates as CI. Usage: make install | make lint | make type | make test | make all
PY ?= python

.PHONY: install lint format type test all clean results

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

# Regenerates docs/RESULTS.md (needs the [dense,hf] extras and a GPU for the LLM part).
results:
	bash scripts/run_experiments.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
