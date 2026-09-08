"""Shared test configuration.

One rule: a test marked ``network`` is skipped whenever the environment says the Hugging Face
hub is off limits. CI sets ``HF_HUB_OFFLINE=1``, so the marker's description -- "skipped in
CI" -- is enforced here rather than left to each test's own guard. Locally, with the variable
unset and the tiny model cached, the test runs.
"""

from __future__ import annotations

import os

import pytest

OFFLINE = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``network`` tests when the hub is unreachable by policy."""
    if not OFFLINE:
        return
    skip = pytest.mark.skip(reason="needs the Hugging Face hub; HF_HUB_OFFLINE is set")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
