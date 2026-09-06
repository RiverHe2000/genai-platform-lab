from __future__ import annotations

import pytest

from ragpipe.prompts import (
    ABSTAIN_MARKER,
    SYSTEM_PROMPT,
    build_qa_prompt,
    extract_citations,
    is_abstention,
)


def test_build_qa_prompt_numbers_contexts_from_one() -> None:
    prompt = build_qa_prompt("What is X?", [("Doc A", " text a "), ("Doc B", "text b")])
    assert "[1] (Doc A)\ntext a" in prompt
    assert "[2] (Doc B)\ntext b" in prompt
    assert prompt.endswith("Question: What is X?\nAnswer:")
    assert "(no passages retrieved)" in build_qa_prompt("q", [])
    assert ABSTAIN_MARKER in SYSTEM_PROMPT


def test_extract_citations_unique_in_order() -> None:
    assert extract_citations("The LVR is 80% [2]. With LMI 95% [1][2]. See [12].") == [2, 1, 12]
    assert extract_citations("no citations here") == []
    assert extract_citations("[1234] is not a citation") == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know", True),
        ("I don’t know.", True),
        ("The context does not contain that information.", True),
        ("I do not know the answer based on the passages.", True),
        ("The maximum LVR is 80% [1].", False),
        ("", False),
    ],
)
def test_is_abstention(answer: str, expected: bool) -> None:
    assert is_abstention(answer) is expected
