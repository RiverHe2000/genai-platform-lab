from __future__ import annotations

import pytest

from ragpipe.textproc import (
    paragraph_spans,
    sentence_spans,
    stem,
    tokenize,
    word_count,
    word_spans,
)


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("The LVR limits are 80% and 95%.") == ["lvr", "limit", "80", "95"]


def test_tokenize_keeps_internal_dots_and_apostrophes() -> None:
    assert tokenize(
        "CET1 rose to 11.5% — don't panic", remove_stopwords=False, apply_stem=False
    ) == [
        "cet1",
        "rose",
        "to",
        "11.5",
        "don't",
        "panic",
    ]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("policies", "policy"),
        ("losses", "loss"),
        ("limits", "limit"),
        ("analysis", "analysis"),
        ("status", "status"),
        ("class", "class"),
        ("gas", "gas"),
        ("is", "is"),
    ],
)
def test_stem_is_conservative(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_sentence_spans_basic_and_slices_match() -> None:
    text = "Rates rose sharply. Prices fell!  Really? Yes."
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "Rates rose sharply.",
        "Prices fell!",
        "Really?",
        "Yes.",
    ]


def test_sentence_spans_guards_abbreviations_and_initials() -> None:
    text = "Use HQLA, e.g. Government bonds. See No. 5 applies. J. Smith signed. Done."
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "Use HQLA, e.g. Government bonds.",
        "See No. 5 applies.",
        "J. Smith signed.",
        "Done.",
    ]


def test_sentence_spans_does_not_split_on_decimal_or_lowercase() -> None:
    text = "CET1 is 11.5% today. it stays. Next sentence."
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == ["CET1 is 11.5% today. it stays.", "Next sentence."]


def test_sentence_spans_empty_and_whitespace() -> None:
    assert sentence_spans("") == []
    assert sentence_spans("   \n ") == []


def test_paragraph_spans() -> None:
    text = "# Title\n\nFirst para line one\nline two.\n\n\n  Second para.  \n"
    spans = paragraph_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "# Title",
        "First para line one\nline two.",
        "Second para.",
    ]
    assert paragraph_spans("\n\n") == []


def test_word_spans_and_count() -> None:
    text = "  alpha beta\tgamma\n"
    assert [text[s:e] for s, e in word_spans(text)] == ["alpha", "beta", "gamma"]
    assert word_count(text) == 3
    assert word_count("") == 0
