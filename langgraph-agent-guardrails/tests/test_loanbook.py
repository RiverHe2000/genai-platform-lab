from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agentguard.tools.base import ToolContext
from agentguard.tools.loanbook import (
    LOANS_SCHEMA_DOC,
    LoanBook,
    QueryArgs,
    SQLDeniedError,
    make_loanbook_tools,
)

CTX = ToolContext(thread_id="t", step=1)


def test_seed_is_deterministic_and_idempotent(book: LoanBook) -> None:
    other = LoanBook(":memory:", seed=7, n_loans=60)
    a = book.query("SELECT loan_id, balance, stage FROM loans ORDER BY loan_id")
    b = other.query("SELECT loan_id, balance, stage FROM loans ORDER BY loan_id")
    assert a.rows == b.rows
    assert len(a.rows) == 50 and a.truncated
    assert book.query("SELECT COUNT(*) FROM loans").rows == [[60]]
    different = LoanBook(":memory:", seed=8, n_loans=60)
    assert different.query("SELECT balance FROM loans ORDER BY loan_id LIMIT 5").rows != a.rows[:5]


def test_stage_is_consistent_with_days_past_due(book: LoanBook) -> None:
    rows = book.query("SELECT dpd, stage FROM loans").rows
    for dpd, stage in rows:
        expected = 3 if dpd >= 90 else 2 if dpd >= 30 else 1
        assert stage == expected


def test_query_allows_select_over_loans_only(book: LoanBook) -> None:
    ok = book.query("SELECT segment, COUNT(*) AS n FROM loans GROUP BY segment ORDER BY segment")
    assert ok.columns == ["segment", "n"]
    assert {r[0] for r in ok.rows} <= {"retail", "sme", "corporate"}
    with pytest.raises(SQLDeniedError, match="outside the loans table"):
        book.query("SELECT name, tfn FROM customers")
    with pytest.raises(SQLDeniedError, match="outside the loans table"):
        book.query(
            "SELECT l.loan_id, c.email FROM loans l JOIN customers c ON c.customer_id = l.customer_id"
        )
    with pytest.raises(SQLDeniedError, match="outside the loans table"):
        book.query("SELECT * FROM review_queue")
    with pytest.raises(SQLDeniedError, match="outside the loans table"):
        book.query("SELECT sql FROM sqlite_master")


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DELETE FROM loans", "only SELECT"),
        ("UPDATE loans SET stage = 1", "only SELECT"),
        ("PRAGMA table_info(loans)", "only SELECT"),
        ("SELECT 1; DROP TABLE loans", "single statement"),
        ("", "empty"),
        ("   ;  ", "empty"),
    ],
)
def test_query_rejects_mutations_and_stacking(book: LoanBook, sql: str, message: str) -> None:
    with pytest.raises(SQLDeniedError, match=message):
        book.query(sql)
    assert book.query("SELECT COUNT(*) FROM loans").rows == [[60]]


def test_query_budget_aborts_runaway_query() -> None:
    small = LoanBook(":memory:", n_loans=60, max_vm_steps=2000)
    with pytest.raises(SQLDeniedError, match="budget"):
        small.query("SELECT COUNT(*) FROM loans a, loans b, loans c, loans d")


def test_query_syntax_errors_propagate_as_sqlite_errors(book: LoanBook) -> None:
    with pytest.raises(sqlite3.DatabaseError):
        book.query("SELECT FROM WHERE")


def test_cte_and_with_are_allowed(book: LoanBook) -> None:
    res = book.query(
        "WITH s AS (SELECT stage, COUNT(*) n FROM loans GROUP BY stage) SELECT * FROM s ORDER BY stage"
    )
    assert [r[0] for r in res.rows] == sorted({r[0] for r in res.rows})


def test_render_and_truncation_flag(book: LoanBook) -> None:
    res = book.query("SELECT loan_id FROM loans ORDER BY loan_id LIMIT 3")
    assert res.render() == "loan_id\nL00001\nL00002\nL00003\n(3 rows)"
    assert book.query("SELECT loan_id FROM loans WHERE 1 = 0").render() == "(0 rows)"
    assert "truncated" in book.query("SELECT loan_id FROM loans").render()


def test_review_queue_is_privileged_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "book.sqlite"
    book = LoanBook(path, n_loans=20)
    assert book.loan_exists("L00001") and not book.loan_exists("L99999")
    ticket = book.add_review("L00001", "45 dpd", requested_by="agent:t", approved_by="alice")
    assert ticket == 1
    book.close()
    reopened = LoanBook(path, n_loans=20)
    queue = reopened.review_queue()
    assert (
        len(queue) == 1 and queue[0]["approved_by"] == "alice" and queue[0]["loan_id"] == "L00001"
    )
    assert reopened.query("SELECT COUNT(*) FROM loans").rows == [[20]]


def test_tool_specs(book: LoanBook) -> None:
    describe, query = make_loanbook_tools(book)
    assert describe.handler(describe.args_model(), CTX).output == LOANS_SCHEMA_DOC
    ok = query.handler(QueryArgs(sql="SELECT COUNT(*) AS n FROM loans WHERE stage = 3"), CTX)
    assert ok.ok and ok.output.startswith("n\n")
    denied = query.handler(QueryArgs(sql="DROP TABLE loans"), CTX)
    assert not denied.ok and denied.output.startswith("query rejected")
    broken = query.handler(QueryArgs(sql="SELECT nope FROM loans"), CTX)
    assert not broken.ok and broken.output.startswith("SQL error")
    assert query.schema()["required"] == ["sql"]
