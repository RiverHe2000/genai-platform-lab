"""A seeded synthetic loan book in SQLite and a *read-only* SQL tool over it.

Defence in depth for model-written SQL:

* one statement only, must start with SELECT/WITH;
* an SQLite **authorizer** callback that permits SELECT/READ on the ``loans`` table only —
  ``customers`` (names, emails, TFNs) and ``review_queue`` are invisible to the model even
  if it asks for them, and INSERT/UPDATE/DROP/PRAGMA/ATTACH are denied at the engine level;
* a progress handler that aborts runaway queries;
* a row cap with an explicit ``truncated`` flag.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentguard.tools.base import ToolContext, ToolResult, ToolSpec

ALLOWED_TABLES = frozenset({"loans"})
SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
    phone TEXT NOT NULL, tfn TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS loans (
    loan_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    segment TEXT NOT NULL, product TEXT NOT NULL, state TEXT NOT NULL,
    balance REAL NOT NULL, ltv REAL, rate REAL NOT NULL, dpd INTEGER NOT NULL,
    stage INTEGER NOT NULL, originated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT, loan_id TEXT NOT NULL, reason TEXT NOT NULL,
    requested_by TEXT NOT NULL, approved_by TEXT NOT NULL, created_at TEXT NOT NULL
);
"""
LOANS_SCHEMA_DOC = (
    "Table loans(loan_id TEXT, customer_id TEXT, segment TEXT in {retail, sme, corporate}, "
    "product TEXT in {mortgage, personal, overdraft, term_loan}, "
    "state TEXT in {NSW, VIC, QLD, WA, SA}, "
    "balance REAL (AUD), ltv REAL (0-1, NULL for unsecured), rate REAL (annual %), "
    "dpd INTEGER (days past due), stage INTEGER in {1, 2, 3} (IFRS 9), "
    "originated TEXT (YYYY-MM-DD)). "
    "Only SELECT statements over this table are permitted."
)

_FIRST = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Isla", "Lucas", "Chloe", "Oliver"]
_LAST = [
    "Nguyen",
    "Smith",
    "Patel",
    "Chen",
    "Williams",
    "Brown",
    "Singh",
    "Taylor",
    "Lee",
    "Wilson",
]
_STATES = ["NSW", "VIC", "QLD", "WA", "SA"]


def _tfn(rng: random.Random) -> str:
    """A syntactically valid 9-digit TFN (weights 1,4,3,7,5,8,6,9,10; sum mod 11 == 0)."""
    weights = [1, 4, 3, 7, 5, 8, 6, 9, 10]
    while True:
        digits = [rng.randint(0, 9) for _ in range(9)]
        if sum(d * w for d, w in zip(digits, weights, strict=True)) % 11 == 0:
            return "".join(str(d) for d in digits)


def seed_loanbook(conn: sqlite3.Connection, *, n_loans: int = 200, seed: int = 7) -> None:
    rng = random.Random(seed)
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM loans").fetchone()[0]:
        return
    n_customers = max(1, n_loans // 2)
    customers = []
    for i in range(n_customers):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        customers.append(
            (
                f"C{i + 1:04d}",
                f"{first} {last}",
                f"{first.lower()}.{last.lower()}{i}@example.com",
                f"04{rng.randint(10_000_000, 99_999_999)}",
                _tfn(rng),
            )
        )
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)
    loans = []
    for i in range(n_loans):
        segment = rng.choices(["retail", "sme", "corporate"], weights=[70, 25, 5])[0]
        product = (
            rng.choice(["mortgage", "personal"])
            if segment == "retail"
            else rng.choice(["overdraft", "term_loan"])
        )
        balance = {
            "mortgage": rng.uniform(150_000, 1_800_000),
            "personal": rng.uniform(2_000, 75_000),
            "overdraft": rng.uniform(20_000, 500_000),
            "term_loan": rng.uniform(100_000, 5_000_000),
        }[product]
        ltv = round(rng.uniform(0.3, 0.95), 3) if product in ("mortgage", "term_loan") else None
        dpd = rng.choices(
            [0, rng.randint(1, 29), rng.randint(30, 89), rng.randint(90, 400)],
            weights=[85, 8, 4, 3],
        )[0]
        stage = 3 if dpd >= 90 else 2 if dpd >= 30 else 1
        loans.append(
            (
                f"L{i + 1:05d}",
                customers[rng.randrange(n_customers)][0],
                segment,
                product,
                rng.choice(_STATES),
                round(balance, 2),
                ltv,
                round(rng.uniform(4.5, 12.5), 2),
                dpd,
                stage,
                f"20{rng.randint(15, 25):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            )
        )
    conn.executemany("INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", loans)
    conn.commit()


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool

    def render(self) -> str:
        if not self.rows:
            return "(0 rows)"
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in self.rows)
        note = f"\n({len(self.rows)} rows{', truncated' if self.truncated else ''})"
        return f"{head}\n{body}{note}"


class SQLDeniedError(PermissionError):
    pass


class LoanBook:
    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        seed: int = 7,
        n_loans: int = 200,
        max_rows: int = 50,
        max_vm_steps: int = 200_000,
    ) -> None:
        self._conn = sqlite3.connect(str(path))
        self._max_rows = max_rows
        self._max_vm_steps = max_vm_steps
        seed_loanbook(self._conn, n_loans=n_loans, seed=seed)

    # ----- read-only query --------------------------------------------------------------

    @staticmethod
    def _authorizer(
        action: int, arg1: str | None, arg2: str | None, db: str | None, src: str | None
    ) -> int:
        del arg2, db, src
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in ALLOWED_TABLES:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    @staticmethod
    def _normalise(sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            msg = "empty query"
            raise SQLDeniedError(msg)
        if ";" in cleaned:
            msg = "only a single statement is allowed"
            raise SQLDeniedError(msg)
        if not cleaned.lower().startswith(("select", "with")):
            msg = "only SELECT queries are allowed"
            raise SQLDeniedError(msg)
        return cleaned

    def query(self, sql: str) -> QueryResult:
        cleaned = self._normalise(sql)
        budget = {"left": self._max_vm_steps}

        def progress() -> int:
            budget["left"] -= 1000
            return 1 if budget["left"] <= 0 else 0

        self._conn.set_authorizer(self._authorizer)
        self._conn.set_progress_handler(progress, 1000)
        try:
            cursor = self._conn.execute(cleaned)
            rows = cursor.fetchmany(self._max_rows + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        except sqlite3.DatabaseError as exc:
            text = str(exc)
            if "not authorized" in text or "prohibited" in text:
                msg = f"query touches something outside the loans table: {text}"
                raise SQLDeniedError(msg) from exc
            if "interrupted" in text:
                msg = "query exceeded its execution budget"
                raise SQLDeniedError(msg) from exc
            raise
        finally:
            self._conn.set_authorizer(None)
            self._conn.set_progress_handler(None, 0)
        truncated = len(rows) > self._max_rows
        return QueryResult(
            columns=columns, rows=[list(r) for r in rows[: self._max_rows]], truncated=truncated
        )

    # ----- privileged operations (never reachable from model SQL) -----------------------

    def loan_exists(self, loan_id: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM loans WHERE loan_id = ?", (loan_id,)).fetchone()
            is not None
        )

    def add_review(self, loan_id: str, reason: str, *, requested_by: str, approved_by: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO review_queue (loan_id, reason, requested_by, approved_by, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                loan_id,
                reason,
                requested_by,
                approved_by,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def review_queue(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, loan_id, reason, requested_by, approved_by, created_at "
            "FROM review_queue ORDER BY id"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ----- tool specs ---------------------------------------------------------------------------


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, description="A single SELECT statement over the loans table.")


def make_loanbook_tools(book: LoanBook) -> Sequence[ToolSpec]:
    def describe(args: BaseModel, ctx: ToolContext) -> ToolResult:
        del args, ctx
        return ToolResult(ok=True, output=LOANS_SCHEMA_DOC)

    def query(args: BaseModel, ctx: ToolContext) -> ToolResult:
        del ctx
        sql = str(getattr(args, "sql", ""))
        try:
            result = book.query(sql)
        except SQLDeniedError as exc:
            return ToolResult(ok=False, output=f"query rejected: {exc}")
        except sqlite3.DatabaseError as exc:
            return ToolResult(ok=False, output=f"SQL error: {exc}")
        return ToolResult(
            ok=True,
            output=result.render(),
            data={"columns": result.columns, "rows": result.rows, "truncated": result.truncated},
        )

    return (
        ToolSpec(
            name="describe_loanbook",
            description="Return the schema of the loan-book table you may query.",
            args_model=NoArgs,
            handler=describe,
        ),
        ToolSpec(
            name="query_loanbook",
            description="Run a read-only SELECT over the loans table (max 50 rows returned).",
            args_model=QueryArgs,
            handler=query,
        ),
    )
