from __future__ import annotations

from agentguard.tools.base import ToolContext
from agentguard.tools.loanbook import LoanBook
from agentguard.tools.policy_search import (
    PolicyClause,
    PolicySearch,
    PolicySearchArgs,
    clauses_from_dicts,
    make_policy_tool,
)
from agentguard.tools.review import FlagArgs, make_review_tool


def test_policy_search_ranks_by_overlap_and_supports_k() -> None:
    search = PolicySearch()
    hits = search.search("maximum LVR without lenders mortgage insurance", k=2)
    assert hits[0][0].id == "CP-1.2"
    assert len(hits) == 2
    assert search.search("", k=3) == []
    assert search.search("zzzz qqqq", k=3) == []
    search.add(PolicyClause("X-1", "Custom", "A bespoke clause about unicorns"))
    assert search.search("unicorns")[0][0].id == "X-1"


def test_policy_tool_renders_ids() -> None:
    spec = make_policy_tool(PolicySearch())
    out = spec.handler(PolicySearchArgs(query="hardship complaints", k=1), ToolContext("t", 1))
    assert out.ok and out.output.startswith("[CH-4.1]")
    assert out.data["ids"] == ["CH-4.1"]
    none = spec.handler(PolicySearchArgs(query="qqqq"), ToolContext("t", 1))
    assert none.output == "no matching policy clauses"
    assert clauses_from_dicts([{"id": "a", "title": "b", "text": "c"}]) == [
        PolicyClause("a", "b", "c")
    ]


def test_review_tool_requires_approver_and_existing_loan(book: LoanBook) -> None:
    spec = make_review_tool(book)
    assert spec.risk == "high"
    args = FlagArgs(loan_id="L00003", reason="45 days past due")
    no_approver = spec.handler(args, ToolContext("t", 1))
    assert not no_approver.ok and "approver" in no_approver.output
    missing = spec.handler(
        FlagArgs(loan_id="L99999", reason="does not exist"), ToolContext("t", 1, approved_by="bob")
    )
    assert not missing.ok and "does not exist" in missing.output
    ok = spec.handler(args, ToolContext("thread-9", 1, approved_by="bob"))
    assert ok.ok and "ticket 1" in ok.output and "approved by bob" in ok.output
    queue = book.review_queue()
    assert queue[0]["requested_by"] == "agent:thread-9"
