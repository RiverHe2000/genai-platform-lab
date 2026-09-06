"""The one *high-risk* tool: flag a loan for watchlist review. It changes state that other
people act on, so the graph interrupts for human approval before it runs, and the approver's
identity is written with the record."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agentguard.tools.base import ToolContext, ToolResult, ToolSpec
from agentguard.tools.loanbook import LoanBook


class FlagArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loan_id: str = Field(min_length=1, description="Loan identifier, e.g. L00042.")
    reason: str = Field(min_length=5, max_length=500, description="Why the loan needs review.")


def make_review_tool(book: LoanBook) -> ToolSpec:
    def handler(args: BaseModel, ctx: ToolContext) -> ToolResult:
        loan_id = str(getattr(args, "loan_id", ""))
        reason = str(getattr(args, "reason", ""))
        if ctx.approved_by is None:
            return ToolResult(ok=False, output="flag_for_review requires an approver")
        if not book.loan_exists(loan_id):
            return ToolResult(ok=False, output=f"loan {loan_id} does not exist")
        ticket = book.add_review(
            loan_id, reason, requested_by=f"agent:{ctx.thread_id}", approved_by=ctx.approved_by
        )
        return ToolResult(
            ok=True,
            output=(
                f"loan {loan_id} added to the review queue "
                f"(ticket {ticket}, approved by {ctx.approved_by})"
            ),
            data={"ticket": ticket},
        )

    return ToolSpec(
        name="flag_for_review",
        description="Add a loan to the watchlist review queue. A human must approve this action.",
        args_model=FlagArgs,
        handler=handler,
        risk="high",
    )
