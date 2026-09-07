"""The domain the MCP tools expose: a small, deterministic wealth-management platform.

The world is generated once from a seed and is then read-only for the duration of a
benchmark run, apart from the two write tools, which append to their own logs. Because
every fact is known, the benchmark can grade an answer without a judge model, and a
"reconciliation" task can plant a discrepancy and check that the agent finds exactly it.

Money is carried as `Decimal` end to end. A benchmark that grades numeric answers must
not itself be the source of a rounding error.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Account",
    "AccountType",
    "Client",
    "FeeSchedule",
    "Holding",
    "Note",
    "Order",
    "PolicyDoc",
    "PriceBar",
    "RiskProfile",
    "Transaction",
    "TransactionKind",
]


class RiskProfile(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BALANCED = "balanced"
    GROWTH = "growth"
    HIGH_GROWTH = "high_growth"


class AccountType(StrEnum):
    SUPER = "super"
    PENSION = "pension"
    INVESTMENT = "investment"


class TransactionKind(StrEnum):
    CONTRIBUTION = "contribution"
    WITHDRAWAL = "withdrawal"
    BUY = "buy"
    SELL = "sell"
    FEE = "fee"
    DISTRIBUTION = "distribution"


class Client(BaseModel):
    model_config = ConfigDict(frozen=True)

    client_id: str
    name: str
    adviser: str
    risk_profile: RiskProfile
    date_of_birth: date
    review_due: date
    state: str


class Account(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    client_id: str
    account_type: AccountType
    opened: date
    cash_balance: Decimal


class Holding(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    ticker: str
    name: str
    asset_class: Literal["equity", "fixed_income", "property", "cash", "alternative"]
    units: Decimal
    cost_base: Decimal


class Transaction(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    account_id: str
    trade_date: date
    kind: TransactionKind
    ticker: str | None
    amount: Decimal
    description: str


class PriceBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_at: date
    close: Decimal


class FeeSchedule(BaseModel):
    """Tiered platform fee, expressed as basis points per annum on each tier."""

    model_config = ConfigDict(frozen=True)

    schedule_id: str
    name: str
    tiers: tuple[tuple[Decimal, Decimal], ...]
    """Each tier is (upper_bound_inclusive, basis_points); the last bound is the sentinel."""
    account_fee: Decimal = Decimal("0")
    capped_at: Decimal | None = None


class PolicyDoc(BaseModel):
    """A platform policy document, addressable as an MCP resource."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    section: str
    body: str
    effective: date

    @property
    def uri(self) -> str:
        return f"policy://{self.doc_id}"


class Note(BaseModel):
    """Appended by the `note_append` write tool."""

    model_config = ConfigDict(frozen=True)

    note_id: str
    client_id: str
    author: str
    body: str


class Order(BaseModel):
    """Appended by the `order_place` write tool; the highest-risk action in the world."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    account_id: str
    side: Literal["buy", "sell"]
    ticker: str
    amount: Decimal
    approved_by: str = Field(default="", description="Empty means the order was never approved.")
