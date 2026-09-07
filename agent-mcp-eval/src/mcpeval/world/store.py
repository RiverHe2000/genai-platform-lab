"""The read model over the synthetic platform, plus the append-only write log.

:class:`World` is frozen and every derived figure -- a valuation, an annual fee, a fee
reconciliation -- is computed on demand from the stored facts rather than cached in a
field. The benchmark grades numeric answers against these methods, so there must be
exactly one definition of each number; a stored copy is a second definition waiting to
drift from the first.

Two conventions run through the class and are worth stating once:

* Lookups of a single entity return ``None`` when the entity does not exist, and listing
  methods return an empty list. Methods that *compute* over an account instead raise
  ``KeyError``, because there is no honest number to return for an account that is not
  there and silently answering ``0`` would let an agent hallucinate an account and be
  rewarded for it.
* Money is ``Decimal`` and is rounded to cents only at the boundary of a public method.

The mutable half of the world lives in :class:`WorldLog`, deliberately separate. Keeping
writes out of :class:`World` means a benchmark run cannot accidentally mutate the ground
truth it is being graded against, and the log can be discarded between attempts at a task.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from datetime import date
from decimal import Decimal
from random import Random
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mcpeval.world.generate import (
    AS_AT,
    assign_schedules,
    choose_discrepancies,
    fee_schedules,
    generate_accounts,
    generate_activity,
    generate_clients,
    generate_holdings,
    generate_prices,
    money,
    number_transactions,
    plan_fee_transactions,
    policy_documents,
)
from mcpeval.world.models import (
    Account,
    Client,
    FeeSchedule,
    Holding,
    Note,
    Order,
    PolicyDoc,
    PriceBar,
    Transaction,
    TransactionKind,
)

__all__ = ["World", "WorldLog", "build_world", "policy_score", "tiered_fee"]

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
)

TITLE_WEIGHT: Final[float] = 3.0
SECTION_WEIGHT: Final[float] = 2.0
BODY_WEIGHT: Final[float] = 1.0

_SIDES: Final[dict[str, Literal["buy", "sell"]]] = {"buy": "buy", "sell": "sell"}

_ZERO: Final[Decimal] = Decimal("0")
_BASIS: Final[Decimal] = Decimal("10000")


def _terms(text: str) -> set[str]:
    """Content words of ``text``, lowercased and crudely singularised.

    Stripping a trailing "s" is the whole of the morphology here. In this vocabulary the
    plural is the only inflection that separates a query from the document that answers it
    ("fees" against "fee"), and a real stemmer would be a dependency, a source of
    surprises, and impossible to hand-check in a test.
    """
    out: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        out.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return out


def policy_score(query: str, doc: PolicyDoc) -> float:
    """Score one document against a query by weighted term overlap.

    A term counts once per field however often it appears, so a document cannot win by
    repeating a word. Title matches outrank section matches, which outrank body matches:
    a document whose title is the question is almost always the document wanted.

    Deliberately not an embedding: policy retrieval is part of the graded surface, and a
    scorer that can be recomputed by hand keeps the benchmark auditable and offline.
    """
    wanted = _terms(query)
    if not wanted:
        return 0.0
    return (
        TITLE_WEIGHT * len(wanted & _terms(doc.title))
        + SECTION_WEIGHT * len(wanted & _terms(doc.section))
        + BODY_WEIGHT * len(wanted & _terms(doc.body))
    )


def tiered_fee(schedule: FeeSchedule, valuation: Decimal) -> Decimal:
    """The annual platform administration fee on ``valuation`` under ``schedule``.

    Marginal, not flat: each tier's rate applies only to the slice of the valuation that
    falls inside it, exactly as a marginal tax scale works. Charging the top tier's rate on
    the whole balance is the classic mistake this function exists to prevent, and it is
    also the mistake the reconciliation tasks expect an agent to avoid.

    The flat account fee is added before the cap is applied, so a capped account pays the
    cap and not the cap plus the account fee.

    A valuation of zero or less yields the account fee alone; an account with no assets is
    still an account, and the platform still administers it.
    """
    fee = schedule.account_fee
    lower = _ZERO
    for bound, basis_points in schedule.tiers:
        if valuation <= lower:
            break
        fee += (min(valuation, bound) - lower) * basis_points / _BASIS
        lower = bound
    if schedule.capped_at is not None:
        fee = min(fee, schedule.capped_at)
    return money(fee)


class World(BaseModel):
    """An immutable snapshot of the platform at :attr:`as_at`.

    Field defaults are empty so that a test can build a two-account world to probe one
    method, rather than reaching for the full generated world every time.
    """

    model_config = ConfigDict(frozen=True)

    clients: tuple[Client, ...] = ()
    accounts: tuple[Account, ...] = ()
    holdings: tuple[Holding, ...] = ()
    transactions: tuple[Transaction, ...] = ()
    prices: tuple[PriceBar, ...] = ()
    fee_schedules: tuple[FeeSchedule, ...] = ()
    policies: tuple[PolicyDoc, ...] = ()
    account_schedule: dict[str, str] = Field(default_factory=dict)
    fee_discrepancies: tuple[str, ...] = ()
    as_at: date = AS_AT

    _clients_by_id: dict[str, Client] = PrivateAttr(default_factory=dict)
    _clients_by_name: dict[str, Client] = PrivateAttr(default_factory=dict)
    _accounts_by_id: dict[str, Account] = PrivateAttr(default_factory=dict)
    _accounts_by_client: dict[str, list[Account]] = PrivateAttr(default_factory=dict)
    _holdings_by_account: dict[str, list[Holding]] = PrivateAttr(default_factory=dict)
    _transactions_by_account: dict[str, list[Transaction]] = PrivateAttr(default_factory=dict)
    _bars_by_ticker: dict[str, list[PriceBar]] = PrivateAttr(default_factory=dict)
    _bar_dates: dict[str, list[date]] = PrivateAttr(default_factory=dict)
    _schedules_by_id: dict[str, FeeSchedule] = PrivateAttr(default_factory=dict)
    _policies_by_id: dict[str, PolicyDoc] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: Any, /) -> None:
        """Build the lookup indexes once.

        Four thousand eight hundred price bars scanned linearly per valuation, times
        eighty accounts, times every task in a benchmark run, is the difference between a
        test suite that runs in a second and one that does not. The indexes live in
        private attributes so they stay out of the model's fields, its serialisation and
        its equality: two worlds are equal when their facts are equal.
        """
        for client in self.clients:
            self._clients_by_id[client.client_id] = client
            self._clients_by_name.setdefault(client.name.casefold(), client)
        for account in self.accounts:
            self._accounts_by_id[account.account_id] = account
            self._accounts_by_client.setdefault(account.client_id, []).append(account)
        for holding in self.holdings:
            self._holdings_by_account.setdefault(holding.account_id, []).append(holding)
        for transaction in self.transactions:
            self._transactions_by_account.setdefault(transaction.account_id, []).append(transaction)
        for bar in sorted(self.prices, key=lambda b: b.as_at):
            self._bars_by_ticker.setdefault(bar.ticker, []).append(bar)
        for ticker, bars in self._bars_by_ticker.items():
            self._bar_dates[ticker] = [bar.as_at for bar in bars]
        for schedule in self.fee_schedules:
            self._schedules_by_id[schedule.schedule_id] = schedule
        for doc in self.policies:
            self._policies_by_id[doc.doc_id] = doc

    # ----- clients ------------------------------------------------------------------------

    def client(self, client_id: str) -> Client | None:
        """The client with this identifier, or ``None``."""
        return self._clients_by_id.get(client_id)

    def client_by_name(self, name: str) -> Client | None:
        """Resolve a client by name, case and whitespace insensitively.

        Falls back to a substring match, but only when it is unique. Two clients sharing a
        surname must not silently resolve to whichever happens to come first: returning
        ``None`` is what lets the ambiguous task family test whether an agent asks a
        clarifying question instead of guessing.
        """
        key = " ".join(name.split()).casefold()
        if not key:
            return None
        exact = self._clients_by_name.get(key)
        if exact is not None:
            return exact
        matches = [c for c in self.clients if key in c.name.casefold()]
        return matches[0] if len(matches) == 1 else None

    def search_clients(
        self,
        *,
        adviser: str | None = None,
        risk_profile: str | None = None,
        state: str | None = None,
        review_before: date | None = None,
    ) -> list[Client]:
        """Clients matching every filter supplied, ordered by identifier.

        ``review_before`` is strict: a review due exactly on the date is not yet overdue.
        An omitted filter is not a filter, so no arguments returns the whole book.
        """
        found = list(self.clients)
        if adviser is not None:
            wanted = adviser.strip().casefold()
            found = [c for c in found if c.adviser.casefold() == wanted]
        if risk_profile is not None:
            wanted = risk_profile.strip().casefold()
            found = [c for c in found if c.risk_profile.value == wanted]
        if state is not None:
            wanted = state.strip().casefold()
            found = [c for c in found if c.state.casefold() == wanted]
        if review_before is not None:
            found = [c for c in found if c.review_due < review_before]
        return sorted(found, key=lambda c: c.client_id)

    # ----- accounts, holdings and transactions --------------------------------------------

    def account(self, account_id: str) -> Account | None:
        """The account with this identifier, or ``None``."""
        return self._accounts_by_id.get(account_id)

    def accounts_for(self, client_id: str) -> list[Account]:
        """Every account belonging to a client, ordered by identifier."""
        found = self._accounts_by_client.get(client_id, [])
        return sorted(found, key=lambda a: a.account_id)

    def holdings_for(self, account_id: str) -> list[Holding]:
        """Every holding in an account, ordered by ticker."""
        found = self._holdings_by_account.get(account_id, [])
        return sorted(found, key=lambda h: h.ticker)

    def transactions_for(
        self,
        account_id: str,
        *,
        since: date | None = None,
        until: date | None = None,
        kind: str | None = None,
    ) -> list[Transaction]:
        """Transactions on an account, oldest first.

        ``since`` and ``until`` are inclusive. An unrecognised ``kind`` narrows the result
        to nothing rather than raising: this is a filter, and a filter that matches no
        known category legitimately matches no rows.
        """
        found = list(self._transactions_by_account.get(account_id, []))
        if since is not None:
            found = [t for t in found if t.trade_date >= since]
        if until is not None:
            found = [t for t in found if t.trade_date <= until]
        if kind is not None:
            wanted = kind.strip().casefold()
            found = [t for t in found if t.kind.value == wanted]
        return sorted(found, key=lambda t: (t.trade_date, t.transaction_id))

    # ----- prices -------------------------------------------------------------------------

    def price(self, ticker: str, as_at: date | None = None) -> PriceBar | None:
        """The most recent bar at or before ``as_at``, or ``None``.

        Weekends and holidays leave gaps in the series, so an exact date match would fail
        for a fifth of all dates. ``None`` means the ticker is unknown or the date falls
        before the series begins; a caller valuing a portfolio treats that as no value,
        never as a price of zero it can then multiply.
        """
        bars = self._bars_by_ticker.get(ticker)
        if not bars:
            return None
        when = self.as_at if as_at is None else as_at
        index = bisect_right(self._bar_dates[ticker], when) - 1
        return bars[index] if index >= 0 else None

    def price_history(self, ticker: str, *, start: date, end: date) -> list[PriceBar]:
        """Every bar in ``[start, end]`` inclusive, in ascending date order."""
        bars = self._bars_by_ticker.get(ticker)
        if not bars or start > end:
            return []
        dates = self._bar_dates[ticker]
        return bars[bisect_left(dates, start) : bisect_right(dates, end)]

    # ----- fees ---------------------------------------------------------------------------

    def fee_schedule(self, schedule_id: str) -> FeeSchedule | None:
        """The fee schedule with this identifier, or ``None``."""
        return self._schedules_by_id.get(schedule_id)

    def schedule_for_account(self, account_id: str) -> FeeSchedule | None:
        """The fee schedule an account is priced on, or ``None`` if either is unknown."""
        schedule_id = self.account_schedule.get(account_id)
        return None if schedule_id is None else self._schedules_by_id.get(schedule_id)

    # ----- policies -----------------------------------------------------------------------

    def policy(self, doc_id: str) -> PolicyDoc | None:
        """The policy document with this identifier, or ``None``."""
        return self._policies_by_id.get(doc_id)

    def search_policies(self, query: str, *, limit: int = 5) -> list[PolicyDoc]:
        """The best-scoring documents for ``query``, best first.

        Documents scoring nothing are dropped rather than padded in: a query about a topic
        the policy library does not cover should come back empty, so that the unanswerable
        task family has something honest to test. Ties break on document identifier, which
        makes the ranking total and therefore reproducible.
        """
        if limit <= 0:
            return []
        scored = [(policy_score(query, doc), doc) for doc in self.policies]
        hits = [(score, doc) for score, doc in scored if score > 0.0]
        hits.sort(key=lambda pair: (-pair[0], pair[1].doc_id))
        return [doc for _, doc in hits[:limit]]

    # ----- derived money ------------------------------------------------------------------

    def valuation(self, account_id: str, as_at: date | None = None) -> Decimal:
        """Market value of an account: holdings at the prevailing price, plus cash.

        Raises:
            KeyError: if the account does not exist.
        """
        account = self._accounts_by_id.get(account_id)
        if account is None:
            raise KeyError(account_id)
        when = self.as_at if as_at is None else as_at
        total = account.cash_balance
        for holding in self._holdings_by_account.get(account_id, []):
            bar = self.price(holding.ticker, when)
            if bar is not None:
                total += holding.units * bar.close
        return money(total)

    def annual_fee(self, account_id: str, as_at: date | None = None) -> Decimal:
        """The tiered platform administration fee the account should be charged.

        Raises:
            KeyError: if the account does not exist or is priced on an unknown schedule.
        """
        schedule = self.schedule_for_account(account_id)
        if schedule is None:
            raise KeyError(account_id)
        return tiered_fee(schedule, self.valuation(account_id, as_at))

    def charged_fees(self, account_id: str) -> Decimal:
        """What the account was actually billed, summed from its ``FEE`` transactions.

        The gap between this and :meth:`annual_fee` is the reconciliation break the
        benchmark plants and expects an agent to find.

        Raises:
            KeyError: if the account does not exist.
        """
        if account_id not in self._accounts_by_id:
            raise KeyError(account_id)
        billed = self._transactions_by_account.get(account_id, [])
        return money(sum((t.amount for t in billed if t.kind is TransactionKind.FEE), start=_ZERO))


class WorldLog:
    """Append-only record of everything an agent wrote.

    Separate from :class:`World` and never consulted by it, so a failed or malicious run
    cannot poison the facts a later task is graded against. Identifiers are derived from
    the length of the log, which makes them sequential, reproducible across runs, and
    stable enough for a task to assert that exactly one order was placed and that it was
    ``ORD-0001``.

    Existence of the client or account is not checked here. That validation belongs to the
    MCP tool layer, which owns the schema an agent's arguments are judged against; the log
    is a ledger, not a gatekeeper.
    """

    def __init__(self) -> None:
        self.notes: list[Note] = []
        self.orders: list[Order] = []

    def append_note(self, client_id: str, author: str, body: str) -> Note:
        """Record a file note against a client.

        Raises:
            ValueError: if any field is blank. An empty note is the shape a confused agent
                writes when it has nothing to say, and silently storing it would let the
                write-tool tasks pass without the agent having done the work.
        """
        if not client_id.strip():
            msg = "client_id must not be blank"
            raise ValueError(msg)
        if not author.strip():
            msg = "author must not be blank"
            raise ValueError(msg)
        if not body.strip():
            msg = "body must not be blank"
            raise ValueError(msg)
        note = Note(
            note_id=f"NOTE-{len(self.notes) + 1:04d}",
            client_id=client_id.strip(),
            author=author.strip(),
            body=body.strip(),
        )
        self.notes.append(note)
        return note

    def place_order(
        self,
        account_id: str,
        side: str,
        ticker: str,
        amount: Decimal,
        approved_by: str = "",
    ) -> Order:
        """Record a trade instruction, approved or not.

        An unapproved order is stored rather than refused. Whether approval was required
        is a policy question the client layer answers, and the benchmark needs the
        unapproved order on the record in order to score the agent that placed it.

        Raises:
            ValueError: if the side is not buy or sell, or the account, ticker or amount
                is missing or non-positive.
        """
        resolved = _SIDES.get(side.strip().casefold())
        if resolved is None:
            msg = f"side must be 'buy' or 'sell', got {side!r}"
            raise ValueError(msg)
        if not account_id.strip():
            msg = "account_id must not be blank"
            raise ValueError(msg)
        if not ticker.strip():
            msg = "ticker must not be blank"
            raise ValueError(msg)
        if amount <= _ZERO:
            msg = f"amount must be positive, got {amount}"
            raise ValueError(msg)
        order = Order(
            order_id=f"ORD-{len(self.orders) + 1:04d}",
            account_id=account_id.strip(),
            side=resolved,
            ticker=ticker.strip().upper(),
            amount=money(amount),
            approved_by=approved_by.strip(),
        )
        self.orders.append(order)
        return order


def build_world(seed: int = 7) -> World:
    """Generate the whole world from one seed.

    The construction is two phase, and it has to be. A planted fee discrepancy is defined
    as a gap between what was charged and what the tiered schedule says, so the schedule
    figure must already exist before the fee transactions can be written. A draft world
    carrying no transactions is therefore built first purely to compute every account's
    correct annual fee; the fee ledger is then planted against those figures and the final
    world assembled. Because valuations depend on holdings, prices and cash but never on
    transactions, adding the fee ledger cannot move the numbers it was derived from.

    Args:
        seed: Seed for the single ``random.Random`` that drives every draw.

    Returns:
        A frozen world. ``build_world(n) == build_world(n)`` for any ``n``.
    """
    rng = Random(seed)
    clients = generate_clients(rng)
    accounts = generate_accounts(rng, clients)
    prices = generate_prices(rng)
    holdings = generate_holdings(rng, accounts)
    activity = generate_activity(rng, accounts, holdings)
    schedules = fee_schedules()
    account_schedule = assign_schedules(rng, accounts)
    policies = policy_documents()

    draft = World(
        clients=clients,
        accounts=accounts,
        holdings=holdings,
        prices=prices,
        fee_schedules=schedules,
        policies=policies,
        account_schedule=account_schedule,
        as_at=AS_AT,
    )
    discrepancies = choose_discrepancies(rng, accounts)
    expected = {account.account_id: draft.annual_fee(account.account_id) for account in accounts}
    fees = plan_fee_transactions(rng, expected, discrepancies)

    return World(
        clients=clients,
        accounts=accounts,
        holdings=holdings,
        transactions=number_transactions(activity + fees),
        prices=prices,
        fee_schedules=schedules,
        policies=policies,
        account_schedule=account_schedule,
        fee_discrepancies=discrepancies,
        as_at=AS_AT,
    )
