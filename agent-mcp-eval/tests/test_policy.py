"""Tests for the client-side permission policy.

The precedence of the four checks is the part of this module most likely to rot --- it is
invisible in any test that only exercises one violation at a time --- so every ordered pair
of checks is tested against an input that trips both.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mcpeval.client.policy import (
    ANALYST,
    DEFAULT_READ_TOOLS,
    DEFAULT_WRITE_TOOLS,
    RESEARCHER,
    RULE_ALLOW,
    RULE_APPROVAL_REQUIRED,
    RULE_BUDGET_CALLS,
    RULE_BUDGET_STEPS,
    RULE_BUDGET_TOKENS,
    RULE_BUDGET_WALL,
    RULE_NOT_ALLOWED,
    RULE_UNKNOWN_ROLE,
    RULE_UNKNOWN_TOOL,
    RULE_WRITE_FORBIDDEN,
    SUPERVISOR,
    WRITER,
    BudgetState,
    PermissionPolicy,
    RolePolicy,
    default_policy,
)
from mcpeval.client.session import connect_in_process
from mcpeval.mcp_server.server import build_server
from mcpeval.schemas import PolicyVerdict, ToolSpec
from mcpeval.world.store import WorldLog, build_world

# Shared because PermissionPolicy is frozen; hypothesis tests cannot take a function-scoped
# fixture, and a module-level constant keeps them and the example tests on one policy.
POLICY = default_policy()

READ_ONLY_ROLES = (RESEARCHER, ANALYST, WRITER)


def fresh_budget(**overrides: Any) -> BudgetState:
    """A budget with generous ceilings, so budget never fires unless a test asks it to."""
    defaults: dict[str, Any] = {
        "max_steps": 10,
        "max_calls": 20,
        "max_tokens": 10_000,
        "max_wall_ms": 10_000.0,
    }
    return BudgetState(**{**defaults, **overrides})


def exhausted_budget() -> BudgetState:
    """A budget with every ceiling reached."""
    return BudgetState(
        max_steps=1,
        max_calls=1,
        max_tokens=1,
        max_wall_ms=1.0,
        steps=1,
        calls=1,
        tokens=1,
        elapsed_ms=1.0,
    )


# --------------------------------------------------------------------------------------
# Scope patterns
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "tool", "expected"),
    [
        ("client_lookup", "client_lookup", True),
        ("client_lookup", "client_lookups", False),
        ("client_lookup", "client_looku", False),
        # The wildcard cases use an abstract prefix on purpose: they test the matcher, not
        # the inventory, and pinning them to real tool names would make a rename of a tool
        # look like a bug in the matcher.
        ("client_*", "client_lookup", True),
        ("client_*", "client_", True),
        ("client_*", "account_holdings", False),
        ("*", "anything_at_all", True),
        ("*", "", True),
    ],
)
def test_permits_matches_literals_and_trailing_wildcards(
    pattern: str, tool: str, expected: bool
) -> None:
    assert RolePolicy(name="r", allow=(pattern,)).permits(tool) is expected


def test_permits_is_a_union_over_patterns() -> None:
    role = RolePolicy(name="r", allow=("policy_*", "price_history"))
    assert role.permits("policy_fetch")
    assert role.permits("price_history")
    assert not role.permits("client_search")


def test_empty_allowlist_permits_nothing() -> None:
    assert not RolePolicy(name="r").permits("client_lookup")


@pytest.mark.parametrize("pattern", ["get_*_price", "*_price", "a*b"])
def test_interior_wildcard_is_rejected_at_construction(pattern: str) -> None:
    with pytest.raises(ValidationError, match="only a trailing"):
        RolePolicy(name="r", allow=(pattern,))


def test_role_policy_is_frozen() -> None:
    role = RolePolicy(name="r", allow=("get_*",))
    # Through setattr because a direct assignment is a static type error, and the point of
    # the test is that it is also a runtime error for code that has no type checker.
    field = "may_write"
    with pytest.raises(ValidationError):
        setattr(role, field, True)


# --------------------------------------------------------------------------------------
# BudgetState
# --------------------------------------------------------------------------------------


def test_fresh_budget_has_headroom() -> None:
    budget = fresh_budget()
    assert not budget.exhausted
    assert budget.exhaustion_rule is None


def test_spending_accumulates() -> None:
    budget = fresh_budget()
    budget.spend_step()
    budget.spend_step(2)
    budget.spend_call()
    budget.spend_tokens(500)
    budget.spend_wall_ms(12.5)
    assert (budget.steps, budget.calls, budget.tokens, budget.elapsed_ms) == (3, 1, 500, 12.5)


@pytest.mark.parametrize(
    ("method", "amount"),
    [("spend_step", -1), ("spend_call", -1), ("spend_tokens", -5), ("spend_wall_ms", -0.5)],
)
def test_negative_spend_is_refused(method: str, amount: float) -> None:
    budget = fresh_budget()
    with pytest.raises(ValueError, match="negative"):
        getattr(budget, method)(amount)


def test_exhaustion_is_at_the_ceiling_not_past_it() -> None:
    # The tally is incremented after admission, so a run sitting exactly on its ceiling has
    # nothing left. A ">" test here would let every run make one call too many.
    budget = BudgetState(max_calls=2, max_steps=99, max_tokens=99, max_wall_ms=99.0)
    budget.spend_call()
    assert not budget.calls_exhausted
    budget.spend_call()
    assert budget.calls_exhausted


def test_zero_ceiling_is_exhausted_from_the_start() -> None:
    assert BudgetState(max_steps=0).steps_exhausted


@pytest.mark.parametrize(
    ("field", "rule"),
    [
        ("steps", RULE_BUDGET_STEPS),
        ("calls", RULE_BUDGET_CALLS),
        ("tokens", RULE_BUDGET_TOKENS),
        ("elapsed_ms", RULE_BUDGET_WALL),
    ],
)
def test_each_ceiling_reports_its_own_rule(field: str, rule: str) -> None:
    budget = BudgetState(max_steps=5, max_calls=5, max_tokens=5, max_wall_ms=5.0)
    setattr(budget, field, 5)
    assert budget.exhausted
    assert budget.exhaustion_rule == rule


def test_exhaustion_rule_order_is_fixed() -> None:
    # Two runs that exhaust the same pair of budgets must be reported identically.
    budget = exhausted_budget()
    assert budget.exhaustion_rule == RULE_BUDGET_STEPS
    budget.max_steps = 99
    assert budget.exhaustion_rule == RULE_BUDGET_CALLS
    budget.max_calls = 99
    assert budget.exhaustion_rule == RULE_BUDGET_TOKENS
    budget.max_tokens = 99
    assert budget.exhaustion_rule == RULE_BUDGET_WALL
    budget.max_wall_ms = 99.0
    assert budget.exhaustion_rule is None


# --------------------------------------------------------------------------------------
# PermissionPolicy construction
# --------------------------------------------------------------------------------------


def test_write_tool_missing_from_inventory_is_rejected() -> None:
    # It would fail open: the scope check would treat it as read-only for every role.
    with pytest.raises(ValidationError, match="missing from known_tools"):
        PermissionPolicy(known_tools=frozenset({"client_lookup"}), write_tools=frozenset({"wipe"}))


def test_approval_may_name_a_tool_that_does_not_exist_yet() -> None:
    # The opposite mismatch fails closed, so it is allowed.
    policy = PermissionPolicy(
        known_tools=frozenset({"client_lookup"}), approval_required=frozenset({"not_shipped"})
    )
    assert policy.requires_approval("not_shipped")


def test_new_budget_takes_the_tighter_of_global_and_role() -> None:
    policy = PermissionPolicy(
        roles={"r": RolePolicy(name="r", max_steps=3, max_tokens=99_000)},
        max_steps=12,
        max_tokens=40_000,
        max_tool_calls=7,
        max_wall_ms=555.0,
    )
    budget = policy.new_budget("r")
    assert budget.max_steps == 3
    assert budget.max_tokens == 40_000
    assert budget.max_calls == 7
    assert budget.max_wall_ms == 555.0
    assert budget.steps == budget.calls == budget.tokens == 0


def test_new_budget_for_an_unknown_role_is_a_wiring_error() -> None:
    with pytest.raises(KeyError, match="unknown role"):
        POLICY.new_budget("auditor")


def test_with_tools_replaces_the_inventory_from_the_advertisement() -> None:
    specs = [
        ToolSpec(name="ping", description="", read_only=True),
        ToolSpec(name="nuke", description="", read_only=False, destructive=True),
    ]
    refreshed = POLICY.with_tools(specs)
    assert refreshed.known_tools == frozenset({"ping", "nuke"})
    assert refreshed.write_tools == frozenset({"nuke"})
    assert not refreshed.is_write("ping")


def test_with_tools_unions_approval_rather_than_replacing_it() -> None:
    specs = [
        ToolSpec(name="note_append", description="", read_only=False),
        ToolSpec(name="order_place", description="", read_only=False),
        ToolSpec(name="client_lookup", description="", read_only=True, requires_approval=True),
    ]
    refreshed = POLICY.with_tools(specs)
    assert refreshed.requires_approval("client_lookup")
    assert refreshed.requires_approval("order_place")


def test_with_tools_keeps_roles_and_ceilings() -> None:
    refreshed = POLICY.with_tools([ToolSpec(name="ping", description="")])
    assert refreshed.roles == POLICY.roles
    assert refreshed.max_steps == POLICY.max_steps
    assert refreshed.max_wall_ms == POLICY.max_wall_ms


def test_default_policy_can_be_seeded_with_a_live_advertisement() -> None:
    policy = default_policy(tools=[ToolSpec(name="ping", description="")])
    assert policy.known_tools == frozenset({"ping"})
    assert policy.roles.keys() == POLICY.roles.keys()


# --------------------------------------------------------------------------------------
# decide: one violation at a time
# --------------------------------------------------------------------------------------


def test_in_scope_read_is_allowed() -> None:
    decision = POLICY.decide(
        RESEARCHER, "client_lookup", {"client_id": "C1"}, budget=fresh_budget(), approved=False
    )
    assert decision.verdict is PolicyVerdict.ALLOW
    assert decision.rule == RULE_ALLOW
    assert decision.allowed


def test_unknown_tool_is_refused_as_unknown() -> None:
    decision = POLICY.decide(
        SUPERVISOR, "get_bank_password", {}, budget=fresh_budget(), approved=True
    )
    assert decision.verdict is PolicyVerdict.REFUSE_UNKNOWN_TOOL
    assert decision.rule == RULE_UNKNOWN_TOOL
    assert not decision.allowed


def test_unknown_role_has_no_scope() -> None:
    decision = POLICY.decide("auditor", "client_lookup", {}, budget=fresh_budget(), approved=False)
    assert decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert decision.rule == RULE_UNKNOWN_ROLE


@pytest.mark.parametrize("role", READ_ONLY_ROLES)
@pytest.mark.parametrize("tool", DEFAULT_WRITE_TOOLS)
def test_read_only_roles_may_never_write(role: str, tool: str) -> None:
    decision = POLICY.decide(role, tool, {}, budget=fresh_budget(), approved=True)
    assert decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert decision.rule == RULE_WRITE_FORBIDDEN


def test_tool_outside_the_role_allowlist_is_refused() -> None:
    decision = POLICY.decide(WRITER, "transactions_list", {}, budget=fresh_budget(), approved=False)
    assert decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert decision.rule == RULE_NOT_ALLOWED


def test_being_the_wrong_kind_of_role_beats_being_off_the_allowlist() -> None:
    # The writer fails both tests on place_order; "you are read-only" is the fact an
    # auditor needs, so it is the one reported.
    decision = POLICY.decide(WRITER, "order_place", {}, budget=fresh_budget(), approved=True)
    assert decision.rule == RULE_WRITE_FORBIDDEN


def test_supervisor_write_without_approval_is_refused() -> None:
    decision = POLICY.decide(
        SUPERVISOR, "order_place", {"amount": "5000"}, budget=fresh_budget(), approved=False
    )
    assert decision.verdict is PolicyVerdict.REFUSE_NO_APPROVAL
    assert decision.rule == RULE_APPROVAL_REQUIRED


def test_supervisor_write_with_approval_is_allowed() -> None:
    decision = POLICY.decide(
        SUPERVISOR, "order_place", {"amount": "5000"}, budget=fresh_budget(), approved=True
    )
    assert decision.allowed


@pytest.mark.parametrize(
    ("overrides", "rule"),
    [
        ({"max_steps": 0}, RULE_BUDGET_STEPS),
        ({"max_calls": 0}, RULE_BUDGET_CALLS),
        ({"max_tokens": 0}, RULE_BUDGET_TOKENS),
        ({"max_wall_ms": 0.0}, RULE_BUDGET_WALL),
    ],
)
def test_each_exhausted_budget_refuses_with_its_own_rule(
    overrides: dict[str, Any], rule: str
) -> None:
    decision = POLICY.decide(
        RESEARCHER, "client_lookup", {}, budget=fresh_budget(**overrides), approved=False
    )
    assert decision.verdict is PolicyVerdict.REFUSE_BUDGET
    assert decision.rule == rule


# --------------------------------------------------------------------------------------
# decide: precedence, one test per ordered pair of checks
# --------------------------------------------------------------------------------------


def test_unknown_tool_beats_out_of_scope() -> None:
    decision = POLICY.decide(WRITER, "made_up_tool", {}, budget=fresh_budget(), approved=False)
    assert decision.rule == RULE_UNKNOWN_TOOL


def test_unknown_tool_beats_approval() -> None:
    policy = POLICY.model_copy(update={"approval_required": frozenset({"made_up_tool"})})
    decision = policy.decide(SUPERVISOR, "made_up_tool", {}, budget=fresh_budget(), approved=False)
    assert decision.rule == RULE_UNKNOWN_TOOL


def test_unknown_tool_beats_budget() -> None:
    decision = POLICY.decide(
        SUPERVISOR, "made_up_tool", {}, budget=exhausted_budget(), approved=True
    )
    assert decision.rule == RULE_UNKNOWN_TOOL


def test_out_of_scope_beats_approval() -> None:
    # A researcher can never place an order, so telling it to go and fetch an approval
    # would be a lie: the approval would not have helped.
    decision = POLICY.decide(RESEARCHER, "order_place", {}, budget=fresh_budget(), approved=False)
    assert decision.rule == RULE_WRITE_FORBIDDEN


def test_out_of_scope_beats_budget() -> None:
    # Otherwise an unauthorised attempt made late in a run is filed as a budget refusal,
    # and raising the budget would appear to "fix" a security violation.
    decision = POLICY.decide(
        RESEARCHER, "order_place", {}, budget=exhausted_budget(), approved=True
    )
    assert decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE


def test_approval_beats_budget() -> None:
    decision = POLICY.decide(
        SUPERVISOR, "order_place", {}, budget=exhausted_budget(), approved=False
    )
    assert decision.verdict is PolicyVerdict.REFUSE_NO_APPROVAL


# --------------------------------------------------------------------------------------
# decide: side effects and reason strings
# --------------------------------------------------------------------------------------


def test_decide_never_spends_the_budget() -> None:
    budget = fresh_budget()
    before = budget.model_dump()
    POLICY.decide(SUPERVISOR, "client_lookup", {"client_id": "C1"}, budget=budget, approved=True)
    POLICY.decide(RESEARCHER, "order_place", {}, budget=budget, approved=False)
    assert budget.model_dump() == before


def test_refusal_reason_names_argument_keys_but_never_their_values() -> None:
    decision = POLICY.decide(
        RESEARCHER,
        "order_place",
        {"account_id": "ACC-SECRET-001", "amount": "125000.00"},
        budget=fresh_budget(),
        approved=False,
    )
    assert "account_id" in decision.reason
    assert "amount" in decision.reason
    assert "ACC-SECRET-001" not in decision.reason
    assert "125000.00" not in decision.reason


def test_refusal_reason_says_none_when_there_are_no_arguments() -> None:
    decision = POLICY.decide(RESEARCHER, "order_place", {}, budget=fresh_budget(), approved=False)
    assert "arguments: none" in decision.reason


# --------------------------------------------------------------------------------------
# The shipped default policy
# --------------------------------------------------------------------------------------


def test_default_policy_has_the_four_benchmark_roles() -> None:
    assert set(POLICY.roles) == {SUPERVISOR, RESEARCHER, ANALYST, WRITER}


def test_only_the_supervisor_may_write() -> None:
    assert POLICY.roles[SUPERVISOR].may_write
    assert not any(POLICY.roles[r].may_write for r in READ_ONLY_ROLES)


@pytest.mark.parametrize("tool", DEFAULT_WRITE_TOOLS)
def test_every_write_tool_needs_an_approval(tool: str) -> None:
    assert POLICY.requires_approval(tool)
    assert POLICY.is_write(tool)


@pytest.mark.parametrize("tool", DEFAULT_READ_TOOLS)
def test_no_read_tool_needs_an_approval(tool: str) -> None:
    assert not POLICY.requires_approval(tool)
    assert not POLICY.is_write(tool)


@pytest.mark.parametrize(
    ("role", "tool", "allowed"),
    [
        (RESEARCHER, "client_search", True),
        (RESEARCHER, "transactions_list", True),
        (RESEARCHER, "portfolio_valuation", False),
        (ANALYST, "fee_reconcile", True),
        (ANALYST, "price_history", True),
        (ANALYST, "client_search", False),
        (WRITER, "client_lookup", True),
        (WRITER, "policy_search", True),
        (WRITER, "price_history", False),
        (SUPERVISOR, "portfolio_valuation", True),
        (SUPERVISOR, "policy_search", True),
    ],
)
def test_default_role_scopes_follow_the_division_of_labour(
    role: str, tool: str, allowed: bool
) -> None:
    decision = POLICY.decide(role, tool, {}, budget=fresh_budget(), approved=False)
    assert decision.allowed is allowed


@pytest.mark.parametrize("role", [SUPERVISOR, *READ_ONLY_ROLES])
def test_every_default_role_gets_a_usable_budget(role: str) -> None:
    budget = POLICY.new_budget(role)
    assert not budget.exhausted
    assert budget.max_steps <= POLICY.max_steps
    assert budget.max_tokens <= POLICY.max_tokens


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------

_ROLES = st.sampled_from([SUPERVISOR, RESEARCHER, ANALYST, WRITER, "auditor", ""])
_TOOLS = st.sampled_from([*DEFAULT_READ_TOOLS, *DEFAULT_WRITE_TOOLS, "made_up", ""])
_ARGS = st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=3)


@given(role=_ROLES, tool=_TOOLS, arguments=_ARGS, approved=st.booleans())
def test_a_decision_is_allowed_exactly_when_the_allow_rule_fired(
    role: str, tool: str, arguments: dict[str, str], approved: bool
) -> None:
    decision = POLICY.decide(role, tool, arguments, budget=fresh_budget(), approved=approved)
    assert decision.allowed == (decision.rule == RULE_ALLOW)
    assert decision.allowed == (decision.verdict is PolicyVerdict.ALLOW)


@given(role=_ROLES, tool=_TOOLS, arguments=_ARGS, approved=st.booleans())
def test_decide_is_total_and_leaves_the_budget_alone(
    role: str, tool: str, arguments: dict[str, str], approved: bool
) -> None:
    # An enforcement point that raises turns a routine denial into a crashed run.
    budget = fresh_budget()
    before = budget.model_dump()
    decision = POLICY.decide(role, tool, arguments, budget=budget, approved=approved)
    assert decision.reason
    assert budget.model_dump() == before


@given(
    role=st.sampled_from(READ_ONLY_ROLES),
    tool=st.sampled_from(DEFAULT_WRITE_TOOLS),
    arguments=_ARGS,
    approved=st.booleans(),
)
def test_no_read_only_role_can_reach_a_write_tool_by_any_route(
    role: str, tool: str, arguments: dict[str, str], approved: bool
) -> None:
    decision = POLICY.decide(role, tool, arguments, budget=fresh_budget(), approved=approved)
    assert decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE


@given(tool=st.sampled_from(DEFAULT_WRITE_TOOLS), arguments=_ARGS)
def test_even_the_supervisor_cannot_write_without_an_approval(
    tool: str, arguments: dict[str, str]
) -> None:
    decision = POLICY.decide(SUPERVISOR, tool, arguments, budget=fresh_budget(), approved=False)
    assert decision.verdict is PolicyVerdict.REFUSE_NO_APPROVAL


async def test_the_default_inventory_is_the_servers() -> None:
    """The names in this module must be the names the server publishes.

    They were not, once, and nothing failed: an earlier draft of the inventory
    (`get_client`, `place_order`) survived here after the server settled on `client_lookup`
    and `order_place`. Every role pattern then matched nothing, so `default_policy()` refused
    every specialist call with `scope.tool_not_allowed` while the supervisor's `"*"` carried
    on working. A multi-agent run under that policy scores near zero for a reason that has
    nothing to do with multi-agent systems, and no test could see it because every test used
    the same wrong names.

    So this one asks the server. It is the only test in the file that starts one.
    """
    server = build_server(build_world(), WorldLog())
    async with connect_in_process(server) as client:
        published = {spec.name for spec in await client.discover()}

    assert set(DEFAULT_READ_TOOLS) | set(DEFAULT_WRITE_TOOLS) == published
    assert set(DEFAULT_WRITE_TOOLS) == {spec for spec in published if spec in DEFAULT_WRITE_TOOLS}


async def test_every_default_role_scope_names_a_tool_that_exists() -> None:
    """A pattern that matches nothing is a role that can do nothing, silently."""
    server = build_server(build_world(), WorldLog())
    async with connect_in_process(server) as client:
        published = [spec.name for spec in await client.discover()]

    policy = default_policy()
    for name, role in policy.roles.items():
        reachable = [tool for tool in published if role.permits(tool)]
        assert reachable, f"role {name!r} can reach none of the server's tools"
