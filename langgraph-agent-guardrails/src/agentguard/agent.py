"""Facade over the compiled graph: run a turn, resume after a human decision, inspect state,
and write the audit trail. Also the factory that assembles everything from ``Settings``."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentguard.audit import AuditLog
from agentguard.config import Settings
from agentguard.graph import GraphDeps, build_graph
from agentguard.llm import ChatModel, FakeChatModel, HFChatModel, OpenAICompatibleChatModel
from agentguard.schemas import GuardrailEvent, ToolRecord
from agentguard.tools.base import ToolRegistry
from agentguard.tools.calculator import CALCULATOR
from agentguard.tools.loanbook import LoanBook, make_loanbook_tools
from agentguard.tools.policy_search import PolicyClause, PolicySearch, make_policy_tool
from agentguard.tools.review import make_review_tool


@dataclass(slots=True)
class AgentResult:
    thread_id: str
    turn: int
    status: str
    answer: str | None
    steps: int
    tool_records: list[ToolRecord] = field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    latency_s: float = 0.0
    model: str = ""

    @property
    def tools_called(self) -> list[str]:
        return [r.tool for r in self.tool_records]

    @property
    def rails_fired(self) -> list[str]:
        return [e.rail for e in self.guardrail_events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "turn": self.turn,
            "status": self.status,
            "answer": self.answer,
            "steps": self.steps,
            "tool_records": [r.model_dump() for r in self.tool_records],
            "guardrail_events": [e.model_dump() for e in self.guardrail_events],
            "pending_action": self.pending_action,
            "latency_s": self.latency_s,
            "model": self.model,
        }


class Agent:
    def __init__(self, graph: Any, deps: GraphDeps, audit: AuditLog) -> None:
        self._graph = graph
        self._deps = deps
        self._audit = audit

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str, values: dict[str, Any], latency: float) -> AgentResult:
        turn = int(values.get("turn", 0))
        interrupted = bool(values.get("__interrupt__"))
        status = "awaiting_approval" if interrupted else str(values.get("status", "ok"))
        records = [
            ToolRecord.model_validate({k: v for k, v in r.items() if k != "turn"})
            for r in values.get("tool_records", [])
            if r.get("turn") == turn
        ]
        events = [
            GuardrailEvent.model_validate({k: v for k, v in e.items() if k != "turn"})
            for e in values.get("guardrail_events", [])
            if e.get("turn") == turn
        ]
        return AgentResult(
            thread_id=thread_id,
            turn=turn,
            status=status,
            answer=None if interrupted else values.get("final_answer"),
            steps=int(values.get("steps", 0)),
            tool_records=records,
            guardrail_events=events,
            pending_action=values.get("pending_action") if interrupted else None,
            latency_s=latency,
            model=self._deps.model.name,
        )

    def _log_turn(self, result: AgentResult, user_input: str) -> None:
        self._audit.record(
            "turn",
            result.thread_id,
            turn=result.turn,
            user_input=user_input,
            status=result.status,
            steps=result.steps,
            answer=result.answer,
            pending_action=result.pending_action,
            tool_records=[r.model_dump() for r in result.tool_records],
            guardrail_events=[e.model_dump() for e in result.guardrail_events],
            latency_s=result.latency_s,
            model=result.model,
        )

    def run(self, thread_id: str, user_input: str) -> AgentResult:
        started = time.perf_counter()
        values = self._graph.invoke(
            {"thread_id": thread_id, "user_input": user_input}, config=self._config(thread_id)
        )
        result = self._result(thread_id, values, time.perf_counter() - started)
        self._log_turn(result, user_input)
        return result

    def resume(
        self, thread_id: str, *, approved: bool, approver: str, note: str = ""
    ) -> AgentResult:
        snapshot = self._graph.get_state(self._config(thread_id))
        if not snapshot.next:
            msg = f"thread {thread_id!r} is not waiting for a decision"
            raise RuntimeError(msg)
        self._audit.record("approval", thread_id, approved=approved, approver=approver, note=note)
        started = time.perf_counter()
        values = self._graph.invoke(
            Command(resume={"approved": approved, "approver": approver, "note": note}),
            config=self._config(thread_id),
        )
        result = self._result(thread_id, values, time.perf_counter() - started)
        self._log_turn(result, str(values.get("user_input", "")))
        return result

    def state(self, thread_id: str) -> dict[str, Any]:
        snapshot = self._graph.get_state(self._config(thread_id))
        values: dict[str, Any] = dict(snapshot.values or {})
        values["waiting"] = bool(snapshot.next)
        return values

    def history(self, thread_id: str) -> list[dict[str, str]]:
        return list(self.state(thread_id).get("messages", []))


# ----- factory ------------------------------------------------------------------------------


def build_model(settings: Settings) -> ChatModel:
    cfg = settings.model
    if cfg.kind == "fake":
        return FakeChatModel()
    if cfg.kind == "openai":
        return OpenAICompatibleChatModel(
            cfg.base_url,
            cfg.model,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    return HFChatModel(cfg.model, device=cfg.device, seed=settings.seed)


def build_registry(book: LoanBook, policy_search: PolicySearch) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CALCULATOR)
    for spec in make_loanbook_tools(book):
        registry.register(spec)
    registry.register(make_policy_tool(policy_search))
    registry.register(make_review_tool(book))
    return registry


def build_checkpointer(settings: Settings) -> Any:
    if settings.checkpoint_path is None:
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_agent(
    settings: Settings,
    *,
    model: ChatModel | None = None,
    extra_clauses: Iterable[PolicyClause] = (),
    loanbook: LoanBook | None = None,
) -> Agent:
    if settings.loanbook_path is not None:
        settings.loanbook_path.parent.mkdir(parents=True, exist_ok=True)
    book = loanbook or LoanBook(
        settings.loanbook_path or ":memory:",
        seed=settings.loanbook_seed,
        n_loans=settings.loanbook_size,
        max_rows=settings.sql_max_rows,
    )
    search = PolicySearch()
    for clause in extra_clauses:
        search.add(clause)
    deps = GraphDeps(
        model=model or build_model(settings),
        registry=build_registry(book, search),
        policy=settings.guardrails,
        max_steps=settings.max_steps,
        max_parse_retries=settings.max_parse_retries,
        require_approval=settings.require_approval_for_high_risk,
        max_tokens=settings.model.max_tokens,
        temperature=settings.model.temperature,
    )
    graph = build_graph(deps, checkpointer=build_checkpointer(settings))
    return Agent(graph, deps, AuditLog(settings.audit_path))
