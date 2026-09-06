from __future__ import annotations

from pathlib import Path

from agentguard.audit import AuditLog


def test_audit_redacts_and_filters_and_replays(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit" / "log.jsonl")
    log.record(
        "turn",
        "t1",
        turn=1,
        user_input="TFN 123 456 782 please",
        status="ok",
        steps=1,
        latency_s=0.5,
        answer="done",
        tool_records=[
            {
                "tool": "search_policy",
                "args": {"q": "ava@example.com"},
                "ok": True,
                "output": "x",
                "approved": None,
            }
        ],
        guardrail_events=[
            {"rail": "pii", "stage": "input", "action": "redact", "score": 1.0, "detail": "TFN"}
        ],
    )
    log.record("approval", "t1", approved=True, approver="alice@bank.example", note="fine")
    log.record(
        "turn",
        "t2",
        turn=1,
        user_input="other",
        status="blocked",
        steps=0,
        latency_s=0.1,
        answer="no",
    )
    log.record("custom", "t1", foo=[1, "0412 345 678"])
    entries = log.entries("t1")
    assert [e["kind"] for e in entries] == ["turn", "approval", "custom"]
    assert entries[0]["user_input"] == "TFN [TFN] please"
    assert entries[0]["tool_records"][0]["args"]["q"] == "[EMAIL]"
    assert entries[2]["foo"] == [1, "[PHONE]"]
    assert len(log.entries()) == 4
    assert entries[1]["approver"] == "alice@bank.example"  # accountable staff identity kept
    text = log.replay("t1")
    assert (
        "turn 1 status=ok" in text
        and "by alice@bank.example" in text
        and "rail pii@input -> redact" in text
    )
    assert "custom:" in text
    assert AuditLog(tmp_path / "missing.jsonl").entries() == []


def test_in_memory_audit() -> None:
    log = AuditLog()
    log.record("turn", "t", turn=1, user_input="x", status="ok", steps=0, latency_s=0, answer="y")
    assert len(log.entries("t")) == 1 and log.entries("other") == []
    assert log.replay("t").startswith("# thread t")
