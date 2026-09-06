"""Append-only JSON-lines audit trail. Every string is passed through the PII redactor
before it is written, so the log itself cannot become a leak (a real incident pattern)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentguard.guardrails import pii

# Fields that identify the *staff member* accountable for a decision. Keeping them intact is
# an audit requirement; customer PII never legitimately appears in them.
PRESERVE_KEYS = frozenset({"approver", "approved_by", "requested_by"})


def _redact_any(value: Any) -> Any:
    if isinstance(value, str):
        return pii.redact(value)[0]
    if isinstance(value, dict):
        return {
            k: (v if k in PRESERVE_KEYS and isinstance(v, str) else _redact_any(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_any(v) for v in value]
    return value


class AuditLog:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._memory: list[dict[str, Any]] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, thread_id: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": kind,
            "thread_id": thread_id,
            **_redact_any(fields),
        }
        if self._path is None:
            self._memory.append(entry)
        else:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def entries(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        if self._path is None:
            rows = list(self._memory)
        elif not self._path.exists():
            rows = []
        else:
            with self._path.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        if thread_id is not None:
            rows = [r for r in rows if r.get("thread_id") == thread_id]
        return rows

    def replay(self, thread_id: str) -> str:
        """Human-readable timeline of one conversation."""
        lines = [f"# thread {thread_id}"]
        for e in self.entries(thread_id):
            kind = e.get("kind")
            if kind == "turn":
                lines.append(
                    f"[{e['ts']}] turn {e.get('turn')} status={e.get('status')} "
                    f"steps={e.get('steps')} latency={float(e.get('latency_s', 0)):.2f}s"
                )
                lines.append(f"  user: {e.get('user_input')}")
                for r in e.get("tool_records", []):
                    flag = "" if r.get("approved") is None else f" approved={r.get('approved')}"
                    lines.append(
                        f"  tool {r.get('tool')}({json.dumps(r.get('args'))}) "
                        f"ok={r.get('ok')}{flag} -> {str(r.get('output'))[:120]!r}"
                    )
                for g in e.get("guardrail_events", []):
                    lines.append(
                        f"  rail {g.get('rail')}@{g.get('stage')} -> {g.get('action')} "
                        f"(score={g.get('score')}) {g.get('detail', '')}"
                    )
                lines.append(f"  answer: {e.get('answer')}")
            elif kind == "approval":
                lines.append(
                    f"[{e['ts']}] approval decision approved={e.get('approved')} "
                    f"by {e.get('approver')} note={e.get('note')!r}"
                )
            else:
                rest = {k: v for k, v in e.items() if k not in ("ts", "kind", "thread_id")}
                lines.append(f"[{e['ts']}] {kind}: {json.dumps(rest)}")
        return "\n".join(lines) + "\n"
