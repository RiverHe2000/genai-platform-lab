"""Guardrails. Each rail is a pure function returning what it found; the graph decides what
to do with it according to ``GuardrailPolicy`` and records a ``GuardrailEvent`` either way."""
