"""agentguard: a LangGraph tool-using agent with layered guardrails for a bank's credit desk.

Layers:

* ``llm`` — chat-model protocol with scripted, OpenAI-compatible and Hugging Face backends
* ``schemas`` — typed agent state and the action contract the model must follow
* ``tools`` — sandboxed calculator, read-only SQL over a loan book, policy search, and a
  high-risk action that needs human approval
* ``guardrails`` — PII, prompt-injection, topic and output rails, each returning an event
* ``graph`` / ``agent`` — the LangGraph state machine, checkpointing, interrupts, resume
* ``audit`` — JSON-lines audit trail with redaction and replay
* ``evaluation`` — scenario and red-team harness with a CI gate
"""

__version__ = "0.1.0"
