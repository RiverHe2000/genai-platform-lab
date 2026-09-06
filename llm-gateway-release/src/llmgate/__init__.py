"""llmgate: an OpenAI-compatible LLM gateway and an evaluation-gated release workflow.

Layers:

* ``protocol`` — the OpenAI wire contract (chat / completions / streaming / errors)
* ``backends`` — vLLM, any OpenAI-compatible server, an in-process Hugging Face model, and a
  deterministic fake, all behind one async protocol
* ``resilience`` / ``router`` — retries, circuit breaker, bulkhead, primary/fallback,
  weighted canary and shadow routing
* ``guardrails`` / ``auth`` / ``ratelimit`` — request and response rails, API keys, token
  buckets and daily budgets
* ``observability`` / ``api`` — Prometheus metrics, JSON logs, request ids, the FastAPI app
* ``loadtest`` / ``evaluation`` — load generator, evaluation suites, paired statistics and
  the promotion decision
"""

__version__ = "0.1.0"
