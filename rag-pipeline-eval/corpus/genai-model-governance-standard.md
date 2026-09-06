# Generative AI Model Governance Standard

Classification. Every generative AI use case is classified before build. Tier A use cases
are customer-facing or influence a decision about a customer and require a human in the
loop for every output. Tier B use cases are internal assistants for staff. Tier C use
cases are sandboxed experiments with synthetic or public data only.

Release evaluation. Before a Tier A or Tier B assistant is released, it must be evaluated
on a held-out set of at least 200 questions that is not used for prompt development. A
grounded assistant must achieve a faithfulness score of at least 0.85, an answer relevancy
score of at least 0.70 and an unsupported-claim (hallucination) rate of no more than 5%.
Results are recorded on the model card together with the evaluation dataset version.

Re-evaluation. The full evaluation is re-run on every change to the underlying model, the
retrieval corpus or the system prompt. A degradation of more than 5 percentage points on
any release metric blocks the change.

Red teaming. Each assistant is red-teamed before release and quarterly thereafter,
covering prompt injection, data exfiltration, harmful content and out-of-scope requests.

Guardrails. Inputs and outputs pass through guardrails that detect personally identifiable
information, prompt-injection patterns and off-topic requests. Guardrail events are logged.

Logging and retention. Prompts, retrieved passages and outputs are logged with the
request identifier and retained for 12 months, with personally identifiable information
redacted before storage.

External models. Vendor-hosted models may only be called through the approved model
gateway. Customer personal information may not be sent to an external model provider
without a completed data protection impact assessment.
