# Scripted harness regression fixtures

These two files record all 72 deterministic outcomes for each architecture. They are a
contract for the harness, not a model leaderboard: the scripted agents deliberately include
failures, refused calls and incomplete trajectories. A newly green result caused by a
weakened grader is a change that needs review too.

`make gate` runs both architectures, checks task and policy identity and each task's success,
answer score, call F1, approval, forbidden calls, steps and failure classes, then renders the
architecture comparison. A mismatch fails with exit 3; missing or malformed evidence fails
with exit 1. Wall-clock timings are excluded because machines differ.

The final research comparison may return **HOLD** and exit 0. Use `bench compare --gate`
when explicitly testing a deployment promotion policy, where non-promotion must block a
release. That is a separate decision from whether the harness still grades its fixtures
correctly.

The initial fixtures were reproduced locally with the shipped scripted model, default seed
and step budget. They agree with the committed scripted experiment grades. Change them only
alongside a reviewed change to tasks, policy, orchestration or grading, explaining which
outcomes changed; CI never rewrites its own expectations.
