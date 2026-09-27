# v1 candidate resource feasibility stop

The Qwen3-4B run was stopped by the operator because its process used 12.22 GB of dedicated GPU memory plus 13.28 GB of shared GPU memory and system commit headroom became low. This was not a score-based stop and no CUDA out-of-memory exception was observed.

The execution session received Ctrl+C and returned exit 1. On this Windows launcher, Python did not finish its `finally` handler. The original candidate receipt therefore still says `running`; it is preserved unchanged and cannot pass complete-study verification. All completed calls and task trajectories remain available. There is no complete v1 cross-model comparison, and the partial candidate is not scored as a full study.

`FEASIBILITY_STOP.json` records the operator decision, resource counters and exact partial-file checksums. The completed 1.5B study remains separate. v1 source and protocol are frozen at `29d7ee3c350c47491db83d9f46536af326f993b5`; verify its completed evidence using that source. Any revised attention implementation requires a new version and full reruns of both models, with no mixing or selective retries.
