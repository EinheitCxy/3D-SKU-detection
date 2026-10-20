# Independent evidence review — 2026-10-20

Reviewer: separate medium subagent `promptda_evidence_review`, read-only. This file records the coordinator's synthesis of the review; the reviewer did not implement the candidates or run these experiments.

| Claim | Finding | Supported scope |
|---|---|---|
| Both fresh-process delivery arms completed | PASS | Both result files are complete, both launch exit codes are 0. |
| Combined fresh outputs are equal | PASS | `evaluate_outputs.py` and its result compare nine NPZ fields including key/dtype/shape/values; one million GLB positions, colors, geometry associations and node transforms are equal, maximum coordinate difference 0. |
| Selected projection is valid on this input | PASS, input-limited | The optimized arm requires certificate=true and includes its cost in delivery. It does not accept an unknown domain. |
| Raw delivery became faster | CONDITIONAL | This pair observed 23.779→17.680 seconds, about 25.65% lower latency. Each arm n=1, ordered, shared GPU; this is not a population or controlled-service estimate. |
| GPU memory stayed at the observed reference level | PASS, sampled | Both task driver sampled peaks are 4,996 MiB. Sampling does not establish a continuous exact peak. |
| Generic failure semantics / production default readiness | FAIL, insufficient | The unselected-overflow counterexample remains. One certified input and equal artifacts do not prove whole-domain or cross-architecture replacement. |

The reviewer withdrew the earlier “combined numerical equivalence pending” finding after the new direct output comparison passed. No issue was found that overturns the limited conclusion: this input produced equal payloads and this isolated combined prototype was faster in the observed pair.

CPU sampling excludes certificate/display/GLB; it cannot be added to request or loader gains from separate windows. The old 2.603-second resident request is not this run's comparison baseline. Actual counts are 25 requests and 24 reference comparisons.

Additional coordinator inspection of launch receipts: an unrelated GPU1 workload, PID3856119, started during the native delivery arm and was resident before the optimized arm. This strengthens the shared-load limitation; it does not affect the direct artifact comparison. All unrelated workloads were preserved.

The same independent reviewer subsequently checked the final README, summary and this review, including the additional PID3856119 limitation, and found no material misleading claim. Its conclusion remains conditional observed latency improvement with current-input payload equality, not a controlled causal estimate or service guarantee.
