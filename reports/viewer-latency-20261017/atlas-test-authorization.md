# Atlas threading test — 2026-10-17

Rick's instruction “帮我测试。think hard” authorizes the prepared atlas 1/2/4-thread comparison; the later “how” asks for progress and does not cancel it. This is focused validation of the implemented exporter, not a restart of the closed RGB research budget or authorization for other geometry experiments.

Scope: read the fixed full video3 generation and existing source, run the prepared smoke and complete-export comparisons, and write task-local scripts/logs/reports. Preserve source arrays, input sidecars, production defaults, and unrelated dirty edits. No GPU, network install, service/COS calls, publication, commit or push.

Ceilings carried from the proposal for this atlas subset: 7,200 CPU seconds and 7,200 elapsed seconds; each comparison at most 900 seconds; RSS at most 12 GiB per runner / 24 GiB aggregate; new files at most 2 GiB; one heavy comparison at a time, fixed four-CPU affinity. At most two official candidates (2 and 4 workers), one debug repair and one retry. The execution worker receives a smaller 5,000 CPU-second / 2,400 elapsed-second allocation and stops before exceeding it.

Keep the prepared single-variable parameters, B1/C1/C2/B2 order, complete-page and binary equality checks, and both-positive-pairs criterion. Add per-export CPU/RSS diagnostics before observing results if needed; do not change acceptance after observing timing. A second worker independently reviews the reports. These focused engineering benchmark records are not a new managed ResearchState or an end-to-end service baseline.
