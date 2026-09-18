# Performance and memory optimization plan

## Objective

Implement the approved deterministic RANSAC, matching/SAM3, and DA3 cache
publication optimizations without changing metric-cache or matching semantics.

## Constraints

- Preserve support-plane seed, candidate ordering, and gates.
- Preserve DA3 schema-v3 metric cache and atomic publication behavior.
- Do not introduce fallback or compatibility behavior.
- Add only focused regression tests and update the root README.

## Tasks

1. Vectorize/batch deterministic support-plane RANSAC and prove seeded outcomes
   match the scalar implementation.
2. Remove INFO-level projected-point host synchronization, reuse hit scores, move
   SAM3 postprocessing off GPU, and avoid persistent DA3 RGB GPU tensors.
3. Publish DA3 runner cache directly through a same-directory partial file and
   atomically replace it after validation.
4. Integrate, update documentation, and run the owned focused and root test gates.
