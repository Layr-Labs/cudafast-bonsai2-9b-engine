# Engineering notes (organizers)

Partner-facing measurements and design records for the Bonsai 2 9B CUDA track.
These documents support staging, benchd integration, and fleet posture reviews.

| Document | Contents |
|---|---|
| [engine-probe.md](engine-probe.md) | GB10 llama-bench, determinism, and speculation probes on rehearsal weights |
| [resident-vs-server.md](resident-vs-server.md) | Resident draft-MTP loop aligned with llama-server / speculative-simple |
| [resident-validation.md](resident-validation.md) | Adapter e2e, oracle checks, and what is proven on-box |
| [benchd-decode-metric.md](benchd-decode-metric.md) | `noop_decode_speedup` and the official iterate scoring path |
| [replay-policy-measurements.md](replay-policy-measurements.md) | Timed replay measurements behind the two fixture keys |
| [account-boundary-audit.md](account-boundary-audit.md) | Job-account posture on contest Sparks (2026-10-08 audit) |
| [runner-isolation-gap-list.md](runner-isolation-gap-list.md) | Gaps vs Eigen-style runner isolation (reference comparison) |
| [nemotron-harness-inventory.md](nemotron-harness-inventory.md) | Nemotron → Bonsai port file inventory (historical) |
