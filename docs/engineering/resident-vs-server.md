# Resident vs llama-server draft-MTP (commit 4fbda129)

Reference: `tools/server/server-context.cpp` (`pre_decode` / `post_decode` speculative path) and
`examples/speculative-simple/speculative-simple.cpp`. Resident: `harness/llama_resident/engine_llama.cpp` and `harness/llama_resident/speculative_round.cpp`.
One speculative round runs in `run_speculative_round()` (`speculative_round.cpp`). It makes the llama.cpp calls through `LlamaSpeculativeRoundBackend` (`engine_llama.cpp`). `LlamaEngine::decode_speculative_round()` calls it.

| Server / speculative-simple step | Resident |
|----------------------------------|----------|
| Prefix prompt on target (`inp[0..n-2]`, no logits) + `common_speculative_process` per chunk | `LlamaEngine::start` prefix loop (`mtp prefix prefill`) |
| `id_last = inp.back()`, `prompt_tgt` without last, `n_past = n-1` | Same (`id_last_`, `prompt_tgt_`, `n_past_`) |
| Full prompt prefill on target + `common_speculative_process` per chunk; logits on last prompt row | `start()` MTP loop (same as depth-0 prefill shape, plus `process`) |
| `common_speculative_begin(spec, seq, full prompt text tokens)` at `DONE_PROMPT` | `common_speculative_begin` on full `inp` after prefill |
| First generated token: `common_sampler_sample` on last prompt logits (not a verify batch) | Same in `start()`; `decode_speculative_round()` from `step()` only |
| Draft round: verify `[sampled, draft…]` → `sample_and_accept_n` | `run_speculative_round()`; greedy verify uses `emit_logits` (same argmax/tie-break as serial `top_logits`) |
| Target context backend sampler off for reproducibility | `sampling.backend_sampling = false`, `llama_set_sampler(ctx_tgt, …, nullptr)` |
| `n_draft_max = min(ctx room, budget, spec n_max)` | `min(ctx room, depth_)` in `LlamaSpeculativeRoundBackend::prepare_draft` |
| Draft: `update_pos`, optional `update_dft`, `common_speculative_draft`, restore dft KV past `pos_max+1` | Same block in `LlamaSpeculativeRoundBackend::prepare_draft`; `run_speculative_round` calls it when the draft is empty |
| Target checkpoint before verify when draft non-empty (FULL or RS overflow) | Same `use_ckpt_tgt_` / `use_ckpt_dft_` rules, in `LlamaSpeculativeRoundBackend::prepare_draft` |
| Partial accept: `load_tgt`/`load_dft`, trim KV, `prompt_tgt.resize(ckpt.n_tokens)`, restore sampler | Same: `LlamaSpeculativeRoundBackend::rollback_partial_accept` loads the checkpoints, trims KV and resizes `prompt_tgt`; `run_speculative_round` then restores the sampler clone |
| Full accept: `common_speculative_accept`, extend `prompt_tgt`, trim KV at `n_past` | Same, in `run_speculative_round`; emits per batch index via `emit_logits(i)` |
| Serial (no spec): sample one column after prompt | `decode_serial_one` / depth 0 `start` |
| `n_batch` / `n_ubatch` from params; MTP prompt fits one ubatch | `make_engine_params`: `n_ubatch = n_batch` when `draft_depth > 0` |
| New task: fresh slot / reset KV + re-`common_speculative_init` | `clear_kv` / `abort` on `invalidate` |

Wire note: `Session::sync` still exposes the first emitted token from the first speculative round; that now matches server timing (no pre-round sampler accept on a lone seed decode).
