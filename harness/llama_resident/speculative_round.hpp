#pragma once

#include "engine_iface.hpp"

#include <cstdint>
#include <memory>
#include <vector>

/// Mutable state the speculative-round control flow shares between backends.
struct SpeculativeRoundState {
    std::vector<int32_t> *draft = nullptr;
    bool *use_ckpt_tgt = nullptr;
    bool *use_ckpt_dft = nullptr;
    int *n_past = nullptr;
    int32_t *id_last = nullptr;
    std::vector<int32_t> *prompt_tgt = nullptr;
    std::vector<int32_t> *tokens = nullptr;
    int64_t *last_step_drafts = nullptr;
    int64_t *last_step_hits = nullptr;
};

/// Per-verify-attempt counters (MTP tests introspect these).
struct SpecRoundAttempt {
    int64_t drafts = 0;
    int64_t hits = 0;
};

/// Backend for one speculative decode round: draft, checkpoint, verify, accept.
class SpeculativeRoundBackend {
  public:
    virtual ~SpeculativeRoundBackend() = default;

    virtual int depth() const = 0;

    /// When draft is empty: draft tokens and configure checkpoint flags.
    virtual void prepare_draft(SpeculativeRoundState &st) = 0;

    /// When reusing a partial draft with target checkpointing enabled.
    virtual void require_checkpoint_present() const = 0;

    /// Target verify decode + speculative process for the current draft.
    virtual void verify_decode(SpeculativeRoundState &st) = 0;

    /// Clone sampler before a possible partial-accept rollback (no-op if nullptr).
    virtual std::unique_ptr<void, void (*)(void *)> clone_sampler() = 0;

    virtual void restore_sampler(std::unique_ptr<void, void (*)(void *)> saved) = 0;

    /// Greedy verify accept; result stays in backend storage until the next accept call.
    virtual const std::vector<int32_t> &greedy_accept(const std::vector<int32_t> &draft) = 0;

    /// Partial accept: moves the last greedy_accept result into draft and restores checkpoints.
    virtual void rollback_partial_accept(SpeculativeRoundState &st) = 0;

    virtual void accept_spec_prefix(size_t n_draft_accepted) = 0;

    virtual Emit emit_logits_at_verify_index(int idx) = 0;

    virtual void sampler_accept_token(int32_t token) = 0;

    virtual void trim_kv_tail(int n_past) = 0;

    /// After a full accept, before emitting (fake rolls back verify prompt extension).
    virtual void finish_verify_before_emits(SpeculativeRoundState &st, size_t n_committed_ids) { (void)st; (void)n_committed_ids; }

    virtual void on_round_complete() {}

    /// Optional hook so fakes can record per-attempt draft/hit increments.
    virtual void on_verify_attempt(int64_t drafts_added, int64_t hits_added) { (void)drafts_added; (void)hits_added; }
};

std::vector<Emit> run_speculative_round(SpeculativeRoundState st, SpeculativeRoundBackend &backend);
