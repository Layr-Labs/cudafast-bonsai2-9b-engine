#pragma once

#include "engine_iface.hpp"
#include "speculative_round.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// MTP session tests: production speculative-round loop with a scripted backend.
class SpecRoundTestEngine : public IEngine {
  public:
    explicit SpecRoundTestEngine(int draft_depth = 4, int32_t vocab = 131072, int32_t ctx = 8192);
    ~SpecRoundTestEngine() override;

    int32_t vocab_size() const override { return vocab_size_; }
    int32_t eos_token() const override { return 2; }
    int depth() const override { return depth_; }
    int32_t ctx_size() const override { return ctx_size_; }
    const std::string &ident() const override { return ident_; }
    const std::string &model_path() const override { return model_path_; }

    std::vector<Emit> start(const std::vector<int32_t> &tokens) override;
    std::vector<Emit> step() override;
    std::vector<Emit> decode_forced(int32_t token) override;
    void abort() override;
    std::pair<int64_t, int64_t> last_spec_step_counters() const override {
        return {last_step_drafts_, last_step_hits_};
    }

    const std::vector<int32_t> &prompt_tgt() const { return prompt_tgt_; }
    const std::vector<int32_t> &tokens() const { return tokens_; }
    int sim_tgt_kv_pos() const { return loop_n_past_; }

    using SpecAttemptCounters = SpecRoundAttempt;
    const std::vector<SpecAttemptCounters> &last_round_attempts() const { return last_round_attempts_; }

    void arm_verify_tie(int32_t low_id, int32_t high_id);

    /// Address of the vector the fake last returned from greedy_accept.
    const std::vector<int32_t> *greedy_accept_returned_addr() const { return greedy_accept_returned_addr_; }
    /// Before the next full-accept emits, write marker through the returned address (last id).
    void overwrite_last_accepted_id_after_return(int32_t marker) {
        overwrite_marker_ = marker;
        overwrite_armed_ = true;
    }

    int last_trim_kv_n_past() const { return last_trim_kv_n_past_; }
    int trim_kv_tail_calls() const { return trim_kv_tail_calls_; }

    int sampler_clones_created() const { return sampler_clones_created_; }
    int sampler_clones_freed() const { return sampler_clones_freed_; }
    int sampler_restores() const { return sampler_restores_; }
    void throw_in_next_greedy_accept() { throw_in_next_greedy_accept_ = true; }
    void set_use_ckpt_tgt(bool on) { use_ckpt_tgt_ = on; }
    int rollbacks() const { return rollbacks_; }

    int starts = 0;

  private:
    friend class FakeBackend;
    class FakeBackend;

    Emit emit_frontier() const;
    Emit emit_at_verify_index(int idx) const;
    std::vector<int32_t> make_full_draft(bool reject_at_one) const;
    std::vector<int32_t> greedy_sample_and_accept(const std::vector<int32_t> &draft) const;
    SpeculativeRoundState round_state();

    int depth_;
    int32_t vocab_size_;
    int32_t ctx_size_;
    std::string ident_;
    std::string model_path_;

    std::vector<int32_t> tokens_;
    std::vector<int32_t> prompt_tgt_;
    std::vector<int32_t> draft_;
    int32_t id_last_ = 0;
    int loop_n_past_ = 0;
    bool use_ckpt_tgt_ = true;
    bool use_ckpt_dft_ = false;

    size_t verify_ckpt_tokens_ = 0;

    bool next_verify_tie_ = false;
    int32_t tie_low_ = 0;
    int32_t tie_high_ = 0;
    bool reject_next_full_draft_at_one_ = false;

    int64_t last_step_drafts_ = 0;
    int64_t last_step_hits_ = 0;
    std::vector<SpecAttemptCounters> last_round_attempts_;

    std::vector<int32_t> *greedy_accept_returned_addr_ = nullptr;
    bool overwrite_armed_ = false;
    int32_t overwrite_marker_ = 0;

    int last_trim_kv_n_past_ = -1;
    int trim_kv_tail_calls_ = 0;

    int sampler_clones_created_ = 0;
    int sampler_clones_freed_ = 0;
    int sampler_restores_ = 0;
    int rollbacks_ = 0;
    bool throw_in_next_greedy_accept_ = false;

    std::unique_ptr<SpeculativeRoundBackend> backend_;
};
