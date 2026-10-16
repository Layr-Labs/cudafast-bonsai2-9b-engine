#pragma once

#include "engine_iface.hpp"

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class LlamaSpeculativeRoundBackend;

class LlamaEngine : public IEngine {
    friend class LlamaSpeculativeRoundBackend;

  public:
    /// spec_mode "serial", "mtp" (the target GGUF's own MTP layer) or "dflash"
    /// (the separate block drafter at draft_model_path).
    LlamaEngine(const std::string &model_path, const std::string &spec_mode, int draft_depth,
                const std::string &draft_model_path, int32_t ctx_size, int32_t n_batch, int32_t n_ubatch,
                const std::string &ident);
    ~LlamaEngine() override;

    int32_t vocab_size() const override { return vocab_size_; }
    int32_t eos_token() const override { return eos_token_; }
    int depth() const override { return depth_; }
    std::string spec_mode() const override { return spec_mode_; }
    std::string draft_model_path() const override { return draft_model_path_; }
    int32_t ctx_size() const override { return ctx_size_; }
    const std::string &ident() const override { return ident_; }
    const std::string &model_path() const override { return model_path_; }

    std::vector<Emit> start(const std::vector<int32_t> &tokens) override;
    std::vector<Emit> start_frontier_only(const std::vector<int32_t> &tokens) override;
    std::vector<Emit> step() override;
    std::vector<Emit> decode_forced(int32_t token) override;
    void abort() override;
    std::pair<int64_t, int64_t> last_spec_step_counters() const override {
        return {last_step_drafts_, last_step_hits_};
    }

  private:
    Emit emit_logits(int idx) const;
    // Target sequence and bookkeeping only; clear_kv() also resets the MTP draft side.
    void clear_target();
    void clear_kv();
    void prefill_prompt(const llama_tokens &inp);
    std::vector<Emit> decode_serial_one();
    std::vector<Emit> decode_speculative_round();

    std::string model_path_;
    std::string ident_;
    std::string spec_mode_;
    std::string draft_model_path_;
    int depth_;
    int32_t ctx_size_;
    int32_t n_batch_;
    int32_t vocab_size_;
    int32_t eos_token_;

    // main_init_ must outlive contexts; destroyed after spec_/sampler.
    common_init_result_ptr main_init_;
    common_speculative_init_result_ptr draft_init_;
    common_speculative_ptr spec_;
    common_sampler *smpl_ = nullptr;
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> target_topk_{nullptr, llama_sampler_free};

    common_params base_params_{};
    common_params_speculative spec_params_{};
    common_prompt_checkpoint ckpt_{};
    llama_seq_id seq_id_ = 0;
    llama_tokens tokens_;
    int n_past_ = 0;
    int logits_batch_idx_ = 0;
    llama_token id_last_ = 0;
    llama_tokens prompt_tgt_;
    llama_tokens draft_;
    bool use_ckpt_tgt_ = false;
    bool use_ckpt_dft_ = false;
    common_context_seq_rm_type seq_rm_tgt_ = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type seq_rm_dft_ = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    llama_model *model_ = nullptr;
    llama_context *ctx_tgt_ = nullptr;
    llama_context *ctx_dft_ = nullptr;

    // DFlash: the target layers whose inputs the drafter reads, and whether
    // their extraction is on (the speculative state turns it on when it is
    // created; start_frontier_only turns it off).
    std::vector<uint32_t> dflash_target_layers_;
    bool dflash_layer_inp_on_ = false;

    int64_t last_step_drafts_ = 0;
    int64_t last_step_hits_ = 0;
};
