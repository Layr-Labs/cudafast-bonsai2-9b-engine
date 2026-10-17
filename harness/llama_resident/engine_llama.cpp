#include "engine_llama.hpp"

#include "common.h"
// Staging API: llama_set_embeddings_nextn, llama_set_logits_topk (common/speculative.cpp includes it the same way).
#include "../src/llama-ext.h"
#include "logits_rank.hpp"
#include "speculative_round.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

static void sampler_unique_free(common_sampler *p) {
    if (p) {
        common_sampler_free(p);
    }
}

class LlamaSpeculativeRoundBackend : public SpeculativeRoundBackend {
  public:
    explicit LlamaSpeculativeRoundBackend(LlamaEngine *eng) : eng_(eng) {
        verify_batch_ = llama_batch_init(eng_->depth_ + 1, 0, 1);
    }

    ~LlamaSpeculativeRoundBackend() override { llama_batch_free(verify_batch_); }

    int depth() const override { return eng_->depth_; }

    void prepare_draft(SpeculativeRoundState &st) override {
        if (!st.draft->empty()) {
            return;
        }
        eng_->ckpt_.update_pos(eng_->prompt_tgt_.size(),
                               llama_memory_seq_pos_min(llama_get_memory(eng_->ctx_tgt_), eng_->seq_id_),
                               llama_memory_seq_pos_max(llama_get_memory(eng_->ctx_tgt_), eng_->seq_id_));
        if (eng_->seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            eng_->ckpt_.update_dft(eng_->ctx_dft_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        int n_draft_max = static_cast<int>(llama_n_ctx(eng_->ctx_tgt_)) - *st.n_past - 2;
        n_draft_max = std::max(0, std::min(n_draft_max, eng_->depth_));
        common_speculative_get_draft_params(eng_->spec_.get(), eng_->seq_id_) = {
            /*drafting*/ true,
            /*n_max*/ n_draft_max,
            /*n_past*/ *st.n_past,
            /*id_last*/ *st.id_last,
            /*prompt*/ &eng_->prompt_tgt_,
            /*result*/ st.draft,
        };
        common_speculative_draft(eng_->spec_.get());
        // DFlash2 device walk: the ids stay on the device and verify_decode imports them, so the host
        // does not wait for the draft graph here; the verify graph queues behind it on the device
        draft_on_device_ = common_speculative_draft_on_device(eng_->spec_.get());
        if (eng_->ctx_dft_ && !draft_on_device_) {
            llama_synchronize(eng_->ctx_dft_);
        }
        if (eng_->ctx_dft_) {
            if (eng_->seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                eng_->ckpt_.load_dft(eng_->ctx_dft_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            llama_memory_seq_rm(llama_get_memory(eng_->ctx_dft_), eng_->seq_id_, eng_->ckpt_.pos_max + 1, -1);
        }
        if (!st.draft->empty()) {
            *st.use_ckpt_tgt = eng_->seq_rm_tgt_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                               (eng_->seq_rm_tgt_ == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                                st.draft->size() > llama_n_rs_seq(eng_->ctx_tgt_));
            const bool ckpt_dft_rs = eng_->seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                                       st.draft->size() > llama_n_rs_seq(eng_->ctx_dft_);
            *st.use_ckpt_dft = eng_->seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ckpt_dft_rs;
            if (*st.use_ckpt_tgt) {
                eng_->ckpt_.update_tgt(eng_->ctx_tgt_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (ckpt_dft_rs) {
                eng_->ckpt_.update_dft(eng_->ctx_dft_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
    }

    void require_checkpoint_present() const override {
        if (eng_->ckpt_.empty()) {
            throw std::runtime_error("speculative checkpoint missing for partial draft reuse");
        }
    }

    void verify_decode(SpeculativeRoundState &st) override {
        common_batch_clear(verify_batch_);
        common_batch_add(verify_batch_, *st.id_last, (*st.n_past)++, {eng_->seq_id_}, true);
        for (size_t i = 0; i < st.draft->size(); ++i) {
            common_batch_add(verify_batch_, (*st.draft)[i], *st.n_past + static_cast<int>(i), {eng_->seq_id_}, true);
        }
        const bool on_device = draft_on_device_ && !st.draft->empty();
        draft_on_device_ = false;
        if (on_device && !llama_set_token_import_dflash_walk(eng_->ctx_tgt_, eng_->ctx_dft_, 1,
                                                             static_cast<int32_t>(st.draft->size()))) {
            throw std::runtime_error("DFlash2 device walk import failed");
        }
        if (llama_decode(eng_->ctx_tgt_, verify_batch_) != 0) {
            throw std::runtime_error("speculative verify decode failed");
        }
        if (on_device) {
            // the verify graph already holds the device ids; the host takes the same ids for the accept,
            // the emits and the bookkeeping (a prefix of the walk when the draft was truncated)
            std::vector<int32_t> walk(static_cast<size_t>(eng_->depth_) + 1);
            const int32_t n_walk = llama_get_dflash_walk(eng_->ctx_dft_, walk.data(), static_cast<int32_t>(walk.size()));
            if (n_walk < static_cast<int32_t>(st.draft->size())) {
                throw std::runtime_error("DFlash2 device walk returned too few ids");
            }
            for (size_t i = 0; i < st.draft->size(); ++i) {
                (*st.draft)[i] = walk[i];
                verify_batch_.token[i + 1] = walk[i];
            }
        }
        // Defer DFlash feature injection until the committed prefix is known.
        // A checkpoint retry does not consume features from its failed attempt.
        if (eng_->spec_mode_ != "dflash" && !common_speculative_process(eng_->spec_.get(), verify_batch_)) {
            throw std::runtime_error("speculative process failed");
        }
    }

  private:
    bool draft_on_device_ = false;

  public:
    std::unique_ptr<void, void (*)(void *)> clone_sampler() override {
        if (!eng_->use_ckpt_tgt_) {
            return {nullptr, noop_sampler_holder_free};
        }
        common_sampler *cloned = common_sampler_clone(eng_->smpl_);
        return {cloned, sampler_ptr_free};
    }

    void restore_sampler(std::unique_ptr<void, void (*)(void *)> saved) override {
        sampler_unique_free(eng_->smpl_);
        eng_->smpl_ = static_cast<common_sampler *>(saved.release());
    }

    const std::vector<int32_t> &greedy_accept(const std::vector<int32_t> &draft) override {
        // One top-8 scan per verify row: the accept reads its token, the emit reuses it.
        verify_emits_.clear();
        greedy_accept_ids_.clear();
        for (size_t i = 0; i <= draft.size(); ++i) {
            verify_emits_.push_back(eng_->emit_logits(static_cast<int>(i)));
            greedy_accept_ids_.push_back(verify_emits_.back().token);
            if (i == draft.size() || draft[i] != greedy_accept_ids_.back()) {
                break;
            }
        }
        return static_cast<const std::vector<int32_t> &>(greedy_accept_ids_);
    }

    void rollback_partial_accept(SpeculativeRoundState &st) override {
        *st.draft = std::move(greedy_accept_ids_);
        eng_->ckpt_.load_tgt(eng_->ctx_tgt_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        llama_memory_seq_rm(llama_get_memory(eng_->ctx_tgt_), eng_->seq_id_, eng_->ckpt_.pos_max + 1, -1);
        if (eng_->ctx_dft_) {
            if (*st.use_ckpt_dft) {
                eng_->ckpt_.load_dft(eng_->ctx_dft_, eng_->seq_id_, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            llama_memory_seq_rm(llama_get_memory(eng_->ctx_dft_), eng_->seq_id_, eng_->ckpt_.pos_max + 1, -1);
        }
        st.prompt_tgt->resize(eng_->ckpt_.n_tokens);
        *st.n_past = static_cast<int>(st.prompt_tgt->size());
    }

    void accept_spec_prefix(size_t n_draft_accepted) override {
        if (eng_->spec_mode_ == "dflash") {
            // The correction/bonus is next round's anchor, not a verified input.
            if (n_draft_accepted + 1 > static_cast<size_t>(verify_batch_.n_tokens)) {
                throw std::runtime_error("accepted DFlash prefix exceeds verify batch");
            }
            verify_batch_.n_tokens = static_cast<int32_t>(n_draft_accepted + 1);
            if (!common_speculative_process(eng_->spec_.get(), verify_batch_)) {
                throw std::runtime_error("accepted DFlash prefix process failed");
            }
        }
        common_speculative_accept(eng_->spec_.get(), eng_->seq_id_, n_draft_accepted);
    }

    Emit emit_logits_at_verify_index(int idx) override { return verify_emits_.at(static_cast<size_t>(idx)); }

    void sampler_accept_token(int32_t token) override { common_sampler_accept(eng_->smpl_, token, true); }

    void trim_kv_tail(int n_past) override {
        llama_memory_seq_rm(llama_get_memory(eng_->ctx_tgt_), eng_->seq_id_, n_past, -1);
        if (eng_->ctx_dft_) {
            llama_memory_seq_rm(llama_get_memory(eng_->ctx_dft_), eng_->seq_id_, n_past, -1);
        }
    }

  private:
    static void noop_sampler_holder_free(void *) {}
    static void sampler_ptr_free(void *p) { sampler_unique_free(static_cast<common_sampler *>(p)); }

    LlamaEngine *eng_;
    llama_batch verify_batch_{};
    llama_tokens greedy_accept_ids_;
    std::vector<Emit> verify_emits_;
};

static common_params make_engine_params(const std::string &model_path, const std::string &spec_mode,
                                        int draft_depth, const std::string &draft_model_path, int32_t ctx_size,
                                        int32_t n_batch, int32_t n_ubatch) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = ctx_size;
    params.n_batch = n_batch;
    // MTP prompt process() must see the full prefix in one ubatch (GB10 nondeterminism otherwise).
    params.n_ubatch = draft_depth > 0 ? n_batch : n_ubatch;
    params.n_parallel = 1;
    params.n_gpu_layers = -1;
    // llama.cpp keeps the token embedding table in host memory, which puts a CPU split (the row
    // lookup) and a synchronous upload in front of every graph; on the GPU the lookup runs inside
    // the graph and dequantizes the same way (test/token_embd_equivalence.cpp)
    if (ggml_backend_dev_t gpu = ggml_backend_dev_by_name("CUDA0")) {
        params.tensor_buft_overrides.push_back({"^token_embd\\.weight$", ggml_backend_dev_buffer_type(gpu)});
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }
    constexpr int n_threads = 4;
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.cache_type_k = GGML_TYPE_F16;
    params.cache_type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.sampling.temp = 0.0f;
    params.sampling.top_k = 1;
    params.sampling.seed = 1234;
    params.sampling.backend_sampling = false;
    if (draft_depth > 0) {
        if (spec_mode == "dflash") {
            params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
            params.speculative.draft.mparams.path = draft_model_path;
            params.speculative.draft.n_gpu_layers = -1;
        } else {
            params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        }
        params.speculative.draft.n_max = draft_depth;
        // CPU draft sampling only: backend top-k offload is not reproducible on GB10.
        params.speculative.draft.backend_sampling = false;
        // verify_decode imports DFlash2 drafts walked on the device
        params.speculative.draft.device_draft = true;
    }
    return params;
}

LlamaEngine::LlamaEngine(const std::string &model_path, const std::string &spec_mode, int draft_depth,
                         const std::string &draft_model_path, int32_t ctx_size, int32_t n_batch, int32_t n_ubatch,
                         const std::string &ident)
    : model_path_(model_path), ident_(ident), spec_mode_(draft_depth > 0 ? spec_mode : "serial"),
      draft_model_path_(spec_mode_ == "dflash" ? draft_model_path : std::string()), depth_(draft_depth),
      ctx_size_(ctx_size), n_batch_(n_batch) {
    if (spec_mode_ != "serial" && spec_mode_ != "mtp" && spec_mode_ != "dflash") {
        throw std::runtime_error("unknown spec mode " + spec_mode_);
    }
    if (spec_mode_ == "dflash" && draft_model_path_.empty()) {
        throw std::runtime_error("dflash needs a draft model");
    }
    llama_backend_init();

    base_params_ = make_engine_params(model_path, spec_mode_, depth_, draft_model_path_, ctx_size_, n_batch,
                                      n_ubatch);
    main_init_ = common_init_from_params(base_params_);
    if (!main_init_ || !main_init_->model() || !main_init_->context()) {
        throw std::runtime_error("failed to load model");
    }
    model_ = main_init_->model();
    ctx_tgt_ = main_init_->context();
    llama_set_sampler(ctx_tgt_, seq_id_, nullptr);
    logits_topk_ = llama_set_logits_topk(ctx_tgt_, kTopK);

    const llama_vocab *vocab = llama_model_get_vocab(model_);
    vocab_size_ = llama_vocab_n_tokens(vocab);
    eos_token_ = llama_vocab_eos(vocab);

    if (depth_ > 0) {
        common_params params_dft = common_base_params_to_speculative(base_params_);
        draft_init_ = common_speculative_init_from_params(params_dft, model_, ctx_tgt_);
        if (!draft_init_ || !draft_init_->context()) {
            throw std::runtime_error("failed to initialize the " + spec_mode_ + " draft context");
        }
        ctx_dft_ = draft_init_->context();
        if (spec_mode_ == "dflash") {
            // The drafter decodes [anchor, mask * (block_size - 1)], so it drafts at most
            // block_size - 1 tokens. llama.cpp clamps a longer request; refuse it instead, so
            // the depth the resident reports is the depth that runs.
            const llama_model *model_dft = llama_get_model(ctx_dft_);
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) < 0) {
                throw std::runtime_error("the draft model has no dflash.block_size");
            }
            const int block_size = std::atoi(buf);
            if (depth_ > block_size - 1) {
                throw std::runtime_error("draft-len " + std::to_string(depth_) + " exceeds the drafter's block size " +
                                         std::to_string(block_size) + " minus the anchor");
            }
            for (uint32_t k = 0; k < llama_model_target_layer_ids_n(model_dft); ++k) {
                dflash_target_layers_.push_back(static_cast<uint32_t>(llama_model_target_layer_ids(model_dft)[k]));
            }
        }
        spec_params_ = base_params_.speculative;
        spec_params_.draft.ctx_tgt = ctx_tgt_;
        spec_params_.draft.ctx_dft = ctx_dft_;
        spec_.reset(common_speculative_init(spec_params_, 1));
        if (!spec_) {
            throw std::runtime_error("failed to initialize speculative decoding");
        }
        dflash_layer_inp_on_ = !dflash_target_layers_.empty();
        seq_rm_tgt_ = common_context_can_seq_rm(ctx_tgt_);
        seq_rm_dft_ = common_context_can_seq_rm(ctx_dft_);
        use_ckpt_tgt_ = seq_rm_tgt_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        use_ckpt_dft_ = seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        if (dflash_layer_inp_on_) {
            // Reserve the target scheduler at boot for both target-layer extraction settings (input-independent):
            // the decode setting the speculative state turned on, then the serial one a timed prefill uses.
            // A later switch between them restores the reserved state instead of reserving again.
            llama_sched_reserve_now(ctx_tgt_);
            for (const uint32_t lid : dflash_target_layers_) {
                llama_set_embeddings_layer_inp(ctx_tgt_, lid, false);
            }
            llama_sched_reserve_now(ctx_tgt_);
            dflash_layer_inp_on_ = false;
        }
    }

    common_params_sampling sp = base_params_.sampling;
    smpl_ = common_sampler_init(model_, sp);
}

LlamaEngine::~LlamaEngine() {
    if (smpl_) {
        common_sampler_free(smpl_);
        smpl_ = nullptr;
    }
    spec_.reset();
    draft_init_.reset();
    main_init_.reset();
    llama_backend_free();
}

void LlamaEngine::clear_target() {
    llama_memory_clear(llama_get_memory(ctx_tgt_), true);
    tokens_.clear();
    prompt_tgt_.clear();
    draft_.clear();
    n_past_ = 0;
    id_last_ = 0;
    ckpt_.clear();
    if (smpl_) {
        common_sampler_reset(smpl_);
    }
    use_ckpt_tgt_ = seq_rm_tgt_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    use_ckpt_dft_ = seq_rm_dft_ == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    llama_synchronize(ctx_tgt_);
}

void LlamaEngine::clear_kv() {
    clear_target();
    if (ctx_dft_) {
        llama_memory_clear(llama_get_memory(ctx_dft_), true);
    }
    if (spec_) {
        spec_.reset(common_speculative_init(spec_params_, 1));
        dflash_layer_inp_on_ = !dflash_target_layers_.empty();
    }
    if (ctx_dft_) {
        llama_synchronize(ctx_dft_);
    }
}

void LlamaEngine::abort() { clear_kv(); }

void LlamaEngine::prefill_prompt(const llama_tokens &inp) {
    const int32_t n = static_cast<int32_t>(inp.size());
    int32_t pos = 0;
    logits_batch_idx_ = 0;
    for (int32_t off = 0; off < n;) {
        const int32_t chunk = std::min(n_batch_, n - off);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        for (int32_t i = 0; i < chunk; ++i) {
            const bool want_logits = (off + i == n - 1);
            if (want_logits) {
                logits_batch_idx_ = i;
            }
            common_batch_add(batch, inp[off + i], pos++, {seq_id_}, want_logits);
        }
        if (llama_decode(ctx_tgt_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("prefill decode failed");
        }
        llama_batch_free(batch);
        off += chunk;
    }
}

Emit LlamaEngine::emit_logits(int idx) const {
    llama_synchronize(ctx_tgt_);
    if (logits_topk_) {
        const int32_t *ids = nullptr;
        const float *top = nullptr;
        const float *head = nullptr;
        Emit e;
        if (llama_get_logits_topk_ith(ctx_tgt_, idx, &ids, &top, &head) && top_from_selection(ids, top, head, e)) {
            return e;
        }
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_));
    const float *logits = llama_get_logits_ith(ctx_tgt_, idx);
    if (!logits) {
        throw std::runtime_error("no logits at frontier");
    }
    return top_from_logits(logits, n_vocab);
}

std::vector<Emit> LlamaEngine::start(const std::vector<int32_t> &tokens) {
    if (static_cast<int>(tokens.size()) + depth_ + 2 > ctx_size_) {
        throw std::runtime_error("context full");
    }
    abort();

    llama_tokens inp;
    inp.reserve(tokens.size());
    for (int32_t t : tokens) {
        inp.push_back(t);
    }
    if (inp.empty()) {
        throw std::runtime_error("empty prefill");
    }

    if (depth_ == 0) {
        prefill_prompt(inp);
        tokens_ = inp;
        n_past_ = static_cast<int>(inp.size());
        id_last_ = inp.back();
        Emit first = emit_logits(logits_batch_idx_);
        id_last_ = first.token;
        tokens_.push_back(first.token);
        return {first};
    }

    if (inp.size() < 2) {
        throw std::runtime_error("mtp sync needs at least two tokens");
    }

    const int32_t n = static_cast<int32_t>(inp.size());
    int32_t pos = 0;
    logits_batch_idx_ = 0;
    for (int32_t off = 0; off < n;) {
        const int32_t chunk = std::min(n_batch_, n - off);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        for (int32_t i = 0; i < chunk; ++i) {
            const bool want_logits = (off + i == n - 1);
            if (want_logits) {
                logits_batch_idx_ = i;
            }
            common_batch_add(batch, inp[off + i], pos++, {seq_id_}, want_logits);
        }
        if (llama_decode(ctx_tgt_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("mtp prompt prefill failed");
        }
        if (!common_speculative_process(spec_.get(), batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("speculative prompt process failed");
        }
        llama_batch_free(batch);
        off += chunk;
    }

    prompt_tgt_ = inp;
    n_past_ = n;
    tokens_ = inp;
    common_speculative_begin(spec_.get(), seq_id_, prompt_tgt_);

    // The drafter's pass over the prompt, queued by common_speculative_process above (DFlash: the
    // encoder over the prompt rows' target-layer inputs and the drafter K/V of those rows), runs on
    // the draft context's own stream, which the target synchronize in emit_logits does not wait
    // for. It is part of this sync's prompt processing, so the reply waits for it: the request
    // returns with the prompt fully processed. No token is drafted here; the first draft runs in
    // the first decode cycle.
    if (ctx_dft_) {
        llama_synchronize(ctx_dft_);
    }

    Emit first = emit_logits(logits_batch_idx_);
    id_last_ = first.token;
    common_sampler_reset(smpl_);
    common_sampler_accept(smpl_, id_last_, true);
    tokens_.push_back(first.token);
    return {first};
}

std::vector<Emit> LlamaEngine::start_frontier_only(const std::vector<int32_t> &tokens) {
    if (depth_ == 0) {
        return start(tokens);
    }
    if (static_cast<int>(tokens.size()) + depth_ + 2 > ctx_size_) {
        throw std::runtime_error("context full");
    }
    llama_tokens inp(tokens.begin(), tokens.end());
    if (inp.size() < 2) {
        throw std::runtime_error("mtp sync needs at least two tokens");
    }
    // The target pass is the serial one. Only later decode steps use the MTP
    // side, and a timed prefill takes none, so this pass skips all of it:
    // - the draft reset (draft memory clear, speculative state re-create);
    //   start() does it with its abort() before the first draft;
    // - the MTP layer's pass over the prompt (common_speculative_process) and
    //   common_speculative_begin;
    // - the nextn hidden-state extraction (the MTP layer's input). With it on,
    //   the last target layer runs over every prompt row instead of the
    //   frontier row only, and every row is copied to the host. start()'s
    //   abort() turns it on again: the speculative state it re-creates
    //   (common_speculative_impl_draft_mtp) sets it on the target context.
    clear_target();
    llama_set_embeddings_nextn(ctx_tgt_, false, false);
    // DFlash: the same for the target-layer inputs the drafter reads (eight
    // layers, every prompt row copied to the host). The speculative state
    // start() re-creates turns them on again. Each set call makes the next
    // target pass re-reserve its compute graph (llama_context::
    // set_embeddings_layer_inp), so they are turned off only when they are on:
    // the first prefill pass after an invalidate pays the one re-reserve, and
    // the passes after it run the serial graph.
    if (dflash_layer_inp_on_) {
        for (const uint32_t lid : dflash_target_layers_) {
            llama_set_embeddings_layer_inp(ctx_tgt_, lid, false);
        }
        dflash_layer_inp_on_ = false;
    }
    prefill_prompt(inp);
    // tokens_ stays empty, so step() and decode_forced() refuse: the draft
    // context has not seen the prompt. Session::sync runs start() again
    // before any eval or cycle.
    return {emit_logits(logits_batch_idx_)};
}

std::vector<Emit> LlamaEngine::decode_serial_one() {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, id_last_, n_past_, {seq_id_}, true);
    if (llama_decode(ctx_tgt_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("decode failed");
    }
    llama_batch_free(batch);
    n_past_++;
    Emit e = emit_logits(0);
    id_last_ = e.token;
    tokens_.push_back(e.token);
    return {e};
}

std::vector<Emit> LlamaEngine::decode_speculative_round() {
    LlamaSpeculativeRoundBackend backend(this);
    SpeculativeRoundState st;
    st.draft = &draft_;
    st.use_ckpt_tgt = &use_ckpt_tgt_;
    st.use_ckpt_dft = &use_ckpt_dft_;
    st.n_past = &n_past_;
    st.id_last = &id_last_;
    st.prompt_tgt = &prompt_tgt_;
    st.tokens = &tokens_;
    st.last_step_drafts = &last_step_drafts_;
    st.last_step_hits = &last_step_hits_;
    return run_speculative_round(st, backend);
}

std::vector<Emit> LlamaEngine::decode_forced(int32_t token) {
    if (tokens_.empty()) {
        throw std::runtime_error("decode_forced with no live request");
    }
    draft_.clear();
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, n_past_, {seq_id_}, true);
    if (llama_decode(ctx_tgt_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("forced decode failed");
    }
    if (depth_ > 0 && !common_speculative_process(spec_.get(), batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("forced speculative process failed");
    }
    llama_batch_free(batch);
    n_past_++;
    tokens_.push_back(token);
    if (depth_ > 0) {
        prompt_tgt_.push_back(token);
    }
    Emit e = emit_logits(0);
    id_last_ = e.token;
    tokens_.push_back(e.token);
    if (depth_ > 0) {
        common_sampler_reset(smpl_);
        common_sampler_accept(smpl_, id_last_, true);
    }
    return {e};
}

std::vector<Emit> LlamaEngine::step() {
    if (tokens_.empty()) {
        throw std::runtime_error("step with no live request");
    }
    last_step_drafts_ = 0;
    last_step_hits_ = 0;
    if (depth_ == 0) {
        return decode_serial_one();
    }
    return decode_speculative_round();
}
