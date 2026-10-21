#include "engine_spec_round_test.hpp"

#include "logits_rank.hpp"

#include <stdexcept>
#include <utility>

namespace {
struct FakeSamplerClone {
    int *freed;
};

void fake_sampler_free(void *p) {
    auto *clone = static_cast<FakeSamplerClone *>(p);
    (*clone->freed)++;
    delete clone;
}
} // namespace

class SpecRoundTestEngine::FakeBackend : public SpeculativeRoundBackend {
  public:
    explicit FakeBackend(SpecRoundTestEngine *owner) : owner_(owner) {}

    int depth() const override { return owner_->depth_; }

    void prepare_draft(SpeculativeRoundState &st) override {
        if (st.draft->empty()) {
            *st.draft = owner_->make_full_draft(owner_->reject_next_full_draft_at_one_);
            owner_->reject_next_full_draft_at_one_ = false;
        }
    }

    void require_checkpoint_present() const override {}

    void verify_decode(SpeculativeRoundState &st) override {
        owner_->verify_ckpt_tokens_ = st.prompt_tgt->size();
        const size_t n_draft = st.draft->size();
        for (size_t i = 0; i < n_draft; ++i) {
            st.prompt_tgt->push_back((*st.draft)[i]);
        }
        (*st.n_past)++;
    }

    std::unique_ptr<void, void (*)(void *)> clone_sampler() override {
        if (!owner_->use_ckpt_tgt_) {
            return {nullptr, fake_sampler_free};
        }
        owner_->sampler_clones_created_++;
        return {new FakeSamplerClone{&owner_->sampler_clones_freed_}, fake_sampler_free};
    }

    void restore_sampler(std::unique_ptr<void, void (*)(void *)> saved) override {
        if (saved) {
            owner_->sampler_restores_++;
        }
    }

    const std::vector<int32_t> &greedy_accept(const std::vector<int32_t> &draft) override {
        if (owner_->throw_in_next_greedy_accept_) {
            owner_->throw_in_next_greedy_accept_ = false;
            throw std::runtime_error("injected greedy_accept failure");
        }
        greedy_accept_ids_ = owner_->greedy_sample_and_accept(draft);
        owner_->greedy_accept_returned_addr_ = &greedy_accept_ids_;
        return greedy_accept_ids_;
    }

    void rollback_partial_accept(SpeculativeRoundState &st) override {
        owner_->rollbacks_++;
        *st.draft = std::move(greedy_accept_ids_);
        st.prompt_tgt->resize(owner_->verify_ckpt_tokens_);
        *st.n_past = static_cast<int>(st.prompt_tgt->size());
    }

    void accept_spec_prefix(size_t) override {}

    Emit emit_logits_at_verify_index(int idx) override {
        Emit e = owner_->emit_at_verify_index(idx);
        e.token = owner_->id_last_;
        if (!e.top.empty()) {
            e.top[0] = {owner_->id_last_, e.top[0].second};
        }
        return e;
    }

    void sampler_accept_token(int32_t) override {}

    void trim_kv_tail(int n_past) override {
        owner_->last_trim_kv_n_past_ = n_past;
        owner_->trim_kv_tail_calls_++;
    }

    void finish_verify_before_emits(SpeculativeRoundState &st, size_t n_committed_ids) override {
        st.prompt_tgt->resize(owner_->verify_ckpt_tokens_);
        (void)n_committed_ids;
        if (owner_->overwrite_armed_) {
            owner_->overwrite_armed_ = false;
            owner_->greedy_accept_returned_addr_->back() = owner_->overwrite_marker_;
        }
    }

    void on_round_complete() override { owner_->next_verify_tie_ = false; }

    void on_verify_attempt(int64_t drafts_added, int64_t hits_added) override {
        owner_->last_round_attempts_.push_back({drafts_added, hits_added});
    }

  private:
    SpecRoundTestEngine *owner_;
    std::vector<int32_t> greedy_accept_ids_;
};

SpecRoundTestEngine::SpecRoundTestEngine(int draft_depth, int32_t vocab, int32_t ctx)
    : depth_(draft_depth), vocab_size_(vocab), ctx_size_(ctx),
      ident_("spec-round-test depth=" + std::to_string(draft_depth)), model_path_("/fake/spec-round-model") {
    backend_ = std::make_unique<SpecRoundTestEngine::FakeBackend>(this);
}

SpecRoundTestEngine::~SpecRoundTestEngine() = default;

void SpecRoundTestEngine::arm_verify_tie(int32_t low_id, int32_t high_id) {
    next_verify_tie_ = true;
    tie_low_ = low_id;
    tie_high_ = high_id;
}

SpeculativeRoundState SpecRoundTestEngine::round_state() {
    SpeculativeRoundState s;
    s.draft = &draft_;
    s.use_ckpt_tgt = &use_ckpt_tgt_;
    s.use_ckpt_dft = &use_ckpt_dft_;
    s.n_past = &loop_n_past_;
    s.id_last = &id_last_;
    s.prompt_tgt = &prompt_tgt_;
    s.tokens = &tokens_;
    s.last_step_drafts = &last_step_drafts_;
    s.last_step_hits = &last_step_hits_;
    return s;
}

Emit SpecRoundTestEngine::emit_frontier() const {
    const int32_t token = id_last_ * 31 + static_cast<int32_t>(prompt_tgt_.size()) + 601;
    Emit e;
    e.token = token % 9000 + 1000;
    for (int i = 0; i < kTopK; ++i) {
        const int32_t id = (i == 0) ? e.token : (e.token + i) % vocab_size_;
        e.top.push_back({id, 10.0f - static_cast<float>(i)});
    }
    return e;
}

Emit SpecRoundTestEngine::emit_at_verify_index(int idx) const {
    if (next_verify_tie_ && idx == 0) {
        std::vector<float> logits(static_cast<size_t>(vocab_size_), -20.0f);
        logits[static_cast<size_t>(tie_low_)] = 5.0f;
        logits[static_cast<size_t>(tie_high_)] = 5.0f;
        return top_from_logits(logits.data(), vocab_size_);
    }
    Emit e = emit_frontier();
    e.token = id_last_ + static_cast<int32_t>(idx) + 11;
    if (e.top.empty()) {
        e.top.push_back({e.token, 9.0f});
    } else {
        e.top[0] = {e.token, e.top[0].second};
    }
    return e;
}

std::vector<int32_t> SpecRoundTestEngine::make_full_draft(bool reject_at_one) const {
    std::vector<int32_t> draft(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        draft[static_cast<size_t>(i)] = id_last_ + static_cast<int32_t>(i) + 11;
        if (reject_at_one && i == 1) {
            draft[static_cast<size_t>(i)] = -1;
        }
    }
    return draft;
}

std::vector<int32_t> SpecRoundTestEngine::greedy_sample_and_accept(const std::vector<int32_t> &draft) const {
    std::vector<int32_t> result;
    result.reserve(draft.size() + 1);
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        const int32_t id = emit_at_verify_index(static_cast<int>(i)).token;
        result.push_back(id);
        if (draft[static_cast<size_t>(i)] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        result.push_back(emit_at_verify_index(static_cast<int>(i)).token);
    }
    return result;
}

std::vector<Emit> SpecRoundTestEngine::step() {
    if (tokens_.empty()) {
        throw std::runtime_error("step with no live request");
    }
    if (depth_ == 0) {
        last_step_drafts_ = 0;
        last_step_hits_ = 0;
        last_round_attempts_.clear();
        Emit e = emit_frontier();
        id_last_ = e.token;
        tokens_.push_back(e.token);
        return {e};
    }
    last_round_attempts_.clear();
    return run_speculative_round(round_state(), *backend_);
}

std::vector<Emit> SpecRoundTestEngine::start(const std::vector<int32_t> &tokens) {
    if (static_cast<int>(tokens.size()) + depth_ + 2 > ctx_size_) {
        throw std::runtime_error("context full");
    }
    starts++;
    abort();
    tokens_ = tokens;
    prompt_tgt_ = tokens;
    id_last_ = tokens.back();
    loop_n_past_ = static_cast<int>(prompt_tgt_.size());
    draft_.clear();
    reject_next_full_draft_at_one_ = true;
    Emit first = emit_frontier();
    id_last_ = first.token;
    tokens_.push_back(first.token);
    return {first};
}

std::vector<Emit> SpecRoundTestEngine::decode_forced(int32_t token) {
    if (tokens_.empty()) {
        throw std::runtime_error("decode_forced with no live request");
    }
    tokens_.push_back(token);
    if (depth_ > 0) {
        prompt_tgt_.push_back(token);
        loop_n_past_ = static_cast<int>(prompt_tgt_.size());
    }
    id_last_ = token;
    Emit e = emit_frontier();
    id_last_ = e.token;
    tokens_.push_back(e.token);
    return {e};
}

void SpecRoundTestEngine::abort() {
    tokens_.clear();
    prompt_tgt_.clear();
    draft_.clear();
    id_last_ = 0;
    loop_n_past_ = 0;
    last_step_drafts_ = 0;
    last_step_hits_ = 0;
    last_round_attempts_.clear();
    reject_next_full_draft_at_one_ = false;
    next_verify_tie_ = false;
    last_trim_kv_n_past_ = -1;
    trim_kv_tail_calls_ = 0;
}
