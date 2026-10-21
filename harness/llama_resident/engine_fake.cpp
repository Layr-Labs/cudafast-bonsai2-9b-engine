#include "engine_fake.hpp"

#include <stdexcept>

FakeEngine::FakeEngine(int draft_depth, int32_t vocab, int32_t ctx, const std::string &spec_mode,
                       const std::string &draft_model_path)
    : depth_(draft_depth), vocab_size_(vocab), ctx_size_(ctx),
      ident_("fake depth=" + std::to_string(draft_depth)), model_path_("/fake/model"), spec_mode_(spec_mode),
      draft_model_path_(draft_model_path) {}

int32_t FakeEngine::next_token(const std::vector<int32_t> &ctx) const {
    const int32_t a = ctx.empty() ? 0 : ctx.back();
    const int32_t b = ctx.size() > 1 ? ctx[ctx.size() - 2] : 0;
    return (a * 31 + b * 17 + 7) % 1000 + 20;
}

Emit FakeEngine::emit_at(const std::vector<int32_t> &ctx) const {
    const int32_t token = next_token(ctx);
    Emit e;
    e.token = token;
    for (int i = 0; i < kTopK; ++i) {
        const int32_t id = (i == 0) ? token : (token + i) % vocab_size_;
        e.top.push_back({id, 10.0f - static_cast<float>(i)});
    }
    return e;
}

std::vector<Emit> FakeEngine::start(const std::vector<int32_t> &tokens) {
    if (static_cast<int>(tokens.size()) + depth_ + 2 > ctx_size_) {
        throw std::runtime_error("context full");
    }
    starts++;
    ctx_ = tokens;
    cycle_ = 0;
    const Emit first = emit_at(ctx_);
    ctx_.push_back(first.token);
    return {first};
}

std::vector<Emit> FakeEngine::start_frontier_only(const std::vector<int32_t> &tokens) {
    if (static_cast<int>(tokens.size()) + depth_ + 2 > ctx_size_) {
        throw std::runtime_error("context full");
    }
    frontier_starts++;
    // Like the llama engine at a depth: the frontier, and no live request to
    // step from.
    ctx_.clear();
    return {emit_at(tokens)};
}

std::vector<Emit> FakeEngine::step() {
    if (ctx_.empty()) {
        throw std::runtime_error("step with no live request");
    }
    cycle_++;
    const int n = depth_ ? 1 + (cycle_ % (depth_ + 1)) : 1;
    last_step_drafts_ = depth_ > 0 ? depth_ : 0;
    last_step_hits_ = depth_ > 0 ? static_cast<int64_t>(n) - 1 : 0;
    std::vector<Emit> out;
    for (int i = 0; i < n; ++i) {
        const Emit e = emit_at(ctx_);
        ctx_.push_back(e.token);
        out.push_back(e);
    }
    return out;
}

std::vector<Emit> FakeEngine::decode_forced(int32_t token) {
    if (ctx_.empty()) {
        throw std::runtime_error("decode_forced with no live request");
    }
    if (!ctx_.empty()) {
        ctx_.pop_back();
    }
    ctx_.push_back(token);
    const Emit e = emit_at(ctx_);
    ctx_.push_back(e.token);
    return {e};
}

void FakeEngine::abort() { ctx_.clear(); }
