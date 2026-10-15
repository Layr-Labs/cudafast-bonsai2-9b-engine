#include "session.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;

Session::Session(std::unique_ptr<IEngine> engine, bool spec_run, int64_t load_epoch)
    : engine_(std::move(engine)), spec_run_offered_(spec_run), load_epoch_(load_epoch) {
    invalidate();
}

void Session::invalidate() {
    engine_->abort();
    ctx_.clear();
    frontier_ = nullptr;
    frontier_storage_ = {};
    ahead_.clear();
    drafts_ = 0;
    hits_ = 0;
    bounded_ = false;
}

const Emit &Session::need_frontier() const {
    if (!frontier_) {
        throw RefusedError("no live sequence; sync first");
    }
    return *frontier_;
}

std::vector<Emit> Session::engine_step() {
    auto emitted = engine_->step();
    if (emitted.empty()) {
        throw RefusedError("the engine step emitted nothing");
    }
    if (engine_->depth() > 0) {
        const auto [d, h] = engine_->last_spec_step_counters();
        drafts_ += d;
        hits_ += h;
    }
    return emitted;
}

void Session::check_ids(const std::vector<int32_t> &tokens) const {
    for (const int32_t t : tokens) {
        if (t < 0 || t >= engine_->vocab_size()) {
            throw RefusedError("token is outside the vocabulary");
        }
    }
}

int32_t Session::sync(const std::vector<int32_t> &tokens, bool frontier_only) {
    if (tokens.empty()) {
        throw RefusedError("sync needs a non-empty token list");
    }
    check_ids(tokens);
    auto emitted = frontier_only ? engine_->start_frontier_only(tokens) : engine_->start(tokens);
    bounded_ = frontier_only;
    ctx_ = tokens;
    frontier_storage_ = emitted.front();
    frontier_ = &frontier_storage_;
    ahead_.clear();
    for (size_t i = 1; i < emitted.size(); ++i) {
        ahead_.push_back(emitted[i]);
    }
    return frontier_->token;
}

void Session::resync_if_bounded() {
    if (bounded_) {
        const std::vector<int32_t> again = ctx_;
        sync(again);
    }
}

void Session::feed(int32_t token) {
    resync_if_bounded();
    const Emit &frontier = need_frontier();
    if (token != frontier.token) {
        if (!ahead_.empty()) {
            std::vector<int32_t> reprefill = ctx_;
            reprefill.push_back(token);
            sync(reprefill);
            return;
        }
        ctx_.push_back(token);
        auto emitted = engine_->decode_forced(token);
        if (emitted.empty()) {
            throw RefusedError("the engine forced decode emitted nothing");
        }
        frontier_storage_ = emitted.front();
        frontier_ = &frontier_storage_;
        ahead_.clear();
        return;
    }
    ctx_.push_back(token);
    if (ahead_.empty()) {
        auto more = engine_step();
        for (const auto &e : more) {
            ahead_.push_back(e);
        }
    }
    frontier_storage_ = ahead_.front();
    frontier_ = &frontier_storage_;
    ahead_.pop_front();
}

int32_t Session::eval(int32_t token) {
    check_ids({token});
    feed(token);
    return need_frontier().token;
}

std::vector<int32_t> Session::cycle(int32_t fed, int budget) {
    if (budget <= 0) {
        throw RefusedError("a cycle needs a positive budget");
    }
    check_ids({fed});
    resync_if_bounded();
    const Emit &frontier = need_frontier();
    if (fed != frontier.token) {
        if (!ahead_.empty()) {
            std::vector<int32_t> reprefill = ctx_;
            reprefill.push_back(fed);
            sync(reprefill);
            return {fed};
        }
        feed(fed);
        return {fed};
    }
    if (ahead_.empty()) {
        auto more = engine_step();
        for (const auto &e : more) {
            ahead_.push_back(e);
        }
    }
    const size_t m = std::min(ahead_.size(), static_cast<size_t>(budget));
    std::vector<Emit> taken;
    for (size_t i = 0; i < m; ++i) {
        taken.push_back(ahead_.front());
        ahead_.pop_front();
    }
    std::vector<int32_t> committed;
    committed.push_back(fed);
    for (size_t i = 0; i + 1 < taken.size(); ++i) {
        committed.push_back(taken[i].token);
    }
    ctx_.insert(ctx_.end(), committed.begin(), committed.end());
    frontier_storage_ = taken.back();
    frontier_ = &frontier_storage_;
    return committed;
}

std::pair<std::vector<std::vector<int32_t>>, std::vector<int32_t>> Session::spec_run(int32_t first,
                                                                                      int count) {
    std::vector<std::vector<int32_t>> rounds;
    std::vector<int32_t> frontiers;
    int32_t pending = first;
    int produced = 0;
    while (produced < count) {
        auto committed = cycle(pending, count - produced);
        rounds.push_back(committed);
        pending = need_frontier().token;
        frontiers.push_back(pending);
        produced += static_cast<int>(committed.size());
    }
    return {rounds, frontiers};
}

std::pair<std::vector<int32_t>, std::vector<float>> Session::top_logits(int k) {
    const auto &top = need_frontier().top;
    if (static_cast<int>(top.size()) < k) {
        throw RefusedError("the engine returned fewer top logits than requested");
    }
    std::vector<int32_t> ids;
    std::vector<float> logits;
    for (int i = 0; i < k; ++i) {
        ids.push_back(top[i].first);
        logits.push_back(top[i].second);
    }
    return {ids, logits};
}

HelloInfo Session::hello() const {
    HelloInfo h;
    h.vocab_size = engine_->vocab_size();
    h.eos_token = engine_->eos_token();
    h.spec_mode = engine_->spec_mode();
    h.mtp_armed = h.spec_mode == "mtp";
    h.draft_tokens = engine_->depth() + 1;
    h.ctx_size = engine_->ctx_size();
    h.load_epoch = load_epoch_;
    h.spec_run = spec_run_offered_;
    h.ident = engine_->ident();
    h.model_path = engine_->model_path();
    h.mtp_head_path = h.mtp_armed ? engine_->model_path() : "";
    h.draft_model_path = engine_->draft_model_path();
    return h;
}

static int64_t json_int(const json &req, const char *name, const std::string &op) {
    if (!req.contains(name) || !req[name].is_number_integer()) {
        throw RefusedError(op + " needs a \"" + name + "\" integer");
    }
    return req[name].get<int64_t>();
}

static int32_t json_i32_in_range(const json &req, const char *name, const std::string &op, int32_t lo, int32_t hi) {
    const int64_t v = json_int(req, name, op);
    if (v < lo || v > hi) {
        throw RefusedError(op + " needs a \"" + name + "\" integer in " + std::to_string(lo) + ".." +
                           std::to_string(hi));
    }
    return static_cast<int32_t>(v);
}

static int32_t json_vocab_token(const json &req, const char *name, const std::string &op, int32_t vocab_size) {
    return json_i32_in_range(req, name, op, 0, vocab_size - 1);
}

bool Session::handle_op(const std::string &op, const void *json_req, std::string &reply_json) {
    const json &req = *static_cast<const json *>(json_req);
    json reply;
    reply["ok"] = true;

    if (op == "hello") {
        const HelloInfo h = hello();
        reply["vocab_size"] = h.vocab_size;
        reply["eos_token"] = h.eos_token;
        reply["mtp_armed"] = h.mtp_armed;
        reply["spec_mode"] = h.spec_mode;
        reply["draft_tokens"] = h.draft_tokens;
        reply["ctx_size"] = h.ctx_size;
        reply["load_epoch"] = h.load_epoch;
        reply["spec_run"] = h.spec_run;
        reply["ident"] = h.ident;
        reply["model_path"] = h.model_path;
        reply["mtp_head_path"] = h.mtp_head_path;
        reply["draft_model_path"] = h.draft_model_path;
    } else if (op == "invalidate") {
        invalidate();
    } else if (op == "sync") {
        if (!req.contains("tokens") || !req["tokens"].is_array()) {
            throw RefusedError("sync needs a \"tokens\" integer array");
        }
        std::vector<int32_t> tokens;
        for (const auto &v : req["tokens"]) {
            if (!v.is_number_integer()) {
                throw RefusedError("sync needs a \"tokens\" integer array");
            }
            const int64_t raw = v.get<int64_t>();
            if (raw < 0 || raw > std::numeric_limits<int32_t>::max()) {
                throw RefusedError("sync needs a \"tokens\" integer array");
            }
            tokens.push_back(static_cast<int32_t>(raw));
        }
        // Optional "max_tokens": 1 means the caller wants the frontier token
        // and nothing after it (a timed prefill). Any other positive value is
        // an unbounded sync.
        bool frontier_only = false;
        if (req.contains("max_tokens")) {
            frontier_only = json_i32_in_range(req, "max_tokens", op, 1, std::numeric_limits<int32_t>::max()) == 1;
        }
        reply["token"] = sync(tokens, frontier_only);
    } else if (op == "eval") {
        reply["token"] = eval(json_vocab_token(req, "token", op, engine_->vocab_size()));
    } else if (op == "argmax") {
        reply["token"] = need_frontier().token;
    } else if (op == "top_logits") {
        const int k = json_i32_in_range(req, "k", op, 1, kMaxTopK);
        auto [ids, logits] = top_logits(k);
        reply["ids"] = ids;
        reply["logits"] = logits;
    } else if (op == "eval_speculative") {
        auto committed = cycle(json_vocab_token(req, "first_token", op, engine_->vocab_size()),
                               json_i32_in_range(req, "budget", op, 1, kMaxRun));
        reply["tokens"] = committed;
        reply["token"] = need_frontier().token;
    } else if (op == "spec_run") {
        if (!spec_run_offered_) {
            throw RefusedError("spec_run is not offered");
        }
        const int count = json_i32_in_range(req, "count", op, 1, kMaxRun);
        auto [rounds, frontiers] =
            spec_run(json_vocab_token(req, "first_token", op, engine_->vocab_size()), count);
        reply["rounds"] = rounds;
        reply["frontiers"] = frontiers;
        reply["token"] = need_frontier().token;
    } else if (op == "spec_counters") {
        reply["drafts"] = drafts_;
        reply["hits"] = hits_;
        reply["quenches"] = 0;
        reply["disagreements"] = 0;
    } else if (op == "bye") {
        reply_json = reply.dump();
        return false;
    } else {
        throw RefusedError("unknown op \"" + op + "\"");
    }

    reply_json = reply.dump();
    return true;
}
