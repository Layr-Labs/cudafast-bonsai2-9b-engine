#include "engine_fake.hpp"
#include "engine_spec_round_test.hpp"
#include "logits_rank.hpp"
#include "session.hpp"
#include "wire.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <random>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static const std::vector<int32_t> kPrompt = [] {
    std::vector<int32_t> p;
    for (int i = 100; i < 164; ++i) {
        p.push_back(i);
    }
    return p;
}();

static std::vector<int32_t> serial_stream(int n) {
    Session s(std::make_unique<FakeEngine>(0));
    std::vector<int32_t> out;
    int32_t t = s.sync(kPrompt);
    for (int i = 0; i < n; ++i) {
        t = s.eval(t);
        out.push_back(t);
    }
    return out;
}

static void expect(bool ok, const char *msg) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void test_serial_eval_advances_one_token() {
    Session s(std::make_unique<FakeEngine>(0));
    const int32_t first = s.sync(kPrompt);
    expect(s.eval(first) == s.frontier()->token, "eval advances frontier");
    expect(s.ctx().size() == kPrompt.size() + 1, "ctx length");
}

static void test_spec_run_equals_serial_at_every_depth() {
    const auto want = serial_stream(128);
    for (int depth = 1; depth <= kImplementedDepth; ++depth) {
        Session s(std::make_unique<FakeEngine>(depth));
        const int32_t seed = s.sync(kPrompt);
        auto [rounds, frontiers] = s.spec_run(seed, 128);
        std::vector<int32_t> got;
        for (size_t i = 0; i < rounds.size(); ++i) {
            for (size_t j = 1; j < rounds[i].size(); ++j) {
                got.push_back(rounds[i][j]);
            }
            got.push_back(frontiers[i]);
        }
        expect(got == want, "spec_run matches serial stream");
        int total = 0;
        for (const auto &r : rounds) {
            total += static_cast<int>(r.size());
        }
        expect(total == 128, "committed count");
        expect(s.drafts() > 0, "drafts counted");
    }
}

static void test_spec_run_never_overshoots() {
    for (int count : {1, 2, 3, 7, 129}) {
        Session s(std::make_unique<FakeEngine>(6));
        const int32_t seed = s.sync(kPrompt);
        auto [rounds, frontiers] = s.spec_run(seed, count);
        int total = 0;
        for (const auto &r : rounds) {
            total += static_cast<int>(r.size());
        }
        expect(total == count, "no overshoot");
    }
}

static void test_overhang_consumed() {
    const auto want = serial_stream(20);
    Session s(std::make_unique<FakeEngine>(6));
    int32_t pending = s.sync(kPrompt);
    std::vector<int32_t> got;
    for (int count : {3, 5, 12}) {
        auto [rounds, frontiers] = s.spec_run(pending, count);
        for (size_t i = 0; i < rounds.size(); ++i) {
            for (size_t j = 1; j < rounds[i].size(); ++j) {
                got.push_back(rounds[i][j]);
            }
            got.push_back(frontiers[i]);
        }
        pending = frontiers.back();
    }
    expect(got == want, "overhang");
}

static int32_t fake_next_token(const std::vector<int32_t> &ctx) {
    const int32_t a = ctx.empty() ? 0 : ctx.back();
    const int32_t b = ctx.size() > 1 ? ctx[ctx.size() - 2] : 0;
    return (a * 31 + b * 17 + 7) % 1000 + 20;
}

static void test_teacher_forcing_incremental() {
    auto backend = std::make_unique<FakeEngine>(0);
    FakeEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t first = s.sync(kPrompt);
    const int32_t forced = (first + 1) % 1000 + 20;
    s.eval(forced);
    std::vector<int32_t> want = kPrompt;
    want.push_back(forced);
    expect(s.ctx() == want, "teacher ctx");
    expect(raw->starts == 1, "incremental teacher force (no reprefill)");
}

static void test_teacher_forcing_reprefill_with_ahead() {
    auto backend = std::make_unique<FakeEngine>(4);
    FakeEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t seed = s.sync(kPrompt);
    s.cycle(seed, 1);
    expect(raw->starts == 1, "ahead primed without reprefill");
    const int32_t forced = seed + 77;
    s.eval(forced);
    expect(raw->starts == 2, "teacher force with ahead reprefills");
}

static void test_fake_forced_decode_chain() {
    Session s(std::make_unique<FakeEngine>(0));
    const int32_t first = s.sync(kPrompt);
    const int32_t forced = 855;
    const int32_t frontier = s.eval(forced);
    std::vector<int32_t> ctx = kPrompt;
    ctx.push_back(forced);
    expect(frontier == fake_next_token(ctx), "forced frontier from ctx");
    const int32_t next = s.eval(frontier);
    ctx.push_back(frontier);
    expect(next == fake_next_token(ctx), "normal eval after force");
    expect(next != frontier, "forced decode does not repeat frontier");
}

static void test_fake_consecutive_forced_tokens() {
    Session s(std::make_unique<FakeEngine>(0));
    const int32_t first = s.sync(kPrompt);
    const int32_t f1 = 855;
    const int32_t t1 = s.eval(f1);
    const int32_t f2 = t1 + 111;
    expect(f2 != first && f2 != t1, "second force differs from frontier and first forced frontier");
    const int32_t t2 = s.eval(f2);
    std::vector<int32_t> ctx = kPrompt;
    ctx.push_back(f1);
    expect(t1 == fake_next_token(ctx), "first forced frontier");
    ctx.push_back(f2);
    expect(t2 == fake_next_token(ctx), "second forced frontier");
}

static void test_spec_counters_one_mtp_step() {
    Session s(std::make_unique<FakeEngine>(4));
    const int32_t seed = s.sync(kPrompt);
    s.cycle(seed, 1);
    expect(s.drafts() == 4, "draft counter per verify attempt");
    expect(s.hits() == 1, "hit counter from partial multi-emit step");
}

static void test_spec_run_committed_only() {
    Session s(std::make_unique<FakeEngine>(6));
    const int32_t seed = s.sync(kPrompt);
    auto [rounds, frontiers] = s.spec_run(seed, 11);
    int total = 0;
    for (const auto &r : rounds) {
        total += static_cast<int>(r.size());
    }
    expect(total == 11, "spec_run committed count after internal retries");
    expect(frontiers.size() == rounds.size(), "frontier per round");
}

static void test_spec_round_greedy_accept_reuses_backend_buffer() {
    SpecRoundTestEngine eng(4);
    eng.start(kPrompt);
    const int32_t marker = 424242;
    // The fake writes the marker through the address it returned from greedy_accept, after the
    // return and before the emits. A loop that copied the ids would commit the original token.
    eng.overwrite_last_accepted_id_after_return(marker);
    const auto out = eng.step();
    expect(eng.greedy_accept_returned_addr() != nullptr, "fake recorded the greedy_accept result address");
    expect(!out.empty() && eng.tokens().back() == marker,
           "loop reads ids from the object greedy_accept returned (no per-attempt copy)");
}

static void test_spec_round_sampler_restored_on_partial_accept() {
    SpecRoundTestEngine eng(4);
    eng.start(kPrompt);
    eng.step();
    expect(eng.last_round_attempts().size() == 2, "partial accept then retry");
    expect(eng.sampler_clones_created() == 2, "one sampler clone per verify attempt");
    expect(eng.sampler_restores() == 1, "partial accept restores the cloned sampler");
    expect(eng.sampler_clones_freed() == 2, "every sampler clone freed after the round");
}

static void test_spec_round_no_target_checkpoint_accepts_partial() {
    SpecRoundTestEngine eng(4);
    eng.set_use_ckpt_tgt(false);
    const auto first = eng.start(kPrompt);
    const int32_t seed = first.at(0).token;
    const auto out = eng.step();
    expect(eng.last_round_attempts().size() == 1, "no target checkpoint: one verify attempt");
    expect(eng.sampler_clones_created() == 0, "no target checkpoint: no sampler clone");
    expect(eng.sampler_restores() == 0, "no target checkpoint: no sampler restore");
    expect(eng.rollbacks() == 0, "no target checkpoint: no rollback");
    const std::vector<int32_t> want = {seed + 11, seed + 12};
    std::vector<int32_t> got;
    for (const auto &e : out) {
        got.push_back(e.token);
    }
    expect(got == want, "no target checkpoint: partial accept returns the accepted tokens");
}

static void test_spec_round_sampler_clone_freed_on_throw() {
    SpecRoundTestEngine eng(4);
    eng.start(kPrompt);
    eng.throw_in_next_greedy_accept();
    bool threw = false;
    try {
        eng.step();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect(threw, "greedy_accept failure propagates out of the round");
    expect(eng.sampler_clones_created() == 1, "sampler cloned before the throw");
    expect(eng.sampler_clones_freed() == 1, "sampler clone freed when the round throws");
    expect(eng.sampler_restores() == 0, "no restore on throw");
}

static void test_mtp_partial_rejection_retry_counters() {
    auto backend = std::make_unique<SpecRoundTestEngine>(4);
    SpecRoundTestEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t seed = s.sync(kPrompt);
    s.cycle(seed, 1);
    expect(s.drafts() == 8, "drafts accumulate both verify attempts");
    expect(s.hits() == 3, "hits sum partial accept then prefix reuse accept");
    const auto &attempts = raw->last_round_attempts();
    expect(attempts.size() == 2, "two verify attempts in one step");
    expect(attempts[0].drafts == 4 && attempts[0].hits == 1, "first attempt partial at index 1");
    expect(attempts[1].drafts == 4 && attempts[1].hits == 2, "retry reuses accepted prefix draft");
}

static void test_mtp_partial_rejection_emits_only_committed() {
    auto backend = std::make_unique<SpecRoundTestEngine>(4);
    SpecRoundTestEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t seed = s.sync(kPrompt);
    const int32_t rejected_draft_id = -1;
    // Literal expectations from the fake's target rule (next = previous + 11 + index): the
    // accepted prefix plus the correction, and nothing from the rejected draft.
    const std::vector<int32_t> want = {seed, seed + 11, seed + 12};
    const auto committed = s.cycle(seed, 8);
    expect(committed == want, "partial retry commits prefix plus correction only");
    expect(std::find(committed.begin(), committed.end(), rejected_draft_id) == committed.end(),
           "invalid draft slot never committed");
    expect(s.frontier()->token == seed + 13, "frontier after partial retry");
    expect(raw->last_round_attempts().size() == 2, "internal retry during step");
}

static void test_mtp_frontier_matches_top_logits_with_tie() {
    auto backend = std::make_unique<SpecRoundTestEngine>(3);
    SpecRoundTestEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t seed = s.sync(kPrompt);
    raw->arm_verify_tie(17, 42);
    s.cycle(seed, 1);
    auto [ids, logits] = s.top_logits(8);
    expect(ids[0] == 17, "depth>0 verify tie picks lower id via top_from_logits");
    expect(logits[0] == logits[1], "tied logits at verify frontier");
    expect(s.frontier()->token == 17, "frontier equals lower-id argmax after verify");
    expect(seed != 17, "sync frontier unrelated to verify tie");
}

static void test_mtp_teacher_force_prompt_survives_verify_rollback() {
    auto backend = std::make_unique<SpecRoundTestEngine>(4);
    SpecRoundTestEngine *raw = backend.get();
    Session s(std::move(backend));
    const int32_t seed = s.sync(kPrompt);
    const int32_t forced = seed + 77;
    s.eval(forced);
    const std::vector<int32_t> prompt_after_force = raw->prompt_tgt();
    const size_t forced_index = prompt_after_force.size() - 1;
    expect(prompt_after_force[forced_index] == forced, "forced token at prompt tail");
    const int kv_after_force = raw->sim_tgt_kv_pos();
    const int32_t frontier = s.frontier()->token;
    s.cycle(frontier, 1);
    const auto &after_step = raw->prompt_tgt();
    expect(after_step.size() >= prompt_after_force.size(), "verify accept may extend prompt after rollback");
    for (size_t i = 0; i < prompt_after_force.size(); ++i) {
        expect(after_step[i] == prompt_after_force[i], "rollback keeps teacher-forced prefix intact");
    }
    expect(after_step[forced_index] == forced, "forced token kept at original index");
    expect(std::find(after_step.begin(), after_step.end(), -1) == after_step.end(),
           "rejected verify draft pollution rolled back");
    const int expect_kv = kv_after_force + 3;
    expect(raw->last_trim_kv_n_past() == expect_kv,
           "trim_kv_tail receives loop n_past after accept increment (not fake KV math)");
    expect(raw->trim_kv_tail_calls() == 1, "one tail trim on full accept after partial retry");
    expect(s.frontier() != nullptr, "retry exposes frontier after rollback");
}

static void test_invalidate_resets_session() {
    Session s(std::make_unique<FakeEngine>(3));
    const int32_t seed = s.sync(kPrompt);
    s.cycle(seed, 2);
    expect(s.drafts() > 0, "drafts before invalidate");
    s.invalidate();
    expect(s.drafts() == 0, "drafts cleared");
    expect(s.hits() == 0, "hits cleared");
    expect(s.frontier() == nullptr, "frontier cleared");
    try {
        s.eval(seed);
        expect(false, "eval after invalidate");
    } catch (const RefusedError &) {
    }
}

static void test_frontier_only_sync_then_feed_syncs_again() {
    for (int depth = 0; depth <= kImplementedDepth; ++depth) {
        Session want(std::make_unique<FakeEngine>(depth));
        const int32_t want_seed = want.sync(kPrompt);
        auto [want_rounds, want_frontiers] = want.spec_run(want_seed, 32);

        auto engine = std::make_unique<FakeEngine>(depth);
        FakeEngine *fake = engine.get();
        Session s(std::move(engine));
        const int32_t seed = s.sync(kPrompt, /*frontier_only*/ true);
        expect(seed == want_seed, "a frontier-only sync gives the full sync's frontier");
        expect(fake->frontier_starts == 1 && fake->starts == 0, "a frontier-only sync runs start_frontier_only once");
        auto [rounds, frontiers] = s.spec_run(seed, 32);
        expect(fake->starts == 1, "feeding a bounded sequence syncs its context again");
        expect(rounds == want_rounds && frontiers == want_frontiers, "the stream after a bounded sync is the full sync's");
    }
}

static void test_sync_max_tokens_on_the_wire() {
    auto engine = std::make_unique<FakeEngine>(3);
    FakeEngine *fake = engine.get();
    Session s(std::move(engine));
    std::string reply;
    nlohmann::json req = {{"op", "sync"}, {"tokens", kPrompt}, {"max_tokens", 1}};
    s.handle_op("sync", &req, reply);
    expect(fake->frontier_starts == 1 && fake->starts == 0, "sync max_tokens 1 is frontier-only");
    req = {{"op", "sync"}, {"tokens", kPrompt}, {"max_tokens", 7}};
    s.handle_op("sync", &req, reply);
    expect(fake->frontier_starts == 1 && fake->starts == 1, "sync max_tokens 7 is a full sync");
    req = {{"op", "sync"}, {"tokens", kPrompt}};
    s.handle_op("sync", &req, reply);
    expect(fake->starts == 2, "sync without max_tokens is a full sync");
    for (const nlohmann::json &bad : {nlohmann::json(0), nlohmann::json(-1), nlohmann::json("1")}) {
        req = {{"op", "sync"}, {"tokens", kPrompt}, {"max_tokens", bad}};
        try {
            s.handle_op("sync", &req, reply);
            expect(false, "sync with a bad max_tokens");
        } catch (const RefusedError &) {
        }
    }
}

// Reference copy of the pre-single-pass top_from_logits (unsorted buffer + per-entry worst scan).
// One deliberate difference: the old code passed best.end() to partial_sort, so for
// n_vocab < kTopK it sorted the unset zero entries in with the real ones and could return a
// fake {id 0, logit 0} entry. This copy sorts only the first k, which is the intended result.
static Emit top_from_logits_reference(const float *logits, int n_vocab) {
    const int k = std::min<int>(kTopK, n_vocab);
    std::array<std::pair<float, int32_t>, kTopK> best{};
    for (int i = 0; i < k; ++i) {
        best[i] = {logits[i], i};
    }
    for (int i = k; i < n_vocab; ++i) {
        int worst = 0;
        for (int j = 1; j < k; ++j) {
            if (logit_id_before(best[worst].first, best[worst].second, best[j].first, best[j].second)) {
                worst = j;
            }
        }
        const float li = logits[i];
        if (logit_id_before(li, i, best[worst].first, best[worst].second)) {
            best[worst] = {li, i};
        }
    }
    std::sort(best.begin(), best.begin() + k,
              [](const auto &a, const auto &b) { return logit_id_before(a.first, a.second, b.first, b.second); });
    Emit e;
    e.token = best[0].second;
    for (int i = 0; i < k; ++i) {
        e.top.push_back({best[i].second, best[i].first});
    }
    return e;
}

static bool emit_equal(const Emit &a, const Emit &b) {
    if (a.token != b.token || a.top.size() != b.top.size()) {
        return false;
    }
    for (size_t i = 0; i < a.top.size(); ++i) {
        if (a.top[i].first != b.top[i].first || a.top[i].second != b.top[i].second) {
            return false;
        }
    }
    return true;
}

static void test_top_from_logits_matches_reference() {
    const int full_vocab = 248320;
    std::mt19937 rng(0xC0FFEE42u);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    const auto check = [&](const std::vector<float> &logits) {
        const int n = static_cast<int>(logits.size());
        const Emit got = top_from_logits(logits.data(), n);
        const Emit want = top_from_logits_reference(logits.data(), n);
        expect(emit_equal(got, want), "top_from_logits matches reference");
    };

    for (int n_vocab : {1, 7, 8, 9, full_vocab}) {
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<float> logits(static_cast<size_t>(n_vocab));
            for (float &v : logits) {
                v = dist(rng);
            }
            check(logits);
        }
    }

    for (int n_vocab : {1, 7, 8, 9, 128, full_vocab}) {
        std::vector<float> logits(static_cast<size_t>(n_vocab), 3.0f);
        check(logits);
    }

    for (int n_vocab : {8, 64, full_vocab}) {
        std::vector<float> logits(static_cast<size_t>(n_vocab));
        for (int i = 0; i < n_vocab; ++i) {
            logits[static_cast<size_t>(i)] = static_cast<float>((i * 17) % 11);
        }
        check(logits);
    }

    {
        std::vector<float> logits(64, -std::numeric_limits<float>::infinity());
        logits[10] = 1.0f;
        logits[20] = 1.0f;
        logits[30] = 0.0f;
        check(logits);
    }

    // No NaN case: logit_id_before is not a strict order when NaN is present, so neither the old
    // nor the new code defines a result for NaN logits (llama.cpp does not produce them).

    // Below kTopK every entry is returned, best first, with no unset entries.
    for (int n_vocab : {1, 3, 7}) {
        std::vector<float> logits(static_cast<size_t>(n_vocab));
        for (int i = 0; i < n_vocab; ++i) {
            logits[static_cast<size_t>(i)] = -5.0f - static_cast<float>(i);
        }
        const Emit e = top_from_logits(logits.data(), n_vocab);
        expect(static_cast<int>(e.top.size()) == n_vocab, "top size equals n_vocab below kTopK");
        for (int i = 0; i < n_vocab; ++i) {
            expect(e.top[static_cast<size_t>(i)].first == i && e.top[static_cast<size_t>(i)].second == logits[static_cast<size_t>(i)],
                   "below kTopK the top list is the real entries in order");
        }
    }

    for (int extra = 0; extra < 200; ++extra) {
        const int n_vocab = 1 + static_cast<int>(rng() % 512u);
        std::vector<float> logits(static_cast<size_t>(n_vocab));
        const int tie_val = static_cast<int>(rng() % 7u);
        for (int i = 0; i < n_vocab; ++i) {
            if (rng() % 4u == 0u) {
                logits[static_cast<size_t>(i)] = static_cast<float>(tie_val);
            } else {
                logits[static_cast<size_t>(i)] = dist(rng);
            }
        }
        check(logits);
    }
}

static void test_invalidate_clears_queued_ahead() {
    Session s(std::make_unique<FakeEngine>(4));
    const int32_t seed = s.sync(kPrompt);
    s.cycle(seed, 1);
    expect(s.queued_ahead() == 1, "partial cycle leaves one emit queued");
    expect(s.frontier() != nullptr, "frontier still live before invalidate");
    s.invalidate();
    expect(s.queued_ahead() == 0, "invalidate clears ahead queue before sync");
    expect(s.ctx().empty(), "invalidate clears session context before sync");
    expect(s.frontier() == nullptr, "invalidate drops frontier");
    const int32_t fresh = s.sync(kPrompt);
    const std::vector<int32_t> want = serial_stream(4);
    std::vector<int32_t> got;
    int32_t t = fresh;
    for (int i = 0; i < 4; ++i) {
        t = s.eval(t);
        got.push_back(t);
    }
    expect(got == want, "fresh stream matches serial baseline after invalidate");
}

static void test_logit_tie_break_lower_id() {
    std::vector<float> logits(128, -10.0f);
    logits[42] = 5.0f;
    logits[17] = 5.0f;
    const Emit e = top_from_logits(logits.data(), static_cast<int>(logits.size()));
    expect(e.token == 17, "equal logits pick lower id");
    expect(e.top[0].first == 17, "top logits argmax tie");
}

static void test_frontier_matches_top_logits() {
    Session s(std::make_unique<FakeEngine>(0));
    const int32_t first = s.sync(kPrompt);
    auto [ids, logits] = s.top_logits(8);
    expect(ids[0] == first, "frontier equals top argmax");
    expect(s.frontier()->token == ids[0], "frontier token matches top_logits");
}

static void test_json_integer_range_refusals() {
    Session s(std::make_unique<FakeEngine>(0));
    s.sync(kPrompt);
    nlohmann::json req = {{"op", "eval"}, {"token", static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1}};
    std::string reply;
    try {
        s.handle_op("eval", &req, reply);
        expect(false, "eval int64 overflow");
    } catch (const RefusedError &) {
    }
    req = {{"op", "eval"}, {"token", -1}};
    try {
        s.handle_op("eval", &req, reply);
        expect(false, "eval negative token");
    } catch (const RefusedError &) {
    }
}

static void test_top_logits() {
    Session s(std::make_unique<FakeEngine>(0));
    const int32_t first = s.sync(kPrompt);
    auto [ids, logits] = s.top_logits(8);
    expect(ids[0] == first, "top logits argmax");
    expect(ids.size() == 8, "top k size");
    for (size_t i = 1; i < logits.size(); ++i) {
        expect(logits[i - 1] >= logits[i], "sorted logits");
    }
}

static void test_refusals() {
    Session s(std::make_unique<FakeEngine>(0));
    try {
        s.eval(5);
        expect(false, "eval without sync");
    } catch (const RefusedError &) {
    }
    try {
        s.top_logits(8);
        expect(false, "top_logits without sync");
    } catch (const RefusedError &) {
    }
}

static void test_out_of_vocab() {
    Session s(std::make_unique<FakeEngine>(0));
    try {
        s.sync({1, 2, 131072});
        expect(false, "oob sync");
    } catch (const RefusedError &) {
    }
}

static void test_hello_depth_convention() {
    HelloInfo h0 = Session(std::make_unique<FakeEngine>(0)).hello();
    expect(h0.draft_tokens == 1, "serial draft_tokens");
    HelloInfo h3 = Session(std::make_unique<FakeEngine>(3)).hello();
    expect(h3.draft_tokens == 4, "mtp draft_tokens");
    expect(h3.mtp_armed, "mtp armed");
    expect(h3.spec_mode == "mtp", "mtp spec_mode");
    expect(h3.draft_model_path.empty(), "mtp has no separate drafter");
}

static void test_hello_dflash() {
    HelloInfo h = Session(std::make_unique<FakeEngine>(15, 131072, 8192, "dflash", "/fake/dflash2.gguf")).hello();
    expect(h.spec_mode == "dflash", "dflash spec_mode");
    expect(!h.mtp_armed, "dflash does not arm the mtp drafter");
    expect(h.draft_tokens == 16, "dflash draft_tokens");
    expect(h.draft_model_path == "/fake/dflash2.gguf", "dflash draft_model_path");
    expect(h.mtp_head_path.empty(), "dflash has no mtp head path");
}

static std::ifstream open_adapter_session_backend() {
    std::vector<std::filesystem::path> candidates;
#ifdef ADAPTER_SESSION_BACKEND
    candidates.emplace_back(ADAPTER_SESSION_BACKEND);
#endif
    const auto exe = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    candidates.push_back(exe / ".." / ".." / "checkout" / "harness" / "protocol-adapter" / "src" /
                         "session_backend.rs");
    candidates.push_back(std::filesystem::path("harness/protocol-adapter/src/session_backend.rs"));
    for (const auto &path : candidates) {
        std::ifstream in(path);
        if (in) {
            return in;
        }
    }
    return std::ifstream();
}

static void test_implemented_depth_agrees_with_adapter() {
    std::ifstream in = open_adapter_session_backend();
    if (!in) {
        expect(false, "cannot open session_backend.rs");
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("pub const ENGINE_IMPLEMENTED_DEPTH") != std::string::npos) {
            const auto pos = line.rfind('=');
            const int depth = std::stoi(line.substr(pos + 1));
            expect(depth == kImplementedDepth, "implemented depth");
            return;
        }
    }
    expect(false, "ENGINE_IMPLEMENTED_DEPTH not found");
}

static std::string recv_line(int fd) {
    std::string reply;
    char ch = 0;
    while (read(fd, &ch, 1) == 1 && ch != '\n') {
        reply.push_back(ch);
    }
    return reply;
}

static std::string send_recv(int fd, const std::string &line) {
    const std::string out = line + "\n";
    expect(write(fd, out.data(), out.size()) == static_cast<ssize_t>(out.size()), "write");
    return recv_line(fd);
}

static void write_line(int fd, const std::string &line) {
    const std::string out = line + "\n";
    expect(write(fd, out.data(), out.size()) == static_cast<ssize_t>(out.size()), "write");
}

static void send_bytes_nosignal(int fd, const void *data, size_t len) {
    const char *buf = static_cast<const char *>(data);
    size_t off = 0;
    while (off < len) {
        const ssize_t n = send(fd, buf + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN) {
                usleep(1000);
                continue;
            }
            break;
        }
        if (n == 0) {
            break;
        }
        off += static_cast<size_t>(n);
    }
    expect(off == len, "full payload sent before expecting server close");
}

static bool peer_closed(int fd) {
    char ch = 0;
    const ssize_t n = read(fd, &ch, 1);
    return n == 0;
}

static void expect_peer_closed_no_reply(int fd, const char *msg, int max_wait_ms = 600000) {
    const int flags = fcntl(fd, F_GETFL, 0);
    expect(flags >= 0, "fcntl getfl");
    expect(fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0, "nonblocking read");
    ssize_t payload = 0;
    char buf[4096];
    int waited_ms = 0;
    while (waited_ms < max_wait_ms) {
        pollfd pfd {};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int pr = poll(&pfd, 1, 100);
        waited_ms += 100;
        if (pr < 0 && errno != EINTR) {
            break;
        }
        if (pr == 0) {
            continue;
        }
        if (pr < 0 && errno == EINTR) {
            continue;
        }
        if (pfd.revents & (POLLIN | POLLHUP | POLLERR)) {
            const ssize_t n = read(fd, buf, sizeof(buf));
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN) {
                    if (pfd.revents & POLLHUP) {
                        expect(payload == 0, msg);
                        return;
                    }
                    continue;
                }
                break;
            }
            if (n == 0) {
                expect(payload == 0, msg);
                return;
            }
            payload += n;
        }
    }
    expect(false, msg);
}

static int connect_resident(const std::string &sock) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock.c_str(), sizeof(addr.sun_path) - 1);
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        usleep(50000);
    }
    close(fd);
    return -1;
}

static pid_t spawn_fake_resident(const std::string &sock, const std::string &ready, const std::string &bin) {
    const pid_t child = fork();
    if (child == 0) {
        setenv("RESIDENT_BACKEND", "fake", 1);
        setenv("RESIDENT_SPEC_DEPTH", "2", 1);
        setenv("RESIDENT_SOCKET", sock.c_str(), 1);
        setenv("RESIDENT_READY_FILE", ready.c_str(), 1);
        execl(bin.c_str(), bin.c_str(), "--fake", "--socket", sock.c_str(), "--spec", "mtp", "--draft-len", "2",
              nullptr);
        _exit(1);
    }
    for (int i = 0; i < 400 && !std::filesystem::exists(ready); ++i) {
        usleep(50000);
    }
    expect(std::filesystem::exists(ready), "resident ready");
    return child;
}

static constexpr size_t kWireMaxRequestLine = 4 * 1024 * 1024;

static void test_wire_two_requests_one_write() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire pipelined test: llama-resident not built\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-pipe-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    const int fd = connect_resident(sock);
    expect(fd >= 0, "connect pipelined");
    const std::string batch = std::string(R"({"op":"hello"})") + "\n" + R"({"op":"hello"})" + "\n";
    expect(write(fd, batch.data(), batch.size()) == static_cast<ssize_t>(batch.size()), "pipelined write");
    const auto first = nlohmann::json::parse(recv_line(fd));
    const auto second = nlohmann::json::parse(recv_line(fd));
    expect(first["ok"].get<bool>(), "first pipelined hello");
    expect(second["ok"].get<bool>(), "second pipelined hello");
    close(fd);
    kill(child, SIGTERM);
}

static void test_wire_request_split_across_writes() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire split test: llama-resident not built\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-split-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    const int fd = connect_resident(sock);
    expect(fd >= 0, "connect split");
    const std::string req = R"({"op":"hello"})";
    expect(write(fd, req.data(), 5) == 5, "split write prefix");
    usleep(50000);
    expect(write(fd, req.data() + 5, req.size() - 5) == static_cast<ssize_t>(req.size() - 5), "split write suffix");
    expect(write(fd, "\n", 1) == 1, "split write newline");
    const auto reply = nlohmann::json::parse(recv_line(fd));
    expect(reply["ok"].get<bool>(), "split hello");
    close(fd);
    kill(child, SIGTERM);
}

static void test_wire_max_request_line_boundary() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire boundary test: llama-resident not built\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-boundary-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    const int fd = connect_resident(sock);
    expect(fd >= 0, "connect boundary");
    // Exactly the cap (the newline is not counted), with the newline in a separate write.
    const std::string max_line(kWireMaxRequestLine, 'x');
    send_bytes_nosignal(fd, max_line.data(), max_line.size());
    send_bytes_nosignal(fd, "\n", 1);
    const auto bad = nlohmann::json::parse(recv_line(fd));
    expect(!bad["ok"].get<bool>(), "max-sized line still gets a wire reply");
    const auto ok = nlohmann::json::parse(send_recv(fd, R"({"op":"hello"})"));
    expect(ok["ok"].get<bool>(), "hello after max-sized line");
    std::string over_line(kWireMaxRequestLine + 1, 'y');
    over_line.push_back('\n');
    send_bytes_nosignal(fd, over_line.data(), over_line.size());
    expect_peer_closed_no_reply(fd, "over-max line closes with no response");
    close(fd);
    kill(child, SIGTERM);
}

static void test_wire_phase_round_trip() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire test: llama-resident not built\n");
        return;
    }

    const pid_t child = spawn_fake_resident(sock, ready, bin);

    for (int phase = 0; phase < 2; ++phase) {
        const int fd = connect_resident(sock);
        expect(fd >= 0, "connect");
        auto hello = nlohmann::json::parse(send_recv(fd, R"({"op":"hello"})"));
        expect(hello["ok"].get<bool>(), "hello ok");
        expect(hello["draft_tokens"].get<int>() == 3, "depth wire");
        auto bad_argmax = nlohmann::json::parse(send_recv(fd, R"({"op":"argmax"})"));
        expect(!bad_argmax["ok"].get<bool>(), "argmax refused");
        auto sync = nlohmann::json::parse(send_recv(fd, "{\"op\":\"sync\",\"tokens\":" +
                                                           nlohmann::json(kPrompt).dump() + "}"));
        expect(sync["ok"].get<bool>(), "sync ok");
        const int seed = sync["token"].get<int>();
        auto run = nlohmann::json::parse(
            send_recv(fd, "{\"op\":\"spec_run\",\"first_token\":" + std::to_string(seed) + ",\"count\":32}"));
        expect(run["ok"].get<bool>(), "spec_run ok");
        int total = 0;
        for (const auto &r : run["rounds"]) {
            total += static_cast<int>(r.size());
        }
        expect(total == 32, "spec_run count");
        expect(run["frontiers"].back().get<int>() == run["token"].get<int>(), "frontier token");
        auto top = nlohmann::json::parse(send_recv(fd, R"({"op":"top_logits","k":8})"));
        expect(top["ids"].size() == 8, "top_logits");
        auto nope = nlohmann::json::parse(send_recv(fd, R"({"op":"nope"})"));
        expect(!nope["ok"].get<bool>(), "unknown op");
        auto bye = nlohmann::json::parse(send_recv(fd, R"({"op":"bye"})"));
        expect(bye["ok"].get<bool>(), "bye");
        close(fd);
    }
    kill(child, SIGTERM);
}

static void test_wire_oversize_line_and_survival() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire oversize test: llama-resident not built\n");
        return;
    }
    struct sigaction old_pipe {};
    struct sigaction ign_pipe {};
    ign_pipe.sa_handler = SIG_IGN;
    sigemptyset(&ign_pipe.sa_mask);
    sigaction(SIGPIPE, &ign_pipe, &old_pipe);
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-oversize-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    {
        const int fd = connect_resident(sock);
        expect(fd >= 0, "connect oversize");
        std::string huge(4 * 1024 * 1024 + 65, 'x');
        send_bytes_nosignal(fd, huge.data(), huge.size());
        expect_peer_closed_no_reply(fd, "oversize line closes with no response", 120000);
        close(fd);
    }
    {
        const int fd = connect_resident(sock);
        expect(fd >= 0, "connect after oversize");
        auto hello = nlohmann::json::parse(send_recv(fd, R"({"op":"hello"})"));
        expect(hello["ok"].get<bool>(), "hello after oversize");
        close(fd);
    }
    const int status = kill(child, 0);
    expect(status == 0, "server survived oversize client");
    kill(child, SIGTERM);
    sigaction(SIGPIPE, &old_pipe, nullptr);
}

static void test_wire_disconnect_mid_reply() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire disconnect test: llama-resident not built\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-disc-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    {
        const int fd = connect_resident(sock);
        expect(fd >= 0, "connect disconnect");
        int rcvbuf = 64;
        expect(setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) == 0, "shrink rcvbuf");
        write_line(fd, "{\"op\":\"sync\",\"tokens\":" + nlohmann::json(kPrompt).dump() + "}");
        std::string sync_line;
        char ch = 0;
        while (read(fd, &ch, 1) == 1 && ch != '\n') {
            sync_line.push_back(ch);
        }
        auto sync = nlohmann::json::parse(sync_line);
        expect(sync["ok"].get<bool>(), "sync ok before large reply");
        const int seed = sync["token"].get<int>();
        write_line(fd, "{\"op\":\"spec_run\",\"first_token\":" + std::to_string(seed) +
                              ",\"count\":256}");
        char run_buf[16] = {};
        const ssize_t run_got = read(fd, run_buf, sizeof(run_buf));
        expect(run_got > 0, "large spec_run reply started before hang-up");
        close(fd);
        usleep(200000);
    }
    expect(kill(child, 0) == 0, "server alive after mid-reply disconnect");
    {
        const int fd = connect_resident(sock);
        expect(fd >= 0, "connect after disconnect");
        auto hello = nlohmann::json::parse(send_recv(fd, R"({"op":"hello"})"));
        expect(hello["ok"].get<bool>(), "hello after disconnect");
        close(fd);
    }
    kill(child, SIGTERM);
}

static void test_wire_integer_out_of_range() {
#ifndef LLAMA_RESIDENT_BIN
#define LLAMA_RESIDENT_BIN ""
#endif
    const std::string bin = LLAMA_RESIDENT_BIN;
    if (bin.empty() || !std::filesystem::exists(bin)) {
        fprintf(stderr, "skip wire integer test: llama-resident not built\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() /
                     ("llama-resident-wire-int-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const std::string sock = (dir / "r.sock").string();
    const std::string ready = (dir / "ready").string();
    const pid_t child = spawn_fake_resident(sock, ready, bin);
    const int fd = connect_resident(sock);
    expect(fd >= 0, "connect int");
    send_recv(fd, "{\"op\":\"sync\",\"tokens\":" + nlohmann::json(kPrompt).dump() + "}");
    const auto bad = nlohmann::json::parse(
        send_recv(fd, "{\"op\":\"eval\",\"token\":" + std::to_string(static_cast<int64_t>(1) << 40) + "}"));
    expect(!bad["ok"].get<bool>(), "wire rejects huge eval token");
    close(fd);
    kill(child, SIGTERM);
}

int main() {
    test_serial_eval_advances_one_token();
    test_spec_run_equals_serial_at_every_depth();
    test_spec_run_never_overshoots();
    test_overhang_consumed();
    test_teacher_forcing_incremental();
    test_teacher_forcing_reprefill_with_ahead();
    test_fake_forced_decode_chain();
    test_fake_consecutive_forced_tokens();
    test_spec_counters_one_mtp_step();
    test_spec_run_committed_only();
    test_spec_round_greedy_accept_reuses_backend_buffer();
    test_spec_round_sampler_restored_on_partial_accept();
    test_spec_round_sampler_clone_freed_on_throw();
    test_spec_round_no_target_checkpoint_accepts_partial();
    test_mtp_partial_rejection_retry_counters();
    test_mtp_partial_rejection_emits_only_committed();
    test_mtp_frontier_matches_top_logits_with_tie();
    test_mtp_teacher_force_prompt_survives_verify_rollback();
    test_invalidate_resets_session();
    test_frontier_only_sync_then_feed_syncs_again();
    test_sync_max_tokens_on_the_wire();
    test_top_from_logits_matches_reference();
    test_invalidate_clears_queued_ahead();
    test_logit_tie_break_lower_id();
    test_frontier_matches_top_logits();
    test_json_integer_range_refusals();
    test_top_logits();
    test_refusals();
    test_out_of_vocab();
    test_hello_depth_convention();
    test_hello_dflash();
    test_implemented_depth_agrees_with_adapter();
    test_wire_two_requests_one_write();
    test_wire_request_split_across_writes();
    test_wire_max_request_line_boundary();
    test_wire_phase_round_trip();
    test_wire_oversize_line_and_survival();
    test_wire_disconnect_mid_reply();
    test_wire_integer_out_of_range();
    puts("ok");
    return 0;
}
