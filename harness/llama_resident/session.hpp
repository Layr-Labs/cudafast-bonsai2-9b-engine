#pragma once

#include "engine_iface.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class RefusedError : public std::runtime_error {
  public:
    explicit RefusedError(const std::string &msg) : std::runtime_error(msg) {}
};

struct HelloInfo {
    int32_t vocab_size = 0;
    int32_t eos_token = 0;
    bool mtp_armed = false;
    std::string spec_mode = "serial";
    int32_t draft_tokens = 1;
    int32_t ctx_size = 0;
    int64_t load_epoch = 0;
    bool spec_run = true;
    std::string ident;
    std::string model_path;
    std::string mtp_head_path;
    std::string draft_model_path;
};

class Session {
  public:
    Session(std::unique_ptr<IEngine> engine, bool spec_run = true, int64_t load_epoch = 0);

    void invalidate();
    int32_t sync(const std::vector<int32_t> &tokens, bool frontier_only = false);
    int32_t eval(int32_t token);
    std::vector<int32_t> cycle(int32_t fed, int budget);
    std::pair<std::vector<std::vector<int32_t>>, std::vector<int32_t>> spec_run(int32_t first, int count);
    std::pair<std::vector<int32_t>, std::vector<float>> top_logits(int k);
    HelloInfo hello() const;

    int64_t drafts() const { return drafts_; }
    int64_t hits() const { return hits_; }

    const std::vector<int32_t> &ctx() const { return ctx_; }
    const Emit *frontier() const { return frontier_; }
    size_t queued_ahead() const { return ahead_.size(); }

    bool handle_op(const std::string &op, const void *json_req, std::string &reply_json);

  private:
    const Emit &need_frontier() const;
    std::vector<Emit> engine_step();
    void feed(int32_t token);
    void check_ids(const std::vector<int32_t> &tokens) const;
    void resync_if_bounded();

    std::unique_ptr<IEngine> engine_;
    bool spec_run_offered_;
    int64_t load_epoch_;
    int64_t drafts_ = 0;
    int64_t hits_ = 0;
    std::vector<int32_t> ctx_;
    const Emit *frontier_ = nullptr;
    Emit frontier_storage_{};
    std::deque<Emit> ahead_;
    // The live sequence came from a frontier-only sync: the engine holds no
    // state to step from, so the next eval or cycle syncs ctx_ again.
    bool bounded_ = false;
};
