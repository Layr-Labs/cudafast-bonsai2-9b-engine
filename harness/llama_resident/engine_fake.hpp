#pragma once

#include "engine_iface.hpp"

#include <cstdint>
#include <string>
#include <vector>

class FakeEngine : public IEngine {
  public:
    explicit FakeEngine(int draft_depth = 0, int32_t vocab = 131072, int32_t ctx = 8192,
                        const std::string &spec_mode = "mtp", const std::string &draft_model_path = "");

    int32_t vocab_size() const override { return vocab_size_; }
    int32_t eos_token() const override { return 2; }
    int depth() const override { return depth_; }
    std::string spec_mode() const override { return depth_ > 0 ? spec_mode_ : "serial"; }
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

    int starts = 0;
    int frontier_starts = 0;

  private:
    int32_t next_token(const std::vector<int32_t> &ctx) const;
    Emit emit_at(const std::vector<int32_t> &ctx) const;

    int depth_;
    int32_t vocab_size_;
    int32_t ctx_size_;
    std::string ident_;
    std::string model_path_;
    std::string spec_mode_;
    std::string draft_model_path_;
    std::vector<int32_t> ctx_;
    int cycle_ = 0;
    int64_t last_step_drafts_ = 0;
    int64_t last_step_hits_ = 0;
};
