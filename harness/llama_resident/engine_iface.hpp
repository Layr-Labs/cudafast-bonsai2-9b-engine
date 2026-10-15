#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

constexpr int kImplementedDepth = 6;
// DFlash drafts a block of block_size tokens: the anchor plus block_size - 1 drafts.
// The cap is the drafter's own dflash.block_size - 1, checked when it loads.
constexpr int kTopK = 8;
constexpr int kMaxTopK = 64;
constexpr int kMaxRun = 65536;

struct Emit {
    int32_t token = 0;
    std::vector<std::pair<int32_t, float>> top;
};

class IEngine {
  public:
    virtual ~IEngine() = default;

    virtual int32_t vocab_size() const = 0;
    virtual int32_t eos_token() const = 0;
    virtual int depth() const = 0;
    /// "serial", "mtp" or "dflash": the speculative method the engine was booted with.
    virtual std::string spec_mode() const { return depth() > 0 ? "mtp" : "serial"; }
    /// The separate draft model file (dflash only), else empty.
    virtual std::string draft_model_path() const { return std::string(); }
    virtual int32_t ctx_size() const = 0;
    virtual const std::string &ident() const = 0;
    virtual const std::string &model_path() const = 0;

    virtual std::vector<Emit> start(const std::vector<int32_t> &tokens) = 0;
    /// start() for a caller that needs the frontier token and nothing after it
    /// (a timed prefill). The engine may skip work that only later steps use;
    /// the frontier is the one start() gives. No step() or decode_forced() may
    /// follow it: the session syncs again first.
    virtual std::vector<Emit> start_frontier_only(const std::vector<int32_t> &tokens) { return start(tokens); }
    virtual std::vector<Emit> step() = 0;
    /// One incremental decode with a caller-chosen token (teacher forcing).
    virtual std::vector<Emit> decode_forced(int32_t token) = 0;
    virtual void abort() = 0;

    /// Draft/hit increments from the last MTP step() (0 when serial).
    virtual std::pair<int64_t, int64_t> last_spec_step_counters() const { return {0, 0}; }
};
