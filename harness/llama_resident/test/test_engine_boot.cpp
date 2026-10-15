// Boot-path test of LlamaEngine at an MTP depth (CPU only, no weight file).
// Real code under test: LlamaEngine (engine_llama.cpp) on a tiny qwen35 model with one MTP layer that
// llama_model_init_from_user builds from metadata (random F32 weights, "none" tokenizer with n_vocab ids),
// saved to a GGUF file. The MTP layer uses the shared output.weight as its head, as in the target GGUF.
#include "engine_llama.hpp"

#include "gguf.h"
#include "llama-model.h" // internal: to drop the optional MTP shared head
#include "llama-model-saver.h" // internal: test_frontier_only writes the tiny model to a GGUF file
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

static int g_fail = 0;
static void expect(bool ok, const char *msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_fail;
    }
}

static std::mt19937 rng; // reseeded per model: same seed -> same weights

static void fill_random(ggml_tensor *t, void *) {
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> v(ggml_nelements(t));
    for (auto &x : v) x = nd(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

constexpr uint32_t kVocab = 64;

// A tiny qwen35: 2 trunk layers (one gated delta net, one attention) plus one MTP layer, with no
// MTP shared head (as in the target GGUF), so the MTP layer drafts through output.weight.
static llama_model *tiny_qwen35_mtp() {
    gguf_context *g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qwen35");
    gguf_set_val_str(g, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(g, "qwen35.vocab_size", kVocab);
    gguf_set_val_u32(g, "qwen35.context_length", 256);
    gguf_set_val_u32(g, "qwen35.embedding_length", 32);
    gguf_set_val_u32(g, "qwen35.block_count", 3);
    gguf_set_val_u32(g, "qwen35.nextn_predict_layers", 1);
    gguf_set_val_u32(g, "qwen35.feed_forward_length", 64);
    gguf_set_val_u32(g, "qwen35.attention.head_count", 2);
    gguf_set_val_u32(g, "qwen35.attention.head_count_kv", 1);
    gguf_set_val_u32(g, "qwen35.attention.key_length", 16);
    gguf_set_val_u32(g, "qwen35.attention.value_length", 16);
    gguf_set_val_f32(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(g, "qwen35.rope.dimension_count", 8);
    const int32_t sections[4] = {2, 1, 1, 0};
    gguf_set_arr_data(g, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    gguf_set_val_u32(g, "qwen35.full_attention_interval", 2);
    gguf_set_val_u32(g, "qwen35.ssm.conv_kernel", 4);
    gguf_set_val_u32(g, "qwen35.ssm.state_size", 8);
    gguf_set_val_u32(g, "qwen35.ssm.group_count", 2);
    gguf_set_val_u32(g, "qwen35.ssm.time_step_rank", 2);
    gguf_set_val_u32(g, "qwen35.ssm.inner_size", 16);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    rng.seed(7);
    llama_model *m = llama_model_init_from_user(g, fill_random, nullptr, mp);
    gguf_free(g);
    if (m) {
        for (auto &layer : m->layers) layer.nextn.shared_head_head = nullptr; // init_from_user creates it
    }
    return m;
}

static std::vector<int32_t> stream_tokens(LlamaEngine &e, int rounds) {
    std::vector<int32_t> out;
    for (int i = 0; i < rounds; ++i) {
        for (const Emit &x : e.step()) out.push_back(x.token);
    }
    return out;
}

// The frontier-only prefill (a timed prefill, sync max_tokens 1) at an MTP depth turns the nextn
// extraction off for its pass; the full sync after it must turn it on again. That full sync still
// drafts, and gives the frontier and stream a full sync alone gives. Mutation checks: drop the
// speculative re-create in clear_kv -> the full sync after a frontier-only one throws; make
// start_frontier_only call start() -> "step after a frontier-only sync refuses" fails. That its logits equal the serial ones bit for bit
// is a GPU property (the narrowed last layer runs other kernels); the lane log records it.
static void test_frontier_only(const fs::path &tmp) {
    llama_model *m = tiny_qwen35_mtp();
    expect(m != nullptr, "tiny qwen35 for the frontier-only test builds");
    if (!m) return;
    const std::string path = (tmp / "tiny-mtp.gguf").string();
    for (auto &layer : m->layers) { // the dropped shared head's scales: the loader asks for none of them
        layer.nextn.shared_head_head_s = nullptr;
        layer.nextn.shared_head_head_in_s = nullptr;
    }
    {
        llama_model_saver ms(m);
        ms.add_kv_from_model();
        // The saver writes per-layer arrays for the trunk layers only; the loader wants every layer.
        gguf_set_arr_data(ms.gguf_ctx, "qwen35.attention.recurrent_layers", GGUF_TYPE_UINT32,
                          m->hparams.is_recr_impl.data(), m->hparams.n_layer_all);
        ms.add_tensors_from_model();
        ms.save(path);
    }
    llama_model_free(m);
    const std::vector<int32_t> prompt = {3, 8, 13, 18, 23, 28, 33, 38, 43, 48};
    try {
        LlamaEngine mtp(path, "mtp", 3, "", 64, 16, 16, "mtp3");
        const Emit full = mtp.start(prompt).front();
        const std::vector<int32_t> ref = stream_tokens(mtp, 4);

        const Emit bounded = mtp.start_frontier_only(prompt).front();
        expect(bounded.token == full.token, "frontier-only sync at depth 3 gives the full sync's frontier token");
        bool refused = false;
        try {
            mtp.step();
        } catch (const std::exception &) {
            refused = true;
        }
        expect(refused, "step after a frontier-only sync refuses");

        const Emit again = mtp.start(prompt).front();
        expect(again.token == full.token && again.top == full.top, "full sync after a frontier-only one: same frontier");
        expect(stream_tokens(mtp, 4) == ref, "full sync after a frontier-only one: same MTP stream");
        expect(mtp.last_spec_step_counters().first > 0, "full sync after a frontier-only one still drafts");

        // A new engine whose first call is a frontier-only sync, then a full sync.
        LlamaEngine leg(path, "mtp", 3, "", 64, 16, 16, "mtp3");
        expect(leg.start_frontier_only(prompt).front().token == full.token,
               "frontier-only sync as the first call gives the full sync's frontier token");
        const Emit first = leg.start(prompt).front();
        expect(first.token == full.token && first.top == full.top, "full sync after a first frontier-only one: same frontier");
        expect(stream_tokens(leg, 4) == ref, "full sync after a first frontier-only one: same MTP stream");
        expect(leg.last_spec_step_counters().first > 0, "full sync after a first frontier-only one drafts");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "threw: %s\n", e.what());
        expect(false, "frontier-only then full sync at depth 3 runs");
    }
}

int main() {
    llama_backend_init();
    const fs::path tmp = fs::temp_directory_path() / ("engine_boot_" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    test_frontier_only(tmp);
    fs::remove_all(tmp);
    llama_backend_free();
    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("test_engine_boot: OK\n");
    return 0;
}
