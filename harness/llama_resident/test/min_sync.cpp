// Minimal repro of LlamaEngine::start + 32 greedy decodes at depth 0.
// Mirrors make_engine_params / prefill_prompt / decode_serial_one with nothing
// else in the way, so a failure here is llama.cpp usage and not the resident.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static int32_t argmax_ith(llama_context *ctx, const llama_vocab *vocab, int idx) {
    const float *logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        fprintf(stderr, "min_sync: no logits at %d\n", idx);
        exit(1);
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int32_t best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [n_ctx] [n_batch] [n_ubatch]\n", argv[0]);
        return 1;
    }
    const int32_t n_ctx = argc > 2 ? atoi(argv[2]) : 8192;
    const int32_t n_batch = argc > 3 ? atoi(argv[3]) : 2048;
    const int32_t n_ubatch = argc > 4 ? atoi(argv[4]) : 512;

    common_init();
    llama_backend_init();

    // identical to LlamaEngine::make_engine_params(depth = 0)
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_parallel = 1;
    params.n_gpu_layers = -1;
    const int n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    params.cpuparams.n_threads = n_threads;
    params.cpuparams_batch.n_threads = n_threads;
    params.cache_type_k = GGML_TYPE_F16;
    params.cache_type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.sampling.temp = 0.0f;
    params.sampling.top_k = 1;

    common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        fprintf(stderr, "min_sync: failed to load model\n");
        return 1;
    }
    llama_model *model = init->model();
    llama_context *ctx = init->context();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    printf("min_sync: loaded model=%p ctx=%p n_ctx=%u n_batch=%u n_vocab=%d\n", (void *) model, (void *) ctx,
           llama_n_ctx(ctx), llama_n_batch(ctx), llama_vocab_n_tokens(vocab));
    fflush(stdout);

    llama_tokens inp = common_tokenize(ctx, "The capital of France is", true, true);
    printf("min_sync: prompt tokens = %zu\n", inp.size());
    fflush(stdout);

    // prefill in n_batch chunks, logits on the last token only
    const int32_t n = static_cast<int32_t>(inp.size());
    int32_t pos = 0;
    for (int32_t off = 0; off < n;) {
        const int32_t chunk = std::min(n_batch, n - off);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        for (int32_t i = 0; i < chunk; ++i) {
            common_batch_add(batch, inp[off + i], pos++, {0}, off + i == n - 1);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "min_sync: prefill decode failed rc=%d\n", rc);
            return 1;
        }
        off += chunk;
    }

    int32_t id = argmax_ith(ctx, vocab, -1);
    printf("min_sync: first token = %d '%s'\n", id, common_token_to_piece(ctx, id).c_str());
    fflush(stdout);

    for (int i = 0; i < 32; ++i) {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, id, pos++, {0}, true);
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "min_sync: decode %d failed rc=%d\n", i, rc);
            return 1;
        }
        id = argmax_ith(ctx, vocab, -1);
        printf("min_sync: step %2d token = %6d '%s'\n", i, id, common_token_to_piece(ctx, id).c_str());
        fflush(stdout);
    }

    printf("min_sync: OK\n");
    init.reset();
    llama_backend_free();
    return 0;
}
