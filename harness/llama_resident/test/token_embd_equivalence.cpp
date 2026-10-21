// Exact check of the token embedding lookup on the GPU against the host lookup it replaces: every
// row of the model's token_embd.weight, gathered with ggml_get_rows on the CUDA backend, must have
// the bits the CPU backend's lookup produces (the type's to_float row dequantization).
//
// usage: token-embd-equivalence MODEL.gguf
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf = gguf_init_from_file(argv[1], gp);
    if (!gguf) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    const int64_t tid = gguf_find_tensor(gguf, "token_embd.weight");
    ggml_tensor * w_meta = ggml_get_tensor(meta, "token_embd.weight");
    if (tid < 0 || !w_meta) {
        fprintf(stderr, "no token_embd.weight\n");
        return 2;
    }
    const int64_t n_embd = w_meta->ne[0];
    const int64_t n_rows = w_meta->ne[1];
    const size_t  row_size = ggml_row_size(w_meta->type, n_embd);

    std::vector<uint8_t> data(ggml_nbytes(w_meta));
    FILE * f = fopen(argv[1], "rb");
    if (!f || fseek(f, (long) (gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tid)), SEEK_SET) != 0 ||
        fread(data.data(), 1, data.size(), f) != data.size()) {
        fprintf(stderr, "cannot read token_embd.weight\n");
        return 2;
    }
    fclose(f);

    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) {
        fprintf(stderr, "no CUDA backend\n");
        return 2;
    }

    // the table on the device once, then the rows in chunks
    const int64_t chunk = 4096;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w    = ggml_new_tensor_2d(ctx, w_meta->type, n_embd, n_rows);
    ggml_tensor * ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk);
    ggml_tensor * rows = ggml_get_rows(ctx, w, ids);
    ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, data.data(), 0, data.size());

    const ggml_to_float_t to_float = ggml_get_type_traits(w_meta->type)->to_float;

    std::vector<int32_t> id(chunk);
    std::vector<float>   got((size_t) chunk * n_embd);
    std::vector<float>   want(n_embd);
    int64_t n_checked = 0;
    int64_t n_failed  = 0;
    for (int64_t r0 = 0; r0 < n_rows; r0 += chunk) {
        const int64_t n = std::min(chunk, n_rows - r0);
        for (int64_t i = 0; i < chunk; ++i) {
            id[i] = (int32_t) (r0 + (i < n ? i : 0));
        }
        ggml_backend_tensor_set(ids, id.data(), 0, id.size() * sizeof(int32_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 2;
        }
        ggml_backend_tensor_get(rows, got.data(), 0, got.size() * sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            to_float(data.data() + (size_t) (r0 + i) * row_size, want.data(), n_embd);
            if (std::memcmp(want.data(), got.data() + (size_t) i * n_embd, n_embd * sizeof(float)) != 0) {
                if (n_failed++ < 8) {
                    fprintf(stderr, "FAIL row %lld\n", (long long) (r0 + i));
                }
            }
            ++n_checked;
        }
    }

    printf("token-embd-equivalence: %s %lld rows x %lld, %lld exact, %lld failures\n", ggml_type_name(w_meta->type),
           (long long) n_rows, (long long) n_embd, (long long) (n_checked - n_failed), (long long) n_failed);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    gguf_free(gguf);
    ggml_free(meta);
    return n_failed == 0 && n_checked == n_rows ? 0 : 1;
}
