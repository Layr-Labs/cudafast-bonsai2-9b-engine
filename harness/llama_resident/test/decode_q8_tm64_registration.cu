// Exercise the model-load registry, exact layout bytes, production dispatch,
// graph replay, excluded shapes/columns, and teardown at a reused address.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "mmq.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
template <typename T> struct allocation {
    T * data = nullptr;
    explicit allocation(size_t count) { check(cudaMalloc(&data, count*sizeof(T))); }
    ~allocation() { cudaFree(data); }
    allocation(const allocation &) = delete;
};
static uint32_t random_word(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static void same(const std::vector<float> & expected, const std::vector<float> & got,
                 int rows, int cols, int k, const char * stage) {
    require(expected.size() == got.size(), "output size");
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isfinite(got[i]) && std::memcmp(&expected[i], &got[i], sizeof(float)) == 0) continue;
        std::fprintf(stderr, "M=%d N=%d K=%d stage=%s index=%zu expected=%a actual=%a\n",
                     rows, cols, k, stage, i, expected[i], got[i]);
        require(false, "bitwise output equivalence");
    }
}
static void layout_bytes(const std::vector<block_q8_0> & original, const ggml_cuda_dl_entry & e) {
    const int rows = int(e.nrows), blocks = int(e.ncols/QK8_0), steps = blocks/4;
    const size_t bytes = original.size()*sizeof(block_q8_0);
    std::vector<block_q8_0> copy(original.size());
    for (int tile_rows : {128, 64}) {
        const char * source = tile_rows == 128 ? e.dl : e.dl_tm64;
        if (source == nullptr) continue;
        check(cudaMemcpy(copy.data(), source, bytes, cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            for (int block = 0; block < blocks; ++block) {
                const size_t dest = (((size_t(row/tile_rows)*steps + block/4)*tile_rows + row%tile_rows)*4 + block%4);
                if (std::memcmp(&original[size_t(row)*blocks + block], &copy[dest], sizeof(block_q8_0)) != 0) {
                    std::fprintf(stderr, "layout=%d M=%d K=%lld row=%d block=%d\n",
                                 tile_rows, rows, (long long) e.ncols, row, block);
                    require(false, "every weight byte must be preserved");
                }
            }
        }
    }
}

using register_fn_t = size_t (*)(const ggml_tensor *);

static void weight_case(ggml_backend_t backend, register_fn_t reg, int rows, int k, bool enabled, bool ring) {
    ggml_init_params ip = {ggml_tensor_overhead()*2, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, rows);
    ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(weight_buffer != nullptr, "weight allocation");
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t bytes = ggml_nbytes(w);
    const bool eligible = ggml_cuda_dl_q8_tm64_shape(rows, k);
    const bool has_tm64 = enabled && eligible;
    std::vector<block_q8_0> weights(size_t(rows)*(k/QK8_0));
    uint32_t rng = uint32_t(13*rows + k + 7);
    for (size_t i = 0; i < weights.size(); ++i) {
        float scale = float(1 + random_word(rng)%100)/1000;
        if (i%127 == 0) scale = 0;
        if (i%131 == 0) scale = -scale;
        if (i%137 == 0) scale = 3e-7f;
        const ggml_fp16_t bits = ggml_fp32_to_fp16(scale);
        std::memcpy(&weights[i].d, &bits, sizeof(bits));
        for (auto & q : weights[i].qs) q = int(random_word(rng)%256) - 128;
    }
    ggml_backend_tensor_set(w, weights.data(), 0, bytes);
    require(ggml_cuda_dl_find(w->data) == nullptr, "new source must not find stale layout");
    require(reg(w) == bytes*(has_tm64 ? 2 : 1), "registered allocation accounting");
    const ggml_cuda_dl_entry * e = ggml_cuda_dl_find(w->data);
    require(e != nullptr && e->type == GGML_TYPE_Q8_0 && e->dl_type == GGML_TYPE_Q8_0, "Q8 registry entry");
    require((e->dl_tm64 != nullptr) == has_tm64, "selective registration");
    const char * original_copy = e->dl;
    const char * tm64_copy = e->dl_tm64;
    require(reg(w) == 0 && e->dl == original_copy && e->dl_tm64 == tm64_copy, "idempotent registration");
    layout_bytes(weights, *e);
    ggml_cuda_dl_unregister_range((const char *) w->data + bytes, 1);
    ggml_cuda_dl_unregister_range(w->data, 0);
    require(ggml_cuda_dl_find(w->data) == e, "outside and empty ranges preserve entry");

    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    for (int cols : {1, 7, 15, 16}) {
        std::vector<block_q8_1_mmq> y(size_t(k/128)*cols);
        for (auto & b : y) {
            for (auto & d : b.d4) d = float(1 + random_word(rng)%100)/2000;
            for (auto & q : b.qs) q = int(random_word(rng)%256) - 128;
        }
        allocation<int> dy(y.size()*sizeof(y[0])/sizeof(int));
        allocation<float> output(size_t(rows)*cols);
        check(cudaMemcpyAsync(dy.data, y.data(), y.size()*sizeof(y[0]), cudaMemcpyHostToDevice, stream));
        mmq_args args = {};
        args.x = (const char *) w->data; args.type_x = GGML_TYPE_Q8_0; args.y = dy.data; args.dst = output.data;
        args.ncols_x = k; args.nrows_x = rows; args.ncols_dst = cols; args.stride_row_x = k/QK8_0;
        args.ncols_y = cols; args.nrows_dst = rows; args.ncols_max = cols;
        args.nchannels_x = args.nchannels_y = args.nsamples_x = args.nsamples_y = 1;
        args.stride_channel_dst = args.stride_sample_dst = size_t(rows)*cols;
        require(ggml_cuda_mmq_dl_supported(args, *e), "production eligibility");
        require(ggml_cuda_dl_q8_tm64_use(args, *e) == (has_tm64 && cols == 16), "column selection");
        mmq_args unsupported = args;
        unsupported.dst_hi = output.data;
        require(!ggml_cuda_dl_q8_tm64_use(unsupported, *e), "stacked requests keep original layout");
        unsupported = args; unsupported.nrows_x /= 2;
        require(!ggml_cuda_dl_q8_tm64_use(unsupported, *e), "partial row views keep original layout");
        unsupported = args; unsupported.ncols_dst = unsupported.ncols_y = unsupported.ncols_max = 32;
        require(!ggml_cuda_dl_q8_tm64_use(unsupported, *e), "J32 excluded");

        const int grid = ggml_cuda_mmq_pq2_j16_grid(args, nsm);
        allocation<float> fixup(size_t(grid)*16*128);
        auto execute = [&](const ggml_cuda_dl_entry & entry) {
            check(cudaMemsetAsync(output.data, 0x7e, size_t(rows)*cols*sizeof(float), stream));
            check(cudaMemsetAsync(fixup.data, 0x7e, size_t(grid)*16*128*sizeof(float), stream));
            ggml_cuda_mmq_dl_run<GGML_TYPE_Q8_0>(args, entry, grid, nsm, fixup.data, stream);
            check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
            std::vector<float> out(size_t(rows)*cols);
            check(cudaMemcpy(out.data(), output.data, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
            return out;
        };
        ggml_cuda_dl_entry baseline = *e;
        baseline.dl_tm64 = nullptr; // same state as unavailable optional storage
        const auto expected = execute(baseline);
        // CUDA's half union makes this block non-copy-constructible. The
        // independent fixture below fills every activation scale and code.
        std::vector<block_q8_1_mmq> changed_y(y.size());
        for (auto & b : changed_y) {
            for (auto & d : b.d4) d = float(1 + random_word(rng)%100)/3000;
            for (auto & q : b.qs) q = int(random_word(rng)%256) - 128;
        }
        check(cudaMemcpyAsync(dy.data, changed_y.data(), changed_y.size()*sizeof(changed_y[0]), cudaMemcpyHostToDevice, stream));
        const auto expected_changed = execute(baseline);
        require(std::memcmp(expected.data(), expected_changed.data(), expected.size()*sizeof(float)) != 0,
                "changed activation fixture must change the output");
        check(cudaMemcpyAsync(dy.data, y.data(), y.size()*sizeof(y[0]), cudaMemcpyHostToDevice, stream));
        // An excluded column count or disabled ring must never read the TM64
        // allocation, even when a valid registered pointer is present.
        const bool selected = has_tm64 && cols == 16 && ring;
        if (has_tm64 && !selected) check(cudaMemsetAsync(e->dl_tm64, 0, bytes, stream));
        same(expected, execute(*e), rows, cols, k, "registry-dispatch");
        if (selected) {
            // Prove this request reads the new copy, not the original tensor
            // or the retained TM128 copy. This is the final column case.
            check(cudaMemsetAsync(w->data, 0, bytes, stream));
            check(cudaMemsetAsync(e->dl, 0, bytes, stream));
            same(expected, execute(*e), rows, cols, k, "old-copies-cleared");
        }
        cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ggml_cuda_mmq_dl_run<GGML_TYPE_Q8_0>(args, *e, grid, nsm, fixup.data, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto & input = repeat%2 ? changed_y : y;
            const auto & reference = repeat%2 ? expected_changed : expected;
            check(cudaMemcpyAsync(dy.data, input.data(), input.size()*sizeof(input[0]), cudaMemcpyHostToDevice, stream));
            check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
            std::vector<float> got(expected.size());
            check(cudaMemcpy(got.data(), output.data, got.size()*sizeof(float), cudaMemcpyDeviceToHost));
            same(reference, got, rows, cols, k, "changed-input-graph-replay");
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        if (has_tm64 && !selected) {
            ggml_cuda_dl_repack<4, 34>((const char *) w->data, e->dl_tm64, rows, k/QK8_0, GGML_CUDA_DL_TM64, stream);
            check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
        }
        std::printf("registration M=%d N=%d K=%d enabled=%d ring=%d selected=%d exact=1 changed_input_graph=1 fallback=1\n",
                    rows, cols, k, enabled, ring, selected);
        std::fflush(stdout);
    }
    check(cudaStreamDestroy(stream));
    // Explicit unregister while the source remains allocated, then register
    // different bytes at exactly the same address. No stale entry may survive.
    ggml_cuda_dl_unregister_range(w->data, bytes);
    require(ggml_cuda_dl_find(w->data) == nullptr, "explicit range teardown");
    weights[0].qs[0] ^= 0x5a;
    ggml_backend_tensor_set(w, weights.data(), 0, bytes);
    require(reg(w) == bytes*(has_tm64 ? 2 : 1), "same-address registration");
    e = ggml_cuda_dl_find(w->data);
    require(e != nullptr, "new generation found");
    layout_bytes(weights, *e);
    const void * source_address = w->data;
    ggml_backend_buffer_free(weight_buffer); // real model buffer teardown hook
    require(ggml_cuda_dl_find(source_address) == nullptr, "backend teardown removes layouts");
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const bool enabled = argc == 2 && std::strcmp(argv[1], "--enabled") == 0;
    const bool ring_off = argc == 2 && std::strcmp(argv[1], "--ring-off") == 0;
    require(argc == 1 || enabled || ring_off, "usage: decode-q8-tm64-registration [--enabled|--ring-off]");
    if (enabled || ring_off) setenv("GGML_CUDA_DL_Q8_TM64", "1", 1);
    else setenv("GGML_CUDA_DL_Q8_TM64", "0", 1); // explicit opt-out
    setenv("GGML_CUDA_DECODE_LAYOUT", "1", 1);
    setenv("GGML_CUDA_DL_RING", ring_off ? "0" : "1", 1);
    setenv("GGML_CUDA_DL_RING_MIN_ROWS", "4096", 1);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    require(backend != nullptr, "CUDA backend");
    require(ggml_cuda_info().devices[0].cc == GGML_CUDA_CC_DGX_SPARK, "requires sm_121");
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    auto reg = (register_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_register_decode_layout");
    require(reg != nullptr, "model-load registration function");
    weight_case(backend, reg, 12288, 4096, enabled || ring_off, !ring_off);
    weight_case(backend, reg, 4096, 12288, enabled || ring_off, !ring_off);
    weight_case(backend, reg, 4096, 32768, enabled || ring_off, !ring_off);
    weight_case(backend, reg, 4096, 4096, enabled || ring_off, !ring_off);
    ggml_backend_free(backend);
    std::printf("DECODE_Q8_TM64_REGISTRATION_PASS enabled=%d ring=%d shapes=4\n", enabled || ring_off, !ring_off);
    return 0;
}
