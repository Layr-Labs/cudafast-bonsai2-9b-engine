// Production registry/dispatch gate for the measured pure-tile PTQ FFN path.
// This deliberately uses the normal TM128 allocation, including the real
// registered gate/up pair, and inspects captured kernel functions for routing.
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
struct guarded {
    static constexpr size_t guard = 32;
    size_t count;
    allocation<float> storage;
    explicit guarded(size_t count) : count(count), storage(count + 2*guard) {}
    float * data() const { return storage.data + guard; }
    void poison(cudaStream_t stream) { check(cudaMemsetAsync(storage.data, 0x7e, (count + 2*guard)*sizeof(float), stream)); }
    std::vector<float> read() const {
        std::vector<float> all(count + 2*guard);
        check(cudaMemcpy(all.data(), storage.data, all.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) {
            uint32_t before, after;
            std::memcpy(&before, &all[i], 4); std::memcpy(&after, &all[guard + count + i], 4);
            require(before == 0x7e7e7e7e && after == 0x7e7e7e7e, "output/fixup guard");
        }
        return {all.begin() + guard, all.begin() + guard + count};
    }
};
static uint32_t next_random(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static void fill_weights(std::vector<block_pq2_0> & weights, uint32_t & random) {
    uint8_t trits[81];
    for (int i = 0; i < 81; ++i) {
        int value = i; trits[i] = 0;
        for (int j = 0; j < 4; ++j) { trits[i] |= uint8_t(value%3) << (2*j); value /= 3; }
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        float scale = float(1 + next_random(random)%100)/1000;
        if (i%127 == 0) scale = 0;
        if (i%131 == 0) scale = -scale;
        if (i%137 == 0) scale = 3e-7f;
        const ggml_fp16_t bits = ggml_fp32_to_fp16(scale);
        std::memcpy(&weights[i].d, &bits, sizeof(bits));
        for (auto & q : weights[i].qs) q = trits[next_random(random)%81];
    }
}
static void fill_activation(std::vector<block_q8_1_mmq> & blocks, uint32_t & random, int phase) {
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (auto & d : blocks[i].d4) d = i%127 == 0 ? 0 : float(1 + next_random(random)%100)/(phase ? 3000 : 2000);
        for (auto & q : blocks[i].qs) q = int(next_random(random)%256) - 128;
    }
}
static void same(const std::vector<float> & expected, const std::vector<float> & got,
                 int rows, int cols, int k, const char * stage, bool partial = false) {
    require(expected.size() == got.size(), "comparison size");
    for (size_t i = 0; i < got.size(); ++i) {
        // Only live columns of stream-K partial tiles are defined.
        if (partial && (i/128)%16 >= size_t(cols)) continue;
        if (std::isfinite(got[i]) && std::memcmp(&expected[i], &got[i], sizeof(float)) == 0) continue;
        std::fprintf(stderr, "M=%d N=%d K=%d stage=%s index=%zu expected=%a actual=%a\n",
                     rows, cols, k, stage, i, expected[i], got[i]);
        require(false, "bitwise equivalence");
    }
}
static void inspect_route(cudaGraph_t graph, bool selected) {
    size_t count = 0; check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    int kernels = 0, jit = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type; check(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams params = {}; check(cudaGraphKernelNodeGetParams(node, &params));
        ++kernels;
        if (params.func == (void *) mul_mat_q_j16_dl_ring<GGML_CUDA_DL_FMT_PTQ1S, 3, 2, true, 2, true>) ++jit;
    }
    require(kernels > 0 && jit == int(selected), "captured kernel must prove the expected route");
}
static void resources() {
    const auto original = mul_mat_q_j16_dl_ring<GGML_CUDA_DL_FMT_PTQ1S, 3, 2>;
    const auto jit = mul_mat_q_j16_dl_ring<GGML_CUDA_DL_FMT_PTQ1S, 3, 2, true, 2, true>;
    constexpr size_t shared = ggml_cuda_dl_ring_smem<GGML_CUDA_DL_FMT_PTQ1S, 2>(3);
    for (auto kernel : {original, jit}) {
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, int(cudaSharedmemCarveoutMaxShared)));
        cudaFuncAttributes attr; check(cudaFuncGetAttributes(&attr, kernel));
        int resident; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, shared));
        std::printf("resource variant=%s regs=%d local_bytes=%zu dynamic_shared=%zu resident_ctas=%d\n",
                    kernel == jit ? "jit-pure-bound2" : "original", attr.numRegs, attr.localSizeBytes, shared, resident);
        if (kernel == jit) require(attr.numRegs <= 64 && attr.localSizeBytes == 0 && resident >= 2,
                                   "production specialization must retain measured two-CTA residency");
    }
}
using register_fn_t = size_t (*)(const ggml_tensor *);
using pair_fn_t = size_t (*)(const ggml_tensor *, const ggml_tensor *);

static void weight_case(ggml_backend_t backend, register_fn_t reg, pair_fn_t reg_pair,
                        int half_rows, int k, bool stacked, bool enabled, bool ring, bool signed_mma) {
    const int rows = half_rows*(stacked ? 2 : 1);
    ggml_init_params ip = {ggml_tensor_overhead()*3, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, k, half_rows);
    ggml_tensor * u = stacked ? ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, k, half_rows) : nullptr;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "weight buffer");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t bytes = ggml_nbytes(w), packed = size_t(half_rows)*(k/128)*sizeof(block_ptq1_0);
    std::vector<block_pq2_0> weights(size_t(half_rows)*(k/128));
    uint32_t random = uint32_t(rows + 13*k + 7);
    fill_weights(weights, random); ggml_backend_tensor_set(w, weights.data(), 0, bytes);
    if (u) {
        std::vector<block_pq2_0> up(weights.size());
        fill_weights(up, random); ggml_backend_tensor_set(u, up.data(), 0, bytes);
    }
    auto register_all = [&]() {
        require(reg(w) == packed, "only the existing PTQ TM128 allocation is needed");
        const auto * entry = ggml_cuda_dl_find(w->data);
        require(entry && entry->type == GGML_TYPE_PQ2_0 && entry->dl_type == GGML_TYPE_PTQ1_0 &&
                entry->dl_tm64 == nullptr, "existing signed PTQ registry layout");
        const auto * old_copy = entry->dl;
        require(reg(w) == 0 && ggml_cuda_dl_find(w->data)->dl == old_copy, "idempotent registration");
        if (u) {
            require(reg(u) == packed, "up PTQ registration");
            require(reg_pair(w, u) == 2*(bytes + packed), "existing stacked-pair allocations only");
            const auto * pair = ggml_cuda_glu_pair_find(w->data);
            require(pair && pair->up == u->data && pair->dl_stacked, "real registered gate/up pair");
            require(reg_pair(w, u) == 0, "idempotent pair registration");
        }
    };
    require(ggml_cuda_dl_find(w->data) == nullptr, "new buffer must not retain stale layout");
    register_all();
    ggml_cuda_dl_entry entry = *ggml_cuda_dl_find(w->data);
    if (u) { entry.dl = ggml_cuda_glu_pair_find(w->data)->dl_stacked; entry.nrows = rows; }
    ggml_cuda_dl_unregister_range((const char *) w->data + bytes, 0);
    require(ggml_cuda_dl_find(w->data) != nullptr, "empty teardown preserves registration");
    std::vector<char> old_block(sizeof(block_ptq1_0));
    check(cudaMemcpy(old_block.data(), entry.dl, old_block.size(), cudaMemcpyDeviceToHost));

    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    for (int cols : {1, 7, 15, 16}) {
        std::vector<block_q8_1_mmq> y(size_t(k/128)*cols), changed(y.size());
        fill_activation(y, random, 0); fill_activation(changed, random, 1);
        allocation<int> dy(y.size()*sizeof(y[0])/sizeof(int));
        guarded output(size_t(rows)*cols);
        mmq_args args = {};
        args.x = (const char *) w->data; args.type_x = GGML_TYPE_PQ2_0; args.y = dy.data; args.dst = output.data();
        args.ncols_x = k; args.nrows_x = rows; args.ncols_dst = cols; args.stride_row_x = k/128;
        args.ncols_y = cols; args.nrows_dst = half_rows; args.ncols_max = cols;
        args.nchannels_x = args.nchannels_y = args.nsamples_x = args.nsamples_y = 1;
        args.stride_channel_dst = args.stride_sample_dst = size_t(half_rows)*cols;
        args.dst_hi = stacked ? args.dst + size_t(half_rows)*cols : nullptr;
        args.nrows_lo = stacked ? half_rows : rows;
        require(ggml_cuda_mmq_dl_supported(args, entry), "ordinary production eligibility");
        const int grid = ggml_cuda_mmq_pq2_j16_grid(args, nsm);
        const bool shape = half_rows == 12288 && k == 4096 && cols == 16;
        const bool selected = enabled && ring && signed_mma && shape;
        require(ggml_cuda_dl_ptq_ffn_jit_use(args, entry, grid, nsm) == (enabled && shape), "narrow shape predicate");
        mmq_args excluded = args; excluded.ncols_dst = excluded.ncols_y = excluded.ncols_max = 32;
        require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "J32 excluded");
        excluded = args; excluded.ncols_x += 128;
        require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "different K excluded");
        excluded = args; excluded.nrows_x /= 2;
        require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "leading-row view excluded");
        require(!ggml_cuda_dl_ptq_ffn_jit_use(args, entry, grid + 1, nsm), "different virtual partition excluded");
        excluded = args; excluded.nchannels_y = 2;
        require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "batched request excluded");
        if (stacked) {
            excluded = args; excluded.dst_hi = nullptr; excluded.nrows_dst = rows;
            require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "unstacked M24576 excluded");
            excluded = args; excluded.nrows_lo -= 128;
            require(!ggml_cuda_dl_ptq_ffn_jit_use(excluded, entry, grid, nsm), "different stack ownership excluded");
        }
        guarded fixup(size_t(grid)*16*128);
        auto execute = [&](const ggml_cuda_dl_entry & e) {
            output.poison(stream); fixup.poison(stream);
            ggml_cuda_mmq_dl_run<GGML_TYPE_PQ2_0>(args, e, grid, nsm, fixup.data(), stream);
            check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
            return output.read();
        };
        // A leading-row view of a larger registry entry takes the unchanged
        // dispatcher. The kernel's dimensions, packed bytes and grid remain
        // exactly the same, making this an original-path control in every mode.
        ggml_cuda_dl_entry baseline = entry; baseline.nrows += 128;
        check(cudaMemcpyAsync(dy.data, y.data(), y.size()*sizeof(y[0]), cudaMemcpyHostToDevice, stream));
        const auto expected = execute(baseline), expected_fixup = fixup.read();
        check(cudaMemcpyAsync(dy.data, changed.data(), changed.size()*sizeof(changed[0]), cudaMemcpyHostToDevice, stream));
        const auto expected_changed = execute(baseline), changed_fixup = fixup.read();
        require(std::memcmp(expected.data(), expected_changed.data(), expected.size()*sizeof(float)) != 0,
                "changed activation fixture must change output");
        check(cudaMemcpyAsync(dy.data, y.data(), y.size()*sizeof(y[0]), cudaMemcpyHostToDevice, stream));
        same(expected, execute(entry), rows, cols, k, "production-route");
        same(expected_fixup, fixup.read(), rows, cols, k, "production-partials", true);
        cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ggml_cuda_mmq_dl_run<GGML_TYPE_PQ2_0>(args, entry, grid, nsm, fixup.data(), stream);
        check(cudaStreamEndCapture(stream, &graph));
        inspect_route(graph, selected);
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 9; ++replay) {
            const auto & input = replay%2 ? changed : y;
            const auto & reference = replay%2 ? expected_changed : expected;
            const auto & partial_reference = replay%2 ? changed_fixup : expected_fixup;
            check(cudaMemcpyAsync(dy.data, input.data(), input.size()*sizeof(input[0]), cudaMemcpyHostToDevice, stream));
            output.poison(stream); fixup.poison(stream);
            check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
            same(reference, output.read(), rows, cols, k, "changed-input-graph");
            same(partial_reference, fixup.read(), rows, cols, k, "graph-partials", true);
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        std::printf("ptq_ffn M=%d N=%d K=%d stacked=%d enabled=%d ring=%d signed=%d selected=%d exact=1 route_graph=1 replay=9\n",
                    rows, cols, k, stacked, enabled, ring, signed_mma, selected);
        std::fflush(stdout);
    }
    check(cudaStreamDestroy(stream));
    ggml_cuda_dl_unregister_range(w->data, bytes);
    require(ggml_cuda_dl_find(w->data) == nullptr, "explicit source teardown");
    if (u) {
        require(ggml_cuda_glu_pair_find(w->data) == nullptr, "source teardown removes stacked pair");
        ggml_cuda_dl_unregister_range(u->data, bytes);
    }
    // Change a legal trit and re-register the exact same address. Repacking
    // must produce the new bytes even when a previous graph used that source.
    weights[0].qs[0] = uint8_t((weights[0].qs[0] & ~3u) | (((weights[0].qs[0] & 3u) + 1)%3));
    ggml_backend_tensor_set(w, weights.data(), 0, bytes);
    register_all();
    std::vector<char> new_block(old_block.size());
    check(cudaMemcpy(new_block.data(), ggml_cuda_dl_find(w->data)->dl, new_block.size(), cudaMemcpyDeviceToHost));
    require(old_block != new_block, "same-address registration must repack changed bytes");
    const void * source = w->data;
    ggml_backend_buffer_free(buffer);
    require(ggml_cuda_dl_find(source) == nullptr && ggml_cuda_glu_pair_find(source) == nullptr,
            "backend buffer teardown removes source and stacked copies");
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const bool on = argc == 2 && std::strcmp(argv[1], "--enabled") == 0;
    const bool ring_off = argc == 2 && std::strcmp(argv[1], "--ring-off") == 0;
    const bool unsigned_mma = argc == 2 && std::strcmp(argv[1], "--unsigned") == 0;
    require(argc == 1 || on || ring_off || unsigned_mma,
            "usage: decode-ptq-ffn-route [--enabled|--ring-off|--unsigned]");
    const bool enabled = on || ring_off || unsigned_mma;
    if (enabled) setenv("GGML_CUDA_PTQ_FFN_JIT", "1", 1);
    else setenv("GGML_CUDA_PTQ_FFN_JIT", "0", 1);
    setenv("GGML_CUDA_DECODE_LAYOUT", "1", 1); setenv("GGML_CUDA_DECODE_PTQ1", "1", 1);
    setenv("GGML_CUDA_DL_RING_PT", ring_off ? "0" : "1", 1);
    setenv("GGML_CUDA_PTQ1_SIGNED", unsigned_mma ? "0" : "1", 1);
    setenv("GGML_CUDA_PREFILL_PLANE", "0", 1);
    setenv("GGML_CUDA_DECODE_GU_EPILOGUE", "0", 1); // isolate stacked JIT copy accounting
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    require(backend != nullptr && ggml_cuda_info().devices[0].cc == GGML_CUDA_CC_DGX_SPARK, "requires GB10 CUDA backend");
    resources();
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    auto reg = (register_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_register_decode_layout");
    auto reg_pair = (pair_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_register_prefill_glu_pair");
    require(reg && reg_pair, "model-load registration APIs");
    weight_case(backend, reg, reg_pair, 12288, 4096, false, enabled, !ring_off, !unsigned_mma);
    weight_case(backend, reg, reg_pair, 12288, 4096, true, enabled, !ring_off, !unsigned_mma);
    weight_case(backend, reg, reg_pair, 4096, 4096, false, enabled, !ring_off, !unsigned_mma);
    weight_case(backend, reg, reg_pair, 12288, 8192, false, enabled, !ring_off, !unsigned_mma);
    ggml_backend_free(backend);
    std::printf("DECODE_PTQ_FFN_ROUTE_PASS enabled=%d ring=%d signed=%d cases=16 graph_replays=144\n",
                enabled, !ring_off, !unsigned_mma);
}
