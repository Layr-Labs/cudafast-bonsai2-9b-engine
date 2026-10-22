// GB10: real FA dispatch with half scratch off/on, unchanged captured geometry,
// wordwise outputs, guarded outputs and changed-input graph replays.
#include "ggml-cuda.h"
#include "fattn-mma-f16.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int D = 256, HEADS = 16, KV_HEADS = 4, GUARD = 64;
constexpr uint32_t SENTINEL = 0x47123456;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
void require(bool value, const char * why) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}
template<class T> struct Buffer {
    T * ptr = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&ptr, count*sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
};
struct Guarded {
    size_t count;
    Buffer<uint32_t> storage;
    std::vector<uint32_t> poison;
    explicit Guarded(size_t count) : count(count), storage(count + 2*GUARD), poison(count + 2*GUARD, SENTINEL) {}
    float * data() const { return (float *) (storage.ptr + GUARD); }
    void reset(cudaStream_t stream) {
        check(cudaMemcpyAsync(storage.ptr, poison.data(), poison.size()*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    }
    std::vector<uint32_t> read() const {
        std::vector<uint32_t> words(poison.size());
        check(cudaMemcpy(words.data(), storage.ptr, words.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < GUARD; ++i) {
            require(words[i] == SENTINEL && words[GUARD + count + i] == SENTINEL, "output buffer guards");
        }
        return words;
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaKernelNodeParams producer = {};
    cudaKernelNodeParams fixup = {};
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
    void inspect() {
        size_t count = 0;
        check(cudaGraphGetNodes(graph, nullptr, &count));
        require(count == 2, "FA plus uniform fixup must be captured");
        std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(graph, nodes.data(), &count));
        for (auto node : nodes) {
            cudaGraphNodeType type;
            check(cudaGraphNodeGetType(node, &type));
            require(type == cudaGraphNodeTypeKernel, "only kernel nodes expected");
            cudaKernelNodeParams params = {};
            check(cudaGraphKernelNodeGetParams(node, &params));
            if (params.blockDim.y == 4) producer = params;
            else fixup = params;
        }
        require(producer.func && fixup.func, "producer and fixup found");
        require(producer.gridDim.x > KV_HEADS && producer.gridDim.x % KV_HEADS == 0, "uniform stream-K partitions");
        require(fixup.gridDim.x == KV_HEADS && fixup.gridDim.y == 16 && fixup.gridDim.z == 4, "uniform fixup grid");
    }
};
bool same_dim(dim3 a, dim3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

// Exercise all 65536 half patterns, including signed zeros, subnormals and NaNs.
// The reference repeats the old producer expression and materializes its F32 store.
static __global__ void transport_reference(float2 * ref, half2 * packed) {
    const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    const half2 value = __halves2half2(__ushort_as_half(2*i), __ushort_as_half(2*i + 1));
    float2 dst_val = make_float2(0.0f, 0.0f);
    const float KQ_crs = 1.0f;
    const float2 dst_val_add = __half22float2(value);
    dst_val.x += dst_val_add.x*KQ_crs;
    dst_val.y += dst_val_add.y*KQ_crs;
    ref[i] = dst_val;
    packed[i] = value;
}
static __global__ void transport_reconstruct(const half * packed, float * got) {
    const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    got[i] = fattn_half_partial_to_float(packed[i]);
}
void transport_test(cudaStream_t stream) {
    Buffer<float2> ref(32768), got(32768);
    Buffer<half2> packed(32768);
    transport_reference<<<128, 256, 0, stream>>>(ref.ptr, packed.ptr);
    check(cudaGetLastError());
    transport_reconstruct<<<256, 256, 0, stream>>>((const half *) packed.ptr, (float *) got.ptr);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    std::vector<uint32_t> a(65536), b(65536);
    check(cudaMemcpy(a.data(), ref.ptr, a.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), got.ptr, b.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    require(a == b, "all half patterns reconstruct the original F32 bits");
}

void shape(ggml_backend_cuda_context & cuda, int kv, int rows, bool strided) {
    auto * ctx = ggml_init({ggml_tensor_overhead()*16, nullptr, true});
    require(ctx != nullptr, "ggml context");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, rows, HEADS, 1);
    const size_t nq = size_t(D)*rows*HEADS*(strided ? 2 : 1);
    if (strided) {
        // Q view of a token-major joint Q|gate projection, with unused gate lanes.
        q->nb[1] = 2*D*HEADS*sizeof(float);
        q->nb[2] = 2*D*sizeof(float);
        q->nb[3] = nq*sizeof(float);
    }
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, kv, KV_HEADS, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, kv, KV_HEADS, 1);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, 32, 1, 1);
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/std::sqrt(float(D)), 0.0f, 0.0f);
    Buffer<float> dq(nq);
    Buffer<ggml_fp16_t> dk(ggml_nelements(k)), dv(ggml_nelements(v)), dm(ggml_nelements(mask));
    Guarded reference(ggml_nelements(out)), candidate(ggml_nelements(out));
    q->data = dq.ptr; k->data = dk.ptr; v->data = dv.ptr; mask->data = dm.ptr;
    std::vector<float> hq(nq);
    std::vector<ggml_fp16_t> hk(ggml_nelements(k)), hv(ggml_nelements(v)), hm(ggml_nelements(mask));
    for (int j = 0; j < 32; ++j) for (int i = 0; i < kv; ++i) {
        hm[size_t(j)*kv + i] = ggml_fp32_to_fp16(i <= kv - rows + j ? 0.0f : -INFINITY);
    }
    check(cudaMemcpy(dm.ptr, hm.data(), hm.size()*sizeof(ggml_fp16_t), cudaMemcpyHostToDevice));
    auto upload = [&](int phase) {
        std::mt19937 rng(20261021 + phase);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & x : hq) x = dist(rng);
        for (auto & x : hk) x = ggml_fp32_to_fp16(dist(rng));
        for (auto & x : hv) x = ggml_fp32_to_fp16(dist(rng));
        if (phase == 2) {
            const ggml_fp16_t edges[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                                         0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe01};
            for (size_t i = 0; i < hv.size(); ++i) hv[i] = edges[i % (sizeof(edges)/sizeof(edges[0]))];
        }
        check(cudaMemcpyAsync(dq.ptr, hq.data(), hq.size()*sizeof(float), cudaMemcpyHostToDevice, cuda.stream()));
        check(cudaMemcpyAsync(dk.ptr, hk.data(), hk.size()*sizeof(ggml_fp16_t), cudaMemcpyHostToDevice, cuda.stream()));
        check(cudaMemcpyAsync(dv.ptr, hv.data(), hv.size()*sizeof(ggml_fp16_t), cudaMemcpyHostToDevice, cuda.stream()));
    };
    auto launch = [&](int mode) {
        // Exercise default-on as well as explicit 1, always against the explicit 0 path.
        if (mode && strided) unsetenv("GGML_CUDA_FATTN_HALF_PARTIALS");
        else setenv("GGML_CUDA_FATTN_HALF_PARTIALS", mode ? "1" : "0", 1);
        out->data = mode ? candidate.data() : reference.data();
        ggml_cuda_flash_attn_ext_mma_f16_case<D, D, 16, 4>(cuda, out);
    };
    auto compare = [&](int phase) {
        check(cudaStreamSynchronize(cuda.stream()));
        const auto a = reference.read(), b = candidate.read();
        if (a != b) {
            size_t first = 0;
            while (a[first] == b[first]) ++first;
            std::fprintf(stderr, "KV=%d rows=%d strided=%d phase=%d word=%zu ref=%08x got=%08x\n",
                         kv, rows, strided, phase, first, a[first], b[first]);
            std::exit(1);
        }
        for (size_t i = 0; i < reference.count; ++i) {
            require(a[GUARD + i] != SENTINEL, "every output written");
            if (phase != 2) {
                float value;
                std::memcpy(&value, &a[GUARD + i], sizeof(value));
                require(std::isfinite(value), "random output finite");
            }
        }
    };
    upload(0); reference.reset(cuda.stream()); candidate.reset(cuda.stream());
    // Initialize both attributes, the pool and PDL launch caches before capture.
    launch(0);
    check(cudaStreamSynchronize(cuda.stream()));
    launch(1); compare(0);
    Graph graphs[2];
    for (int mode = 0; mode < 2; ++mode) {
        check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
        launch(mode);
        check(cudaStreamEndCapture(cuda.stream(), &graphs[mode].graph));
        check(cudaGraphInstantiate(&graphs[mode].exec, graphs[mode].graph, nullptr, nullptr, 0));
        graphs[mode].inspect();
    }
    require(same_dim(graphs[0].producer.gridDim, graphs[1].producer.gridDim) &&
            same_dim(graphs[0].producer.blockDim, graphs[1].producer.blockDim) &&
            graphs[0].producer.sharedMemBytes == graphs[1].producer.sharedMemBytes, "unchanged producer geometry");
    require(same_dim(graphs[0].fixup.gridDim, graphs[1].fixup.gridDim) &&
            same_dim(graphs[0].fixup.blockDim, graphs[1].fixup.blockDim), "unchanged fixup geometry");
    require((graphs[0].producer.func != graphs[1].producer.func) == (rows == 16), "half producer selected only for guarded shape");
    require((graphs[0].fixup.func != graphs[1].fixup.func) == (rows == 16), "matching fixup reader selected");
    const size_t blocks_per_tile = graphs[1].producer.gridDim.x / KV_HEADS;
    const size_t planned_bytes = (2*(blocks_per_tile - 1) + 1)*sizeof(float);
    require(graphs[0].fixup.sharedMemBytes == 0, "F32 fixup keeps its original shared-memory contract");
    require(graphs[1].fixup.sharedMemBytes == (rows == 16 ? planned_bytes : 0),
            "half fixup carries one block-wide scalar combine plan");
    for (int phase : {0, 1, 2, 1, 0}) {
        upload(phase); reference.reset(cuda.stream()); candidate.reset(cuda.stream());
        check(cudaStreamSynchronize(cuda.stream()));
        // Both captures may reuse the pool's scratch address; finish one before starting the other.
        for (auto & graph : graphs) {
            check(cudaGraphLaunch(graph.exec, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        }
        compare(phase);
    }
    std::printf("FA_HALF_EXACT KV=%d rows=%d strided=%d blocks=%u graph_replays=5\n",
                kv, rows, strided, graphs[0].producer.gridDim.x);
    ggml_free(ctx);
}
} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    require(backend != nullptr, "CUDA backend initialization");
    require(ggml_cuda_info().devices[0].cc == GGML_CUDA_CC_DGX_SPARK, "GB10 required");
    ggml_backend_cuda_context cuda(0);
    transport_test(cuda.stream());
    for (int kv : {512, 768, 1024}) for (bool strided : {false, true}) {
        shape(cuda, kv, 16, strided);
        shape(cuda, kv, 15, strided); // Query tail must keep F32 scratch.
    }
    unsetenv("GGML_CUDA_FATTN_HALF_PARTIALS");
    ggml_backend_free(backend);
    std::printf("FATTN_HALF_PARTIALS_PASS shapes=12 transport_patterns=65536 exact_output_bits=true\n");
    return 0;
}
