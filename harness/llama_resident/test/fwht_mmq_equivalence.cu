// Actual standalone FWHT -> MMQ D4 versus fused writer: every byte/guard.
// Wide16/512, signs, dtype, padding, changed-input/sign multi-replay.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "fwht.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(2);
    }
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261015);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    int cases = 0;
    for (int n : {4096}) for (int multiple : {1, 2, 3})
    for (int rows : {16, 512}) for (bool signs_on : {false, true})
    for (bool half_on : {false, true}) for (bool padded : {false, true}) {
        const int k = n * multiple;
        const int kp = k + (padded ? n : 0);
        const size_t count = size_t(k) * rows;
        std::vector<float> input(count), signs(k);
        for (size_t i = 0; i < count; ++i) {
            input[i] = i % 17 == 0 ? 0.0f : dist(rng);
        }
        for (int i = 0; i < k; ++i) signs[i] = (rng() & 1) ? 1.0f : -1.0f;
        std::vector<ggml_fp16_t> half_input(count);
        ggml_fp32_to_fp16_row(input.data(), half_input.data(), count);
        const size_t input_bytes = count * (half_on ? sizeof(ggml_fp16_t) : sizeof(float));
        void * dx = nullptr;
        float * ds = nullptr, * dt = nullptr;
        void * qr = nullptr, * qf = nullptr;
        const size_t qbytes = size_t(kp) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);
        constexpr size_t guard = 64;
        check(cudaMalloc(&dx, input_bytes));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&dt, count * sizeof(float)));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        check(cudaMemcpy(dx, half_on ? static_cast<void *>(half_input.data()) : static_cast<void *>(input.data()), input_bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(ds, signs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
        ggml_tensor * x = ggml_new_tensor_2d(ctx, half_on ? GGML_TYPE_F16 : GGML_TYPE_F32, n, multiple * rows);
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, multiple * rows);
        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        x->data = dx; t->data = dt; s->data = ds;
        std::vector<unsigned char> ref(qbytes + guard), fused(qbytes + guard);
        for (bool capture : {false, true}) {
            check(cudaMemset(qr, 0xa5, qbytes + guard));
            check(cudaMemset(qf, 0xa5, qbytes + guard));
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            const bool ok = signs_on ? ggml_cuda_op_fwht_signed(cuda, x, s, t) : ggml_cuda_op_fwht(cuda, x, t);
            if (!ok) return 3;
            quantize_mmq_q8_1_cuda(dt, nullptr, qr, GGML_TYPE_PQ2_0, k, k, k*rows, k*rows, kp, rows, 1, 1, cuda.stream());
            fwht_quantize_mmq_d4_cuda(dx, x->type, signs_on ? ds : nullptr, qf, k, kp, rows, cuda.stream());
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (capture) {
                check(cudaStreamEndCapture(cuda.stream(), &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            for (int replay=0; replay<(capture ? 3 : 1); ++replay) {
                if (capture) {
                    if (replay) {
                        for (size_t i=0; i<count; ++i) input[i]=i%19==0 ? 0.0f : dist(rng);
                        for (int i=0; i<k; ++i) signs[i]=(rng()&1) ? 1.0f : -1.0f;
                        ggml_fp32_to_fp16_row(input.data(),half_input.data(),count);
                        check(cudaMemcpy(dx,half_on ? static_cast<void *>(half_input.data()) : static_cast<void *>(input.data()),input_bytes,cudaMemcpyHostToDevice));
                        check(cudaMemcpy(ds,signs.data(),k*sizeof(float),cudaMemcpyHostToDevice));
                    }
                    check(cudaGraphLaunch(exec,cuda.stream()));
                }
            check(cudaStreamSynchronize(cuda.stream()));
            check(cudaMemcpy(ref.data(), qr, qbytes + guard, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(fused.data(), qf, qbytes + guard, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < qbytes + guard; ++i) {
                if (ref[i] != fused[i] || (i >= qbytes && (ref[i] != 0xa5 || fused[i] != 0xa5))) {
                    std::fprintf(stderr, "mismatch n=%d k=%d rows=%d signs=%d half=%d pad=%d graph=%d byte=%zu ref=%u fused=%u\n", n,k,rows,signs_on,half_on,padded,capture,i,ref[i],fused[i]);
                    return 1;
                }
            }
            ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dx)); check(cudaFree(ds)); check(cudaFree(dt)); check(cudaFree(qr)); check(cudaFree(qf));
    }
    ggml_backend_free(backend);
    std::printf("fwht_mmq_equivalence_ok cases=%d bytes_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
