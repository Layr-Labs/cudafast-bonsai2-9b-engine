#pragma once
// Frozen promoted759 full-kernel reference.
template <bool GROUPED>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_dod_qkv_reference(ggml_cuda_dod_qkv_args a, float * tmp_fixup) {
    constexpr int GB = sizeof(block_ptq1_0), I = 128, J = 16, S = 3, SPS = 2;
    constexpr int K = 4096, nkb = K/QK_PQ2_0, XB = I*GB;
    constexpr int YB = J*int(sizeof(block_q8_1_mmq));
    constexpr int SB = ggml_cuda_dl_ring_stage_bytes<GGML_CUDA_DL_FMT_PTQ1S, SPS>();
    const int problem = GROUPED ? int(blockIdx.x)/48 : 0;
    const int b = GROUPED ? int(blockIdx.x)%48 : int(blockIdx.x);
    const int rows = GROUPED ? (problem == 0 ? 8192 : 1024) : 10240;
    const int row_base = GROUPED ? (problem == 0 ? 0 : 8192 + (problem - 1)*1024) : 0;
    const int begin = ggml_cuda_dod_qkv_start(b, rows, nkb);
    const int total = ggml_cuda_dod_qkv_start(b + 1, rows, nkb) - begin;
    int it = begin/nkb, st = begin % nkb;
    const int nstages = (total + SPS - 1)/SPS;
    const char * x = a.x + int64_t(row_base)*nkb*GB;
    const int * y = a.y;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const uint32_t yb_bytes = uint32_t(a.ncols*int(sizeof(block_q8_1_mmq)));
    extern __shared__ __align__(128) char ggml_cuda_dod_qkv_smem[];
    char * smem = ggml_cuda_dod_qkv_smem;
    uint64_t * full = (uint64_t *)(smem + S*SB);
    if (tid == 0) {
#pragma unroll
        for (int k = 0; k < S; ++k) { ggml_cuda_dl_mbar_init(&full[k], 1); }
        ggml_cuda_dl_mbar_init_fence();
    }
    __syncthreads();
    const uint64_t wpol = ggml_cuda_dl_policy_evict_first();
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            ggml_cuda_dl_mbar_expect_tx(&full[k], uint32_t(n)*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + k*SB, x + int64_t(begin + k*SPS)*XB, uint32_t(n)*XB, &full[k], wpol);
        }
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            for (int j = 0; j < n; ++j) {
                const int stj = (begin + k*SPS + j) % nkb;
                ggml_cuda_dl_bulk_g2s(smem + k*SB + SPS*XB + j*YB, (const char *)y + int64_t(stj)*yb_bytes, yb_bytes, &full[k]);
            }
        }
    }
    const int w = threadIdx.y, g = threadIdx.x/4, c = threadIdx.x%4;
    float sum[1][4] = {{0.0f}};
#pragma unroll 1
    for (int k = 0; k < nstages; ++k) {
        const int s = k%S, n = min(SPS, total - k*SPS);
        ggml_cuda_dl_mbar_wait(&full[s], uint32_t((k/S)&1));
#pragma unroll
        for (int j = 0; j < SPS; ++j) {
            if (j < n) {
                const char * sx = smem + s*SB + j*XB + (w*8 + g)*GB;
                const char * sy = smem + s*SB + SPS*XB + j*YB + g*int(sizeof(block_q8_1_mmq));
                ggml_cuda_dl_ring_step<GGML_CUDA_DL_FMT_PTQ1S>(sum, sx, sy, c);
                ++st;
                if (st == nkb || k*SPS + j == total - 1) {
                    if (st == nkb) {
                        int row = row_base + it*I + w*8, stride;
                        float * dst = ggml_cuda_dod_qkv_dst(a, row, stride);
                        ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride, a.ncols - 1);
                        st = 0;
                        ++it;
                    } else {
                        ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + int64_t(blockIdx.x)*(J*I), w*8, I, J - 1);
                    }
#pragma unroll
                    for (int l = 0; l < 4; ++l) { sum[0][l] = 0.0f; }
                }
            }
        }
        if (k + S < nstages) { __syncthreads(); }
        if (tid == 0 && k + S < nstages) {
            const int kk = k + S, nn = min(SPS, total - kk*SPS), gsk = begin + kk*SPS;
            ggml_cuda_dl_mbar_expect_tx(&full[s], uint32_t(nn)*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + s*SB, x + int64_t(gsk)*XB, uint32_t(nn)*XB, &full[s], wpol);
            for (int j = 0; j < nn; ++j) {
                ggml_cuda_dl_bulk_g2s(smem + s*SB + SPS*XB + j*YB,
                    (const char *)y + int64_t((gsk + j)%nkb)*yb_bytes, yb_bytes, &full[s]);
            }
        }
    }
}

