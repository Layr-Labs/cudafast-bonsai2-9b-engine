// Frozen promoted239e6e8 compact preparation, before warp query staging.
template <bool RAW, bool G_PRECOMPUTED>
// Leave more register room than i4's spill-producing four-CTA/64-register cap.
static __global__ void __launch_bounds__(256, 3) reference239_gdn_compact32_prepare(
        const float * q, const float * k, const float * g, const float * beta,
        const float * raw_dt_bias, const float * raw_a, gdn_compact32 * work,
        int H, int prefix_tokens, int64_t sq1, int64_t sq2,
        int64_t sb1, int64_t sb2, uint3 neqk1_magic) {
    constexpr int C = GDN_CHUNK;
    constexpr int PITCH = 132;
    const int h = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x;
    const int t0 = ch*C;
    const int n = min(C, prefix_tokens-t0);
    const int kh = fastmodulo(h, neqk1_magic);
    gdn_compact32 & out = work[ch*H+h];
    __shared__ __align__(16) float ks[C][PITCH];
    __shared__ float lower[C][C+1];
    __shared__ float decay[C], b[C];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < C*32; i += 256) {
        const int t = i/32, r = (i%32)*4;
        const float4 k4 = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
        *(float4 *) &ks[t][r] = k4;
    }
    if (tid < C) {
        float gv = 1.0f, bv = 0.0f;
        if (tid < n) {
            float g0 = g[h*sb1+(t0+tid)*sb2];
            bv = beta[h*sb1+(t0+tid)*sb2];
            static_assert(!(RAW && G_PRECOMPUTED), "raw and precomputed gates are exclusive");
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h];
                g0 = raw_a[h] * (x > 20.0f ? x : logf(1.0f+expf(x)));
                bv = 1.0f/(1.0f+expf(-bv));
            }
            gv = G_PRECOMPUTED ? g0 : expf(g0);
        }
        decay[tid] = gv;
        b[tid] = out.beta[tid] = bv;
    }
    __syncthreads();
    if (tid < C) {
        float p = 1.0f, s = 1.0f;
        for (int j = 0; j <= tid; ++j) { p *= decay[j]; }
        for (int j = tid+1; j < n; ++j) { s *= decay[j]; }
        out.prefix[tid] = p;
        out.suffix[tid] = s;
    }
    for (int ij = tid; ij < C*C; ij += 256) {
        const int i = ij/C, j = ij%C;
        float kk = 0.0f, qk = 0.0f, ratio = 1.0f;
        // i is warp-uniform, j is the lane. ALL lanes must execute the
        // shuffles, including j>i; only the Gram FMAs are triangular.
        if (i < n) {
            const float * qi = q+kh*sq1+(t0+i)*sq2;
            const float4 q_lane = make_float4(gdn_chunk32_load1(qi+j), gdn_chunk32_load1(qi+j+32),
                gdn_chunk32_load1(qi+j+64), gdn_chunk32_load1(qi+j+96));
            if (j <= i) { for (int t = j+1; t <= i; ++t) { ratio *= decay[t]; } }
#pragma unroll
            for (int r = 0; r < 128; r += 4) {
                const float qr = r < 32 ? q_lane.x : r < 64 ? q_lane.y : r < 96 ? q_lane.z : q_lane.w;
                const float4 q4 = make_float4(__shfl_sync(0xffffffffu, qr, (r%32)+0),
                    __shfl_sync(0xffffffffu, qr, (r%32)+1), __shfl_sync(0xffffffffu, qr, (r%32)+2),
                    __shfl_sync(0xffffffffu, qr, (r%32)+3));
                if (j <= i) {
                    const float4 ki = *(const float4 *) &ks[i][r];
                    const float4 kj = *(const float4 *) &ks[j][r];
                    kk = fmaf(ki.x, kj.x, kk); qk = fmaf(q4.x, kj.x, qk);
                    kk = fmaf(ki.y, kj.y, kk); qk = fmaf(q4.y, kj.y, qk);
                    kk = fmaf(ki.z, kj.z, kk); qk = fmaf(q4.z, kj.z, qk);
                    kk = fmaf(ki.w, kj.w, kk); qk = fmaf(q4.w, kj.w, qk);
                }
            }
        }
        lower[i][j] = j < i ? b[i]*ratio*kk : 0.0f;
        out.m[i][j] = ratio*qk;
    }
    __syncthreads();
    // Only 32 inverse columns, rather than 128 W + 128 U columns.
    // Solve the unit-diagonal system directly; there is no division or gate
    // quotient. Each inverse entry uses ascending-j f32 FMAs.
    if (tid < C) {
        float x[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            float rhs = i == tid ? 1.0f : 0.0f;
#pragma unroll
            for (int j = 0; j < i; ++j) { rhs = fmaf(-lower[i][j], x[j], rhs); }
            x[i] = rhs;
        }
#pragma unroll
        for (int i = 0; i < C; ++i) { out.inv[i][tid] = x[i]; }
    }
}
