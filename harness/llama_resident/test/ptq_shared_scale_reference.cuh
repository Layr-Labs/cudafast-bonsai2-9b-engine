#pragma once
template <bool SGN = false>
static __device__ __forceinline__ void reference168_ptq1_j16_compute(float (&sum)[1][4], const ggml_cuda_ptq1_j16_stage & st, const int c) {
    // dA of the C rows 2c and 2c+1 of the 8-row tile: the scale of row r was loaded by lanes 4r..4r+3.
    const int xd   = (int) (st.w6 >> 16);
    const int d_lo = __shfl_sync(0xFFFFFFFF, xd, 8*c + 0);
    const int d_hi = __shfl_sync(0xFFFFFFFF, xd, 8*c + 4);
    const float dA[2] = {
        __half2float(__ushort_as_half((unsigned short) d_lo)),
        __half2float(__ushort_as_half((unsigned short) d_hi)),
    };
    const float dB[2][4] = {
        {st.yd[0].x, st.yd[0].y, st.yd[0].z, st.yd[0].w},
        {st.yd[1].x, st.yd[1].y, st.yd[1].z, st.yd[1].w},
    };
    uint32_t b[4][2];
    ggml_cuda_ptq1_j16_decode(b, st, c);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
        if constexpr (SGN) {
            const int b0 = (int) ((b[s][0] + 0x7F7F7F7Fu) ^ 0x80808080u);
            const int b1 = (int) ((b[s][1] + 0x7F7F7F7Fu) ^ 0x80808080u);
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], b0, b1);    // MAGIC + sum(a*(t-1))
        } else {
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], -1, -1);       // MAGIC - sum(a)
            ggml_cuda_ptq1_j16_mma_u8(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], (int) b[s][0], (int) b[s][1]);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float cf = __int_as_float(C[l]) - 12582912.0f; // == (float) idot, exactly
            sum[0][l] += cf*dA[l % 2]*dB[l / 2][s];
        }
    }
}
