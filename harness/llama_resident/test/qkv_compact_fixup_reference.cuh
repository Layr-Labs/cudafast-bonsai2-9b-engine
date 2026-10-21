#pragma once
template <bool GROUPED>
__launch_bounds__(128, 1)
static __global__ void reference203_qkv_fixup(ggml_cuda_dod_qkv_args a, const float * partials) {
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    constexpr int nkb = 4096/QK_PQ2_0;
    const int problem = GROUPED ? int(blockIdx.x)/48 : 0;
    const int b = GROUPED ? int(blockIdx.x)%48 : int(blockIdx.x);
    const int rows = GROUPED ? (problem == 0 ? 8192 : 1024) : 10240;
    const int row_base = GROUPED ? (problem == 0 ? 0 : 8192 + (problem - 1)*1024) : 0;
    const int begin = ggml_cuda_dod_qkv_start(b, rows, nkb);
    const int end = ggml_cuda_dod_qkv_start(b + 1, rows, nkb);
    if (begin == end || begin%nkb == 0 || (begin/nkb == end/nkb && end%nkb != 0)) { return; }
    float sum[4] = {0.0f};
    int stop = begin;
    for (int prev = b - 1; prev >= 0; --prev) {
        const int start = ggml_cuda_dod_qkv_start(prev, rows, nkb);
        if (start != stop) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum[j] += partials[int64_t((GROUPED ? problem*48 : 0) + prev)*2048 +
                                   (4*j + threadIdx.y)*128 + blockIdx.y*32 + threadIdx.x];
            }
            if (start%nkb == 0 || start/nkb < begin/nkb) { break; }
        }
        stop = start;
    }
    int row = row_base + (begin/nkb)*128 + blockIdx.y*32 + threadIdx.x, stride;
    float * dst = ggml_cuda_dod_qkv_dst(a, row, stride);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = 4*j + threadIdx.y;
        if (col < a.ncols) { dst[col*stride + row] += sum[j]; }
    }
}
