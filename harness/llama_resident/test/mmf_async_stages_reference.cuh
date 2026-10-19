// Frozen promoted221 paired kernel; uses untouched leader chain/helper.
static __global__ void reference235_mul_mat_f_ksplit_pair(
        const float * x0, const float * x1, const float * y, float * dst0, float * dst1,
        const int ncols, const int nrb, const int stride_row0, const int stride_row1, const int stride_col_y,
        const int stride_col_dst0, const int stride_col_dst1) {
#if defined(AMPERE_MMA_AVAILABLE) && !defined(VOLTA_MMA_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    typedef tile<16, 8, float> tile_A;
    typedef tile<8,  8, float> tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int rows_per_block = MMF_ROWS_PER_BLOCK;
    constexpr int nwarps         = mmf_ksplit_nwarps;
    constexpr int warp_size      = ggml_cuda_get_physical_warp_size();
    constexpr int ntA            = rows_per_block / tile_A::I;
    constexpr int ntB            = (mmf_ksplit_cols + tile_B::I - 1) / tile_B::I;

    int b = blockIdx.x;
    const int w   = b % nwarps;  b /= nwarps;
    const int itB = b % ntB;     b /= ntB;
    const int itA = b % ntA;     b /= ntA;
    const int rb  = b % nrb;
    const int m   = b / nrb;

    const float * x              = m ? x1 : x0;
    float *       dst            = m ? dst1 : dst0;
    const int     stride_row     = m ? stride_row1 : stride_row0;
    const int     stride_col_dst = m ? stride_col_dst1 : stride_col_dst0;

    x += (int64_t(rb)*rows_per_block + itA*tile_A::I)*stride_row;
    y += int64_t(itB)*tile_B::I*stride_col_y;

    mul_mat_f_ksplit_prefetch_x(x, ncols, stride_row, w);
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();

    extern __shared__ char data_mmf_ksplit[];
    float * tile_xy = (float *) data_mmf_ksplit;
    __shared__ float acc[tile_C::ne][warp_size];

    tile_C C;
    mul_mat_f_ksplit_chain(x, y, tile_xy, ncols, stride_row, stride_col_y, w, itB, C);

#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        acc[l][threadIdx.x] = C.x[l];
    }
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (w == 0) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < nwarps; ++r) {
                sum += *cluster.map_shared_rank(&acc[l][threadIdx.x], r);
            }
            const int i = itA*tile_C::I + tile_C::get_i(l);
            const int j = itB*tile_C::J + tile_C::get_j(l);
            dst[j*stride_col_dst + rb*rows_per_block + i] = sum;
        }
    }
    cluster.sync(); // the other ranks' shared memory stays valid until rank 0 has read it
#else
    GGML_UNUSED_VARS(x0, x1, y, dst0, dst1, ncols, nrb, stride_row0, stride_row1, stride_col_y, stride_col_dst0,
        stride_col_dst1);
    NO_DEVICE_CODE;
#endif
}
