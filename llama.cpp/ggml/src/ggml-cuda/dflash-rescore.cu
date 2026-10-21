#include "dflash-rescore.cuh"

#define DFLASH_RESCORE_WARPS     16
#define DFLASH_RESCORE_PER_WARP  4

// One warp per (column r, slot s): the dot product of head row q*chunk + offset with column r of b.
// The block stages column r in shared memory. In each 128-wide PQ2_0 block, lane l reads code byte l
// (elements 4l .. 4l+3, two bits each, low bits first) and the block's scale.
static __global__ void dflash_rescore_pq2_0(const char * a, const int64_t nb_a1, const int64_t R,
                                            const float * b, const int64_t s_b1, const int32_t * c,
                                            const int kc, const int n_chunks, const int chunk, const int K,
                                            const int S, const int n, float * out) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    extern __shared__ float xs[];
    const int r    = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const float * x = b + r*s_b1;
    for (int k = warp*WARP_SIZE + lane; k < K; k += DFLASH_RESCORE_WARPS*WARP_SIZE) {
        xs[k] = x[k];
    }
    __syncthreads();

    const int nb = K / QK_PQ2_0;
    for (int j = 0; j < DFLASH_RESCORE_PER_WARP; ++j) {
        const int s = (blockIdx.x*DFLASH_RESCORE_PER_WARP + j)*DFLASH_RESCORE_WARPS + warp;
        if (s >= S) {
            break;
        }
        const int     q      = s / kc;
        const int32_t offset = c[(s % kc) + (int64_t) (q + r*n_chunks)*kc];
        const int64_t id     = (int64_t) q*chunk + offset;

        float value = -INFINITY;
        if (offset >= 0 && offset < chunk && id < R) {
            const block_pq2_0 * row = (const block_pq2_0 *) (a + id*nb_a1);
            float sum = 0.0f;
            for (int ib = 0; ib < nb; ++ib) {
                const float   d  = __half2float(row[ib].d);
                const int     qb = row[ib].qs[lane];
                const float * xv = xs + ib*QK_PQ2_0 + 4*lane;
                const float   t  = ((qb      & 3) - 1)*xv[0] + (((qb >> 2) & 3) - 1)*xv[1]
                                 + (((qb >> 4) & 3) - 1)*xv[2] + (((qb >> 6) & 3) - 1)*xv[3];
                sum += d*t;
            }
            value = warp_reduce_sum(sum);
        }
        if (lane == 0) {
            out[s + (int64_t) r*S]                = value;
            out[(int64_t) S*n + s + (int64_t) r*S] = (float) (id < R ? id : 0);
        }
    }
}

bool ggml_cuda_dflash_rescore_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    const ggml_tensor * c = op->src[2];
    return a->type == GGML_TYPE_PQ2_0 && a->nb[0] == ggml_type_size(a->type) && a->ne[0] % QK_PQ2_0 == 0 &&
           a->ne[0] <= 12288 && b->type == GGML_TYPE_F32 && b->nb[0] == sizeof(float) && b->nb[1] % sizeof(float) == 0 &&
           c->type == GGML_TYPE_I32 && ggml_is_contiguous(c) && ggml_is_contiguous(op);
}

void ggml_cuda_op_dflash_rescore(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];
    GGML_ASSERT(ggml_cuda_dflash_rescore_supported(dst));

    const int chunk    = ggml_get_op_params_i32(dst, 0);
    const int K        = (int) a->ne[0];
    const int n        = (int) b->ne[1];
    const int kc       = (int) c->ne[0];
    const int n_chunks = (int) (c->ne[1] / n);
    const int S        = kc*n_chunks;

    const int    blocks_x = (S + DFLASH_RESCORE_WARPS*DFLASH_RESCORE_PER_WARP - 1) / (DFLASH_RESCORE_WARPS*DFLASH_RESCORE_PER_WARP);
    const size_t smem     = (size_t) K*sizeof(float);

    const ggml_cuda_kernel_launch_params launch_params(dim3(blocks_x, n, 1), dim3(WARP_SIZE, DFLASH_RESCORE_WARPS, 1),
                                                       smem, ctx.stream());
    ggml_cuda_kernel_launch(dflash_rescore_pq2_0, launch_params, (const char *) a->data, (int64_t) a->nb[1],
                            (int64_t) a->ne[1], (const float *) b->data, (int64_t) (b->nb[1]/sizeof(float)),
                            (const int32_t *) c->data, kc, n_chunks, chunk, K, S, n, (float *) dst->data);
}
