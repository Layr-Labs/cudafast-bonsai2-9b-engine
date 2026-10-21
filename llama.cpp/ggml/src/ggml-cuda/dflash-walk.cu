#include "dflash-walk.cuh"

#include <cstdlib>

// One thread runs the walk: n_steps dependent argmaxes over top_k scores each. It keeps
// std::max_element's rule (replace only on a strict "best < candidate", so the first maximum wins and a
// NaN never replaces the best) and the host walk's float -> int32 truncation, so every id equals the one
// common/speculative.cpp would pick from the same lattice.
static __global__ void dflash_walk_f32(const char * lattice, const int64_t nb1, const int top_k, const int row0,
                                       const int n_steps, int32_t * out) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (threadIdx.x != 0) {
        return;
    }
    int predecessor = 0;
    for (int i = 0; i < n_steps; ++i) {
        const float * row    = (const float *) (lattice + (int64_t) (row0 + i)*nb1);
        const float * scores = row + top_k + (int64_t) predecessor*top_k;
        int   best   = 0;
        float best_v = scores[0];
        for (int k = 1; k < top_k; ++k) {
            const float v = scores[k];
            if (best_v < v) {
                best_v = v;
                best   = k;
            }
        }
        predecessor = best;
        out[i] = (int32_t) row[predecessor];
    }
}

// The same walk with its argmaxes taken in parallel: every (step i, predecessor slot p) pair gets the
// first maximum of its top_k scores with the serial rule above, and the step's candidate ids are cast to
// int32 the same way; all of that is independent of the path. Thread 0 then follows the path through
// shared memory only, so the walk's critical path is one round of global loads instead of n_steps.
#define DFLASH_WALK_PAR_THREADS 256
#define DFLASH_WALK_PAR_MAX     4096 // n_steps*top_k entries held in shared memory

static __global__ void dflash_walk_par_f32(const char * lattice, const int64_t nb1, const int top_k, const int row0,
                                           const int n_steps, int32_t * out) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    extern __shared__ int32_t walk_smem[];
    int32_t * best_slot = walk_smem;                    // [n_steps][top_k]: best successor of predecessor p
    int32_t * cand_id   = walk_smem + n_steps*top_k;    // [n_steps][top_k]: candidate ids

    for (int e = threadIdx.x; e < n_steps*top_k; e += blockDim.x) {
        const int     i   = e / top_k;
        const int     p   = e % top_k;
        const float * row = (const float *) (lattice + (int64_t) (row0 + i)*nb1);
        const float * scores = row + top_k + (int64_t) p*top_k;
        int   best   = 0;
        float best_v = scores[0];
        for (int k = 1; k < top_k; ++k) {
            const float v = scores[k];
            if (best_v < v) {
                best_v = v;
                best   = k;
            }
        }
        best_slot[e] = best;
        cand_id[e]   = (int32_t) row[p];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int predecessor = 0;
        for (int i = 0; i < n_steps; ++i) {
            predecessor = best_slot[i*top_k + predecessor];
            out[i]      = cand_id[i*top_k + predecessor];
        }
    }
}

void ggml_cuda_op_dflash_walk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32);

    const int top_k   = ggml_get_op_params_i32(dst, 0);
    const int row0    = ggml_get_op_params_i32(dst, 1);
    const int n_steps = ggml_get_op_params_i32(dst, 2);

    // LLAMA_DFLASH_WALK_SERIAL=1 keeps the one-thread walk.
    static const bool serial = getenv("LLAMA_DFLASH_WALK_SERIAL") != nullptr && atoi(getenv("LLAMA_DFLASH_WALK_SERIAL")) != 0;
    if (!serial && (int64_t) n_steps*top_k <= DFLASH_WALK_PAR_MAX) {
        const size_t smem = 2*(size_t) n_steps*top_k*sizeof(int32_t);
        const ggml_cuda_kernel_launch_params par_params(dim3(1, 1, 1), dim3(DFLASH_WALK_PAR_THREADS, 1, 1), smem,
                                                        ctx.stream());
        ggml_cuda_kernel_launch(dflash_walk_par_f32, par_params, (const char *) src0->data, (int64_t) src0->nb[1], top_k,
                                row0, n_steps, (int32_t *) dst->data);
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params(dim3(1, 1, 1), dim3(WARP_SIZE, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(dflash_walk_f32, launch_params, (const char *) src0->data, (int64_t) src0->nb[1], top_k, row0,
                            n_steps, (int32_t *) dst->data);
}
