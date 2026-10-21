#include "common.cuh"

// GGML_OP_DFLASH_WALK: the greedy DFlash2 selector walk on the device (see ggml_dflash_walk).
void ggml_cuda_op_dflash_walk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
