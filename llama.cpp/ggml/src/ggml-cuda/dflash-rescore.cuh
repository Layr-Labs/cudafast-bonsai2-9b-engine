#include "common.cuh"

// GGML_OP_DFLASH_RESCORE: exact dot products of shortlisted drafter-head rows (see ggml_dflash_rescore).
void ggml_cuda_op_dflash_rescore(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_dflash_rescore_supported(const ggml_tensor * op);
