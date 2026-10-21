#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Returns false without launching when the existing streaming selector is unavailable.
bool ggml_cuda_op_top_k_values(ggml_backend_cuda_context & ctx, ggml_tensor * ids, ggml_tensor * values);
