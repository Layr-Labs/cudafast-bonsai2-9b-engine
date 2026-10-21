#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// C62: RMS_NORM * w -> MROPE/IMROPE (n_offs 0, no freq factors) [-> K SET_ROWS [+ V SET_ROWS of the same heads]]
bool ggml_cuda_rms_norm_mul_mrope_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * rope);
void ggml_cuda_op_rms_norm_mul_mrope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul,
        ggml_tensor * rope, ggml_tensor * set_rows, ggml_tensor * v_view, ggml_tensor * v_set_rows);

void ggml_cuda_op_rms_norm_mul_mrope_qk(ggml_backend_cuda_context & ctx,
        ggml_tensor * qrms, ggml_tensor * qmul, ggml_tensor * qrope,
        ggml_tensor * krms, ggml_tensor * kmul, ggml_tensor * krope,
        ggml_tensor * kstore, ggml_tensor * vview, ggml_tensor * vstore);
