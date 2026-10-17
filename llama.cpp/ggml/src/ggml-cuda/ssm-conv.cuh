#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#define GGML_CUDA_SSM_CONV_MAX_SNAPS 32
#define GGML_CUDA_SSM_CONV_MAX_HOLD  4

// conv-state snapshots taken from conv_input = CONCAT(conv_states, TRANSPOSE(x)): snapshot k is the contiguous
// copy of conv_input columns [col[k], col[k] + width) of every channel, element (j, c) at dst[k][c * width + j]
struct ggml_cuda_ssm_conv_snaps {
    float * dst[GGML_CUDA_SSM_CONV_MAX_SNAPS];
    int     col[GGML_CUDA_SSM_CONV_MAX_SNAPS];
    int     n;
    int     width;
};

// SSM_CONV (+ SILU) of conv_input = CONCAT(conv_states, TRANSPOSE(x)) along dim 0 together with the snapshot copies,
// read from conv_states and x directly: conv_input itself is never written. The per-output arithmetic is the one of
// ssm_conv_f32 / ssm_conv_long_token_f32. out is the SSM_CONV node, or its SILU node when apply_silu. out may alias
// x shifted back by hold token rows (out(t, c) at x(t - hold, c), 0 <= hold <= GGML_CUDA_SSM_CONV_MAX_HOLD, same token
// stride); copy_states when it overlaps conv_states. It must not overlap a snapshot.
void ggml_cuda_ssm_conv_front(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_input, const ggml_tensor * ssm,
                              const ggml_tensor * out, bool apply_silu, const ggml_cuda_ssm_conv_snaps & snaps,
                              int hold, bool copy_states);
