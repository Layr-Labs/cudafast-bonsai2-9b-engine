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

// optional parts of the conv front-end kernel
struct ggml_cuda_ssm_conv_front_ext {
    int             hold        = 0;       // out(t, c) at x(t - hold, c) (see below), 0 when disjoint
    bool            copy_states = false;   // out overlaps conv_states: read a copy
    const float *   gather_base = nullptr; // conv_states is row gather_ids[0] (of gather_row floats) of gather_base:
    const int32_t * gather_ids  = nullptr; //   the GET_ROWS that produced it is skipped
    int64_t         gather_row  = 0;
    float *         qk          = nullptr; // channels [0, qk_channels) go l2-normalized (eps) to qk [128, heads, T]
    int             qk_layout   = 0;       // 2 = packed row + head-major; 3 = C16 F16 prefix + token-major F32 tail
    int64_t         qk_channels = 0;       //   instead of out (their L2_NORM is skipped)
    float           eps         = 0.0f;
    int64_t         x_tail_t0   = INT64_MAX; // x rows [x_tail_t0, T) (that qk lands on) are read from a copy
    const float *   zero_row    = nullptr;   // with gather: when the gathered row is zero_row, its values are
    float           zero_scale  = 0.0f;      //   zero_scale * x + zero_bias (a deferred in-place SCALE; a snapshot
    float           zero_bias   = 0.0f;      //   of this kernel overwrites the row)
};

// SSM_CONV (+ SILU) of conv_input = CONCAT(conv_states, TRANSPOSE(x)) along dim 0 together with the snapshot copies,
// read from conv_states and x directly: conv_input itself is never written. The per-output arithmetic is the one of
// ssm_conv_f32 / ssm_conv_long_token_f32 (and l2_norm_f32 for qk). out is the SSM_CONV node, or its SILU node when
// apply_silu. out may alias x shifted back by hold token rows (out(t, c) at x(t - hold, c), 0 <= hold <=
// GGML_CUDA_SSM_CONV_MAX_HOLD, same token stride). It must not overlap a snapshot; qk must not overlap out, a
// snapshot or x rows before x_tail_t0. A snapshot may be the gathered conv-state row itself.
void ggml_cuda_ssm_conv_front(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_input, const ggml_tensor * ssm,
                              const ggml_tensor * out, bool apply_silu, const ggml_cuda_ssm_conv_snaps & snaps,
                              const ggml_cuda_ssm_conv_front_ext & ext);
