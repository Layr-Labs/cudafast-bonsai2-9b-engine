#pragma once

#include "ggml.h"

struct llm_last_attn_rows {
    ggml_tensor * query = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask = nullptr;
};

// Select a requested query, not a hardcoded final position. Sixteen identical
// queries retain the ordinary PQ2 J16 MMQ path and its Q8 D4 FP32 scales;
// a one-row matmul would select MMVQ with fp16 activation scales instead.
// No K/V or feature tensor is changed. All gathers execute on every replay.
// J16 and the smaller attention shape can change FP32 reduction order, so
// this algebraically equivalent graph still requires model replay checks.
inline llm_last_attn_rows llm_build_last_attn_rows(ggml_context * ctx,
        ggml_tensor * input, ggml_tensor * positions, ggml_tensor * mask,
        ggml_tensor * output_ids, int64_t n_seqs, bool enabled) {
    if (!enabled || !ctx || !input || !positions || !mask || !output_ids || n_seqs != 1 ||
            input->type != GGML_TYPE_F32 || input->ne[0] != 4096 || input->ne[1] <= 16 ||
            input->ne[2] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            output_ids->type != GGML_TYPE_I32 || ggml_nelements(output_ids) != 1 ||
            !ggml_is_contiguous(output_ids) || positions->type != GGML_TYPE_I32 ||
            positions->ne[0] != 4*input->ne[1] || !ggml_is_vector(positions) ||
            !ggml_is_contiguous(positions) || mask->type != GGML_TYPE_F16 ||
            mask->ne[0] <= 0 || mask->ne[1] != input->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || !ggml_is_contiguous(mask)) {
        return {};
    }

    // CUDA REPEAT does not implement I32. CONCAT retains the exact integer
    // index and stays on device; the graph topology is independent of its value.
    ggml_tensor * ids = output_ids;
    for (int i = 0; i < 4; ++i) ids = ggml_concat(ctx, ids, ids, 0);
    ggml_tensor * pos_ids = ggml_concat(ctx, ids, ids, 1);
    pos_ids = ggml_concat(ctx, pos_ids, pos_ids, 1); // [16, 4]
    ggml_tensor * pos = ggml_reshape_3d(ctx, positions, 1, input->ne[1], 4);
    pos = ggml_reshape_1d(ctx, ggml_get_rows(ctx, pos, pos_ids), 4*16);

    // GET_ROWS returns F32 for an F16 input. The cast is an exact round trip
    // for the original mask values (including its +/-infinity entries).
    ggml_tensor * selected_mask = ggml_cast(ctx, ggml_get_rows(ctx, mask, ids), GGML_TYPE_F16);
    return {ggml_get_rows(ctx, input, ids), pos, selected_mask};
}
