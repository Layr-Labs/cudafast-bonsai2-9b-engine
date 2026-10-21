#pragma once

#include "ggml.h"

// Move only the original 16-row F32 beta/alpha projections. The backend may
// execute them on a side stream; their outputs therefore need the ordinary
// graph allocator to see the entire early-to-GDN lifetime.
inline bool llm_schedule_decode_gates(ggml_cgraph * graph, ggml_tensor * input,
        ggml_tensor * beta, ggml_tensor * alpha, int64_t tokens, int64_t sequences, bool enabled) {
    if (!enabled || !graph || !input || tokens != 16 || sequences != 1 ||
        input->type != GGML_TYPE_F32 || input->ne[0] != 4096 || input->ne[1] != tokens ||
        input->ne[2] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input)) return false;
    ggml_tensor * pair[] = { beta, alpha };
    for (ggml_tensor * t : pair) {
        if (!t || t->op != GGML_OP_MUL_MAT || t->src[1] != input || !t->src[0] ||
            t->type != GGML_TYPE_F32 || t->src[0]->type != GGML_TYPE_F32 ||
            t->src[0]->ne[0] != 4096 || t->src[0]->ne[1] != 32 ||
            t->src[0]->ne[2] != 1 || t->src[0]->ne[3] != 1 ||
            t->ne[0] != 32 || t->ne[1] != tokens || t->ne[2] != 1 || t->ne[3] != 1 ||
            !ggml_is_contiguous(t) || !ggml_is_contiguous(t->src[0])) return false;
    }
    ggml_build_forward_expand(graph, beta);
    ggml_build_forward_expand(graph, alpha);
    return true;
}

// src[9] is a lifetime dependency only, not a numerical GDN operand. Attach it
// before the first forward expansion: graph traversal, cloning, CUDA graph
// fingerprints and gallocr all enumerate GGML_MAX_SRC. This keeps the F32
// activation alive until the side-stream completion join at GDN, even if QKV
// or conv would otherwise reuse it. The math kernels read only src[0..8].
inline void llm_hold_decode_gate_input(ggml_tensor * gdn, ggml_tensor * input) {
    if (!input) return;
    GGML_ASSERT(GGML_MAX_SRC > 9 && gdn->op == GGML_OP_GATED_DELTA_NET && !gdn->src[9]);
    GGML_ASSERT(gdn->src[6] == nullptr && gdn->op_params[1] != 0);
    GGML_ASSERT(gdn->src[2]->ne[0] == 128 && gdn->src[2]->ne[1] == 32 &&
                gdn->src[2]->ne[2] == 16 && gdn->src[2]->ne[3] == 1);
    gdn->src[9] = input;
}
