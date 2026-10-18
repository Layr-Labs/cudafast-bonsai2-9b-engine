#pragma once

#include "ggml.h"

// The two unchanged GDN gate projections read the F32 normalized activation.
// Expanding them before QKV/conv keeps that activation hot and lets the normal
// graph allocator account for their longer lifetimes. There is no early write
// into an allocation whose lifetime was planned for a later node, and no
// auxiliary stream, scratch registry, or cross-execution state.
inline bool llm_schedule_prefill_gates(ggml_cgraph * graph, ggml_tensor * input,
        ggml_tensor * beta, ggml_tensor * alpha, int64_t n_tokens, int64_t n_seqs, bool enabled) {
    if (!enabled || !graph || !input || n_seqs != 1 || n_tokens <= 16 ||
            input->type != GGML_TYPE_F32 || input->ne[0] != 4096 ||
            input->ne[1] != n_tokens || input->ne[2] != 1 || input->ne[3] != 1) {
        return false;
    }
    ggml_tensor * pair[] = { beta, alpha };
    for (ggml_tensor * t : pair) {
        if (!t || t->op != GGML_OP_MUL_MAT || t->src[1] != input || !t->src[0] ||
                t->type != GGML_TYPE_F32 || t->src[0]->type != GGML_TYPE_F32 ||
                t->src[0]->ne[0] != 4096 || t->src[0]->ne[1] != 32 ||
                t->src[0]->ne[2] != 1 || t->src[0]->ne[3] != 1 ||
                t->ne[0] != 32 || t->ne[1] != n_tokens || t->ne[2] != 1 || t->ne[3] != 1) {
            return false;
        }
    }
    // Preserve beta-before-alpha and every original matmul argument. Only
    // their topological position changes; both still execute on every call.
    ggml_build_forward_expand(graph, beta);
    ggml_build_forward_expand(graph, alpha);
    return true;
}
