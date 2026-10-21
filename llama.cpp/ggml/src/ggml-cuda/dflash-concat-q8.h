#pragma once

#include "ggml-impl.h"
#include <cstdint>
#include <initializer_list>

// A deliberately narrow graph-only matcher. Quantization is launched at the
// original CONCAT position, while both source slabs are still live. Only its
// two ordinary Q8_0 K/V consumers may observe the virtual concatenated tensor.
static inline int ggml_cuda_dflash_concat_q8_match(const ggml_cgraph * graph, int index) {
    if (index < 0 || index >= graph->n_nodes) return -1;
    const ggml_tensor * cat = graph->nodes[index];
    const ggml_tensor * a = cat->src[0], * b = cat->src[1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 1 || !a || !b ||
            cat->type != GGML_TYPE_F32 || cat->view_src || !cat->data ||
            (cat->flags & GGML_TENSOR_FLAG_OUTPUT) || !(cat->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            cat->ne[0] != 4096 || cat->ne[1] != 32 || cat->ne[2] != 1 || cat->ne[3] != 1 ||
            !ggml_is_contiguous(cat)) return -1;
    for (const ggml_tensor * x : {a, b}) {
        if (x->type != GGML_TYPE_F32 || !x->data || !ggml_is_contiguous(x) ||
                x->ne[0] != 4096 || x->ne[1] != 16 || x->ne[2] != 1 || x->ne[3] != 1 ||
                (uintptr_t(x->data) & 15) != 0) return -1;
    }
    int consumers = 0, first = -1;
    for (int j = 0; j < graph->n_nodes; ++j) {
        const ggml_tensor * mm = graph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) if (mm->src[s] == cat) {
            const ggml_tensor * w = mm->src[0];
            if (j <= index || s != 1 || mm->op != GGML_OP_MUL_MAT || !w ||
                    w->type != GGML_TYPE_Q8_0 || w->ne[0] != 4096 || w->ne[1] != 1024 ||
                    w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
                    mm->type != GGML_TYPE_F32 || mm->ne[0] != 1024 || mm->ne[1] != 32 ||
                    mm->ne[2] != 1 || mm->ne[3] != 1 || !ggml_is_contiguous(mm) ||
                    !(mm->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                    ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) return -1;
            if (++consumers > 2) return -1;
            if (first < 0) first = j;
        }
    }
    return consumers == 2 && ggml_node_get_use_count(graph, index) == 2 ? first : -1;
}
