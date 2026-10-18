#pragma once

#include "ggml-impl.h"

#include <cstdint>

// Range arithmetic avoids ordering/subtracting unrelated C++ pointers and
// does not overflow when one interval ends at the address-space boundary.
static inline bool ggml_cuda_fixup_ranges_overlap(const void * a, size_t na, const void * b, size_t nb) {
    const uintptr_t pa=(uintptr_t)a,pb=(uintptr_t)b;
    return na && nb && (pa<=pb ? pb-pa<na : pa-pb<nb);
}

// Only the first ADD operand is the projection: retain even NaN payload and
// signed-zero operand order. The original projection and every reshape must
// have no observable use except this ADD. The remaining RMS/FWHT chain is
// independently validated by the existing production fusion predicate.
static inline int ggml_cuda_decode_fixup_add_rms_match(const ggml_cgraph * graph,int i) {
    if (i<0 || i>=graph->n_nodes) return -1;
    const ggml_tensor * mm=graph->nodes[i];
    const ggml_tensor * w=mm->src[0], * x=mm->src[1];
    if (mm->op!=GGML_OP_MUL_MAT || !w || !x || w->type!=GGML_TYPE_PQ2_0 ||
            mm->type!=GGML_TYPE_F32 || x->type!=GGML_TYPE_F32 ||
            (w->ne[0]!=4096 && w->ne[0]!=12288) || w->ne[1]!=4096 ||
            w->ne[2]!=1 || w->ne[3]!=1 || x->ne[0]!=w->ne[0] ||
            x->ne[1]!=16 || x->ne[2]!=1 || x->ne[3]!=1 ||
            mm->ne[0]!=4096 || mm->ne[1]!=16 || mm->ne[2]!=1 || mm->ne[3]!=1 ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(mm) ||
            !mm->data || (mm->flags&GGML_TENSOR_FLAG_OUTPUT)) return -1;
    const ggml_tensor * last=mm;
    int ia=i+1;
    while (ia<graph->n_nodes && ia-i<3 && graph->nodes[ia]->op==GGML_OP_RESHAPE) {
        const ggml_tensor * view=graph->nodes[ia];
        if (view->src[0]!=last || view->data!=mm->data || view->ne[0]!=4096 ||
                ggml_nelements(view)!=ggml_nelements(mm) || !ggml_is_contiguous(view) ||
                (view->flags&GGML_TENSOR_FLAG_OUTPUT)) return -1;
        last=view;
        ++ia;
    }
    if (ia+2>=graph->n_nodes) return -1;
    const ggml_tensor * add=graph->nodes[ia], * res=add->src[1];
    if (add->op!=GGML_OP_ADD || add->src[0]!=last || !res ||
            !(add->flags&GGML_TENSOR_FLAG_COMPUTE) || add->type!=GGML_TYPE_F32 || res->type!=GGML_TYPE_F32 ||
            !ggml_are_same_shape(last,add) || !ggml_are_same_shape(last,res) ||
            !ggml_is_contiguous(add) || !ggml_is_contiguous(res) || !res->data ||
            ggml_cuda_fixup_ranges_overlap(mm->data,ggml_nbytes(mm),res->data,ggml_nbytes(res))) return -1;
    ggml_op ops[4];
    for (int j=i;j<=ia;++j) ops[j-i]=graph->nodes[j]->op;
    const int outputs[]={ia};
    return ggml_can_fuse_subgraph(graph,i,ia-i+1,ops,outputs,1) ? ia : -1;
}
