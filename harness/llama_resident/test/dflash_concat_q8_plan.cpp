#include "dflash-concat-q8.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void need(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
struct fixture {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor * a, * b, * cat, * w, * mm0, * mm1;
    int index = -1, first = -1;
    fixture() {
        ctx = ggml_init({ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true});
        graph = ggml_new_graph(ctx);
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 16);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 16);
        a->data = (void *) uintptr_t(0x1000000); b->data = (void *) uintptr_t(0x2000000);
        cat = ggml_concat(ctx, a, b, 1); cat->data = (void *) uintptr_t(0x3000000);
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 1024);
        mm0 = ggml_mul_mat(ctx, w, cat); mm1 = ggml_mul_mat(ctx, w, cat);
        ggml_set_output(mm0); ggml_set_output(mm1);
        ggml_build_forward_expand(graph, mm0); ggml_build_forward_expand(graph, mm1);
        for (int i=0; i<graph->n_nodes; ++i) {
            if (graph->nodes[i] == cat) index=i;
            if (graph->nodes[i] == mm0) first=i;
        }
    }
    ~fixture() { ggml_free(ctx); }
    bool match() { return ggml_cuda_dflash_concat_q8_match(graph,index)==first; }
};
int main() {
    int guards=0;
    fixture f; need(f.match(), "actual two-consumer shape"); ++guards;
    for (auto * x : {f.a,f.b,f.cat,f.w,f.mm0,f.mm1}) {
        auto saved=*x;
        x->type=GGML_TYPE_F16; need(!f.match(), "unsupported tensor type"); ++guards; *x=saved;
        x->nb[1]+=16; need(!f.match(), "noncontiguous rows"); ++guards; *x=saved;
        x->ne[2]=2; need(!f.match(), "batched tensor"); ++guards; *x=saved;
    }
    for (auto * x : {f.a,f.b}) {
        auto saved=*x;
        x->data=(void *)(uintptr_t(x->data)+4); need(!f.match(), "unaligned float4 source"); ++guards; *x=saved;
        x->data=nullptr; need(!f.match(), "missing input allocation"); ++guards; *x=saved;
        x->ne[1]=15; need(!f.match(), "partial source row count"); ++guards; *x=saved;
    }
    auto saved=*f.cat;
    f.cat->flags |= GGML_TENSOR_FLAG_OUTPUT; need(!f.match(), "observable concat"); ++guards; *f.cat=saved;
    f.cat->view_src=f.a; need(!f.match(), "concat output view"); ++guards; *f.cat=saved;
    ggml_set_op_params_i32(f.cat,0,0); need(!f.match(), "wrong concat axis"); ++guards; *f.cat=saved;
    f.cat->ne[0]=2048; need(!f.match(), "other width"); ++guards; *f.cat=saved;
    f.cat->ne[1]=16; need(!f.match(), "other merged row count"); ++guards; *f.cat=saved;
    saved=*f.mm1;
    f.mm1->op=GGML_OP_ADD; need(!f.match(), "non-MMQ observer"); ++guards; *f.mm1=saved;
    ggml_set_op_params_i32(f.mm1,1,GGML_HINT_SRC0_IS_HADAMARD); need(!f.match(), "transform observer"); ++guards; *f.mm1=saved;
    f.mm1->src[1]=f.b; need(!f.match(), "one consumer"); ++guards; *f.mm1=saved;
    {
        fixture g;
        auto * extra=ggml_scale(g.ctx,g.cat,1.0f); ggml_build_forward_expand(g.graph,extra);
        need(!g.match(), "extra arithmetic reader"); ++guards;
    }
    {
        fixture g;
        auto * extra=ggml_view_2d(g.ctx,g.cat,4096,16,4096*4,0); ggml_set_output(extra);
        ggml_build_forward_expand(g.graph,extra);
        need(!g.match(), "view observer"); ++guards;
    }
    {
        fixture g;
        auto * extra=ggml_mul_mat(g.ctx,g.w,g.cat); ggml_build_forward_expand(g.graph,extra);
        need(!g.match(), "third MMQ reader"); ++guards;
    }
    size_t words=0, blocks=0;
    // Production is the 4096/16+16 case; the standalone input selector also
    // exercises asymmetric split boundaries. Quantization never crosses rows.
    for (int width : {512,4096,8192}) for (int na : {1,7,16}) for (int nb : {1,15,16}) {
        const int rows=na+nb;
        std::vector<unsigned char> owned(size_t(width/128)*rows*128,0);
        for (int r=0;r<rows;++r) for (int x=0;x<width;x+=4) {
            const size_t packed_input=size_t(r)*width+x;
            const size_t selected_input=size_t(r<na?r:r-na)*width+x;
            need(selected_input+(r<na?0:size_t(na)*width)==packed_input,"selected float4 equals concat");
            const size_t ib=size_t(x/128)*rows+r, q=x%128;
            for (int j=0;j<4;++j) {need(owned[ib*128+q+j]++==0,"one writer per quant value");++words;}
            if (q%32==0) ++blocks;
        }
        for (auto x:owned) need(x==1,"complete k-major N-interleaved output");
    }
    std::printf("DFLASH_CONCAT_Q8_PLAN_PASS guards=%d input_words=%zu scale_blocks=%zu always_on=true\n",guards,words,blocks);
}
