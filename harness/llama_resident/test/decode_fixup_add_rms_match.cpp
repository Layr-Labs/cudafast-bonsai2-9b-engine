#include "decode-fixup-add-rms.h"

#include <cstdio>
#include <cstdlib>

static void require(bool value,const char * message) {
    if (!value) { std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1); }
}
struct fixture {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_tensor * mm,*res,*add,*weight,*input;
    int index=-1,add_index=-1;
    fixture(int k,int rows,int views) {
        ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true});
        graph=ggml_new_graph(ctx);
        weight=ggml_new_tensor_2d(ctx,GGML_TYPE_PQ2_0,k,4096);
        input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,rows);
        res=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,rows);
        auto * nw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4096);
        mm=ggml_mul_mat(ctx,weight,input);
        auto * last=mm;
        for (int i=0;i<views;++i) last=ggml_reshape_2d(ctx,last,4096,rows);
        add=ggml_add(ctx,last,res);
        auto * out=ggml_mul(ctx,ggml_rms_norm(ctx,add,1e-6f),nw);
        ggml_set_output(out);ggml_build_forward_expand(graph,out);
        mm->data=(void *)uintptr_t(0x1000000);
        res->data=(void *)uintptr_t(0x2000000);
        for (int i=0;i<graph->n_nodes;++i) {
            auto * t=graph->nodes[i];
            if (t==mm) index=i;
            if (t==add) add_index=i;
            if (t->op==GGML_OP_RESHAPE) t->data=mm->data;
        }
    }
    ~fixture(){ggml_free(ctx);}
    bool match(){return ggml_cuda_decode_fixup_add_rms_match(graph,index)==add_index;}
};
int main() {
    int checks=0;
    for (int k:{4096,12288}) for (int views:{0,1,2}) {
        fixture f(k,16,views);
        require(f.match(),"supported graph");++checks;
        for (intptr_t delta:{-262143,-4,0,4,262143}) {
            f.res->data=(void *)(uintptr_t(f.mm->data)+delta);
            require(!f.match(),"full residual interval overlap");++checks;
        }
        for (intptr_t delta:{-262144,262144}) {
            f.res->data=(void *)(uintptr_t(f.mm->data)+delta);
            require(f.match(),"touching but disjoint intervals");++checks;
        }
        f.res->data=(void *)uintptr_t(0x2000000);
        f.mm->flags|=GGML_TENSOR_FLAG_OUTPUT;
        require(!f.match(),"observable projection");++checks;
        f.mm->flags&=~GGML_TENSOR_FLAG_OUTPUT;
        auto * first=f.add->src[0];f.add->src[0]=f.res;f.add->src[1]=first;
        require(!f.match(),"reversed ADD retains original operand order");++checks;
        f.add->src[0]=first;f.add->src[1]=f.res;
        f.res->nb[1]+=4;require(!f.match(),"strided residual");++checks;f.res->nb[1]-=4;
        auto * extra=ggml_scale(f.ctx,f.mm,0.5f);
        ggml_build_forward_expand(f.graph,extra);
        require(!f.match(),"additional graph reader");++checks;
    }
    for (int k:{4096,8192,12288}) for (int n:{1,7,15,16,32}) {
        fixture f(k,n,0);
        require(f.match()==(n==16 && k!=8192),"full shape fallback");++checks;
    }
    fixture f(4096,16,3);require(!f.match(),"bounded view chain");++checks;
    require(!ggml_cuda_fixup_ranges_overlap((void *)uintptr_t(16),16,(void *)uintptr_t(32),8),"range endpoints");
    require(ggml_cuda_fixup_ranges_overlap((void *)uintptr_t(UINTPTR_MAX-7),8,(void *)uintptr_t(UINTPTR_MAX-3),4),
            "range endpoint overflow safe");checks+=2;
    std::printf("DECODE_FIXUP_ADD_RMS_MATCH_PASS checks=%d\n",checks);
}
