// Full connected production graph, including graph-allocated norm input,
// original PQ2/F32 projections, conv/L2, raw GDN and both rollback caches.
// The CPU target checks allocation/clone invariants without executing kernels.
#include "llama-decode-gates.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef DECODE_GATES_TEST_CUDA
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {
constexpr int WIDTH=4096, T=16, D=128, H=32, HK=16, C=8192, K=16, GUARD=64;
constexpr int64_t STATE=int64_t(D)*D*H, CONV=3*C;
constexpr uint32_t SENTINEL=0x46a98700u;
enum Mode { LATE, EARLY_SERIAL, SIDE, NO_KEEPALIVE };
const char * names[]={"late", "early-serial", "side", "no-keepalive-fallback"};
void require(bool ok,const char * why) {
    if (!ok) { std::fprintf(stderr,"FAIL %s\n",why); std::exit(1); }
}
bool overlaps(const ggml_tensor * a,const ggml_tensor * b) {
    const uintptr_t x=(uintptr_t)a->data,y=(uintptr_t)b->data;
    return x<y+ggml_nbytes(b) && y<x+ggml_nbytes(a);
}
bool computes(const ggml_tensor * t) {
    return t->op!=GGML_OP_NONE && t->op!=GGML_OP_VIEW && t->op!=GGML_OP_RESHAPE &&
           t->op!=GGML_OP_TRANSPOSE && t->op!=GGML_OP_PERMUTE;
}
struct Result { std::vector<uint32_t> out,conv,state; };
struct Pipeline {
    Mode mode; bool zero;
    ggml_backend_t backend;
    ggml_context * input_ctx,* graph_ctx;
    ggml_backend_buffer_t inputs;
    ggml_gallocr_t allocator;
    ggml_cgraph * graph;
    ggml_tensor * x,* residual,* nw,* signs,* rotation,* wq,* wz,* wb,* wa,* cw,* bias,* a,* ids;
    ggml_tensor * conv_raw,* state_raw,* conv_cache,* state_cache;
    ggml_tensor * norm,* beta,* alpha,* qkv,* z_out,* gdn,* out;
    int norm_aliases=0,protected_nodes=0;
    Pipeline(Mode m,bool z=false) : mode(m),zero(z) {
#ifdef DECODE_GATES_TEST_CUDA
        backend=ggml_backend_cuda_init(0);
#else
        backend=ggml_backend_init_by_name("CPU",nullptr);
#endif
        require(backend,"backend initialization");
        input_ctx=ggml_init({ggml_tensor_overhead()*32,nullptr,true});
        graph_ctx=ggml_init({ggml_tensor_overhead()*512+ggml_graph_overhead(),nullptr,true});
        require(input_ctx && graph_ctx,"contexts");
        x=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,WIDTH,T);
        residual=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,WIDTH,T);
        nw=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,WIDTH);
        signs=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,WIDTH);
        rotation=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,WIDTH,WIDTH);
        wq=ggml_new_tensor_2d(input_ctx,GGML_TYPE_PQ2_0,WIDTH,C);
        wz=ggml_new_tensor_2d(input_ctx,GGML_TYPE_PQ2_0,WIDTH,WIDTH);
        wb=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,WIDTH,H);
        wa=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,WIDTH,H);
        cw=ggml_new_tensor_2d(input_ctx,GGML_TYPE_F32,4,C);
        bias=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,H);
        a=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,H);
        ids=ggml_new_tensor_1d(input_ctx,GGML_TYPE_I32,1);
        conv_raw=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,CONV*(K+1)+2*GUARD);
        state_raw=ggml_new_tensor_1d(input_ctx,GGML_TYPE_F32,STATE*(K+1)+2*GUARD);
        inputs=ggml_backend_alloc_ctx_tensors(input_ctx,backend);
        require(inputs,"input/weight allocation");
        ggml_backend_buffer_clear(inputs,0);
        auto * c=graph_ctx; graph=ggml_new_graph(c);
        conv_cache=ggml_view_2d(c,conv_raw,CONV,K+1,CONV*4,GUARD*4);
        state_cache=ggml_view_2d(c,state_raw,STATE,K+1,STATE*4,GUARD*4);
        norm=ggml_mul(c,ggml_rms_norm(c,ggml_add(c,x,residual),1e-6f),nw);
        ggml_set_name(norm,"norm-lifetime");
        ggml_build_forward_expand(graph,norm);
        beta=ggml_mul_mat(c,wb,norm); alpha=ggml_mul_mat(c,wa,norm);
        ggml_set_name(beta,"beta-projection"); ggml_set_name(alpha,"alpha-projection");
        const int before=ggml_graph_n_nodes(graph);
        require(!llm_schedule_decode_gates(graph,norm,beta,alpha,T,2,true),"sequence guard");
        require(!llm_schedule_decode_gates(graph,norm,beta,alpha,15,1,true),"token guard");
        require(!llm_schedule_decode_gates(graph,norm,beta,alpha,T,1,false),"default-off guard");
        require(!llm_schedule_decode_gates(graph,norm,beta,ggml_mul_mat(c,wa,x),T,1,true),"same-input guard");
        require(ggml_graph_n_nodes(graph)==before,"declined schedule leaves graph untouched");
        require(llm_schedule_decode_gates(graph,norm,beta,alpha,T,1,mode!=LATE)==(mode!=LATE),"selected schedule");
        auto * rotated=ggml_reshape_2d(c,ggml_mul(c,norm,signs),WIDTH,WIDTH*T/WIDTH);
        rotated=ggml_mul_mat(c,rotation,rotated);
        ggml_mul_mat_set_hint(rotated,GGML_HINT_SRC0_IS_HADAMARD);
        rotated=ggml_reshape_4d(c,rotated,WIDTH,T,1,1);
        qkv=ggml_mul_mat(c,wq,rotated);
        z_out=ggml_mul_mat(c,wz,rotated);
        if (zero) ggml_build_forward_expand(graph,ggml_scale_inplace(c,ggml_view_1d(c,conv_cache,CONV,0),0));
        auto * cs=ggml_reshape_3d(c,ggml_get_rows(c,conv_cache,ids),3,C,1);
        auto * xt=ggml_transpose(c,ggml_reshape_3d(c,qkv,C,T,1));
        auto * cat=ggml_concat(c,cs,xt,0);
        for (int i=1;i<=K;++i) {
            auto * from=ggml_view_3d(c,cat,3,C,1,cat->nb[1],cat->nb[2],i*4);
            auto * to=ggml_view_2d(c,conv_cache,CONV,1,conv_cache->nb[1],(K-i)*conv_cache->nb[1]);
            ggml_build_forward_expand(graph,ggml_cpy(c,from,to));
        }
        auto * y=ggml_silu(c,ggml_ssm_conv(c,cat,cw));
        auto * qk=ggml_view_4d(c,y,D,2*HK,T,1,D*4,y->nb[1],y->nb[1]*T,0);
        auto * l2=ggml_l2_norm(c,qk,1e-6f);
        auto * q=ggml_view_4d(c,l2,D,HK,T,1,l2->nb[1],l2->nb[2],l2->nb[3],0);
        auto * k=ggml_view_4d(c,l2,D,HK,T,1,l2->nb[1],l2->nb[2],l2->nb[3],HK*l2->nb[1]);
        auto * v=ggml_view_4d(c,y,D,H,T,1,D*4,y->nb[1],y->nb[1]*T,2*D*HK*4);
        if (zero) ggml_build_forward_expand(graph,ggml_scale_inplace(c,ggml_view_1d(c,state_cache,STATE,0),0));
        auto * state=ggml_reshape_4d(c,ggml_get_rows(c,state_cache,ids),D,D,H,1);
        gdn=ggml_gated_delta_net(c,q,k,v,ggml_reshape_4d(c,alpha,1,H,T,1),
            ggml_reshape_4d(c,beta,1,H,T,1),state,K);
        ggml_gated_delta_net_set_raw_gates(gdn,bias,a);
        ggml_set_name(gdn,"gdn-join");
        if (mode==SIDE || mode==EARLY_SERIAL) llm_hold_decode_gate_input(gdn,norm);
        auto * tail=ggml_view_3d(c,gdn,STATE,1,K,STATE*4,STATE*4,size_t(D)*H*T*4);
        auto * dst=ggml_view_3d(c,state_cache,STATE,1,K,STATE*4,STATE*4,0);
        ggml_build_forward_expand(graph,ggml_cpy(c,tail,dst));
        auto * attention=ggml_cont(c,ggml_view_2d(c,gdn,WIDTH,T,WIDTH*4,0));
        out=ggml_concat(c,attention,z_out,0);
        ggml_set_output(out); ggml_build_forward_expand(graph,out);
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(allocator,graph),"graph allocation");
        int ib=-1,ia=-1,iq=-1,ij=-1;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * n=ggml_graph_node(graph,i);
            if(n==beta) ib=i; if(n==alpha) ia=i; if(n==qkv) iq=i; if(n==gdn) ij=i;
        }
        if (!(ib>=0 && ia>=0 && iq>=0 && ij>std::max(ia,ib)))
            std::fprintf(stderr,"topology mode=%s beta=%d alpha=%d qkv=%d gdn=%d\n",names[mode],ib,ia,iq,ij);
        require(ib>=0 && ia>=0 && iq>=0 && ij>std::max(ia,ib),"graph projections/join");
        require(mode==LATE ? iq<std::min(ia,ib) : ib<ia && ia<iq,"selected early/late order");
        for (int i=std::max(ia,ib)+1;i<ij;++i) {
            auto * n=ggml_graph_node(graph,i);
            if (!computes(n)) continue;
            ++protected_nodes;
            norm_aliases+=overlaps(n,norm);
            require(!overlaps(n,beta) && !overlaps(n,alpha),"gate outputs survive until GDN");
            if (mode==SIDE || mode==EARLY_SERIAL) require(!overlaps(n,norm),"held norm cannot alias pre-join writer");
        }
        require(protected_nodes>=4 || mode==LATE,"meaningful allocator-pressure interval");
        if (mode==NO_KEEPALIVE) require(norm_aliases>0,"negative control actually reuses norm before GDN");
    }
    ~Pipeline() {
        ggml_backend_synchronize(backend); ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(inputs); ggml_backend_free(backend);
        ggml_free(graph_ctx); ggml_free(input_ctx);
    }
    void clone_check() {
        auto copy=ggml_backend_graph_copy(backend,graph);
        require(copy.graph,"graph clone");
        ggml_tensor * n=nullptr,* g=nullptr;
        for(int i=0;i<ggml_graph_n_nodes(copy.graph);++i) {
            auto * t=ggml_graph_node(copy.graph,i);
            if(!std::strcmp(t->name,"norm-lifetime")) n=t;
            if(!std::strcmp(t->name,"gdn-join")) g=t;
        }
        require(n && g && g!=gdn && n!=norm && g->src[9]==n,"graph clone preserves lifetime edge");
        ggml_backend_graph_copy_free(copy);
    }
#ifdef DECODE_GATES_TEST_CUDA
    ggml_backend_cuda_context & cuda() { return *(ggml_backend_cuda_context *)backend->context; }
    void weights() {
        std::mt19937 rng(20261018); std::uniform_real_distribution<float> dist(-1,1);
        auto fill=[&](ggml_tensor * t,float scale,float offset) {
            std::vector<float> data(ggml_nelements(t)); for(float & v:data) v=scale*dist(rng)+offset;
            ggml_backend_tensor_set(t,data.data(),0,data.size()*4);
        };
        fill(nw,0.1f,1); fill(wb,0.01f,0); fill(wa,0.01f,0); fill(cw,0.3f,0);
        fill(bias,0.1f,-0.2f); fill(a,0.05f,-0.2f);
        std::vector<float> hs(WIDTH); for(int i=0;i<WIDTH;++i) hs[i]=i%3 ? 1.0f : -1.0f;
        ggml_backend_tensor_set(signs,hs.data(),0,hs.size()*4);
        std::vector<float> hr(size_t(WIDTH)*WIDTH);
        for(int i=0;i<WIDTH;++i) for(int j=0;j<WIDTH;++j)
            hr[size_t(i)*WIDTH+j]=__builtin_parity(unsigned(i&j)) ? -0.015625f : 0.015625f;
        ggml_backend_tensor_set(rotation,hr.data(),0,hr.size()*4);
        auto reg=(size_t (*)(const ggml_tensor *))ggml_backend_reg_get_proc_address(
            ggml_backend_reg_by_name("CUDA"),"ggml_backend_cuda_register_decode_layout");
        require(reg,"production decode-layout registration");
        for(auto * t:{wq,wz}) {
            // In-file PQ2 block format: one half scale followed by 32 packed bytes.
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            for(size_t b=0;b<bytes.size()/34;++b) {
                const float d=b%131 ? float(1+rng()%100)/5000 : 0.0f;
                const ggml_fp16_t bits=ggml_fp32_to_fp16(b%13 ? d : -d);
                std::memcpy(bytes.data()+34*b,&bits,2);
                for(int q=0;q<32;++q) for(int j=0;j<4;++j) bytes[34*b+2+q]|=uint8_t(rng()%3)<<(2*j);
            }
            ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());
            require(reg(t)==size_t(t->ne[1])*(WIDTH/128)*28,"exact PTQ1 registered shape");
        }
    }
    void upload(unsigned seed,const Result * prior=nullptr,int rb=0) {
        std::mt19937 rng(seed); std::uniform_real_distribution<float> dist(-1,1);
        for(auto * t:{x,residual}) {
            std::vector<float> values(ggml_nelements(t)); for(float & v:values) v=0.4f*dist(rng);
            for(size_t i=0;i<values.size();i+=97) values[i]=(i/97)%2 ? 0.0f : -0.0f;
            ggml_backend_tensor_set(t,values.data(),0,values.size()*4);
        }
        auto cache=[&](ggml_tensor * t,const std::vector<uint32_t> * from) {
            if(from) {ggml_backend_tensor_set(t,from->data(),0,from->size()*4); return;}
            std::vector<uint32_t> words(ggml_nelements(t),SENTINEL);
            for(size_t i=GUARD;i<words.size()-GUARD;++i) {float f=0.03f*dist(rng);std::memcpy(&words[i],&f,4);}
            ggml_backend_tensor_set(t,words.data(),0,words.size()*4);
        };
        cache(conv_raw,prior ? &prior->conv : nullptr); cache(state_raw,prior ? &prior->state : nullptr);
        ggml_backend_tensor_set(ids,&rb,0,4);
    }
    void run() {
        setenv("GGML_CUDA_DECODE_GATES_SIDE",mode==SIDE || mode==NO_KEEPALIVE ? "1" : "0",1);
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"connected graph compute");
    }
    void assert_route() {
        require((cuda().gdn_mmf_side_launches>0)==(mode==SIDE),"expected side/serial route");
        require(cuda().side_fork && cuda().side_join,"existing D4 z side branch must also be active");
#ifdef USE_CUDA_GRAPH
        require(cuda().any_cuda_graph_has_instance(),"real graph capture/replay");
        bool z_copy=false;
        for (const auto & item:cuda().cuda_graphs) {
            cudaGraph_t captured=item.second->graph;
            if (!captured) continue;
            size_t count=0;
            require(cudaGraphGetNodes(captured,nullptr,&count)==cudaSuccess,"captured nodes count");
            std::vector<cudaGraphNode_t> nodes(count);
            require(cudaGraphGetNodes(captured,nodes.data(),&count)==cudaSuccess,"captured nodes");
            for(auto node:nodes) {
                cudaGraphNodeType type;
                require(cudaGraphNodeGetType(node,&type)==cudaSuccess,"captured node type");
                if(type!=cudaGraphNodeTypeMemcpy) continue;
                cudaMemcpy3DParms params={};
                require(cudaGraphMemcpyNodeGetParams(node,&params)==cudaSuccess,"captured copy parameters");
                z_copy|=params.dstPtr.ptr==z_out->data && params.srcPtr.ptr!=z_out->data &&
                        params.extent.width==ggml_nbytes(z_out);
            }
        }
        require(z_copy,"captured private z side-output join copy must execute");
#else
        require(false,"CUDA graphs required");
#endif
    }
    Result fetch() {
        auto read=[](ggml_tensor * t) {
            std::vector<uint32_t> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;
        };
        Result r{read(out),read(conv_raw),read(state_raw)};
        for(int z=0;z<8;++z) for(int h=0;h<H;++h) {
            const size_t at=GUARD+size_t(2+z)*STATE+size_t(h)*D*D;
            require(r.state[at]==0x4744314cu && r.state[at+1]==K && r.state[at+2]==0x4e4f4631u,
                    "real GD1L/NOF1 lazy state formation");
        }
        return r;
    }
#endif
};
#ifdef DECODE_GATES_TEST_CUDA
void equal(const char * label,const std::vector<uint32_t> & expected,const std::vector<uint32_t> & got,bool guards) {
    require(expected.size()==got.size(),"comparison size");
    for(size_t i=0;i<got.size();++i) {
        const bool guard=guards && (i<GUARD || i>=got.size()-GUARD);
        if(expected[i]!=got[i] || (guard && got[i]!=SENTINEL)) {
            std::fprintf(stderr,"FAIL %s word=%zu expected=%08x got=%08x guard=%d\n",label,i,expected[i],got[i],guard);
            std::exit(1);
        }
    }
}
void compare(const Result & ref,const Result & got,Mode mode,int rb,const char * scenario) {
    std::printf("compare scenario=%s mode=%s rb=%d\n",scenario,names[mode],rb);std::fflush(stdout);
    for(uint32_t v:got.out) require((v&0x7f800000u)!=0x7f800000u,"finite output");
    equal("attention/z output",ref.out,got.out,false);
    equal("conv cache",ref.conv,got.conv,true); equal("GDN cache",ref.state,got.state,true);
}
void cuda_checks(bool zero) {
    std::array<std::unique_ptr<Pipeline>,4> ps;
    for(int i=0;i<4;++i) {
        ps[i]=std::make_unique<Pipeline>(Mode(i),zero);ps[i]->weights();
        for(int warm=0;warm<2;++warm) {ps[i]->upload(43+warm);ps[i]->run();}
        ps[i]->assert_route();
    }
    int checks=0;
    auto generation=[&](unsigned seed,const Result * prior,int rb,const char * label) {
        Result ref;
        for(int mode=0;mode<4;++mode) {
            auto & p=*ps[mode];p.upload(seed,prior,rb);p.run();p.assert_route();auto got=p.fetch();
            if(mode==0) ref=std::move(got); else {compare(ref,got,Mode(mode),rb,label);++checks;}
        }
        return ref;
    };
    Result seed=generation(701,nullptr,0,zero ? "deferred-zero" : "eager-seed");
    if(!zero) {
        for(int rb=0;rb<=K;++rb) {
            Result previous=generation(1301,&seed,rb,"lazy-A");
            Result changed=generation(1302,&seed,rb,"lazy-B");
            require(previous.out!=changed.out,"changed norm input changes outputs");
            generation(1302,&seed,rb,"lazy-B-repeat");
            generation(1301,&seed,rb,"lazy-A-repeat");
            generation(1401,&changed,rb,"reused-lazy-state");
        }
        Result invalid=seed;
        for(int z=0;z<8;++z) for(int h=0;h<H;++h)
            invalid.state[GUARD+size_t(2+z)*STATE+size_t(h)*D*D]=0;
        // The original eager aliasing bug at rb=2..9 is outside this schedule
        // change. All valid-lazy rb values above remain strict all-cache gates.
        generation(1501,&invalid,0,"invalid-header-eager-rb0");
    } else {
        generation(702,&seed,0,"deferred-zero-changed-input");
        generation(702,&seed,0,"deferred-zero-repeat");
        generation(701,&seed,0,"deferred-zero-return-A");
    }
    std::printf("PASS connected production chain zero=%d comparisons=%d all_cache_words=1 computed_norm=1\n",zero,checks);
}
#endif
}
int main() {
#ifdef DECODE_GATES_TEST_CUDA
    require(!getenv("GGML_CUDA_DISABLE_GRAPHS"),"graph replay enabled");
    setenv("GGML_CUDA_DECODE_LAYOUT","1",1);setenv("GGML_CUDA_DECODE_PTQ1","1",1);
    setenv("GGML_CUDA_DL_RING_PT","1",1);setenv("GGML_CUDA_PTQ1_SIGNED","1",1);
    setenv("GGML_CUDA_D4_SIDE_STREAM","1",1);
    cuda_checks(false);cuda_checks(true);
#else
    Pipeline late(LATE),unsafe(NO_KEEPALIVE),held(SIDE);
    require(unsafe.norm_aliases>0,"negative control must actually reuse norm before GDN");
    require(held.norm_aliases==0,"src9 prevents norm reuse before GDN");
    held.clone_check();
    std::printf("PASS allocator computed_norm=1 unprotected_aliases=%d protected_aliases=%d protected_nodes=%d graph_clone_src9=1\n",
                unsafe.norm_aliases,held.norm_aliases,held.protected_nodes);
#endif
}
