// Exercise the exact production gather helper, full K/V writes, F32 RoPE/FA,
// live residual/feature tensors, and input/index changes in one reused graph.
// CUDA additionally checks the production PQ2 MMQ dispatch and D4 quantizer.
#include "llama-last-attn.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef LAST_ATTN_TEST_CUDA
#include "ggml-backend-impl.h"
#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
template<class T> static std::vector<T> read(const ggml_tensor * t) {
    require(ggml_nbytes(t)%sizeof(T)==0, "read element size");
    std::vector<T> result(ggml_nbytes(t)/sizeof(T));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    return result;
}
template<class T> static void exact(const std::vector<T> & a, const std::vector<T> & b, const char * what) {
    require(a.size()==b.size(), "comparison size");
    require(!std::memcmp(a.data(), b.data(), a.size()*sizeof(T)), what);
}
static void near(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size()==b.size(), "attention comparison size");
    float worst=0;
    for (size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite attention output");
        const float error=std::abs(a[i]-b[i])/(1.0f+std::abs(a[i]));
        worst=std::max(worst,error);
        if (error>2e-4f) {
            std::fprintf(stderr,"attention[%zu] full=%a selected=%a error=%g\n",i,a[i],b[i],error);
            require(false,"same attention mathematics within F32 reduction tolerance");
        }
    }
    std::printf(" attention_scaled_error=%.8g",worst);
}

static void fallback_guards() {
    ggml_context * ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true});
    require(ctx!=nullptr,"guard context");
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,64);
    auto * pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,256);
    auto * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,128,64);
    auto * ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
    auto check=[&](ggml_tensor * a,ggml_tensor * p,ggml_tensor * m,ggml_tensor * i,int seq,bool on) {
        const size_t before=ggml_used_mem(ctx);
        const auto out=llm_build_last_attn_rows(ctx,a,p,m,i,seq,on);
        require(!out.query && !out.positions && !out.mask,"unsupported input falls back");
        require(ggml_used_mem(ctx)==before,"fallback creates no nodes");
    };
    check(x,pos,mask,ids,1,false);
    check(x,pos,mask,ids,2,true);
    check(x,pos,mask,nullptr,1,true);
    check(x,pos,mask,ggml_new_tensor_1d(ctx,GGML_TYPE_I32,2),1,true);
    check(x,pos,mask,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1),1,true);
    check(x,ggml_new_tensor_1d(ctx,GGML_TYPE_I32,64),mask,ids,1,true);
    check(x,pos,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,64),ids,1,true);
    check(x,pos,ggml_new_tensor_2d(ctx,GGML_TYPE_F16,128,63),ids,1,true);
    check(x,pos,ggml_new_tensor_3d(ctx,GGML_TYPE_F16,128,64,2),ids,1,true);
    check(ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,16),pos,mask,ids,1,true);
    check(ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2048,64),pos,mask,ids,1,true);
    auto * padded=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,8192,64);
    check(ggml_view_2d(ctx,padded,4096,64,padded->nb[1],0),pos,mask,ids,1,true);
    auto * padded_mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,256,64);
    check(x,pos,ggml_view_2d(ctx,padded_mask,128,64,padded_mask->nb[1],0),ids,1,true);
    ggml_free(ctx);
    std::puts("PASS fallback_guards=13 no_graph_mutation=1");
}

struct attention_graph {
    ggml_context * ctx=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_gallocr_t alloc=nullptr;
    llm_last_attn_rows selected;
    ggml_tensor * output=nullptr,* feature=nullptr,* query_rope=nullptr;
    attention_graph(ggml_backend_t backend,ggml_tensor * x,ggml_tensor * pos,ggml_tensor * mask,
            ggml_tensor * ids,ggml_tensor * ks,ggml_tensor * vs,ggml_tensor * kc,ggml_tensor * vc,bool narrow) {
        ctx=ggml_init({ggml_tensor_overhead()*160+ggml_graph_overhead(),nullptr,true});
        require(ctx!=nullptr,"attention context");
        graph=ggml_new_graph(ctx);
        selected=llm_build_last_attn_rows(ctx,x,pos,mask,ids,1,narrow);
        require((selected.query!=nullptr)==narrow,"eligible selection");
        const int64_t nq=narrow?16:x->ne[1];
        ggml_tensor * qinput=narrow?selected.query:x;
        ggml_tensor * qpos=narrow?selected.positions:pos;
        if (narrow) {
            require(selected.mask->type==GGML_TYPE_F16 && selected.mask->ne[0]==mask->ne[0] &&
                    selected.mask->ne[1]==16 && ggml_is_contiguous(selected.mask),"selected FA mask shape");
            require(selected.positions->type==GGML_TYPE_I32 && selected.positions->ne[0]==64,"four position axes");
            for (auto * t:{selected.query,selected.positions,selected.mask}) ggml_set_output(t);
        }
        auto * q=ggml_rms_norm(ctx,ggml_reshape_3d(ctx,qinput,256,16,nq),1e-6f);
        int sections[4]={16,24,24,0};
        q=ggml_rope_multi(ctx,q,qpos,nullptr,128,sections,GGML_ROPE_TYPE_MROPE,8192,
                         1000000.0f,1.0f,0.0f,1.0f,32.0f,1.0f);
        query_rope=ggml_reshape_2d(ctx,q,4096,nq);
        ggml_set_output(query_rope);
        ggml_build_forward_expand(graph,query_rope);
        // Real copies retain all original K/V rows in both graphs. They are
        // expanded before attention, as the production build_attn does.
        ggml_build_forward_expand(graph,ggml_cpy(ctx,ks,kc));
        ggml_build_forward_expand(graph,ggml_cpy(ctx,vs,vc));
        auto * attn=ggml_flash_attn_ext(ctx,ggml_permute(ctx,q,0,2,1,3),kc,vc,
                                      narrow?selected.mask:mask,1.0f/16.0f,0.0f,0.0f);
        ggml_flash_attn_ext_set_prec(attn,GGML_PREC_F32);
        attn=ggml_reshape_2d(ctx,attn,4096,nq);
        auto * gated=ggml_mul(ctx,attn,ggml_sigmoid(ctx,qinput));
        auto * row=narrow?ggml_view_2d(ctx,gated,4096,1,gated->nb[1],0):ggml_get_rows(ctx,gated,ids);
        // This full-sized feature lives until after attention and its scratch.
        feature=ggml_scale(ctx,x,0.75f);
        ggml_set_output(feature);
        ggml_build_forward_expand(graph,feature);
        output=ggml_add(ctx,row,ggml_get_rows(ctx,feature,ids));
        ggml_set_output(output);
        ggml_build_forward_expand(graph,output);
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            const auto * node=ggml_graph_node(graph,i);
            require(!(node->op==GGML_OP_REPEAT && node->type==GGML_TYPE_I32),"no unsupported CUDA I32 repeat");
            require(ggml_backend_supports_op(backend,node),"every node supported on the selected backend");
        }
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(alloc && ggml_gallocr_alloc_graph(alloc,graph),"attention graph allocation");
    }
    ~attention_graph() { ggml_gallocr_free(alloc); ggml_free(ctx); }
};

#ifdef LAST_ATTN_TEST_CUDA
static void quantizer_equivalence(ggml_backend_t backend,const ggml_tensor * full,
        const ggml_tensor * selected,int index) {
    auto * context=(ggml_backend_cuda_context *)backend->context;
    const int cc=ggml_cuda_info().devices[context->device].cc;
    require(!ggml_cuda_should_use_mmvq(GGML_TYPE_PQ2_0,cc,16) &&
            ggml_cuda_should_use_mmq(GGML_TYPE_PQ2_0,cc,16,0),"selected queries retain PQ2 MMQ dispatch");
    require(mmq_get_q8_1_ds_layout(GGML_TYPE_PQ2_0)==MMQ_Q8_1_DS_LAYOUT_D4,"PQ2 activation scales remain FP32");
    const int64_t blocks=4096/QK8_1_MMQ, nf=full->ne[1];
    block_q8_1_mmq * fq=nullptr,*sq=nullptr;
    CUDA_CHECK(cudaMalloc(&fq,blocks*nf*sizeof(*fq)));
    CUDA_CHECK(cudaMalloc(&sq,blocks*16*sizeof(*sq)));
    quantize_mmq_q8_1_cuda((const float *)full->data,nullptr,fq,GGML_TYPE_PQ2_0,
                         4096,4096,4096*nf,4096*nf,4096,nf,1,1,context->stream());
    quantize_mmq_q8_1_cuda((const float *)selected->data,nullptr,sq,GGML_TYPE_PQ2_0,
                         4096,4096,4096*16,4096*16,4096,16,1,1,context->stream());
    std::vector<block_q8_1_mmq> a(blocks*nf),b(blocks*16);
    CUDA_CHECK(cudaMemcpyAsync(a.data(),fq,a.size()*sizeof(*fq),cudaMemcpyDeviceToHost,context->stream()));
    CUDA_CHECK(cudaMemcpyAsync(b.data(),sq,b.size()*sizeof(*sq),cudaMemcpyDeviceToHost,context->stream()));
    CUDA_CHECK(cudaStreamSynchronize(context->stream()));
    for (int64_t k=0;k<blocks;++k) for (int j=0;j<16;++j)
        require(!std::memcmp(&a[k*nf+index],&b[k*16+j],sizeof(*fq)),"all D4 scale and quantizer bytes identical");
    CUDA_CHECK(cudaFree(fq)); CUDA_CHECK(cudaFree(sq));
}
static void actual_graph(ggml_backend_t backend,const attention_graph & graph) {
#ifdef USE_CUDA_GRAPH
    auto * context=(ggml_backend_cuda_context *)backend->context;
    bool captured=false;
    for (const auto & entry:context->cuda_graphs) {
        const auto & cached=*entry.second;
        captured|=cached.instance && cached.warmup_complete && !cached.node_props.empty() &&
                  cached.node_props.back().node.data==graph.output->data &&
                  cached.node_props.size()==size_t(ggml_graph_n_nodes(graph.graph));
    }
    require(captured,"actual CUDA graph capture/replay for this graph");
#else
    require(false,"CUDA graph build required");
#endif
}
#endif

int main() {
    fallback_guards();
#ifdef LAST_ATTN_TEST_CUDA
    constexpr bool cuda=true;
    require(getenv("GGML_CUDA_DISABLE_GRAPHS")==nullptr,"CUDA graphs must be enabled");
#else
    constexpr bool cuda=false;
#endif
    ggml_backend_t backend=ggml_backend_init_by_name(cuda?"CUDA0":"CPU",nullptr);
    require(backend!=nullptr,"backend available");
    if (!cuda) {
        ggml_backend_cpu_set_n_threads(backend,2);
        // CPU FA uses a different numerical path above its query tile size
        // (F32 tiled versus F16 vector accumulation). Hold that path fixed for
        // this graph-wiring test. The CUDA target uses normal production FA.
        ggml_backend_cpu_set_use_ref(backend,true);
    }
    const int rows=cuda?512:64;
    const int kv=rows+256; // Includes older KV rows as in a later prefill chunk.
    ggml_context * inputs=ggml_init({ggml_tensor_overhead()*16,nullptr,true});
    auto * x=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,4096,rows);
    auto * pos=ggml_new_tensor_1d(inputs,GGML_TYPE_I32,4*rows);
    auto * mask=ggml_new_tensor_2d(inputs,GGML_TYPE_F16,kv,rows);
    auto * ids=ggml_new_tensor_1d(inputs,GGML_TYPE_I32,1);
    auto * ks=ggml_new_tensor_3d(inputs,GGML_TYPE_F16,256,kv,4);
    auto * vs=ggml_new_tensor_3d(inputs,GGML_TYPE_F16,256,kv,4);
    auto * kc=ggml_dup_tensor(inputs,ks),* vc=ggml_dup_tensor(inputs,vs);
    auto * buffer=ggml_backend_alloc_ctx_tensors(inputs,backend);
    require(buffer!=nullptr,"input and cache allocation");
    std::mt19937 rng(20261018);
    std::uniform_real_distribution<float> random(-1.0f,1.0f);
    {
        attention_graph full(backend,x,pos,mask,ids,ks,vs,kc,vc,false);
        attention_graph narrow(backend,x,pos,mask,ids,ks,vs,kc,vc,true);
        const int indices[8]={rows-1,rows/3,0,rows-2,rows/3,rows-1,0,rows-1};
        for (int replay=0;replay<8;++replay) {
            std::vector<float> values(4096*rows);
            std::vector<int32_t> positions(4*rows);
            std::vector<ggml_fp16_t> masks(kv*rows),keys(256*kv*4),vals(keys.size());
            for (float & v:values) v=random(rng);
            for (int axis=0;axis<4;++axis) for (int row=0;row<rows;++row)
                positions[axis*rows+row]=256+row+17*axis+3*replay;
            for (int row=0;row<rows;++row) for (int k=0;k<kv;++k) {
                const float m=k<=256+row?((k+row)%13==0?-0.5f:0.0f):-std::numeric_limits<float>::infinity();
                masks[row*kv+k]=ggml_fp32_to_fp16(m);
            }
            for (auto * a:{&keys,&vals}) for (auto & v:*a) v=ggml_fp32_to_fp16(random(rng));
            const int32_t id=indices[replay];
            ggml_backend_tensor_set(x,values.data(),0,ggml_nbytes(x));
            ggml_backend_tensor_set(pos,positions.data(),0,ggml_nbytes(pos));
            ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
            ggml_backend_tensor_set(ids,&id,0,sizeof(id));
            ggml_backend_tensor_set(ks,keys.data(),0,ggml_nbytes(ks));
            ggml_backend_tensor_set(vs,vals.data(),0,ggml_nbytes(vs));
            require(ggml_backend_graph_compute(backend,full.graph)==GGML_STATUS_SUCCESS,"full attention compute");
            const auto expected=read<float>(full.output),full_rope=read<float>(full.query_rope);
            ggml_backend_tensor_memset(kc,0x5a,0,ggml_nbytes(kc));
            ggml_backend_tensor_memset(vc,0xa5,0,ggml_nbytes(vc));
            require(ggml_backend_graph_compute(backend,narrow.graph)==GGML_STATUS_SUCCESS,"selected attention compute");
            std::printf("PASS backend=%s rows=%d kv=%d replay=%d out_id=%d",cuda?"CUDA":"CPU",rows,kv,replay,id);
            near(expected,read<float>(narrow.output));
            exact(keys,read<ggml_fp16_t>(kc),"all K cache rows written");
            exact(vals,read<ggml_fp16_t>(vc),"all V cache rows written");
            exact(values,read<float>(x),"full layer input unchanged");
            exact(positions,read<int32_t>(pos),"full position input unchanged");
            exact(masks,read<ggml_fp16_t>(mask),"full mask unchanged");
            const auto sq=read<float>(narrow.selected.query),sr=read<float>(narrow.query_rope);
            const auto sp=read<int32_t>(narrow.selected.positions);
            const auto sm=read<ggml_fp16_t>(narrow.selected.mask);
            for (int copy=0;copy<16;++copy) {
                require(!std::memcmp(sq.data()+copy*4096,values.data()+id*4096,4096*sizeof(float)),"selected query exact");
                require(!std::memcmp(sr.data()+copy*4096,full_rope.data()+id*4096,4096*sizeof(float)),"selected RoPE exact");
                require(!std::memcmp(sm.data()+copy*kv,masks.data()+id*kv,kv*sizeof(ggml_fp16_t)),"selected mask bits exact");
                for (int axis=0;axis<4;++axis)
                    require(sp[axis*16+copy]==positions[axis*rows+id],"selected position axes exact");
            }
            auto features=values;
            for (float & v:features) v*=0.75f;
            exact(features,read<float>(narrow.feature),"all feature rows preserved across scratch lifetime");
#ifdef LAST_ATTN_TEST_CUDA
            quantizer_equivalence(backend,x,narrow.selected.query,id);
#endif
            std::puts(" full_kv=1 full_features=1 exact_gathers=1");
        }
#ifdef LAST_ATTN_TEST_CUDA
        actual_graph(backend,full); actual_graph(backend,narrow);
        std::puts("PASS actual_cuda_graphs=2 changed_input_and_output_id_replays=8 exact_D4_bytes=1");
#endif
    }
    ggml_backend_buffer_free(buffer); ggml_free(inputs); ggml_backend_free(backend);
}
