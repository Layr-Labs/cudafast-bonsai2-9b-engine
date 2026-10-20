#include "s1_test_util.cuh"
#define GGML_CUDA_S1_DFLASH_KV_IMPLEMENTATION
#include "s1-dflash-kv-j8.cuh"
#undef GGML_CUDA_S1_DFLASH_KV_IMPLEMENTATION
#include "quantize.cuh"
#include "rope.cuh"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <fstream>
#include <random>

static int s1_backend_route(int argc,char ** argv);

template <typename T> static void load_exact(const char * path, std::vector<T> & out) {
    std::ifstream f(path,std::ios::binary);
    f.read((char*)out.data(),out.size()*sizeof(T));
    if(!f || f.peek()!=std::ifstream::traits_type::eof()) { fprintf(stderr,"bad tensor file %s\n",path);exit(2); }
}

int main(int argc,char ** argv) {
    for(int i=1;i<argc;++i) if(!strcmp(argv[i],"--route")) return s1_backend_route(argc,argv);
    bool quick=false,bench=false; int tile=16; const char * kwfile=nullptr,*vwfile=nullptr,*yfile=nullptr;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--quick")) quick=true;
        else if(!strcmp(argv[i],"--bench")) bench=true;
        else if(!strcmp(argv[i],"--j8")) tile=8;
        else if(!strcmp(argv[i],"--j16")) tile=16;
        else if(i+1<argc && !strcmp(argv[i],"--k-weights")) kwfile=argv[++i];
        else if(i+1<argc && !strcmp(argv[i],"--v-weights")) vwfile=argv[++i];
        else if(i+1<argc && !strcmp(argv[i],"--d4")) yfile=argv[++i];
        else { fprintf(stderr,"unknown argument %s\n",argv[i]);return 2; }
    }
    if(bool(kwfile)!=bool(vwfile) || bool(kwfile)!=bool(yfile)) return 2;
    const char * active_mode=tile==16 ? "2" : "1";
    setenv("GGML_CUDA_S1_DFLASH_KV_J8",active_mode,1);
    ggml_backend_t backend=ggml_backend_cuda_init(0);if(!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    constexpr int M=1024,K=4096,N=32,HD=128,HK=8,CACHE=128;
    const int banks=quick || kwfile ? 1 : 8;
    constexpr size_t nw=size_t(M)*K/32,ny=size_t(K)*N/128,nout=size_t(M)*N;
    s1_buffer<block_q8_0> wk(nw*banks),wv(nw*banks);
    s1_buffer<float> x(size_t(K)*N),dk(nout+16),dv(nout+16),normw(HD);
    s1_buffer<block_q8_1_mmq> y(ny);
    s1_buffer<int32_t> positions(N*4);s1_buffer<int64_t> indices(N);
    s1_buffer<ggml_fp16_t> kc(size_t(M)*CACHE+32),vc(size_t(M)*CACHE+32);
    std::mt19937 rng(20261019);std::uniform_real_distribution<float> d(-2,2);
    std::vector<block_q8_0> hwk(nw*banks),hwv(nw*banks);
    for(auto * weights:{&hwk,&hwv}) for(auto & b:*weights) {
        b.d=__float2half(0.0001f+float(rng()%8)*0.0001f);
        for(auto & a:b.qs) a=int(rng()%255)-127;
    }
    std::vector<float> hx(size_t(K)*N),hn(HD);for(auto & a:hx) a=d(rng);for(auto & a:hn) a=d(rng);
    std::vector<block_q8_1_mmq> hy(ny);
    if(kwfile) {load_exact(kwfile,hwk);load_exact(vwfile,hwv);load_exact(yfile,hy);y.upload(hy);}
    wk.upload(hwk);wv.upload(hwv);x.upload(hx);normw.upload(hn);
    std::vector<int32_t> hp(N*4);std::vector<int64_t> hi(N);
    for(int t=0;t<N;++t) {hi[t]=2*t+1;for(int s=0;s<4;++s) hp[s*N+t]=500+t;}
    positions.upload(hp);indices.upload(hi);
    auto make_args=[&](bool key,int bank) {
        mmq_args a={};
        a.x=(const char*)((key ? wk.p : wv.p)+bank*nw);a.type_x=GGML_TYPE_Q8_0;
        a.y=(const int*)y.p;a.dst=key ? dk.p : dv.p;
        a.ncols_x=K;a.nrows_x=M;a.ncols_dst=N;a.stride_row_x=K/32;a.ncols_y=N;a.nrows_dst=M;
        a.nchannels_x=a.nchannels_y=a.nsamples_x=a.nsamples_y=1;
        a.ncols_max=N;return a;
    };
    ggml_context * ctx=ggml_init({2*1024*1024,nullptr,true});
    auto kin=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,HD,HK,N);kin->data=dk.p;
    auto vin=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,HD,HK,N);vin->data=dv.p;
    auto wt=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,HD);wt->data=normw.p;
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,N*4);pos->data=positions.p;
    auto rms=ggml_rms_norm(ctx,kin,1e-6f);auto mul=ggml_mul(ctx,rms,wt);
    int sections[4]={64,0,0,0};
    auto rope=ggml_rope_multi(ctx,mul,pos,nullptr,128,sections,GGML_ROPE_TYPE_MROPE,16384,1e7f,1.0f/16,1,1,32,1);
    auto kt=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,M,CACHE);kt->data=kc.p;
    auto vt=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,M,CACHE);vt->data=vc.p;
    auto idx=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,N);idx->data=indices.p;
    auto ks=ggml_set_rows(ctx,kt,ggml_reshape_2d(ctx,rope,M,N),idx);ks->data=kc.p;
    auto vv=ggml_reshape_2d(ctx,vin,M,N);vv->data=dv.p;
    auto vs=ggml_set_rows(ctx,vt,vv,idx);vs->data=vc.p;
    if(!ggml_cuda_rms_norm_mul_mrope_supported(rms,mul,rope)) return 3;
    auto chain=[&](bool candidate,int bank) {
        // 58f135f6 already virtualizes the two 16-row input slabs at CONCAT.
        // Compare against that producer, not an obsolete materialized concat.
        if(!yfile) quantize_mmq_q8_1_concat_cuda(x.p,x.p+size_t(K)*16,y.p,K,16,16,cuda.stream());
        auto ka=make_args(true,bank),va=make_args(false,bank);
        if(candidate) {
            if(!ggml_cuda_s1_kv_j8_launch(cuda,ka,va,cuda.stream())) {fprintf(stderr,"J%d route refused\n",tile);exit(3);}
        } else {
            launch_mul_mat_q<GGML_TYPE_Q8_0,32,false>(cuda,ka,cuda.stream());
            launch_mul_mat_q<GGML_TYPE_Q8_0,32,false>(cuda,va,cuda.stream());
        }
        ggml_cuda_op_rms_norm_mul_mrope_fused(cuda,rms,mul,rope,ks,vv,vs);
    };
    // A real kill path must refuse before launching, with no altered bytes.
    auto ka=make_args(true,0),va=make_args(false,0);
    setenv("GGML_CUDA_S1_DFLASH_KV_J8","0",1);
    if(ggml_cuda_s1_kv_j8_launch(cuda,ka,va,cuda.stream())) return 3;
    setenv("GGML_CUDA_S1_DFLASH_KV_J8",active_mode,1);
    auto invalid=ka;invalid.ncols_dst=16;
    if(ggml_cuda_s1_kv_j8_launch(cuda,invalid,va,cuda.stream())) return 3;
    size_t failures=0,cases=0;
    for(int bank=0;bank<banks;++bank) {
        cudaGraphExec_t graph[2];
        for(int path=0;path<2;++path) {
            chain(path!=0,bank);S1_CHECK(cudaStreamSynchronize(cuda.stream()));
            graph[path]=s1_capture(cuda.stream(),[&](){chain(path!=0,bank);});
        }
        for(int replay=0;replay<3;++replay) {
            if(!yfile) {for(auto & a:hx) a=replay==1 ? (rng()&1 ? 1e-20f : -1e-20f) : d(rng);x.upload(hx);}
            std::vector<float> refk,refv;std::vector<ggml_fp16_t> refkc,refvc;
            std::vector<block_q8_1_mmq> refy;
            for(int path=0;path<2;++path) {
                S1_CHECK(cudaMemsetAsync(dk.p,0xa5,dk.n*4,cuda.stream()));S1_CHECK(cudaMemsetAsync(dv.p,0xa5,dv.n*4,cuda.stream()));
                S1_CHECK(cudaMemsetAsync(kc.p,0xa5,kc.n*2,cuda.stream()));S1_CHECK(cudaMemsetAsync(vc.p,0xa5,vc.n*2,cuda.stream()));
                S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                S1_CHECK(cudaGraphLaunch(graph[path],cuda.stream()));S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                auto hk=dk.read(),hv=dv.read();auto hkc=kc.read(),hvc=vc.read();auto hq=y.read();
                for(size_t i=nout;i<dk.n;++i) {unsigned int kbits,vbits;memcpy(&kbits,&hk[i],4);memcpy(&vbits,&hv[i],4);failures+=kbits!=0xa5a5a5a5u || vbits!=0xa5a5a5a5u;}
                for(size_t row=0;row<CACHE;++row) if(row%2==0 || row>=64) for(int c=0;c<M;++c) failures+=hkc[row*M+c]!=0xa5a5 || hvc[row*M+c]!=0xa5a5;
                for(size_t i=size_t(M)*CACHE;i<kc.n;++i) failures+=hkc[i]!=0xa5a5 || hvc[i]!=0xa5a5;
                if(!path) {refk=hk;refv=hv;refkc=hkc;refvc=hvc;refy=std::move(hq);} else {
                    s1_error ek,ev,ec;size_t cache_words=0;
                    for(size_t i=0;i<nout;++i) {ek.add(refk[i],hk[i],2e-5,2e-4);ev.add(refv[i],hv[i],2e-5,2e-4);}
                    for(size_t i=0;i<size_t(M)*CACHE;++i) {
                        cache_words+=(refkc[i]!=hkc[i])+(refvc[i]!=hvc[i]);
                        ec.add(ggml_fp16_to_fp32(refkc[i]),ggml_fp16_to_fp32(hkc[i]),INFINITY,0);
                        ec.add(ggml_fp16_to_fp32(refvc[i]),ggml_fp16_to_fp32(hvc[i]),INFINITY,0);
                    }
                    failures+=ek.bad+ev.bad+ec.bad;
                    failures+=memcmp(refy.data(),hq.data(),ny*sizeof(block_q8_1_mmq))!=0;
                    printf("S1_KV_ERROR tile=%d bank=%d replay=%d K_max_abs=%.9g K_max_rel=%.9g V_max_abs=%.9g V_max_rel=%.9g cache_words=%zu cache_max_abs=%.9g cache_max_rel=%.9g bad=%zu nonfinite=%zu\n",tile,bank,replay,ek.abs,ek.rel,ev.abs,ev.rel,cache_words,ec.abs,ec.rel,ek.bad+ev.bad+ec.bad,ek.nonfinite+ev.nonfinite+ec.nonfinite);
                    S1_CHECK(cudaGraphLaunch(graph[path],cuda.stream()));S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                    auto againk=dk.read(),againv=dv.read();
                    failures+=memcmp(hk.data(),againk.data(),nout*4)!=0 || memcmp(hv.data(),againv.data(),nout*4)!=0;
                }
                ++cases;
            }
        }
        for(auto g:graph) S1_CHECK(cudaGraphExecDestroy(g));
    }
    if(bench) {
        cudaGraphExec_t graph[2];
        for(int path=0;path<2;++path) graph[path]=s1_capture(cuda.stream(),[&](){for(int b=0;b<banks;++b) chain(path!=0,b);});
        // Current quantizer + two J32 GEMMs + both fixups + actual norm/RoPE/f16
        // K/V stores. Rotating banks, no added eviction traffic.
        for(int compare:{0,1}) {
            std::vector<float> saving;
            for(int pair=0;pair<8;++pair) {
                float us[2];for(int leg=0;leg<2;++leg) {const int side=(pair+leg)&1;us[side]=s1_time(graph[side ? compare : 0],cuda.stream(),8)/banks;}
                saving.push_back(us[0]-us[1]);
                printf("S1_KV_CHAIN candidate=%d tile=%d CTAs=%d pair=%d banks=%d parent_us=%.6f candidate_us=%.6f saving_us=%.6f input=%s quantizer_timed=%d\n",compare,tile,8*(32/tile)*2,pair,banks,us[0],us[1],us[0]-us[1],kwfile ? "captured" : "synthetic",!yfile);
            }
            std::sort(saving.begin(),saving.end());
            const float median=(saving[3]+saving[4])*0.5f;
            float trimmed=0;for(int i=1;i<7;++i) trimmed+=saving[i]/6;
            printf("S1_KV_GATE candidate=%d tile=%d median_saving_us=%.6f trimmed_saving_us=%.6f min=%.6f max=%.6f five_us_screen=%s\n",
                compare,tile,median,trimmed,saving.front(),saving.back(),compare && !yfile && banks==8 && median>=5 && trimmed>=5 ? "PASS_MICRO_ONLY" : "NO");
        }
        for(auto g:graph) S1_CHECK(cudaGraphExecDestroy(g));
    }
    printf("S1_KV cases=%zu failures=%zu tile=%d CTAs=%d scope=micro rel_floor=1e-6\n",cases,failures,tile,8*(32/tile)*2);
    ggml_free(ctx);ggml_backend_free(backend);return failures ? 1 : 0;
}

// Integration gate: actual backend graph capture, input mutation and fallback.
// The microkernel test above remains separate, so a micro PASS cannot be
// confused with a successful graph dispatch.
static void s1_need(bool ok,const char * why) {
    if(!ok) {fprintf(stderr,"S1_KV_BACKEND_FAIL %s\n",why);exit(1);}
}
static std::vector<unsigned char> s1_tensor_bytes(const ggml_tensor * t) {
    std::vector<unsigned char> h(ggml_nbytes(t));ggml_backend_tensor_get(t,h.data(),0,h.size());return h;
}
struct s1_kv_route_fixture {
    ggml_backend_t backend;
    ggml_context * inputs,*ctx;
    ggml_backend_buffer_t input_buffer;
    ggml_gallocr_t alloc;
    ggml_cgraph * graph;
    ggml_tensor * a,*b,*wk,*wv,*cat,*km,*vm,*kc,*vc,*pos,*idx,*normw;
    int na,nb;bool reverse,between,extra;int specimen;
    static constexpr int M=1024,K=4096,N=32,CACHE=128;
    s1_kv_route_fixture(int which) : na(which==6 ? 8 : 16),nb(N-na),
            reverse(which!=1),between(which==3),extra(which==4),specimen(which) {
        backend=ggml_backend_cuda_init(0);s1_need(backend,"backend init");
        inputs=ggml_init({ggml_tensor_overhead()*32,nullptr,true});
        a=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,K,na);
        b=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,K,nb);
        ggml_set_input(a);ggml_set_input(b);
        ggml_set_name(a,which==2 ? "target_same_shape_input" : which==8 ? "CUDA0#dflash_ctx_rows#0" : "dflash_ctx_rows");
        if(which==8) a->flags&=~GGML_TENSOR_FLAG_INPUT;
        wk=ggml_new_tensor_2d(inputs,GGML_TYPE_Q8_0,K,M);wv=ggml_new_tensor_2d(inputs,GGML_TYPE_Q8_0,K,M);
        ggml_set_name(wk,"blk.3.attn_k.weight");
        ggml_set_name(wv,which==5 ? "blk.3.attn_q.weight" : which==7 ? "blk.4.attn_v.weight" : "blk.3.attn_v.weight");
        kc=ggml_new_tensor_2d(inputs,GGML_TYPE_F16,M,CACHE);vc=ggml_new_tensor_2d(inputs,GGML_TYPE_F16,M,CACHE);
        pos=ggml_new_tensor_1d(inputs,GGML_TYPE_I32,N*4);idx=ggml_new_tensor_1d(inputs,GGML_TYPE_I64,N);
        normw=ggml_new_tensor_1d(inputs,GGML_TYPE_F32,128);
        for(auto t:{kc,vc,pos,idx}) ggml_set_input(t);
        input_buffer=ggml_backend_alloc_ctx_tensors(inputs,backend);s1_need(input_buffer,"input allocation");
        ggml_backend_buffer_set_usage(input_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(19);
        for(auto w:{wk,wv}) {
            std::vector<unsigned char> h(ggml_nbytes(w));
            for(size_t off=0;off<h.size();off+=34) {
                auto scale=ggml_fp32_to_fp16(0.0001f+float(rng()%8)*0.0001f);memcpy(h.data()+off,&scale,2);
                for(int q=0;q<32;++q) h[off+2+q]=int(rng()%255)-127;
            }
            ggml_backend_tensor_set(w,h.data(),0,h.size());
        }
        std::vector<float> hn(128,1);ggml_backend_tensor_set(normw,hn.data(),0,hn.size()*4);
        ctx=ggml_init({ggml_tensor_overhead()*96+ggml_graph_overhead(),nullptr,true});graph=ggml_new_graph(ctx);
        cat=ggml_concat(ctx,a,b,1);ggml_set_name(cat,"s1_route_concat");
        km=ggml_mul_mat(ctx,wk,cat);vm=ggml_mul_mat(ctx,wv,cat);
        ggml_set_name(km,"s1_route_K");ggml_set_name(vm,"s1_route_V");
        ggml_set_output(km);ggml_set_output(vm);
        auto kv=ggml_reshape_3d(ctx,km,128,8,N);auto vv=ggml_reshape_3d(ctx,vm,128,8,N);
        auto first=reverse ? vv : kv;auto second=reverse ? kv : vv;
        ggml_build_forward_expand(graph,first);
        if(between) {auto barrier=ggml_scale(ctx,first,0.5f);ggml_set_output(barrier);ggml_build_forward_expand(graph,barrier);}
        ggml_build_forward_expand(graph,second);
        auto rms=ggml_rms_norm(ctx,kv,1e-6f);auto mul=ggml_mul(ctx,rms,normw);
        int sections[4]={64,0,0,0};
        auto rope=ggml_rope_multi(ctx,mul,pos,nullptr,128,sections,GGML_ROPE_TYPE_MROPE,16384,1e7f,1.0f/16,1,1,32,1);
        auto ks=ggml_set_rows(ctx,kc,ggml_reshape_2d(ctx,rope,M,N),idx);
        auto vs=ggml_set_rows(ctx,vc,ggml_reshape_2d(ctx,vv,M,N),idx);
        ggml_set_output(ks);ggml_set_output(vs);
        ggml_build_forward_expand(graph,ks);ggml_build_forward_expand(graph,vs);
        if(extra) {auto observe=ggml_scale(ctx,cat,0.5f);ggml_set_output(observe);ggml_build_forward_expand(graph,observe);}
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        s1_need(alloc && ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
    }
    ~s1_kv_route_fixture() {
        ggml_backend_synchronize(backend);ggml_backend_free(backend);
        ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_buffer_free(input_buffer);ggml_free(inputs);
    }
    ggml_backend_cuda_context & cuda() {return *static_cast<ggml_backend_cuda_context*>(backend->context);}
    void fill(int phase) {
        std::mt19937 rng(193+phase);
        for(auto t:{a,b}) {
            std::vector<float> h(ggml_nelements(t));
            for(auto & x:h) x=phase==1 ? 0.0f : float(int(rng()%2049)-1024)/1024;
            ggml_backend_tensor_set(t,h.data(),0,h.size()*4);
        }
        std::vector<unsigned char> poison(ggml_nbytes(kc),0xa5);
        ggml_backend_tensor_set(kc,poison.data(),0,poison.size());ggml_backend_tensor_set(vc,poison.data(),0,poison.size());
        std::vector<int32_t> hp(N*4);std::vector<int64_t> hi(N);
        for(int t=0;t<N;++t) {hi[t]=2*t+1;for(int section=0;section<4;++section) hp[section*N+t]=500+phase*7+t;}
        ggml_backend_tensor_set(pos,hp.data(),0,hp.size()*4);ggml_backend_tensor_set(idx,hi.data(),0,hi.size()*8);
    }
    void run() {
        s1_need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"backend graph compute");
        ggml_backend_synchronize(backend);
    }
    cudaGraph_t captured() {
#ifdef USE_CUDA_GRAPH
        auto * n=graph->nodes[0];uint64_t key=uintptr_t(n);
        key=key*0x9E3779B97F4A7C15ull^uint64_t(graph->n_nodes);
        for(int i=0;i<GGML_MAX_DIMS;++i) key=key*0x9E3779B97F4A7C15ull^uint64_t(n->ne[i]);
        auto found=cuda().cuda_graphs.find((void*)uintptr_t(key));s1_need(found!=cuda().cuda_graphs.end(),"exact backend graph key");
        auto & c=*found->second;s1_need(c.graph && c.instance && c.warmup_complete,"warm captured graph");return c.graph;
#else
        s1_need(false,"CUDA graphs required");return nullptr;
#endif
    }
    void inspect(bool grouped) {
        auto graph=captured();size_t n=0;S1_CHECK(cudaGraphGetNodes(graph,nullptr,&n));
        std::vector<cudaGraphNode_t> nodes(n);S1_CHECK(cudaGraphGetNodes(graph,nodes.data(),&n));
        int joint=0,ordinary=0,fixups=0,quantizers=0;
        const auto packed=cuda().fwht_q8().plain_q8.find(cat);s1_need(packed!=cuda().fwht_q8().plain_q8.end(),"packed activation exists");
        for(auto node:nodes) {
            cudaGraphNodeType type;S1_CHECK(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams p={};S1_CHECK(cudaGraphKernelNodeGetParams(node,&p));const char * name=nullptr;
            S1_CHECK(cudaFuncGetName(&name,p.func));s1_need(p.kernelParams,"kernel params available");
            if(strstr(name,"quantize_mmq_q8_1")) ++quantizers;
            if(strstr(name,"ggml_cuda_s1_kv_full_k")) {
                ++joint;s1_need(strstr(name,"16"),"J16 specialization");
                s1_need(p.gridDim.x==8 && p.gridDim.y==2 && p.gridDim.z==2 && p.blockDim.x==32 && p.blockDim.y==8,"grouped 32 CTA geometry");
                auto w0=reverse ? wv : wk;auto w1=reverse ? wk : wv;
                auto o0=reverse ? vm : km;auto o1=reverse ? km : vm;
                s1_need(*(void**)p.kernelParams[0]==w0->data && *(void**)p.kernelParams[1]==w1->data,"both model weight pointers");
                s1_need(*(void**)p.kernelParams[2]==packed->second,"same concat D4 bytes");
                s1_need(*(void**)p.kernelParams[3]==o0->data && *(void**)p.kernelParams[4]==o1->data,"both model output pointers");
            } else if(strstr(name,"mul_mat_q_stream_k_fixup")) ++fixups;
            else if(strstr(name,"mul_mat_q") && !strstr(name,"quantize")) ++ordinary;
        }
        s1_need(quantizers==1,"exactly one quantizer for the two projections");
        s1_need(joint==(grouped?1:0) && ordinary==(grouped?0:2) && fixups==(grouped?0:2),"actual grouped/parent route and fixup count");
    }
};

static int s1_backend_route(int argc,char ** argv) {
    bool quick=false,forced_off=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--route")) continue;
        if(!strcmp(argv[i],"--quick")) quick=true;
        else if(!strcmp(argv[i],"--off")) forced_off=true;
        else {fprintf(stderr,"unknown route argument %s\n",argv[i]);return 2;}
    }
    const char * env=getenv("GGML_CUDA_S1_DFLASH_KV_J8");const bool had_env=env!=nullptr;
    const std::string original=env ? env : "";
    const auto candidate_mode=[&]() {
        if(forced_off) setenv("GGML_CUDA_S1_DFLASH_KV_J8","0",1);
        else if(had_env) setenv("GGML_CUDA_S1_DFLASH_KV_J8",original.c_str(),1);
        else unsetenv("GGML_CUDA_S1_DFLASH_KV_J8");
    };
    unsetenv("GGML_CUDA_DFLASH_CONCAT_Q8");candidate_mode();
    const bool enabled=ggml_cuda_s1_dflash_kv_enabled();size_t failures=0,cases=0,refusals=0;
    for(int specimen=0;specimen<9;++specimen) {
        if(quick && specimen!=0 && specimen!=2 && specimen!=3 && specimen!=8) continue;
        s1_kv_route_fixture parent(specimen),candidate(specimen);
        const bool grouped=enabled && (specimen<2 || specimen==8);
        for(int phase=0;phase<3;++phase) {
            parent.fill(phase);candidate.fill(phase);
            setenv("GGML_CUDA_S1_DFLASH_KV_J8","0",1);
            for(int warm=0;warm<3;++warm) parent.run();parent.inspect(false);
            candidate_mode();for(int warm=0;warm<3;++warm) candidate.run();candidate.inspect(grouped);
            auto ka=s1_tensor_bytes(parent.km),kb=s1_tensor_bytes(candidate.km);
            auto va=s1_tensor_bytes(parent.vm),vb=s1_tensor_bytes(candidate.vm);
            auto kca=s1_tensor_bytes(parent.kc),kcb=s1_tensor_bytes(candidate.kc);
            auto vca=s1_tensor_bytes(parent.vc),vcb=s1_tensor_bytes(candidate.vc);
            s1_error ek,ev,ec;size_t cache_words=0,guard_errors=0;
            for(size_t i=0;i<ka.size();i+=4) {
                float x,y,u,v;memcpy(&x,ka.data()+i,4);memcpy(&y,kb.data()+i,4);memcpy(&u,va.data()+i,4);memcpy(&v,vb.data()+i,4);
                ek.add(x,y,2e-5,2e-4);ev.add(u,v,2e-5,2e-4);
            }
            for(size_t i=0;i<kca.size();i+=2) {
                uint16_t x,y,u,v;memcpy(&x,kca.data()+i,2);memcpy(&y,kcb.data()+i,2);memcpy(&u,vca.data()+i,2);memcpy(&v,vcb.data()+i,2);
                cache_words+=(x!=y)+(u!=v);
                ec.add(ggml_fp16_to_fp32(x),ggml_fp16_to_fp32(y),INFINITY,0);ec.add(ggml_fp16_to_fp32(u),ggml_fp16_to_fp32(v),INFINITY,0);
                const size_t row=(i/2)/1024;
                if(row%2==0 || row>=64) guard_errors+=(x!=0xa5a5 || y!=0xa5a5 || u!=0xa5a5 || v!=0xa5a5);
            }
            auto & pc=parent.cuda().fwht_q8().plain_q8;auto & cc=candidate.cuda().fwht_q8().plain_q8;
            if(grouped && phase==0) {
                void * saved=candidate.km->data;candidate.km->data=candidate.vm->data;
                const bool alias=ggml_cuda_s1_dflash_kv_j16(candidate.cuda(),candidate.cat,candidate.vm,candidate.km,cc.at(candidate.cat));
                candidate.km->data=saved;s1_need(!alias,"output alias must refuse before launch");++refusals;
                const int64_t cols=candidate.km->ne[1];candidate.km->ne[1]=31;
                const bool shape=ggml_cuda_s1_dflash_kv_j16(candidate.cuda(),candidate.cat,candidate.vm,candidate.km,cc.at(candidate.cat));
                candidate.km->ne[1]=cols;s1_need(!shape,"unsupported output shape must refuse");++refusals;
                s1_need(!ggml_cuda_s1_dflash_kv_j16(candidate.cuda(),candidate.cat,candidate.vm,candidate.km,
                    (const char*)cc.at(candidate.cat)+1),"unaligned D4 must refuse");++refusals;
            }
            const size_t qbytes=4096*32/128*sizeof(block_q8_1_mmq);
            std::vector<unsigned char> qa(qbytes),qb(qbytes);
            S1_CHECK(cudaMemcpy(qa.data(),pc.at(parent.cat),qbytes,cudaMemcpyDeviceToHost));
            S1_CHECK(cudaMemcpy(qb.data(),cc.at(candidate.cat),qbytes,cudaMemcpyDeviceToHost));
            failures+=ek.bad+ev.bad+ec.bad+guard_errors+(qa!=qb);
            if(!grouped) failures+=(ka!=kb)+(va!=vb)+(kca!=kcb)+(vca!=vcb);
            candidate.run();failures+=s1_tensor_bytes(candidate.km)!=kb || s1_tensor_bytes(candidate.vm)!=vb ||
                s1_tensor_bytes(candidate.kc)!=kcb || s1_tensor_bytes(candidate.vc)!=vcb;
            printf("S1_KV_BACKEND case=%d phase=%d grouped=%d K_max_abs=%.9g K_max_rel=%.9g V_max_abs=%.9g V_max_rel=%.9g cache_words=%zu cache_max_abs=%.9g cache_max_rel=%.9g guard_errors=%zu D4_exact=%d\n",
                specimen,phase,grouped,ek.abs,ek.rel,ev.abs,ev.rel,cache_words,ec.abs,ec.rel,guard_errors,int(qa==qb));++cases;
        }
    }
    candidate_mode();printf("S1_KV_BACKEND cases=%zu failures=%zu enabled=%d guard_refusals=%zu capture_route_checked=true\n",cases,failures,enabled,refusals);
    return failures ? 1 : 0;
}
