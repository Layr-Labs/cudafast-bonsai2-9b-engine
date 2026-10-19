#include "s1_test_util.cuh"
#include "ggml-cuda.h"
#include "fattn.cuh"
#include "s1-fa-splits.cuh"
#include "norm.cuh"
#include "rope.cuh"
#include "quantize.cuh"
#include <random>

int main(int argc, char ** argv) {
    bool quick=false,bench=false;
    int only_d=0,only_t=0,only_kv=0,only_split=-1;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--quick")) quick=true;
        else if(!strcmp(argv[i],"--bench")) bench=true;
        else if(i+1<argc && !strcmp(argv[i],"--d")) only_d=atoi(argv[++i]);
        else if(i+1<argc && !strcmp(argv[i],"--t")) only_t=atoi(argv[++i]);
        else if(i+1<argc && !strcmp(argv[i],"--kv")) only_kv=atoi(argv[++i]);
        else if(i+1<argc && !strcmp(argv[i],"--split")) only_split=atoi(argv[++i]);
        else { fprintf(stderr,"unknown argument %s\n",argv[i]);return 2; }
    }
    if(only_split>0 && !ggml_cuda_s1_fa_split_valid(only_split)) return 2;
    ggml_backend_t backend=ggml_backend_cuda_init(0); if(!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(19);std::uniform_real_distribution<float> d(-1,1);
    int cases=0,candidate_shapes=0;size_t failures=0;
    for(int D:{128,256}) for(int T:{9,15,16}) for(int NK:{512,768,1024})
    for(int mode:{0,1,2}) {
        if((only_d && D!=only_d) || (only_t && T!=only_t) || (only_kv && NK!=only_kv)) continue;
        if(quick && (T!=16 || NK!=512 || mode!=1)) continue;
        if(bench && mode!=0) continue;
        const int H=D==256 ? 16 : 32, HK=H/4;
        const int banks=bench ? 8 : 1, PM=D==256 ? 8192 : 4096;
        const ggml_type ptype=D==256 ? GGML_TYPE_PQ2_0 : GGML_TYPE_Q8_0;
        const size_t weight_bytes=ggml_row_size(ptype,4096)*PM;
        const size_t nq=size_t(D)*T*H,nc=size_t(D)*NK*HK,nm=size_t(NK)*GGML_PAD(T,8);
        s1_buffer<float> seed(nq),x(nq),q(nq),w(D),gate(nq*2),signs(4096);
        s1_buffer<int32_t> positions(T*4);
        s1_buffer<ggml_fp16_t> k(nc*banks),v(nc*banks),mask(nm);
        s1_buffer<unsigned char> pw(bench ? weight_bytes*banks : 1);
        // Include the parent's MMQ read padding. The producer uses the lower
        // launcher, so no synthetic tensor with buffer==nullptr reaches the
        // public ggml_backend_buffer_get_usage() check.
        s1_buffer<unsigned char> producer_q8((nq/128+32)*sizeof(block_q8_1_mmq));
        S1_CHECK(cudaMemset(producer_q8.p,0,producer_q8.n));
        s1_buffer<unsigned char> out(nq*4+64),packed(size_t(4096)*T/128*144+64);
        std::vector<float> hx(nq),hw(D,1),hg(nq*2),hs(4096);
        std::vector<ggml_fp16_t> hk(nc*banks),hv(nc*banks),hm(nm,ggml_fp32_to_fp16(-INFINITY));
        std::vector<int32_t> hp(T*4);
        for(auto & a:hx) a=d(rng);for(auto & a:hg) a=d(rng);
        for(auto & a:hk) a=ggml_fp32_to_fp16(d(rng));for(auto & a:hv) a=ggml_fp32_to_fp16(d(rng));
        for(auto & a:hs) a=rng()&1 ? 1 : -1;
        for(int t=0;t<T;++t) for(int j=0;j<NK;++j) {
            // mode 1 leaves many empty partitions; mode 2 includes an all-masked query.
            const bool hidden=mode==1 ? (t==0 ? j!=NK-1 : t==1 ? j!=0 : j>NK-T+t) :
                (mode==2 && t==0) || (D==256 && j>NK-T+t);
            hm[size_t(t)*NK+j]=ggml_fp32_to_fp16(hidden ? -INFINITY : 0.0f);
        }
        for(int s=0;s<4;++s) for(int t=0;t<T;++t) hp[s*T+t]=NK-T+t;
        seed.upload(hx);w.upload(hw);gate.upload(hg);signs.upload(hs);k.upload(hk);v.upload(hv);mask.upload(hm);positions.upload(hp);
        if(bench) {
            // Rotate actual projection weights AND per-layer K/V, rather than
            // repeating one hot FA or inserting an independent eviction kernel.
            std::vector<unsigned char> hweights(weight_bytes*banks);
            for(size_t off=0;off<hweights.size();off+=34) {
                const ggml_fp16_t scale=ggml_fp32_to_fp16(0.001f);
                memcpy(hweights.data()+off,&scale,2);
                for(int j=2;j<34;++j) {
                    int value=0;
                    if(D==256) for(int bit=0;bit<8;bit+=2) value|=(rng()%3)<<bit;
                    else value=int(rng()%255)-127;
                    hweights[off+j]=(unsigned char)value;
                }
            }
            pw.upload(hweights);
            for(int b=0;b<banks;++b) if(!ggml_cuda_dl_register(pw.p+b*weight_bytes,PM,4096,ptype)) return 3;
        }
        ggml_context * ctx=ggml_init({2*1024*1024,nullptr,true});
        ggml_tensor * raw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H,T);raw->data=x.p;
        if(bench && D==256) {
            raw->data=gate.p;raw->nb[1]=2*D*sizeof(float);raw->nb[2]=8192*sizeof(float);
        }
        ggml_tensor * wt=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,D);wt->data=w.p;
        auto rms=ggml_rms_norm(ctx,raw,1e-6f);auto mul=ggml_mul(ctx,rms,wt);
        auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,4*T);pos->data=positions.p;
        int sections[4]={D==256 ? 11 : 64,D==256 ? 11 : 0,D==256 ? 10 : 0,0};
        auto rope=ggml_rope_multi(ctx,mul,pos,nullptr,D==256 ? 64 : 128,sections,
            D==256 ? GGML_ROPE_TYPE_IMROPE : GGML_ROPE_TYPE_MROPE,D==256 ? 262144 : 16384,
            1e7f,D==256 ? 1.0f : 1.0f/16,D==256 ? 0.0f : 1.0f,1.0f,32.0f,1.0f);
        rope->data=q.p;
        auto Q=ggml_permute(ctx,rope,0,2,1,3);Q->data=q.p;
        auto K=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,D,NK,HK,1);K->data=k.p;
        auto V=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,D,NK,HK,1);V->data=v.p;
        auto M=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,NK,GGML_PAD(T,8),1,1);M->data=mask.p;
        auto fa=ggml_flash_attn_ext(ctx,Q,K,V,M,1/sqrtf(float(D)),0,0);fa->data=out.p;
        if(!ggml_cuda_rms_norm_mul_mrope_supported(rms,mul,rope) || !ggml_cuda_flash_attn_ext_supported(0,fa)) return 3;
        auto chain=[&]() {
          for(int bank=0;bank<banks;++bank) {
            if(bench) {
                K->data=k.p+bank*nc;V->data=v.p+bank*nc;
                quantize_mmq_q8_1_cuda(seed.p,nullptr,producer_q8.p,ptype,
                    4096,4096,nq,nq,4096,T,1,1,cuda.stream());
                mmq_args a={};
                a.x=(const char*)pw.p+bank*weight_bytes;a.type_x=ptype;
                a.y=(const int*)producer_q8.p;a.dst=D==256 ? gate.p : x.p;
                a.ncols_x=4096;a.nrows_x=PM;a.ncols_dst=T;a.stride_row_x=4096/ggml_blck_size(ptype);
                a.ncols_y=T;a.nrows_dst=PM;a.ncols_max=T;
                a.nchannels_x=a.nchannels_y=a.nsamples_x=a.nsamples_y=1;
                if(D==256) launch_mul_mat_q<GGML_TYPE_PQ2_0,16,false>(cuda,a,cuda.stream());
                else launch_mul_mat_q<GGML_TYPE_Q8_0,16,false>(cuda,a,cuda.stream());
            } else s1_copy(seed.p,x.p,nq,cuda.stream());
            ggml_cuda_op_rms_norm_mul_mrope_fused(cuda,rms,mul,rope,nullptr,nullptr,nullptr);
            ggml_cuda_flash_attn_ext(cuda,fa);
            if(D==256) {
                fwht_quantize_mmq_d4_sigmoid_gate_cuda(gate.p+(bench ? D : 0),(const float*)out.p,signs.p,packed.p,
                    4096,4096,T,256,512,8192,4096,cuda.stream());
            } else {
                // DFlash has no attention gate: its Q8 wo consumes ordinary D4.
                quantize_mmq_q8_1_cuda((const float*)out.p,nullptr,packed.p,GGML_TYPE_Q8_0,
                    4096,4096,4096*T,4096*T,4096,T,1,1,cuda.stream());
            }
          }
        };
        std::vector<cudaGraphExec_t> graphs;
        // Frozen sm_121a MMA configuration for these unchanged query tiles:
        // D256 takes 32 KV rows per work tile, D128 takes 64. Never time an
        // invalid override as if it had applied; inspect actual captured grids.
        const int nbatch_fa=D==256 ? 32 : 64, max_splits=NK/nbatch_fa;
        const int tiles=D==256 ? 4 : 8;
        std::vector<int> splits={0};
        for(int candidate:{4,6,8,12,16,24,32,48}) {
            if(only_split>=0 && candidate!=only_split) continue;
            if(candidate>max_splits) {
                printf("S1_FA_SKIP D=%d T=%d KV=%d split=%d max_splits=%d reason=empty_CTA_metadata\n",D,T,NK,candidate,max_splits);
            } else splits.push_back(candidate);
        }
        if(only_split==1 || only_split==2) splits.push_back(only_split);
        candidate_shapes+=int(splits.size())-1;
        void * parent_function=nullptr;dim3 parent_block;size_t parent_smem=0;
        for(int split:splits) {
            setenv("GGML_CUDA_S1_FA_SPLITS",split ? "1" : "0",1);
            const auto s=std::to_string(split);
            setenv("GGML_CUDA_S1_FA_TARGET",s.c_str(),1);setenv("GGML_CUDA_S1_FA_DRAFTER",s.c_str(),1);
            chain(); S1_CHECK(cudaStreamSynchronize(cuda.stream())); // warm allocations before capture
            cudaGraph_t graph;cudaGraphExec_t exec;
            S1_CHECK(cudaStreamBeginCapture(cuda.stream(),cudaStreamCaptureModeGlobal));chain();
            S1_CHECK(cudaStreamEndCapture(cuda.stream(),&graph));
            size_t count=0;S1_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
            std::vector<cudaGraphNode_t> nodes(count);S1_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
            int bodies=0;
            for(auto node:nodes) {
                cudaGraphNodeType type;S1_CHECK(cudaGraphNodeGetType(node,&type));
                if(type!=cudaGraphNodeTypeKernel) continue;
                cudaKernelNodeParams params;S1_CHECK(cudaGraphKernelNodeGetParams(node,&params));
                const void * first=nullptr;
                if(params.kernelParams && params.kernelParams[0]) memcpy(&first,params.kernelParams[0],sizeof(first));
                if(first!=q.p || params.blockDim.x!=32 || params.blockDim.y!=4) continue;
                ++bodies;
                if(!split) {parent_function=params.func;parent_block=params.blockDim;parent_smem=params.sharedMemBytes;}
                failures+=params.func!=parent_function || params.blockDim.x!=parent_block.x || params.blockDim.y!=parent_block.y ||
                    params.blockDim.z!=parent_block.z || params.sharedMemBytes!=parent_smem;
                if(split) failures+=params.gridDim.x!=unsigned(tiles*split) || params.gridDim.y!=1 || params.gridDim.z!=1;
                printf("S1_FA_CAPTURE D=%d T=%d KV=%d split=%d tiles=%d blocks=%u max_splits=%d smem=%u\n",D,T,NK,split,tiles,params.gridDim.x,max_splits,params.sharedMemBytes);
            }
            failures+=bodies!=banks;
            if(bodies!=banks) {fprintf(stderr,"FA route missing bodies=%d expected=%d\n",bodies,banks);return 3;}
            S1_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));S1_CHECK(cudaGraphDestroy(graph));graphs.push_back(exec);
        }
        const int cc=ggml_cuda_info().devices[0].cc;
        const auto last=std::to_string(splits.back());
        setenv("GGML_CUDA_S1_FA_TARGET",last.c_str(),1);setenv("GGML_CUDA_S1_FA_DRAFTER",last.c_str(),1);
        if(ggml_cuda_s1_fa_splits(fa,cc,max_splits)!=splits.back()) ++failures;
        setenv("GGML_CUDA_S1_FA_SPLITS","0",1);
        if(ggml_cuda_s1_fa_splits(fa,cc,max_splits)!=0) ++failures;
        setenv("GGML_CUDA_S1_FA_SPLITS","1",1);
        const int64_t saved_t=Q->ne[1];Q->ne[1]=17;
        if(ggml_cuda_s1_fa_splits(fa,cc,max_splits)!=0) ++failures;
        Q->ne[1]=saved_t;
        for(int replay=0;replay<3;++replay) {
            for(auto & a:hx) a=d(rng);seed.upload(hx);
            std::vector<float> ref;std::vector<unsigned char> ref_packed;
            for(size_t path=0;path<graphs.size();++path) {
                S1_CHECK(cudaMemsetAsync(out.p,0xa5,out.n,cuda.stream()));
                S1_CHECK(cudaMemsetAsync(packed.p,0xa5,packed.n,cuda.stream()));
                S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                S1_CHECK(cudaGraphLaunch(graphs[path],cuda.stream()));S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                auto b=out.read(),p=packed.read();std::vector<float> h(nq);memcpy(h.data(),b.data(),nq*4);
                for(size_t i=nq*4;i<b.size();++i) failures+=b[i]!=0xa5;
                for(size_t i=packed.n-64;i<p.size();++i) failures+=p[i]!=0xa5;
                if(!path) {ref=h;ref_packed=p;} else {
                    s1_error err;
                    for(size_t i=0;i<nq;++i) {
                        // All-masked behavior must match the parent's bit pattern,
                        // including inherited NaNs; report it separately.
                        if(mode==2 && i<size_t(D)*H && !std::isfinite(ref[i]) && !std::isfinite(h[i])) {
                            if(memcmp(&ref[i],&h[i],4)) ++err.bad;
                            ++err.nonfinite;continue;
                        }
                        err.add(ref[i],h[i],1e-3,1e-3);
                    }
                    failures+=err.bad;
                    size_t downstream_bytes=0;
                    for(size_t i=0;i<p.size()-64;++i) downstream_bytes+=p[i]!=ref_packed[i];
                    printf("S1_FA_ERROR D=%d T=%d KV=%d mask=%d split=%d replay=%d max_abs=%.9g max_rel=%.9g nonfinite=%zu bad=%zu downstream_bytes=%zu\n",D,T,NK,mode,splits[path],replay,err.abs,err.rel,err.nonfinite,err.bad,downstream_bytes);
                    // Repeat at the same input for determinism.
                    S1_CHECK(cudaGraphLaunch(graphs[path],cuda.stream()));S1_CHECK(cudaStreamSynchronize(cuda.stream()));
                    auto again=out.read(),again_packed=packed.read();failures+=again!=b || again_packed!=p;
                }
                ++cases;
            }
        }
        if(bench) {
            // Alternating paired chain timings; no explicit eviction between runs.
            // The parent-parent leg detects order drift. Report all samples.
            for(size_t path=0;path<graphs.size();++path) {
                std::vector<float> saving;
                for(int pair=0;pair<8;++pair) {
                    float us[2];for(int leg=0;leg<2;++leg) {
                        const int side=(pair+leg)&1;us[side]=s1_time(graphs[side ? path : 0],cuda.stream())/banks;
                    }
                    saving.push_back(us[0]-us[1]);
                    printf("S1_FA_CHAIN D=%d T=%d KV=%d split=%d pair=%d parent_us=%.6f candidate_us=%.6f saving_us=%.6f banks=%d producer=projection_rms_rope consumer=%s\n",D,T,NK,splits[path],pair,us[0],us[1],us[0]-us[1],banks,D==256 ? "sigmoid_d4" : "q8_d4");
                }
                std::sort(saving.begin(),saving.end());
                const float median=(saving[3]+saving[4])*0.5f;
                float trimmed=0;for(int i=1;i<7;++i) trimmed+=saving[i]/6;
                printf("S1_FA_CHAIN_SUMMARY D=%d T=%d KV=%d split=%d median_saving_us=%.6f trimmed_saving_us=%.6f min=%.6f max=%.6f\n",
                    D,T,NK,splits[path],median,trimmed,saving.front(),saving.back());
            }
        }
        for(auto g:graphs) S1_CHECK(cudaGraphExecDestroy(g));ggml_free(ctx);
        if(bench) ggml_cuda_dl_unregister_range(pw.p,weight_bytes*banks);
    }
    printf("S1_FA cases=%d candidate_shapes=%d failures=%zu rel_floor=1e-6\n",cases,candidate_shapes,failures);
    ggml_backend_free(backend);return failures ? 1 : cases && (candidate_shapes || only_split==0) ? 0 : 2;
}
