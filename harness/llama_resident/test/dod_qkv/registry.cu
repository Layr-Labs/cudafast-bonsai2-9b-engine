// Links the real backend: registration, production dispatch, graph capture,
// changed activations, independent slice teardown, and same-address reload.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "mmq.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void need(bool b,const char * s) { if(!b){std::fprintf(stderr,"FAIL %s\n",s);std::exit(1);} }
static void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);} }
template <typename T> static T * alloc(size_t n){T *p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template <typename T> static std::vector<T> read(const T *p,size_t n){std::vector<T> h(n);ck(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static unsigned random_state=0xD1;
static unsigned rnd(){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state;}
static constexpr int R[3]={8192,1024,1024};
static constexpr size_t RB=32*sizeof(block_ptq1_0), TOTAL=10240*RB;
int main(int argc,char **argv){
    bool off=argc>1 && std::strcmp(argv[1],"--off")==0;
    bool exact=argc>1 && std::strcmp(argv[1],"--exact")==0;
    setenv("GGML_CUDA_DOD_QKV",off?"0":"1",1);
    setenv("GGML_CUDA_DOD_QKV_EXACT",exact?"1":"0",1);
    auto backend=ggml_backend_cuda_init(0);need(backend,"CUDA backend");
    auto &cuda=*static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_context *wc=ggml_init({ggml_tensor_overhead()*4,nullptr,true});
    ggml_tensor *w[3];for(int s=0;s<3;++s)w[s]=ggml_new_tensor_2d(wc,GGML_TYPE_PQ2_0,4096,R[s]);
    auto wb=ggml_backend_alloc_ctx_tensors(wc,backend);need(wb,"weight buffer");
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto reg=ggml_backend_reg_by_name("CUDA");
    using RF=size_t(*)(const ggml_tensor*);
    using QF=size_t(*)(const ggml_tensor*,const ggml_tensor*,const ggml_tensor*);
    auto rf=(RF)ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_register_decode_layout");
    auto qf=(QF)ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_register_dod_qkv");
    need(rf&&qf,"load hooks registered");
    std::vector<std::vector<char>> original(3);
    for(int s=0;s<3;++s){
        std::vector<char> h(ggml_nbytes(w[s]));
        for(size_t i=0;i<h.size();i+=34){
            half d=__float2half(float(int(rnd()%101)-50)*0.001f);std::memcpy(h.data()+i,&d,2);
            for(int j=0;j<32;++j){unsigned q=0;for(int t=0;t<4;++t)q|=(rnd()%3)<<(2*t);h[i+2+j]=char(q);}
        }
        ggml_backend_tensor_set(w[s],h.data(),0,h.size());
        need(rf(w[s])==R[s]*RB,"separate DL allocation bytes");
        original[s]=read(ggml_cuda_dl_find(w[s]->data)->dl,R[s]*RB);
    }
    size_t free_before,total;ck(cudaMemGetInfo(&free_before,&total));
    need(qf(w[0],w[1],w[2])==(off?0:TOTAL),"QKV allocation accounting");
    need(qf(w[0],w[1],w[2])==0,"idempotent QKV registration");
    const auto *g=ggml_cuda_dod_qkv_find(w[0]->data);
    need(bool(g)==!off,"kill switch disables registration");
    for(int s=0;s<3;++s){
        const auto *e=ggml_cuda_dl_find(w[s]->data);
        need(bool(e->dl_owner)==!off,"shared allocation ownership");
        need(read(e->dl,R[s]*RB)==original[s],"every original DL byte preserved");
        if(!off)need(e->dl==g->dl+(s==0?0:s==1?8192:9216)*RB,"slice offsets");
    }
    size_t free_after;ck(cudaMemGetInfo(&free_after,&total));
    std::printf("REGISTRY free_before=%zu free_after=%zu DL_bytes=%zu (transient repack allocation excluded)\n",free_before,free_after,TOTAL);
    auto *partials=alloc<float>(48*2048);
    for(int n:{9,13,16}){
        ggml_context *c=ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true});
        auto *x=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,n);
        ggml_tensor *out[3];for(int s=0;s<3;++s)out[s]=ggml_mul_mat(c,w[s],x);
        auto buf=ggml_backend_alloc_ctx_tensors(c,backend);need(buf,"output allocation");
        auto *y=alloc<block_q8_1_mmq>(size_t(32)*n+16);
        for(int phase=0;phase<2;++phase){
            std::vector<block_q8_1_mmq> hy(size_t(32)*n+16);
            for(auto &b:hy){for(auto &d:b.d4)d=float(int(rnd()%1001)-500)*(phase?.03f:.0001f);for(auto &q:b.qs)q=int(rnd()%255)-127;}
            ck(cudaMemcpy(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice));
            std::vector<std::vector<float>> ref(3);
            for(int s=0;s<3;++s){
                mmq_args a={};a.x=(const char*)w[s]->data;a.type_x=GGML_TYPE_PQ2_0;a.y=(const int*)y;a.dst=(float*)out[s]->data;
                a.ncols_x=4096;a.nrows_x=R[s];a.ncols_dst=a.ncols_y=a.ncols_max=n;a.stride_row_x=32;a.nrows_dst=R[s];
                a.nchannels_x=a.nchannels_y=a.nsamples_x=a.nsamples_y=1;a.stride_channel_dst=a.stride_sample_dst=R[s]*n;
                ggml_cuda_mmq_dl_run<GGML_TYPE_PQ2_0>(a,*ggml_cuda_dl_find(w[s]->data),48,48,partials,cuda.stream());
                ck(cudaStreamSynchronize(cuda.stream()));ref[s]=read((float*)out[s]->data,R[s]*n);
            }
            const auto launch=[&]{return ggml_cuda_mul_mat_q_dod_qkv(cuda,out[0],out[1],out[2],y);};
            need(launch()==!off,"production dispatch / kill switch");
            if(!off){
                ck(cudaStreamSynchronize(cuda.stream())); // initialize pool + kernel attrs before capture
                cudaGraph_t graph;cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(cuda.stream(),cudaStreamCaptureModeGlobal));need(launch(),"capture dispatch");
                ck(cudaStreamEndCapture(cuda.stream(),&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                for(int rep=0;rep<2;++rep){
                    for(int s=0;s<3;++s)ck(cudaMemsetAsync(out[s]->data,0xa5,R[s]*n*4,cuda.stream()));
                    ck(cudaGraphLaunch(exec,cuda.stream()));ck(cudaStreamSynchronize(cuda.stream()));
                    for(int s=0;s<3;++s){
                        auto got=read((float*)out[s]->data,R[s]*n);double ma=0,mr=0,peak=0;size_t mism=0;
                        for(size_t i=0;i<got.size();++i){need(std::isfinite(got[i]),"finite output");double d=std::abs(double(got[i])-ref[s][i]);ma=std::max(ma,d);mr=std::max(mr,d/std::max(1e-6,std::abs(double(ref[s][i]))));peak=std::max(peak,std::abs(double(ref[s][i])));mism+=std::memcmp(&got[i],&ref[s][i],4)!=0;}
                        std::printf("ROUTE N=%d phase=%d rep=%d segment=%d max_abs=%.9g max_rel=%.9g max_ref=%.9g mism=%zu\n",n,phase,rep,s,ma,mr,peak,mism);
                        need(ma<=2e-5*std::max(1.,peak),"route tolerance");if(exact)need(mism==0,"exact grouped route");
                    }
                }
                ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
            }
        }
        // Shape decline must happen before any kernel: a J=8 input.
        x->ne[1]=8;need(!ggml_cuda_mul_mat_q_dod_qkv(cuda,out[0],out[1],out[2],y),"short input excluded");
        ck(cudaFree(y));ggml_backend_buffer_free(buf);ggml_free(c);
    }
    // Independent-range teardown cannot leave a dangling slice or lookup.
    ggml_cuda_dl_unregister_range(w[0]->data,1);
    need(!ggml_cuda_dod_qkv_find(w[0]->data),"group invalidated on first removed source");
    need(!ggml_cuda_dl_find(w[0]->data),"first DL removed");
    for(int s=1;s<3;++s)need(read(ggml_cuda_dl_find(w[s]->data)->dl,R[s]*RB)==original[s],"remaining slices still alive");
    for(int s=1;s<3;++s)ggml_cuda_dl_unregister_range(w[s]->data,1);
    // Same weight addresses, a fresh load, no stale registry entries.
    for(int s=0;s<3;++s){need(!ggml_cuda_dl_find(w[s]->data),"stale entry removed");need(rf(w[s])==R[s]*RB,"same-address reload");}
    need(qf(w[0],w[1],w[2])==(off?0:TOTAL),"same-address QKV reload");
    ggml_backend_buffer_free(wb);
    for(int s=0;s<3;++s)need(!ggml_cuda_dl_find(w[s]->data),"buffer destructor clears slices");
    need(!ggml_cuda_dod_qkv_find(w[0]->data),"buffer destructor clears group");
    ck(cudaFree(partials));ggml_free(wc);ggml_backend_free(backend);
    std::printf("DOD_QKV_REGISTRY_PASS off=%d exact=%d\n",off,exact);
}
