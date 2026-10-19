// Exact existing split-SwiGLU + D4 writer versus existing fused writer.
// Full allocations, specials, changed graph replays; native complete-chain timing.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void ck(cudaError_t e) { if(e != cudaSuccess) { fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(2); } }
static constexpr size_t G=256;
static __global__ void flush_l2(unsigned *p,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) p[i]=p[i]*1664525u+1013904223u;
}
static double med(std::vector<double> v) { std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static void resources(cudaGraph_t graph,const char*name,int expected) {
    size_t n=0; ck(cudaGraphGetNodes(graph,nullptr,&n)); std::vector<cudaGraphNode_t> nodes(n); ck(cudaGraphGetNodes(graph,nodes.data(),&n));
    int kernels=0;
    for(auto node:nodes) { cudaGraphNodeType t; ck(cudaGraphNodeGetType(node,&t)); if(t!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams p{}; ck(cudaGraphKernelNodeGetParams(node,&p)); cudaFuncAttributes a{}; ck(cudaFuncGetAttributes(&a,p.func));
        printf("resource arm=%s kernel=%d regs=%d local=%zu static_shared=%zu dynamic_shared=%u\n",name,kernels++,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,p.sharedMemBytes);
        if(a.localSizeBytes || a.numRegs>128)exit(3);
    }
    if(kernels!=expected) {fprintf(stderr,"unexpected kernel count %s %d\n",name,kernels);exit(3);}
}
int main() {
    setenv("GGML_CUDA_Q8_DECODE_SWIGLU","1",1);
    ggml_backend_t backend=ggml_backend_cuda_init(0); if(!backend)return 2;
    ggml_backend_cuda_context cuda(0); auto stream=cuda.stream(); std::mt19937 rng(20261019);
    const float specials[]={0.f,-0.f,1e-45f,-1e-45f,FLT_MIN,-FLT_MIN,FLT_MAX,-FLT_MAX,INFINITY,-INFINITY,NAN,-NAN,88.f,-88.f,89.f,-89.f,104.f,-104.f,1.f,-1.f};
    int cases=0;
    for(int k:{4096,12288,32768})for(int rows:{7,16,32})for(int mode=0;mode<6;++mode) {
        size_t bytes=size_t(k)*rows*4,qbytes=size_t(k)*rows/QK8_1_MMQ*sizeof(block_q8_1_mmq);
        std::vector<unsigned char> gate(bytes+2*G,0xa5),up(bytes+2*G,0xa5),r(qbytes+2*G),f(qbytes+2*G),back(bytes+2*G);
        unsigned char *dg,*du,*dr,*df; float*dt;
        ck(cudaMalloc(&dg,gate.size()));ck(cudaMalloc(&du,up.size()));ck(cudaMalloc(&dt,bytes));ck(cudaMalloc(&dr,r.size()));ck(cudaMalloc(&df,f.size()));
        auto fill=[&] {
            for(size_t i=0;i<bytes/4;++i) {
                float a=float(int(rng()%16001)-8000)/1000.f,b=float(int(rng()%16001)-8000)/1000.f;
                if(mode==1 && i%7==0) {a=specials[rng()%20];b=specials[rng()%20];}
                if(mode==2) {a=specials[rng()%20];b=specials[rng()%20];}
                if(mode==3) {uint32_t aa=rng(),bb=rng();memcpy(&a,&aa,4);memcpy(&b,&bb,4);}
                if(mode==4) {a=(i&1)?0.f:-0.f;b=(i&2)?1.f:-1.f;}
                if(mode==5) {a=float(int(rng()%33)-16)*1e-40f;b=float(int(rng()%33)-16)*1e-5f;}
                memcpy(gate.data()+G+i*4,&a,4);memcpy(up.data()+G+i*4,&b,4);
            }
            ck(cudaMemcpyAsync(dg,gate.data(),gate.size(),cudaMemcpyHostToDevice,stream));
            ck(cudaMemcpyAsync(du,up.data(),up.size(),cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));
        };
        fill(); auto*ctx=ggml_init({1024*1024,nullptr,true});
        auto*g=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,rows);g->data=dg+G;
        auto*u=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,rows);u->data=du+G;
        auto*glu=ggml_swiglu_split(ctx,g,u);glu->data=dt;
        auto reference=[&](void*out) {ggml_cuda_op_swiglu(cuda,glu);quantize_mmq_q8_1_cuda(dt,nullptr,out,GGML_TYPE_Q8_0,k,k,int64_t(k)*rows,int64_t(k)*rows,k,rows,1,1,stream);};
        auto fused=[&](void*out) {quantize_mmq_q8_1_swiglu_cuda((float*)(du+G),(float*)(dg+G),out,GGML_TYPE_Q8_0,k,k,int64_t(k)*rows,int64_t(k)*rows,k,rows,1,1,stream);};
        auto poison=[&] {ck(cudaMemsetAsync(dr,0xa5,r.size(),stream));ck(cudaMemsetAsync(df,0xa5,f.size(),stream));ck(cudaStreamSynchronize(stream));};
        for(bool capture:{false,true}) {
            poison();cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            if(capture)ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            reference(dr+G);fused(df+G);
            if(capture) {ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));}
            for(int replay=0;replay<(capture?3:1);++replay) {
                if(capture) {if(replay){fill();poison();}ck(cudaGraphLaunch(exec,stream));}
                ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(r.data(),dr,r.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(f.data(),df,f.size(),cudaMemcpyDeviceToHost));
                for(size_t i=0;i<r.size();++i)if(r[i]!=f[i] || ((i<G||i>=qbytes+G)&&r[i]!=0xa5)) {
                    fprintf(stderr,"mismatch k=%d rows=%d mode=%d graph=%d replay=%d byte=%zu ref=%u fused=%u\n",k,rows,mode,capture,replay,i,r[i],f[i]);return 1;
                }
                ck(cudaMemcpy(back.data(),dg,gate.size(),cudaMemcpyDeviceToHost));if(back!=gate)return 4;
                ck(cudaMemcpy(back.data(),du,up.size(),cudaMemcpyDeviceToHost));if(back!=up)return 4;
                ++cases;
            }
            if(capture){ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));}
        }
        if(k==12288&&rows==16&&mode==0) {
            // Both graphs write the very same destination. Flush lies outside events.
            cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
            for(int arm=0;arm<2;++arm) {ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(arm)fused(dr+G);else reference(dr+G);ck(cudaStreamEndCapture(stream,&graphs[arm]));resources(graphs[arm],arm?"fused":"reference",arm?1:2);ck(cudaGraphInstantiate(&execs[arm],graphs[arm],nullptr,nullptr,0));}
            unsigned*cache;size_t cn=72*1024*1024/4;ck(cudaMalloc(&cache,cn*4));ck(cudaMemsetAsync(cache,1,cn*4,stream));ck(cudaStreamSynchronize(stream));
            cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
            auto time=[&](int arm,bool cold) {std::vector<double> values;for(int z=0;z<9;++z) {
                if(cold) {flush_l2<<<1024,256,0,stream>>>(cache,cn);ck(cudaStreamSynchronize(stream));}
                ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(execs[arm],stream));ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));values.push_back(ms*1000.);
            }return med(values);};
            for(int z=0;z<128;++z)for(auto e:execs)ck(cudaGraphLaunch(e,stream));ck(cudaStreamSynchronize(stream));
            for(bool cold:{false,true})for(bool self:{true,false}) {
                std::vector<double> ratios;
                for(int q=0;q<8;++q) {int order[4]={0,self?0:1,self?0:1,0};if((q&1)&&!self){order[0]=order[3]=1;order[1]=order[2]=0;}
                    double sums[2]={0,0};for(int j=0;j<4;++j){double us=time(order[j],cold);int label=(q&1)?(j==0||j==3):(j==1||j==2);sums[label]+=us;}
                    double ratio=sums[0]/sums[1];ratios.push_back(ratio);printf("quartet cold=%d self=%d index=%d ref_us=%.5f candidate_us=%.5f ratio=%.8f\n",cold,self,q,sums[0]/2,sums[1]/2,ratio);
                }
                double logsum=0;int positive=0;for(double x:ratios){logsum+=log(x);positive+=x>1;}
                printf("timing cold=%d self=%d geomean=%.8f positive=%d/8\n",cold,self,exp(logsum/8),positive);
            }
            ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));ck(cudaFree(cache));for(int a=0;a<2;++a){ck(cudaGraphExecDestroy(execs[a]));ck(cudaGraphDestroy(graphs[a]));}
        }
        ggml_free(ctx);ck(cudaFree(dg));ck(cudaFree(du));ck(cudaFree(dt));ck(cudaFree(dr));ck(cudaFree(df));
    }
    ggml_backend_free(backend);printf("q8_glu16_gate_ok cases=%d bytes_exact=true guards=true inputs_unchanged=true changed_graph_replays=3\n",cases);return 0;
}
