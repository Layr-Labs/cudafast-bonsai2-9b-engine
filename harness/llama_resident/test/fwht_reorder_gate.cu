// Reordered scalar F32 FWHT: diagnostic error is separate from ranked replay.
#include "common.cuh"
#include "quantize.cuh"
#include "fwht.cuh"
#include "fwht-reorder.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
template<bool TENSOR>
__global__ __launch_bounds__(256,2) void run(const float*x,const float*signs,float*out,block_q8_1_mmq*q,int k,int rows) {
    constexpr int N=4096,NT=256;__shared__ float shared[N];float v[N/NT];
    const int tid=threadIdx.x,lane=tid%32,base=blockIdx.x*N,row=blockIdx.y;
#pragma unroll
    for(int i=0;i<N/NT;++i)v[i]=(x[row*k+base+i*NT+tid]*(1.0f/64.0f))*signs[base+i*NT+tid];
    if constexpr(TENSOR)fwht_reorder::transform(v,shared,tid);
    else ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
#pragma unroll
    for(int i=0;i<N/NT;++i){
        const int col=base+(TENSOR?fwht_reorder::column(i,tid):i*NT+tid);
        if(out)out[row*k+col]=v[i];
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float inv=127.0f/amax;const int8_t z=roundf(v[i]*inv);
        auto &dst=q[(col/QK8_1_MMQ)*rows+row];dst.qs[col%QK8_1_MMQ]=z;
        if(lane==0)dst.d4[(col%QK8_1_MMQ)/32]=1.0f/inv;
    }
}
static float timed(cudaGraphExec_t graph,cudaEvent_t a,cudaEvent_t b,cudaStream_t stream){ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(graph,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));return ms*1000/20;}
int main(int argc,char**){
    if(argc>1){
        struct Item{const char*name;cudaDeviceAttr attr;};
        const Item items[]={
            {"cluster_launch",cudaDevAttrClusterLaunch},{"SM_count",cudaDevAttrMultiProcessorCount},
            {"L2_bytes",cudaDevAttrL2CacheSize},{"shared_optin",cudaDevAttrMaxSharedMemoryPerBlockOptin},
            {"threads_per_SM",cudaDevAttrMaxThreadsPerMultiProcessor},{"blocks_per_SM",cudaDevAttrMaxBlocksPerMultiprocessor}};
        for(auto item:items){int value=-1;auto e=cudaDeviceGetAttribute(&value,item.attr,0);printf("FEATURE %s value=%d status=%s\n",item.name,value,cudaGetErrorString(e));}
        return 0;
    }
    std::mt19937 rng(20261019);std::uniform_real_distribution<float> dist(-8,8);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int cases=0;double worst=0;
    for(int k:{4096,12288})for(int rows:{16,512}) {
        const size_t count=size_t(k)*rows,qbytes=count/QK8_1_MMQ*sizeof(block_q8_1_mmq);
        std::vector<float>x(count),signs(k),ref(count),got(count);float *dx,*ds,*dr,*dg;block_q8_1_mmq*dq;
        ck(cudaMalloc(&dx,count*4));ck(cudaMalloc(&ds,k*4));ck(cudaMalloc(&dr,count*4));ck(cudaMalloc(&dg,count*4));ck(cudaMalloc(&dq,qbytes+64));
        for(int i=0;i<k;++i)signs[i]=(rng()&1)?1:-1;ck(cudaMemcpy(ds,signs.data(),k*4,cudaMemcpyHostToDevice));
        for(int mode=0;mode<6;++mode){
            for(size_t i=0;i<count;++i)x[i]=mode==0?0.0f:mode==1?((i&1)?128.0f:-128.0f):mode==2?dist(rng)*1024:mode==3?dist(rng)*1e-6f:dist(rng);
            ck(cudaMemcpy(dx,x.data(),count*4,cudaMemcpyHostToDevice));ck(cudaMemset(dq,0xa5,qbytes+64));
            run<false><<<dim3(k/4096,rows),256,0,stream>>>(dx,ds,dr,dq,k,rows);run<true><<<dim3(k/4096,rows),256,0,stream>>>(dx,ds,dg,dq,k,rows);ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(ref.data(),dr,count*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(got.data(),dg,count*4,cudaMemcpyDeviceToHost));
            double mx=0,err=0,l2=0,err2=0;for(size_t i=0;i<count;++i){if(!std::isfinite(got[i]))return 3;mx=std::max(mx,double(std::abs(ref[i])));err=std::max(err,double(std::abs(ref[i]-got[i])));l2+=double(ref[i])*ref[i];err2+=double(ref[i]-got[i])*(ref[i]-got[i]);}
            double relative=err/(1+mx);worst=std::max(worst,relative);if(relative>2e-5 || (l2>0&&sqrt(err2/l2)>2e-5)){fprintf(stderr,"numeric fail %d %d %d %.9g\n",k,rows,mode,relative);return 1;}
            unsigned char guard[64];ck(cudaMemcpy(guard,reinterpret_cast<char*>(dq)+qbytes,64,cudaMemcpyDeviceToHost));for(auto c:guard)if(c!=0xa5)return 4;
            printf("CASE k=%d rows=%d mode=%d scaled_max=%.9g l2=%.9g\n",k,rows,mode,relative,l2?sqrt(err2/l2):0);++cases;
        }
        cudaGraph_t graphs[2];cudaGraphExec_t exec[2];
        for(int which=0;which<2;++which){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int r=0;r<20;++r){if(which)run<true><<<dim3(k/4096,rows),256,0,stream>>>(dx,ds,nullptr,dq,k,rows);else run<false><<<dim3(k/4096,rows),256,0,stream>>>(dx,ds,nullptr,dq,k,rows);}ck(cudaStreamEndCapture(stream,&graphs[which]));ck(cudaGraphInstantiate(&exec[which],graphs[which],nullptr,nullptr,0));}
        cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
        for(int r=0;r<9;++r){float o0=timed(exec[0],a,b,stream),n0=timed(exec[1],a,b,stream),n1=timed(exec[1],a,b,stream),o1=timed(exec[0],a,b,stream);if(r>1)printf("TIMING k=%d rows=%d rep=%d old_us=%.5f new_us=%.5f ratio=%.6f\n",k,rows,r,(o0+o1)/2,(n0+n1)/2,(o0+o1)/(n0+n1));}
        for(int w=0;w<2;++w){ck(cudaGraphExecDestroy(exec[w]));ck(cudaGraphDestroy(graphs[w]));}ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaFree(dx));ck(cudaFree(ds));ck(cudaFree(dr));ck(cudaFree(dg));ck(cudaFree(dq));
    }
    printf("FWHT_REORDER_PASS cases=%d worst_scaled_max=%.9g numerical_not_bitexact=true\n",cases,worst);return 0;
}
