#pragma once
// Independent full-shared-memory stage oracle: no optimized helper or owner map.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static int reorder_seen_new=0,reorder_seen_old=0;
static bool reorder_test_enabled() {const char *v=getenv("GGML_CUDA_FWHT_REORDER");return !v || atoi(v)!=0;}
template<typename T>
__global__ void fwht_reorder_stage_oracle(const T *x,const float *signs,float *out,bool reordered) {
    __shared__ float a[4096];const int row=blockIdx.x,tid=threadIdx.x;
    for(int i=tid;i<4096;i+=256){float value=float(x[row*4096+i])*(1.0f/64.0f);if(signs)value*=signs[i];a[i]=value;}
    __syncthreads();
    for(int stage=0;stage<12;++stage){
        const int bit=reordered?(stage<4?stage+8:stage-4):stage,h=1<<bit;
        for(int pair=tid;pair<2048;pair+=256){const int lo=(pair/h)*2*h+pair%h;const float x=a[lo],y=a[lo+h];a[lo]=x+y;a[lo+h]=x-y;}
        __syncthreads();
    }
    for(int i=tid;i<4096;i+=256)out[row*4096+i]=a[i];
}
bool fwht_reorder_test_signed(ggml_backend_cuda_context &ctx,const ggml_tensor *src,const ggml_tensor *signs,ggml_tensor *dst) {
    GGML_ASSERT((src->type==GGML_TYPE_F32 || src->type==GGML_TYPE_F16) && dst->type==GGML_TYPE_F32 && dst->ne[0]==4096);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int rows=ggml_nelements(dst)/4096;
    const bool reordered=reorder_test_enabled() && rows>=128 && ggml_cuda_info().devices[ggml_cuda_get_device()].cc==GGML_CUDA_CC_DGX_SPARK;
    if(src->type==GGML_TYPE_F32)fwht_reorder_stage_oracle<float><<<rows,256,0,ctx.stream()>>>((const float *)src->data,signs?(const float *)signs->data:nullptr,(float *)dst->data,reordered);
    else fwht_reorder_stage_oracle<half><<<rows,256,0,ctx.stream()>>>((const half *)src->data,signs?(const float *)signs->data:nullptr,(float *)dst->data,reordered);
    return true;
}
bool fwht_reorder_test_plain(ggml_backend_cuda_context &ctx,const ggml_tensor *src,ggml_tensor *dst) {return fwht_reorder_test_signed(ctx,src,nullptr,dst);}
static cudaError_t reorder_capture_end(cudaStream_t stream,cudaGraph_t *graph) {
    auto result=cudaStreamEndCapture(stream,graph);if(result!=cudaSuccess)return result;
    size_t n=0;cudaGraphGetNodes(*graph,nullptr,&n);std::vector<cudaGraphNode_t> nodes(n);cudaGraphGetNodes(*graph,nodes.data(),&n);
    for(auto node:nodes){cudaGraphNodeType type;cudaGraphNodeGetType(node,&type);if(type!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams p{};cudaGraphKernelNodeGetParams(node,&p);const char *name=nullptr;cudaFuncGetName(&name,p.func);
        if(!name || !strstr(name,"fwht_quantize_mmq_d4"))continue;
        const bool changed=strstr(name,"_reordered")!=nullptr;
        const bool fixed=strstr(name,"_rms") || strstr(name,"_gnorm");
        const int64_t width=fixed?4096:*reinterpret_cast<const int64_t *>(p.kernelParams[4]);
        const bool expected=reorder_test_enabled() && width==4096 && p.gridDim.y>=128 && ggml_cuda_info().devices[ggml_cuda_get_device()].cc==GGML_CUDA_CC_DGX_SPARK;
        if(changed!=expected){fprintf(stderr,"ROUTE FAIL %s rows=%u expected=%d\n",name,p.gridDim.y,expected);exit(5);}
        if(changed){++reorder_seen_new;cudaFuncAttributes attr{};cudaFuncGetAttributes(&attr,p.func);int blocks=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,p.func,256,p.sharedMemBytes);if(attr.localSizeBytes || attr.numRegs>64 || blocks<4){fprintf(stderr,"RESOURCE FAIL regs=%d local=%zu blocks=%d\n",attr.numRegs,attr.localSizeBytes,blocks);exit(6);}}
        else ++reorder_seen_old;
    }
    return result;
}
#define ggml_cuda_op_fwht_signed fwht_reorder_test_signed
#define ggml_cuda_op_fwht fwht_reorder_test_plain
#define cudaStreamEndCapture reorder_capture_end
