// Exact old-vs-fused-CTA J128 test. Build for sm_121 using candidate ggml-cuda includes.
#include "mmq.cuh"
#include "j128_paired_glu_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s line %d\n",#x,cudaGetErrorString(e),__LINE__); std::exit(2); } } while(0)

template<bool OLD,bool ADD,bool GLU>
static void launch(const char *x,const int *y,float *dst,const float *res,float *dst_add,
        int rows,int nkb,int xs,int ycols,int stride,int cols) {
    constexpr auto kernel=OLD ? mul_mat_q_pq2_j128_reference<true,ADD,GLU> : mul_mat_q_pq2_j128<true,ADD,GLU>;
    constexpr size_t smem=ggml_cuda_pq2_j128_smem()+4096;
    static bool init=false;
    if(!init) {
        CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
        CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared));init=true;
    }
    const int ntx=(cols+127)/128;
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(2*(rows/128)*ntx);cfg.blockDim=dim3(32,8);
    cfg.dynamicSmemBytes=smem;cfg.attrs=&attr;cfg.numAttrs=1;
    CHECK(cudaLaunchKernelEx(&cfg,kernel,x,y,dst,(float*)nullptr,rows,nkb,xs,ycols,stride,cols,ntx,res,dst_add));
}
template<class F> static float measure(F f) {
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    for(int i=0;i<5;++i)f();std::vector<float> ts;
    for(int r=0;r<3;++r) {
        CHECK(cudaEventRecord(a));for(int i=0;i<20;++i)f();CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));
        float ms;CHECK(cudaEventElapsedTime(&ms,a,b));ts.push_back(ms*50);
    }
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));std::sort(ts.begin(),ts.end());return ts[1];
}
template<bool ADD,bool GLU> static void run_case(int rows,int k,int cols,int pad,bool alias,bool timing) {
    std::mt19937 rng(20261018+rows+cols+k+pad);
    const int nkb=k/128,xs=nkb+(pad?2:0),yc=(cols+127)/128*128,stride=(GLU?rows/2:rows)+pad;
    std::vector<char> hx(size_t(rows)*xs*sizeof(block_pq2_0));
    for(int row=0;row<rows;++row)for(int kb=0;kb<xs;++kb) {
        block_pq2_0 b{};b.d=__float2half(float(int(rng()%33)-16)*.03125f);
        for(int q=0;q<32;++q) { unsigned v=0;for(int t=0;t<4;++t)v|=(rng()%3)<<(2*t);b.qs[q]=v; }
        std::memcpy(hx.data()+(size_t(row)*xs+kb)*sizeof(b),&b,sizeof(b));
    }
    std::vector<int> hy(size_t(nkb)*yc*ggml_cuda_pq2_j128_ys);
    for(int kb=0;kb<nkb;++kb)for(int col=0;col<yc;++col) {
        int *p=hy.data()+(size_t(kb)*yc+col)*ggml_cuda_pq2_j128_ys;
        for(int s=0;s<4;++s) {float d=float(int(rng()%33)-16)*.015625f;std::memcpy(p+s,&d,4);}
        auto *q=reinterpret_cast<signed char*>(p+4);for(int i=0;i<128;++i)q[i]=int(rng()%255)-127;
    }
    const size_t n=size_t(stride)*cols;
    std::vector<float> hd(n),hr(n);for(size_t i=0;i<n;++i){hd[i]=.375f;hr[i]=float(int(rng()%2049)-1024)*.03125f;}
    char *x;int *y;float *a,*b,*ra,*rb;
    CHECK(cudaMalloc(&x,hx.size()));CHECK(cudaMalloc(&y,hy.size()*4));
    CHECK(cudaMalloc(&a,n*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMalloc(&ra,n*4));CHECK(cudaMalloc(&rb,n*4));
    CHECK(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(y,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a,hd.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,hd.data(),n*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ra,hr.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(rb,hr.data(),n*4,cudaMemcpyHostToDevice));
    auto old=[&](){launch<true,ADD,GLU>(x,y,a,ra,alias?ra:a,rows,nkb,xs,yc,stride,cols);};
    auto cur=[&](){launch<false,ADD,GLU>(x,y,b,rb,alias?rb:b,rows,nkb,xs,yc,stride,cols);};
    old();cur();CHECK(cudaDeviceSynchronize());
    std::vector<float> ha(n),hb(n);
    CHECK(cudaMemcpy(ha.data(),alias?ra:a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),alias?rb:b,n*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(ha.data(),hb.data(),n*4)) {
        for(size_t i=0;i<n;++i)if(std::memcmp(&ha[i],&hb[i],4)) {
            fprintf(stderr,"MISMATCH add=%d glu=%d alias=%d rows=%d k=%d cols=%d pad=%d i=%zu old=%a new=%a\n",ADD,GLU,alias,rows,k,cols,pad,i,ha[i],hb[i]);break;
        }std::exit(3);
    }
    if(timing) {float ot=measure(old),nt=measure(cur);printf("J128 add=%d glu=%d rows=%d k=%d cols=%d old_us=%.4f new_us=%.4f gain=%.6f\n",ADD,GLU,rows,k,cols,ot,nt,ot/nt);}
    CHECK(cudaFree(x));CHECK(cudaFree(y));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(ra));CHECK(cudaFree(rb));
}
int main(){
    int cases=0;
    for(int rows:{128,256,512,2048})for(int k:{256,2048,4096})for(int cols:{1,127,128,129,257,512})for(int pad:{0,32}) {
        run_case<false,false>(rows,k,cols,pad,false,false);
        run_case<true,false>(rows,k,cols,pad,false,false);
        run_case<true,false>(rows,k,cols,pad,true,false);
        run_case<false,true>(rows,k,cols,pad,false,false);cases+=4;
    }
    for(int rows:{2048,4096,12288,24576}) {
        run_case<false,false>(rows,4096,512,0,false,true);
        run_case<true,false>(rows,4096,512,0,false,true);
        run_case<false,true>(rows,4096,512,0,false,true);cases+=3;
    }
    printf("J128_PAIRED_GLU_PASS cases=%d exact_output_and_padding_bits=true\n",cases);
}
