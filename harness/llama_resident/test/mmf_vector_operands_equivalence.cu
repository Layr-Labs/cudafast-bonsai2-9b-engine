// Frozen promoted241 paired kernel versus vectorized synchronous operands.
// Full output and padding bits, including cluster8 ordered reduction and fallback guards.
#include "mmf.cuh"
#include "mmf_vector_operands_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
int ggml_cuda_get_device() { int d=0; cudaGetDevice(&d); return d; }
[[noreturn]] void ggml_cuda_error(const char*st,const char*fn,const char*file,int line,const char*msg) { fprintf(stderr,"%s %s %s:%d %s\n",st,fn,file,line,msg); abort(); }
#define CK(x) do { cudaError_t ce=(x); if(ce!=cudaSuccess) { fprintf(stderr,"line%d %s: %s\n",__LINE__,#x,cudaGetErrorString(ce)); exit(2); } } while(0)
static cudaStream_t stream;

__global__ void native_fragment_bits(const float *x, const float *y, float *old, float *direct, int stride) {
    __shared__ float scratch[16*36];
    tile<16,8,float> A[4]; tile<8,8,float> B[4];
    for(int i=0;i<16;++i) scratch[i*36+threadIdx.x]=x[i*stride+threadIdx.x];
    for(int k=0;k<4;++k) load_ldmatrix(A[k],scratch+8*k,36);
    for(int i=0;i<8;++i) scratch[i*36+threadIdx.x]=y[i*stride+threadIdx.x];
    for(int k=0;k<4;++k) load_ldmatrix(B[k],scratch+8*k,36);
    for(int k=0;k<4;++k) {
        for(int l=0;l<4;++l) {
            old[(k*4+l)*32+threadIdx.x]=A[k].x[l];
            direct[(k*4+l)*32+threadIdx.x]=x[(threadIdx.x/4+8*(l%2))*stride+8*k+threadIdx.x%4+4*(l/2)];
        }
        for(int l=0;l<2;++l) {
            old[(16+k*2+l)*32+threadIdx.x]=B[k].x[l];
            direct[(16+k*2+l)*32+threadIdx.x]=y[B[k].get_i(l)*stride+8*k+B[k].get_j(l)];
        }
    }
}

static void same(const void *a,const void *b,size_t bytes,const char *tag,int k,int rows,int pad,int mode) {
    std::vector<unsigned char> x(bytes),y(bytes); CK(cudaMemcpy(x.data(),a,bytes,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(),b,bytes,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<bytes;++i) if(x[i]!=y[i]) { fprintf(stderr,"FAIL %s byte=%zu old=%02x new=%02x k=%d rows=%d pad=%d mode=%d\n",tag,i,x[i],y[i],k,rows,pad,mode); exit(3); }
}
template<bool OLD> static void launch(const float*x0,const float*x1,const float*y,float*d0,float*d1,int k,int rows,int pad,int offset) {
    cudaLaunchAttribute attrs[2] = {};
    attrs[0].id=cudaLaunchAttributeClusterDimension; attrs[0].val.clusterDim={8,1,1};
    attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization; attrs[1].val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t cfg = {}; cfg.gridDim=dim3(2*(rows/32)*2*2*8); cfg.blockDim=dim3(32); cfg.dynamicSmemBytes=16*36*sizeof(float); cfg.stream=stream; cfg.attrs=attrs; cfg.numAttrs=2;
    auto kernel=OLD?reference247_mul_mat_f_ksplit_pair:mul_mat_f_ksplit_pair;
    CK(cudaLaunchKernelEx(&cfg,kernel,x0+offset,x1+offset,y,d0,d1,k,rows/32,k+pad,k+pad,k+pad,rows+pad,rows+pad));
}
static float elapsed(cudaGraphExec_t exec) {
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a,stream));
    for(int r=0;r<16;++r)CK(cudaGraphLaunch(exec,stream));
    CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); return ms*1000/(16*32);
}
int main(int argc,char **argv) {
    bool optimized_only=argc>1 && std::string(argv[1])=="--race-optimized-only";
    bool sanitizer=optimized_only || (argc>1 && std::string(argv[1])=="--sanitizer");
    CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr int K=5120,R=128,P=64; size_t xn=size_t(K+P)*R+1,yn=size_t(K+P)*16,dn=size_t(R+P)*16;
    float *x0,*x1,*y,*d[2][2],*frag[2]; CK(cudaMalloc(&x0,xn*4));CK(cudaMalloc(&x1,xn*4));CK(cudaMalloc(&y,yn*4));
    for(auto &a:d)for(float *&b:a)CK(cudaMalloc(&b,dn*4));for(float *&a:frag)CK(cudaMalloc(&a,24*32*4));
    std::vector<float> hx0(xn),hx1(xn),hy(yn); std::mt19937 mt(233); int cases=0;
    if(argc>1 && std::string(argv[1])=="--mapping") {
        for(size_t i=0;i<xn;++i){uint32_t u=uint32_t(i+1);memcpy(&hx0[i],&u,4);}for(size_t i=0;i<yn;++i){uint32_t u=uint32_t(i+1);memcpy(&hy[i],&u,4);}
        CK(cudaMemcpyAsync(x0,hx0.data(),xn*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(y,hy.data(),yn*4,cudaMemcpyHostToDevice,stream));
        native_fragment_bits<<<1,32,0,stream>>>(x0,y,frag[0],frag[1],64);CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));std::vector<uint32_t> bits(24*32);CK(cudaMemcpy(bits.data(),frag[0],bits.size()*4,cudaMemcpyDeviceToHost));
        for(int k=0;k<4;++k)for(int lane=0;lane<32;++lane){printf("MAP k=%d lane=%d A=",k,lane);for(int l=0;l<4;++l)printf("%u,",bits[(k*4+l)*32+lane]-1);printf(" B=");for(int l=0;l<2;++l)printf("%u,",bits[(16+k*2+l)*32+lane]-1);printf("\n");}return 0;
    }
    for(int mode=0;mode<6;++mode) {
        auto value=[&](size_t i) { float v=float(int(mt()%20001)-10000)/2048.f;
            if(mode==1)v=(i&1)?-0.f:0.f;
            if(mode==2)v=std::ldexp(float(int(mt()%31)-15),int(mt()%50)-25);
            if(mode==3){uint32_t u[]={0x7fc01234u,0xffc02345u,0x7f800000u,0xff800000u,0x00000001u,0x80000001u,0x00800000u,0x80800000u};memcpy(&v,&u[i%8],4);}
            if(mode==4)v=(i%7==0)?65536.f:(i%7==1)?-65536.f:1e-9f;
            if(mode==5){uint32_t u=mt();memcpy(&v,&u,4);}return v;};
        for(size_t i=0;i<xn;++i){hx0[i]=value(i);hx1[i]=value(i+3);}for(size_t i=0;i<yn;++i)hy[i]=value(i+5);
        CK(cudaMemcpyAsync(x0,hx0.data(),xn*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(x1,hx1.data(),xn*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(y,hy.data(),yn*4,cudaMemcpyHostToDevice,stream));
        if (!optimized_only) for(int stride:{32,33,64}) { native_fragment_bits<<<1,32,0,stream>>>(x0,y,frag[0],frag[1],stride);CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));same(frag[0],frag[1],24*32*4,"native-fragment-bits",32,16,stride-32,mode);++cases; }
        for(int k:{256,512,3072,4096,5120})for(int rows:{32,64,128})for(int pad:{0,1,64})for(int offset:{0,1})for(int graph:{0,1}) {
            if (optimized_only && (pad==1 || offset!=0)) continue;
            size_t bytes=size_t(rows+pad)*16*4;
            for(auto &a:d)for(float*b:a)CK(cudaMemsetAsync(b,0xa5,bytes,stream));
            cudaGraph_t gr;cudaGraphExec_t exec;
            if(graph)CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            if (!optimized_only) launch<true>(x0,x1,y,d[0][0],d[0][1],k,rows,pad,offset);launch<false>(x0,x1,y,d[1][0],d[1][1],k,rows,pad,offset);
            if(graph){CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&exec,gr,nullptr,nullptr,0));CK(cudaGraphLaunch(exec,stream));}
            CK(cudaStreamSynchronize(stream));if(!optimized_only){same(d[0][0],d[1][0],bytes,"gate0-full-output",k,rows,pad,mode);same(d[0][1],d[1][1],bytes,"gate1-full-output",k,rows,pad,mode);}++cases;
            if(graph){CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(gr));}
        }
    }
    if(!sanitizer) {
        for(size_t i=0;i<xn;++i){hx0[i]=float(int(mt()%20001)-10000)/2048.f;hx1[i]=float(int(mt()%20001)-10000)/2048.f;}for(size_t i=0;i<yn;++i)hy[i]=float(int(mt()%20001)-10000)/2048.f;
        CK(cudaMemcpyAsync(x0,hx0.data(),xn*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(x1,hx1.data(),xn*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(y,hy.data(),yn*4,cudaMemcpyHostToDevice,stream));
        for(int k:{3072,4096,5120})for(int rows:{32,128}) {
            cudaGraphExec_t exec[2];for(int a=0;a<2;++a){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<32;++i){if(a)launch<false>(x0,x1,y,d[a][0],d[a][1],k,rows,0,0);else launch<true>(x0,x1,y,d[a][0],d[a][1],k,rows,0,0);}CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&exec[a],gr,nullptr,nullptr,0));CK(cudaGraphDestroy(gr));}
            for(int rep=0;rep<10;++rep){float a,b;if(rep&1){b=elapsed(exec[1]);a=elapsed(exec[0]);}else{a=elapsed(exec[0]);b=elapsed(exec[1]);}if(rep>=2)printf("GRAPH k=%d rows=%d rep=%d old_us=%.6f new_us=%.6f gain=%.6f\n",k,rows,rep,a,b,a/b);}
            for(auto e:exec)CK(cudaGraphExecDestroy(e));
        }
    }
    if(optimized_only) printf("MMF_VECTOR_OPERANDS_OPTIMIZED_SANITIZER_COMPLETE cases=%d comparison_not_run=true\n",cases);
    else printf("MMF_VECTOR_OPERANDS_STAGES_PASS cases=%d fragment_bits_and_both_full_outputs=true\n",cases);
    for(auto &a:d)for(float*b:a)CK(cudaFree(b));for(float*a:frag)CK(cudaFree(a));CK(cudaFree(x0));CK(cudaFree(x1));CK(cudaFree(y));CK(cudaStreamDestroy(stream));
}
