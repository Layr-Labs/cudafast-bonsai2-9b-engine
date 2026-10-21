// Frozen promoted203 scalar kernel versus production vector dispatcher.
#include "unary.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(e),__LINE__);exit(2);}}while(0)
template <float (*op)(float), typename T>
static __global__ void reference213_unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);
}

// Focused single-device fixture: use the active CUDA device and abort on errors.
int ggml_cuda_get_device() { int id = -1; auto e = cudaGetDevice(&id); if (e != cudaSuccess) std::abort(); return id; }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
 fprintf(stderr, "%s: %s (%s %s:%d)\n", stmt, msg, func, file, line); std::abort();
}
static cudaStream_t stream;
static void launch(bool old,const float*x,const float*g,float*d,int64_t k,int64_t n,int64_t o0,int64_t o1){
 if(old)ggml_cuda_kernel_launch(reference213_unary_gated_op_kernel<op_silu,float>,
    ggml_cuda_kernel_launch_params(dim3((k+255)/256),256,0,stream),x,g,d,k,n,o0,o1);
 else unary_gated_cuda<op_silu,float>(x,g,d,k,n,o0,o1,stream);
 CK(cudaGetLastError());
}
static cudaGraphExec_t capture(bool old,const float*x,const float*g,float*d,int64_t k,int64_t n,int64_t o0,int64_t o1){
 cudaGraph_t gph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 for(int i=0;i<64;++i)launch(old,x,g,d,k,n,o0,o1);
 CK(cudaStreamEndCapture(stream,&gph));CK(cudaGraphInstantiate(&exec,gph,nullptr,nullptr,0));CK(cudaGraphDestroy(gph));return exec;
}
static float timing(cudaGraphExec_t exec){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int j=0;j<8;++j)CK(cudaGraphLaunch(exec,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/512;}
static void check(int n,int cols,int pad0,int pad1,int off,bool swap,int alias,int tail,std::mt19937 &rng){
 const int k=n*cols-tail,o0=n+pad0,o1=n+pad1;const size_t count=size_t(n+4)*cols+128;
 std::vector<float> hx(count),hg(count);std::vector<uint32_t> hd(count,0x4f314159u);
 const uint32_t edges[]={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00012u,0x7f800001u,0xff800001u};
 for(size_t i=0;i<count;++i){hx[i]=float(int(rng()%2049)-1024)/32;hg[i]=float(int(rng()%2049)-1024)/64;if(i%11==0){auto u=edges[rng()%14];memcpy(&hx[i],&u,4);}if(i%13==0){auto u=edges[rng()%14];memcpy(&hg[i],&u,4);}}
 float *x0,*x1,*g0,*g1,*d0,*d1;
 CK(cudaMalloc(&x0,count*4));CK(cudaMalloc(&x1,count*4));CK(cudaMalloc(&g0,count*4));CK(cudaMalloc(&g1,count*4));CK(cudaMalloc(&d0,count*4));CK(cudaMalloc(&d1,count*4));
 for(auto x:{x0,x1})CK(cudaMemcpy(x,hx.data(),count*4,cudaMemcpyHostToDevice));for(auto g:{g0,g1})CK(cudaMemcpy(g,hg.data(),count*4,cudaMemcpyHostToDevice));for(auto d:{d0,d1})CK(cudaMemcpy(d,hd.data(),count*4,cudaMemcpyHostToDevice));
 float *a=alias==1?x0:alias==2?g0:d0,*b=alias==1?x1:alias==2?g1:d1;
 launch(true,(swap?g0:x0)+32+off,(swap?x0:g0)+32+off,a+32+off,k,n,swap?o1:o0,swap?o0:o1);
 launch(false,(swap?g1:x1)+32+off,(swap?x1:g1)+32+off,b+32+off,k,n,swap?o1:o0,swap?o0:o1);
 CK(cudaStreamSynchronize(stream));std::vector<uint32_t> ha(count),hb(count);CK(cudaMemcpy(ha.data(),a,count*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb.data(),b,count*4,cudaMemcpyDeviceToHost));
 if(ha!=hb){size_t i=0;while(i<count&&ha[i]==hb[i])++i;fprintf(stderr,"BIT_FAIL n%d cols%d pads%d/%d off%d swap%d alias%d tail%d idx%zu old%08x new%08x\n",n,cols,pad0,pad1,off,swap,alias,tail,i,ha[i],hb[i]);exit(1);}
 if(pad0==0&&pad1==0&&off==0&&!swap&&alias==0&&tail==0&&(cols==16||cols==512||n==12288||n==8192||n==16384||n==131068||n==131072||n==262144||n==262148)){
  auto ga=capture(true,x0+32,g0+32,d0+32,k,n,n,n),gb=capture(false,x1+32,g1+32,d1+32,k,n,n,n);
  for(int rep=0;rep<10;++rep){float old,newer;if(rep%2==0){old=timing(ga);newer=timing(gb);}else{newer=timing(gb);old=timing(ga);}if(rep>=2)printf("GRAPH n=%d cols=%d round=%d old_us=%.5f new_us=%.5f gain=%.6f\n",n,cols,rep,old,newer,old/newer);}
  CK(cudaMemcpy(ha.data(),d0,count*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb.data(),d1,count*4,cudaMemcpyDeviceToHost));if(ha!=hb){fprintf(stderr,"GRAPH_BIT_FAIL\n");exit(1);}CK(cudaGraphExecDestroy(ga));CK(cudaGraphExecDestroy(gb));
 }
 for(auto p:{x0,x1,g0,g1,d0,d1})CK(cudaFree(p));
}
int main(){CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::mt19937 rng(20261019);int cases=0;
 for(int n:{4,128,1024,12288})for(int cols:{1,7,9,16})for(int pad0:{0,4})for(int pad1:{0,4})for(int off:{0,1})for(bool swap:{false,true}){check(n,cols,pad0,pad1,off,swap,0,0,rng);++cases;}
 for(int n:{4,128,1024,12288})for(int cols:{1,7,9,16})for(int alias:{1,2})for(bool swap:{false,true}){check(n,cols,0,0,0,swap,alias,0,rng);++cases;}
 for(int n:{4,128,1024,12288})for(int tail:{1,2,3}){check(n,7,0,0,0,false,0,tail,rng);++cases;}
 for(int n:{8192,16384,131068,131072,262144,262148}){check(n,1,0,0,0,false,0,0,rng);++cases;}
 check(8188,1,0,0,0,false,0,0,rng);++cases;
 check(12288,512,0,0,0,false,0,0,rng);++cases;
 CK(cudaStreamDestroy(stream));printf("SWIGLU_VECTOR_EQUIVALENCE_PASS cases=%d all_output_padding_bits=true\n",cases);}
