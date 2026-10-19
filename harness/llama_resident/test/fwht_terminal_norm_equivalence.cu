#include "../../../llama.cpp/ggml/src/ggml-cuda/quantize.cu"
#include "fwht_terminal_norm_reference.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>
int ggml_cuda_get_device(){int d=0;cudaGetDevice(&d);return d;}
[[noreturn]]void ggml_cuda_error(const char*s,const char*f,const char*p,int l,const char*m){fprintf(stderr,"%s %s %s:%d %s\n",s,f,p,l,m);abort();}
#define CK(x) do{cudaError_t ce=(x);if(ce!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(ce));exit(2);}}while(0)
static cudaStream_t stream;
static void same(const void*a,const void*b,size_t n,const char*tag){std::vector<unsigned char>x(n),y(n);CK(cudaMemcpy(x.data(),a,n,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),b,n,cudaMemcpyDeviceToHost));for(size_t i=0;i<n;++i)if(x[i]!=y[i]){fprintf(stderr,"FAIL %s byte=%zu old=%02x new=%02x\n",tag,i,x[i],y[i]);exit(3);}}
template<bool OLD,bool SIGN>static void launch(int kind,const float*x,const float*z,const float*w,const float*s,float*norm,block_q8_1_mmq*y,int rows,float eps,int layout){
 if(kind==0){auto k=OLD?frozen227_rms<SIGN>:fwht_quantize_mmq_d4_rms<SIGN>;k<<<dim3(1,rows),256,0,stream>>>(x,4128,w,4096,eps,s,norm,y,rows);}
 else{auto k=OLD?frozen227_gnorm<SIGN>:fwht_quantize_mmq_d4_gnorm<SIGN>;uint3 fd=init_fastdiv_values(layout?8:32);k<<<dim3(1,rows),256,0,stream>>>(x,z,136,w,128,eps,fd,1,layout?8:32,32,s,y,rows);}
 CK(cudaGetLastError());
}
template<bool OLD>static void run(int kind,int sign,const float*x,const float*z,const float*w,const float*s,float*norm,block_q8_1_mmq*y,int rows,float eps,int layout){if(sign)launch<OLD,true>(kind,x,z,w,s,norm,y,rows,eps,layout);else launch<OLD,false>(kind,x,z,w,s,norm,y,rows,eps,layout);}
static float elapsed(cudaGraphExec_t g){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int i=0;i<8;++i)CK(cudaGraphLaunch(g,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/(8*32);}
int main(int argc,char**argv){
 bool safety=argc>1&&std::string(argv[1])=="--sanitizer";constexpr int R=512;size_t n=R*32*136;size_t qn=32*R*sizeof(block_q8_1_mmq);float*x,*z,*w,*s,*norm[2];block_q8_1_mmq*y[2];CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CK(cudaMalloc(&x,n*4));CK(cudaMalloc(&z,n*4));CK(cudaMalloc(&w,4096*4));CK(cudaMalloc(&s,4096*4));for(int a=0;a<2;++a){CK(cudaMalloc(&norm[a],R*4096*4));CK(cudaMalloc(&y[a],qn));}
 std::mt19937 rng(222);std::vector<float>hx(n),hz(n),hw(4096),hs(4096);int cases=0;
 for(int mode=0;mode<6;++mode){for(size_t i=0;i<n;++i){float v=float(int(rng()%20001)-10000)/2048.f;if(mode==1)v=0.f;if(mode==2)v=(i&1)?-0.f:0.f;if(mode==3)v=(i%9==0)?65536.f:1e-12f;if(mode==4)v=std::ldexp(float(int(rng()%31)-15),int(rng()%80)-40);if(mode==5){uint32_t b=(i%7==0)?0x7fc01234u:(i%7==1)?0x7f800000u:(i%7==2)?0xff800000u:0u;memcpy(&v,&b,4);}hx[i]=v;hz[i]=float(int(rng()%2001)-1000)/512.f;}
 for(int i=0;i<4096;++i){hw[i]=float(int(rng()%1001)-500)/512.f;hs[i]=(rng()&1)?1.f:-1.f;}
 CK(cudaMemcpy(x,hx.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(z,hz.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(w,hw.data(),4096*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s,hs.data(),4096*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());
 for(int rows:{1,16,128,512})for(int kind:{0,1})for(int sign:{0,1})for(float eps:{0.f,1e-6f,1e-3f})for(int option:{0,1}){
 size_t qb=32*size_t(rows)*sizeof(block_q8_1_mmq),nb=size_t(rows)*4096*4;for(int a=0;a<2;++a){CK(cudaMemsetAsync(y[a],0xa5,qb,stream));CK(cudaMemsetAsync(norm[a],0x5a,nb,stream));}
 run<true>(kind,sign,x,z,w,s,option?norm[0]:nullptr,y[0],rows,eps,option);run<false>(kind,sign,x,z,w,s,option?norm[1]:nullptr,y[1],rows,eps,option);CK(cudaStreamSynchronize(stream));same(y[0],y[1],qb,"Q8-full-allocation");same(norm[0],norm[1],nb,"RMS-optional-output");++cases;}
 }
 printf("FWHT_TERMINAL_NORM_EXACT_PASS cases=%d all_output_bits=true\n",cases);fflush(stdout);if(safety)return 0;
 for(size_t i=0;i<n;++i)hx[i]=float(int(rng()%20001)-10000)/2048.f;CK(cudaMemcpy(x,hx.data(),n*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());
 for(int rows:{16,128,512})for(int kind:{0,1}){
 cudaGraphExec_t exec[2];for(int a=0;a<2;++a){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<32;++i){if(a)run<false>(kind,1,x,z,w,s,kind?nullptr:norm[a],y[a],rows,1e-6f,1);else run<true>(kind,1,x,z,w,s,kind?nullptr:norm[a],y[a],rows,1e-6f,1);}CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&exec[a],gr,nullptr,nullptr,0));CK(cudaGraphDestroy(gr));}
 for(int rep=0;rep<10;++rep){float a,b;if(rep&1){b=elapsed(exec[1]);a=elapsed(exec[0]);}else{a=elapsed(exec[0]);b=elapsed(exec[1]);}if(rep>=2)printf("GRAPH kind=%d rows=%d rep=%d old_us=%.6f new_us=%.6f gain=%.6f\n",kind,rows,rep,a,b,a/b);}
 for(auto e:exec)CK(cudaGraphExecDestroy(e));}
 puts("FWHT_TERMINAL_NORM_PASS");
}
