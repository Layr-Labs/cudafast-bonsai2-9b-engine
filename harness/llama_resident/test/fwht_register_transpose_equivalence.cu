#include "../../../llama.cpp/ggml/src/ggml-cuda/quantize.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include "fwht_register_transpose_reference.cuh"
int ggml_cuda_get_device(){int d=0;cudaGetDevice(&d);return d;}
[[noreturn]] void ggml_cuda_error(const char*st,const char*fn,const char*file,int line,const char*msg){fprintf(stderr,"%s %s %s:%d %s\n",st,fn,file,line,msg);abort();}
#define CK(x) do{cudaError_t ce=(x);if(ce!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(ce));exit(2);}}while(0)
static cudaStream_t stream;
template<bool OLD> __global__ void raw_transform(const float*x,float*y){
 __shared__ float scratch[4096]; float v[16];int t=threadIdx.x,l=t%32;int base=4096*blockIdx.x;
 #pragma unroll
 for(int i=0;i<16;++i)v[i]=x[base+256*i+t];
 if constexpr(OLD)frozen221_butterfly<4096,256>(v,scratch,t,l);
 else ggml_cuda_fwht_register_transpose_butterfly<4096,256>(v,scratch,t,l);
 #pragma unroll
 for(int i=0;i<16;++i)y[base+256*i+t]=v[i];
}
static void same(const void*a,const void*b,size_t n,const char*tag){std::vector<unsigned char>x(n),y(n);CK(cudaMemcpy(x.data(),a,n,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),b,n,cudaMemcpyDeviceToHost));for(size_t i=0;i<n;++i)if(x[i]!=y[i]){fprintf(stderr,"FAIL %s byte=%zu old=%02x new=%02x\n",tag,i,x[i],y[i]);exit(3);}}
template<bool OLD,bool SIGN,bool GLU,typename T>static void launch(const T*x,const T*g,const float*s,block_q8_1_mmq*y,int k,int rows){
 auto kern=OLD?frozen221_quantize<T,SIGN,GLU>:fwht_quantize_mmq_d4<T,SIGN,GLU>;
 kern<<<dim3((k+4095)/4096+1,rows),256,0,stream>>>(x,g,s,y,k,k+4096,rows,k,k);CK(cudaGetLastError());
}
template<bool OLD,typename T>static void run(const T*x,const T*g,const float*s,block_q8_1_mmq*y,int k,int rows,int sign,int glu){if(sign){if(glu)launch<OLD,true,true>(x,g,s,y,k,rows);else launch<OLD,true,false>(x,g,s,y,k,rows);}else{if(glu)launch<OLD,false,true>(x,g,s,y,k,rows);else launch<OLD,false,false>(x,g,s,y,k,rows);}}
static float elapsed(cudaGraphExec_t g){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int i=0;i<8;++i)CK(cudaGraphLaunch(g,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/(8*32);}
int main(int argc,char**argv){
 bool sanitizer = argc>1 && std::string(argv[1])=="--sanitizer";
 CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));constexpr int K=32768,R=128;size_t n=size_t(K)*R;float*x,*g,*s,*raw[2];block_q8_1_mmq*y[2];CK(cudaMalloc(&x,n*4));CK(cudaMalloc(&g,n*4));CK(cudaMalloc(&s,K*4));size_t bytes=size_t((K+4096)/128)*R*sizeof(block_q8_1_mmq);for(int a=0;a<2;++a){CK(cudaMalloc(&y[a],bytes));CK(cudaMalloc(&raw[a],n*4));}
 half *xh,*gh; CK(cudaMalloc(&xh,n*2));CK(cudaMalloc(&gh,n*2));std::vector<half>hxh(n),hgh(n);
 std::mt19937 mt(221);std::vector<float>hx(n),hg(n),hs(K);int cases=0;
 for(int mode=0;mode<6;++mode){for(size_t i=0;i<n;++i){float v=float(int(mt()%20001)-10000)/2048.f;if(mode==1)v=0.f;if(mode==2)v=(i&1)?-0.f:0.f;if(mode==3)v=(i%7==0)?65536.f:(i%7==1)?-65536.f:1e-9f;if(mode==4)v=std::ldexp(float(int(mt()%31)-15),int(mt()%80)-40);if(mode==5){uint32_t bits=(i%8==0)?0x7fc01234u:(i%8==1)?0xffc02345u:(i%8==2)?0x7f800000u:(i%8==3)?0xff800000u:0u;memcpy(&v,&bits,4);}hx[i]=v;hg[i]=float(int(mt()%2001)-1000)/2048.f;}
 for(int i=0;i<K;++i)hs[i]=(mt()&1)?-1.f:1.f;
 CK(cudaMemcpy(x,hx.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(g,hg.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s,hs.data(),K*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());
 raw_transform<true><<<32,256,0,stream>>>(x,raw[0]);raw_transform<false><<<32,256,0,stream>>>(x,raw[1]);CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));same(raw[0],raw[1],32*4096*4,"raw-float-butterfly");++cases;
 for(size_t i=0;i<n;++i){hxh[i]=__float2half(hx[i]);hgh[i]=__float2half(hg[i]);}CK(cudaMemcpy(xh,hxh.data(),n*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(gh,hgh.data(),n*2,cudaMemcpyHostToDevice));
 for(int k:{4096,12288,32768})for(int rows:{1,16,128})for(int sign:{0,1})for(int glu:{0,1}){
 size_t nb=size_t((k+4096)/128)*rows*sizeof(block_q8_1_mmq);for(int a=0;a<2;++a)CK(cudaMemsetAsync(y[a],0xa5,nb,stream));run<true>(x,g,s,y[0],k,rows,sign,glu);run<false>(x,g,s,y[1],k,rows,sign,glu);CK(cudaStreamSynchronize(stream));same(y[0],y[1],nb,"f32-quantized-full-padding");++cases;
 for(int a=0;a<2;++a)CK(cudaMemsetAsync(y[a],0xa5,nb,stream));run<true>(xh,gh,s,y[0],k,rows,sign,glu);run<false>(xh,gh,s,y[1],k,rows,sign,glu);CK(cudaStreamSynchronize(stream));same(y[0],y[1],nb,"f16-quantized-full-padding");++cases;}
 }
 if(sanitizer){printf("FWHT_REGISTER_TRANSPOSE_PASS cases=%d raw_float_bits_and_quant_padding=true sanitizer_mode=true\n",cases);return 0;}
 // Fresh finite values for timing; all samples retained, two warm-up pairs.
 for(size_t i=0;i<n;++i)hx[i]=float(int(mt()%20001)-10000)/2048.f;CK(cudaMemcpy(x,hx.data(),n*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());
 for(int k:{4096,12288,32768})for(int rows:{16,128})for(int glu:{0,1}){
 cudaGraphExec_t exec[2];for(int a=0;a<2;++a){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<32;++i){if(a)run<false>(x,g,s,y[a],k,rows,1,glu);else run<true>(x,g,s,y[a],k,rows,1,glu);}CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&exec[a],gr,nullptr,nullptr,0));CK(cudaGraphDestroy(gr));}
 for(int rep=0;rep<10;++rep){float a,b;if(rep&1){b=elapsed(exec[1]);a=elapsed(exec[0]);}else{a=elapsed(exec[0]);b=elapsed(exec[1]);}if(rep>=2)printf("GRAPH k=%d rows=%d glu=%d rep=%d old_us=%.6f new_us=%.6f gain=%.6f\n",k,rows,glu,rep,a,b,a/b);}
 for(auto e:exec)CK(cudaGraphExecDestroy(e));}
 printf("FWHT_REGISTER_TRANSPOSE_PASS cases=%d raw_float_bits_and_quant_padding=true\n",cases);
}
