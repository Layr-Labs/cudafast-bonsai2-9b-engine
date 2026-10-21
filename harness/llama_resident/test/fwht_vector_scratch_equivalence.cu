// Exact candidate223 scalar-plane scratch versus candidate226 float4-interleaved scratch.
#include "quantize.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#define CK(x) do {auto ck_error=(x);if(ck_error!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",#x,__LINE__,cudaGetErrorString(ck_error));exit(2);}}while(0)
int ggml_cuda_get_device() { int d=-1; CK(cudaGetDevice(&d)); return d; }
[[noreturn]] void ggml_cuda_error(const char * stmt,const char * func,const char * file,int line,const char * msg) {
 fprintf(stderr,"%s %s %s %s:%d\n",stmt,msg,func,file,line);std::abort();
}
template <int N, int NT>
__device__ __forceinline__ void frozen226_warp_lane_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    static_assert(N == 4096 && NT == 1024, "warp/lane transpose geometry");
    static_assert(warp_size == 32 && NT == warp_size * warp_size, "square warp/lane geometry required");
    const int warp = tid / warp_size;
    // XOR swizzle gives unique banks for both row and column traversals.
#pragma unroll
    for (int j = 0; j < NE; ++j) s[j*NT + warp*warp_size + (lane ^ warp)] = reg[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) reg[j] = s[j*NT + lane*warp_size + (warp ^ lane)];
    __syncthreads();
    // Lanes now own old warp bits, so original h32..512 stages become
    // shuffle h1..16 in the same order and with the same lower/upper values.
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int j = 0; j < NE; ++j) s[j*NT + lane*warp_size + (warp ^ lane)] = reg[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) reg[j] = s[j*NT + warp*warp_size + (lane ^ warp)];
    // Scratch is dead in the selected terminal caller.

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}


template <int NT, bool SIGNED>
__launch_bounds__(NT, 1)
static __global__ void reference226_fixup_add_rms(const float * a, const float * b, float * sum, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, const int64_t rows, const float * partials) {
    constexpr int N=4096, NE=N/NT, NR=1024, R=NR/NT;
    static_assert(NT==256 || NT==512 || NT==1024, "add_rms fold block size");
    __shared__ float s_sum[NR/WARP_SIZE];
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE], wv[NE], sv[NE];
#pragma unroll
    for(int m=0;m<NE;++m) {
        wv[m]=w[m*NT+tid];
        sv[m]=SIGNED ? signs[m*NT+tid] : 1.0f;
    }
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    a+=row*N;
    b+=row*N;
    sum+=row*N;
    // NT is a multiple of 256, so advancing m increases source by
    // 3*NT/256 exactly. Hoist the thread-local base once, leaving the
    // original floating-point fixup operations and order unchanged.
    const int tile0 = tid/128;
    partials += (tile0 + tile0/2)*2048 + row*128 + tid%128;
    constexpr int PARTIAL_STRIDE = (3*NT/256)*2048;
    float tmp[R];
#pragma unroll
    for(int j=0;j<R;++j) tmp[j]=0.0f;
#pragma unroll
    for(int m=0;m<NE;++m) {
        const int col=m*NT+tid;
        // M4096 / grid48 / K4096 or K12288: one original partial per tile.
        // Keep the original fixup's +0 accumulation and suffix += partial_sum.
        float partial_sum=0.0f;
        partial_sum+=partials[m*PARTIAL_STRIDE];
        float corrected=a[col];
        corrected+=partial_sum;
        const float xi=__fadd_rn(corrected, b[col]);
        sum[col]=xi;
        v[m]=xi;
        tmp[m%R]+=xi*xi;
    }
#pragma unroll
    for(int j=0;j<R;++j) {
        tmp[j]=warp_reduce_sum(tmp[j]);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp[j];
    }
    __syncthreads();
    const float total=warp_reduce_sum(s_sum[lane]);
    const float scale=rsqrtf(total/ncols+eps);
#pragma unroll
    for(int m=0;m<NE;++m) {
        const float xn=scale*v[m]*wv[m];
        if(norm) norm[row*N+m*NT+tid]=xn;
        v[m]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[m]*=sv[m];
    }
    frozen226_warp_lane_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
template<bool OLD>__global__ void raw226(const float*x,float*y){
 __shared__ __align__(16) float scratch[4096];float v[4];int tid=threadIdx.x,lane=tid%32,base=4096*blockIdx.x;
 #pragma unroll
 for(int i=0;i<4;++i)v[i]=x[base+i*1024+tid];
 if constexpr(OLD)frozen226_warp_lane_butterfly<4096,1024>(v,scratch,tid,lane);
 else ggml_cuda_fwht_warp_lane_butterfly<4096,1024>(v,scratch,tid,lane);
 #pragma unroll
 for(int i=0;i<4;++i)y[base+i*1024+tid]=v[i];
}
static cudaStream_t stream;
static void launch(bool old,bool sign,const float*a,const float*b,float*sum,const float*w,float eps,const float*signs,float*norm,block_q8_1_mmq*y,const float*p){
 const ggml_cuda_kernel_launch_params lp(dim3(1,16,1),dim3(1024,1,1),0,stream);
 if(old){
  if(sign)ggml_cuda_kernel_launch(reference226_fixup_add_rms<1024,true>,lp,a,b,sum,w,4096,eps,signs,norm,y,int64_t(16),p);
  else ggml_cuda_kernel_launch(reference226_fixup_add_rms<1024,false>,lp,a,b,sum,w,4096,eps,signs,norm,y,int64_t(16),p);
 }else fwht_quantize_mmq_d4_fixup_add_rms_cuda(a,b,sum,w,eps,sign?signs:nullptr,norm,y,p,stream);
 CK(cudaGetLastError());
}
static void equal(const void*a,const void*b,size_t n,const char*name,int test){
 if(memcmp(a,b,n)){const unsigned char*x=(const unsigned char*)a,*y=(const unsigned char*)b;size_t i=0;while(i<n&&x[i]==y[i])++i;fprintf(stderr,"FAIL case%d %s byte%zu old=%02x new=%02x\n",test,name,i,x[i],y[i]);exit(1);}
}
static float elapsed(cudaGraphExec_t graph){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int i=0;i<8;++i)CK(cudaGraphLaunch(graph,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/128;}
int main(int argc,char**argv){
 bool safety=argc>1&&std::string(argv[1])=="--sanitizer";
 CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 constexpr size_t N=16*4096,P=48*2048,G=64,Q=(4096/128)*16*sizeof(block_q8_1_mmq);
 float *a[2],*b[2],*out[2],*norm[2],*w,*signs,*partial;
 for(int i=0;i<2;++i){CK(cudaMalloc(&a[i],(N+G)*4));CK(cudaMalloc(&b[i],(N+G)*4));CK(cudaMalloc(&out[i],(N+G)*4));CK(cudaMalloc(&norm[i],(N+G)*4));}
 CK(cudaMalloc(&w,4096*4));CK(cudaMalloc(&signs,4096*4));CK(cudaMalloc(&partial,P*4));block_q8_1_mmq*q[2];for(auto &v:q)CK(cudaMalloc(&v,Q+G*4));
 std::mt19937 rng(214);std::vector<float>ha(N+G),hb(N+G),hw(4096),hs(4096),hp(P);
 std::vector<unsigned char>lhs((N+G)*4),rhs(lhs.size());
 int cases=0;
 for(int mode=0;mode<6;++mode)for(int seed=0;seed<3;++seed)for(int sign=0;sign<2;++sign)for(int alias=0;alias<3;++alias)for(int normed=0;normed<2;++normed){
  auto fill=[&](std::vector<float>&x){for(auto &v:x){v=float(int(rng()%20001)-10000)/4096;if(mode==1)v=0.0f;if(mode==2){uint32_t z=(rng()&1)?0x80000000:0;memcpy(&v,&z,4);}if(mode==3)v*=1.e-20f;if(mode==4){const uint32_t z[]={0x7f800000,0xff800000,0x7fc12345,0x7f812345,1,0x80000001,0,0x80000000};uint32_t u=z[rng()%8];memcpy(&v,&u,4);}if(mode==5)v*=1.e15f;}};
  fill(ha);fill(hb);fill(hp);for(size_t i=0;i<hw.size();++i){hw[i]=float(int(rng()%513)-256)/128;hs[i]=(rng()&1)?1.f:-1.f;}
  CK(cudaMemcpy(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(signs,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(partial,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
  for(int i=0;i<2;++i){CK(cudaMemcpy(a[i],ha.data(),ha.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(b[i],hb.data(),hb.size()*4,cudaMemcpyHostToDevice));CK(cudaMemset(out[i],0xa5,(N+G)*4));CK(cudaMemset(norm[i],0xa5,(N+G)*4));CK(cudaMemset(q[i],0xa5,Q+G*4));}
  const float eps=seed==0?1.e-6f:seed==1?1.e-5f:0.f;
  for(int i=0;i<2;++i)launch(i==0,sign,a[i],b[i],alias==1?a[i]:alias==2?b[i]:out[i],w,eps,signs,normed?norm[i]:nullptr,q[i],partial);
  CK(cudaStreamSynchronize(stream));
  for(int what=0;what<5;++what){void*x[2];size_t bytes=(N+G)*4;for(int i=0;i<2;++i)x[i]=what==0?(void*)a[i]:what==1?(void*)b[i]:what==2?(void*)out[i]:what==3?(void*)norm[i]:(void*)q[i];if(what==4)bytes=Q+G*4;lhs.resize(bytes);rhs.resize(bytes);CK(cudaMemcpy(lhs.data(),x[0],bytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(rhs.data(),x[1],bytes,cudaMemcpyDeviceToHost));equal(lhs.data(),rhs.data(),bytes,"full allocation",cases);}
  // Compare raw FP32 butterfly result payloads, stronger than quantized bytes.
  CK(cudaMemcpy(a[0],ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
  raw226<true><<<16,1024,0,stream>>>(a[0],out[0]);raw226<false><<<16,1024,0,stream>>>(a[0],out[1]);CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(lhs.data(),out[0],N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(rhs.data(),out[1],N*4,cudaMemcpyDeviceToHost));equal(lhs.data(),rhs.data(),N*4,"raw FP32 butterfly",cases);
  ++cases;
 }
 // Pure, disjoint buffers and finite inputs for paired production-dispatch graphs.
 for(auto&v:ha)v=float(int(rng()%20001)-10000)/4096;for(auto&v:hb)v=float(int(rng()%20001)-10000)/4096;for(auto&v:hp)v=float(int(rng()%20001)-10000)/4096;
 for(int i=0;i<2;++i){CK(cudaMemcpy(a[i],ha.data(),ha.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(b[i],hb.data(),hb.size()*4,cudaMemcpyHostToDevice));}CK(cudaMemcpy(partial,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
 if(safety){printf("FWHT_VECTOR_SCRATCH_EQUIVALENCE_PASS cases=%d all_allocated_output_bits=true\n",cases);return 0;}
 for(int sign=0;sign<2;++sign)for(int normed=0;normed<2;++normed){
  cudaGraphExec_t exec[2];for(int i=0;i<2;++i){cudaGraph_t g;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int j=0;j<16;++j)launch(i==0,sign,a[i],b[i],out[i],w,1.e-6f,signs,normed?norm[i]:nullptr,q[i],partial);CK(cudaStreamEndCapture(stream,&g));CK(cudaGraphInstantiate(&exec[i],g,nullptr,nullptr,0));CK(cudaGraphDestroy(g));}
  for(int rep=0;rep<10;++rep){float old_us,new_us;if(rep%2){new_us=elapsed(exec[1]);old_us=elapsed(exec[0]);}else{old_us=elapsed(exec[0]);new_us=elapsed(exec[1]);}if(rep>=2)printf("GRAPH signed=%d norm=%d rep=%d old_us=%.6f new_us=%.6f gain=%.6f\n",sign,normed,rep,old_us,new_us,old_us/new_us);}
  for(auto e:exec)CK(cudaGraphExecDestroy(e));
 }
 printf("FWHT_VECTOR_SCRATCH_EQUIVALENCE_PASS cases=%d all_allocated_output_bits=true\n",cases);
 for(int i=0;i<2;++i){CK(cudaFree(a[i]));CK(cudaFree(b[i]));CK(cudaFree(out[i]));CK(cudaFree(norm[i]));CK(cudaFree(q[i]));}CK(cudaFree(w));CK(cudaFree(signs));CK(cudaFree(partial));CK(cudaStreamDestroy(stream));
}
