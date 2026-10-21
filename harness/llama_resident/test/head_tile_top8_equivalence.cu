// nvcc -std=c++17 -O3 -arch=sm_121a -Illama.cpp/ggml/src/ggml-cuda harness/llama_resident/test/head_tile_top8_equivalence.cu -o head-tile-top8-equivalence
// Exact tile ownership, certificate completeness, adversarial ordering, and changed-input graph replays.
#include "head-cert.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define C(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);return 2;} } while(0)
static uint64_t pack(float v,int col) { uint32_t u;memcpy(&u,&v,4);u=u==0x80000000u?0:u;uint32_t o=u&0x80000000u?~u:u|0x80000000u;return(uint64_t(o)<<32)|~uint32_t(col); }
static __global__ void certificates(const float *x,uint64_t *m,int n) {
    __shared__ uint64_t red[16][129];
    const int w=threadIdx.y,c=threadIdx.x%4,g=threadIdx.x/4,t=blockIdx.x;
    float s[1][4];
    for(int h=0;h<2;++h)for(int l=0;l<2;++l)s[0][2*h+l]=x[int64_t(g+8*h)*n+t*128+w*8+2*c+l];
    ggml_cuda_head_cert_store(s,m,red,t,n/128);
}
int main(){
    cudaStream_t stream;C(cudaStreamCreate(&stream));int cases=0,replays=0;
    std::mt19937 gen(3535);
    for(int n:{128,256,1024,8192,8320,248320}){
        const size_t nf=size_t(n)*16,nm=size_t(n/128)*16*8,np=size_t(topk_select_nchunks(n))*16*8;
        float *x;uint64_t *m,*p;int *out;
        C(cudaMalloc(&x,nf*4));C(cudaMalloc(&m,(nm+16)*8));C(cudaMalloc(&p,np*8));C(cudaMalloc(&out,16*8*4));
        cudaGraph_t graph;cudaGraphExec_t exec;
        C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        certificates<<<n/128,dim3(32,16),0,stream>>>(x,m,n);
        topk_tile_partial<<<dim3(topk_select_nchunks(n),16),32,0,stream>>>(p,n/128,m);
        topk_select_merge_warp<<<16,32,0,stream>>>(p,out,topk_select_nchunks(n),8);
        C(cudaStreamEndCapture(stream,&graph));C(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));C(cudaGraphDestroy(graph));
        for(int mode=0;mode<9;++mode){
            std::vector<float> h(nf);std::vector<uint64_t> em(nm),am(nm+16);std::vector<int> got(128),want(128);
            for(int row=0;row<16;++row){
                std::vector<uint64_t> keys(n);
                for(int col=0;col<n;++col){
                    float v=0;
                    if(mode==0)v=float(int(gen()%200000)-100000)/1000;
                    if(mode==1)v=7.0f;
                    if(mode==2)v=col%2?0.0f:-0.0f;
                    if(mode==3)v=float(col);
                    if(mode==4)v=-float(col);
                    if(mode==5){uint32_t u=gen();memcpy(&v,&u,4);}
                    if(mode==6)v=col<8?100.0f+col:-10.0f;
                    if(mode==7)v=col>=n-8?100.0f+col:-10.0f;
                    if(mode==8){uint32_t bits[]={0xff800000u,0x7f800000u,0x7fc00001u,0xffc00001u,1u,0x80000001u};uint32_t u=bits[col%6];memcpy(&v,&u,4);}
                    h[size_t(row)*n+col]=v;keys[col]=pack(v,col);
                    if (col%128 == 127) {
                        uint64_t tile[128];
                        std::copy(keys.begin()+col-127,keys.begin()+col+1,tile);
                        std::partial_sort(tile,tile+8,tile+128,std::greater<uint64_t>());
                        std::copy(tile,tile+8,em.begin()+(size_t(row)*(n/128)+col/128)*8);
                    }
                }
                std::partial_sort(keys.begin(),keys.begin()+8,keys.end(),std::greater<uint64_t>());
                for(int j=0;j<8;++j)want[row*8+j]=int(~uint32_t(keys[j]));
            }
            // Every replay replaces all tile lists; adversarial ties and concentrated winners are exact.
            C(cudaMemcpyAsync(x,h.data(),nf*4,cudaMemcpyHostToDevice,stream));C(cudaMemsetAsync(m,0xa5,(nm+16)*8,stream));
            C(cudaGraphLaunch(exec,stream));++replays;
            C(cudaMemcpyAsync(am.data(),m,(nm+16)*8,cudaMemcpyDeviceToHost,stream));C(cudaMemcpyAsync(got.data(),out,128*4,cudaMemcpyDeviceToHost,stream));C(cudaStreamSynchronize(stream));
            for(size_t i=0;i<nm;++i)if(am[i]!=em[i]){fprintf(stderr,"FAIL local_top8 n=%d mode=%d i=%zu\n",n,mode,i);return 1;}
            for(size_t i=nm;i<am.size();++i)if(am[i]!=0xa5a5a5a5a5a5a5a5ull)return 3;
            if(got!=want){for(int i=0;i<128;++i)if(got[i]!=want[i]){fprintf(stderr,"FAIL top8 n=%d mode=%d i=%d got=%d want=%d\n",n,mode,i,got[i],want[i]);break;}return 1;}
            ++cases;
        }
        C(cudaGraphExecDestroy(exec));C(cudaFree(x));C(cudaFree(m));C(cudaFree(p));C(cudaFree(out));
    }
    C(cudaStreamDestroy(stream));printf("HEAD_TILE_TOP8_EQUIVALENCE_PASS cases=%d exact_rows=%d changed_input_graph_replays=%d local_top8_and_global_top8_and_guards_exact=true\n",cases,cases*16,replays);return 0;
}
