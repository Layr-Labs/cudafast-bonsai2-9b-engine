// Actual promoted PTQ DL48 fixup versus compact exact live-owner reduction.
#include "mmq.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#define CHECK(x) do { auto status=(x); if(status!=cudaSuccess) { fprintf(stderr,"%s: %s line %d\n",#x,cudaGetErrorString(status),__LINE__); std::exit(2); } } while(0)
static cudaStream_t stream;
static void launch(bool ref,float *d,const float *p,int rows,int nkb,int stride,int cols) {
    if(ref) {
        const uint3 one=init_fastdiv_values(1),kb=init_fastdiv_values(nkb);
        mul_mat_q_stream_k_fixup<GGML_TYPE_PQ2_0,16,false,48><<<dim3(48,4),dim3(32,4),0,stream>>>(
            nullptr,nullptr,d,const_cast<float *>(p),kb,rows,cols,stride,one,stride*cols,one,stride*cols,one);
    } else {
        if(rows==2048)mul_mat_q_ptq_dl_fixup_compact<16><<<dim3(16,4),dim3(32,4),0,stream>>>(d,p,stride,cols);
        else mul_mat_q_ptq_dl_fixup_compact<32><<<dim3(32,4),dim3(32,4),0,stream>>>(d,p,stride,cols);
    }
    CHECK(cudaGetLastError());
}
static cudaGraphExec_t graph(bool ref,float *d,const float *p,int rows,int nkb,int stride,int cols) {
    cudaGraph_t g; cudaGraphExec_t e;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<64;++i)launch(ref,d,p,rows,nkb,stride,cols);
    CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));CHECK(cudaGraphDestroy(g));return e;
}
static float time_graph(cudaGraphExec_t g) {
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a,stream));for(int i=0;i<16;++i)CHECK(cudaGraphLaunch(g,stream));
    CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));return ms*1000/(16*64);
}
int main() {
    const int BPI=ggml_cuda_mmq_get_K_vram(GGML_TYPE_PQ2_0,16,false,GGML_CUDA_CC_DGX_SPARK)/QK_PQ2_0;
    int proofs=0;
    for(int rows:{2048,4096})for(int nkb=16*BPI;nkb<=8192;nkb+=BPI) {
        int tiles=rows/128,live=0;
        auto begin=[=](int b){int k=int(int64_t(b)*tiles*nkb/48);return k-(k%nkb)%BPI;};
        for(int b=0;b<48;++b) {
            const int kb=begin(b),ke=begin(b+1),tile=kb/nkb;
            const bool active=kb!=ke && kb%nkb!=0 && !(tile==ke/nkb && ke%nkb!=0);
            if(active) {
                const int owner=tiles==16?3*live+2:3*(live/2)+1+live%2;
                std::vector<int> actual;
                int stop=kb;
                for(int pred=b-1;;--pred) {
                    int k=begin(pred);
                    if(k!=stop) {
                        actual.push_back(pred);
                        if(k%nkb==0||k/nkb<tile)break;
                    }
                    stop=k;
                }
                std::vector<int> expected;
                for(int p=1;p<=(tiles==16?2:1);++p)expected.push_back(owner-p);
                if(owner!=b||tile!=live||actual!=expected) {
                    fprintf(stderr,"PARTITION_PROOF_FAIL rows=%d nkb=%d b=%d live=%d\n",rows,nkb,b,live);return 1;
                }
                ++live;
            }
            ++proofs;
        }
        if(live!=tiles){fprintf(stderr,"LIVE_COUNT_FAIL rows=%d nkb=%d\n",rows,nkb);return 1;}
    }
    printf("HOST_PARTITION_PROOF_PASS cases=%d BPI=%d\n",proofs,BPI);
    CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::mt19937 rng(20261019);int cases=0;
    for(int rows:{2048,4096})
    for(int k:{4096,4352,4608,6144,8192,12288,32768,65536}) for(int pad:{0,2,4,31,32}) for(int cols:{1,2,7,9,16}) {
        const int nkb=k/128,stride=rows+pad; const size_t n=size_t(stride)*cols,np=48*16*128;
        std::vector<float> input(n),partials(np);std::vector<unsigned char> ha(n*4),hb(n*4);
        for(auto &v:input) v=float(int(rng()%2049)-1024)*0.03125f;
        for(auto &v:partials) {v=float(int(rng()%2049)-1024)*0.125f;if(rng()%7==0)v*=32768.0f;
            const uint32_t edges[]={0u,0x80000000u,1u,0x80000001u,0x00800000u,0x80800000u};
            if(rng()%13==0){uint32_t u=edges[rng()%6];std::memcpy(&v,&u,4);}}
        float *a,*b,*p;CHECK(cudaMalloc(&a,n*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMalloc(&p,np*4));
        CHECK(cudaMemcpy(a,input.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,input.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p,partials.data(),np*4,cudaMemcpyHostToDevice));
        launch(true,a,p,rows,nkb,stride,cols);launch(false,b,p,rows,nkb,stride,cols);CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(ha.data(),a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,n*4,cudaMemcpyDeviceToHost));
        if(ha!=hb){size_t z=0;while(z<ha.size()&&ha[z]==hb[z])++z;fprintf(stderr,"FIXUP_FAIL rows=%d k=%d pad=%d byte=%zu\n",rows,k,pad,z);return 1;} ++cases;
        if(pad==0 && (k==4096||k==12288||k==32768)) {
            auto ga=graph(true,a,p,rows,nkb,stride,cols),gb=graph(false,b,p,rows,nkb,stride,cols);
            for(int r=0;r<10;++r) {
                float old_us,new_us;if(r%2==0){old_us=time_graph(ga);new_us=time_graph(gb);}else{new_us=time_graph(gb);old_us=time_graph(ga);}
                if(r>=2)printf("GRAPH rows=%d k=%d cols=%d round=%d old_us=%.5f new_us=%.5f gain=%.6f\n",rows,k,cols,r,old_us,new_us,old_us/new_us);
            }
            CHECK(cudaMemcpy(ha.data(),a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,n*4,cudaMemcpyDeviceToHost));
            if(ha!=hb){fprintf(stderr,"GRAPH_FIXUP_FAIL rows=%d k=%d\n",rows,k);return 1;}
            CHECK(cudaGraphExecDestroy(ga));CHECK(cudaGraphExecDestroy(gb));
        }
        CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(p));
    }
    CHECK(cudaStreamDestroy(stream));printf("PTQ_COMPACT_FIXUP_PASS cases=%d exact_all_output_and_padding_bits=true\n",cases);return 0;
}
