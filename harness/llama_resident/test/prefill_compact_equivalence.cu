// Compare only the retained compact compute with the original J128 kernel.
// Same grid/fixup, bitwise outputs and all scratch words, guarded buffers,
// changed-input graph replays. No model or timing measurement is involved.
#include "mmq.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int GUARD = 64;
constexpr uint32_t SENTINEL = 0x47123456;
void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); }
}
void require(bool value, const char * why) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
template<class T> struct Buffer {
    T * ptr = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&ptr, n*sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
};
struct Guarded {
    size_t count;
    Buffer<uint32_t> storage;
    std::vector<uint32_t> poison;
    explicit Guarded(size_t n) : count(n), storage(n+2*GUARD), poison(n+2*GUARD,SENTINEL) {}
    float * data() const { return reinterpret_cast<float *>(storage.ptr+GUARD); }
    void reset(cudaStream_t stream) {
        check(cudaMemcpyAsync(storage.ptr,poison.data(),poison.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
    }
    std::vector<uint32_t> read() const {
        std::vector<uint32_t> words(poison.size());
        check(cudaMemcpy(words.data(),storage.ptr,words.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for (int i=0;i<GUARD;++i)
            require(words[i]==SENTINEL && words[GUARD+count+i]==SENTINEL,"buffer guards");
        return words;
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (graph) cudaGraphDestroy(graph); }
};
template<bool ADD,bool GLU,bool COMPACT>
void launch(const mmq_args & args,int grid,int tiles,float * fixup,cudaStream_t stream) {
    if constexpr (GLU) {
        const auto kernel=mul_mat_q_pq2_j128<true,ADD,GLU,COMPACT>;
        const int nkb=args.ncols_x/QK_PQ2_0;
        ggml_cuda_kernel_launch(kernel,
            ggml_cuda_kernel_launch_params(dim3(2*tiles),dim3(32,8),ggml_cuda_pq2_j128_smem(),stream),
            args.x,args.y,args.dst,(float *)nullptr,(int)args.nrows_x,nkb,(int)args.stride_row_x,
            (int)args.ncols_y,(int)args.nrows_dst,(int)args.ncols_dst,(int)(args.ncols_max/128),
            args.add_res,args.add_dst);
    } else if (grid==tiles) {
        ggml_cuda_mmq_pq2_j128_launch<GGML_TYPE_PQ2_0,true,ADD,COMPACT>(args,grid,fixup,stream);
    } else {
        ggml_cuda_mmq_pq2_j128_launch<GGML_TYPE_PQ2_0,false,ADD,COMPACT>(args,grid,fixup,stream);
    }
}
template<bool ADD,bool GLU>
void shape(int rows,int cols,int k) {
    const int nkb=k/QK_PQ2_0,outrows=GLU?rows/2:rows,tiles=(rows/128)*(cols/128);
    const size_t outputs=size_t(outrows)*cols;
    std::mt19937 rng(rows+cols+k);
    std::vector<block_pq2_0> weights(size_t(rows)*nkb);
    for (auto & block:weights) {
        const uint16_t scale=ggml_fp32_to_fp16(float(1+rng()%127)/4096.0f);
        std::memcpy(&block.d,&scale,sizeof(scale));
        for (auto & q:block.qs) q=uint8_t(rng());
    }
    const uint16_t edge[]={0x0000,0x8000,0x0001,0x03ff,0x0400,0x8400};
    for (size_t i=0;i<sizeof(edge)/sizeof(edge[0]);++i) std::memcpy(&weights[i].d,&edge[i],2);
    std::vector<block_q8_1_mmq> activation(size_t(nkb)*cols);
    std::vector<float> residual(outputs);
    for (auto & value:residual) value=float(int(rng()%2001)-1000)/1024.0f;
    Buffer<char> dx(weights.size()*sizeof(weights[0]));
    Buffer<int> dy(activation.size()*sizeof(activation[0])/sizeof(int));
    Buffer<float> dr(outputs);
    check(cudaMemcpy(dx.ptr,weights.data(),weights.size()*sizeof(weights[0]),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr.ptr,residual.data(),outputs*sizeof(float),cudaMemcpyHostToDevice));
    Guarded control(outputs),candidate(outputs),scratch_a(size_t(48)*128*128),scratch_b(size_t(48)*128*128);
    mmq_args args={};
    args.x=dx.ptr;args.type_x=GGML_TYPE_PQ2_0;args.y=dy.ptr;args.dst=control.data();
    args.ncols_x=k;args.nrows_x=rows;args.ncols_dst=cols;args.stride_row_x=nkb;
    args.ncols_y=cols;args.nrows_dst=outrows;args.ncols_max=cols;
    args.nchannels_x=args.nchannels_y=args.nsamples_x=args.nsamples_y=1;
    args.stride_channel_dst=args.stride_sample_dst=outputs;
    args.add_res=ADD?dr.ptr:nullptr;args.add_dst=ADD?control.data():nullptr;
    auto other=args;other.dst=candidate.data();other.add_dst=ADD?candidate.data():nullptr;
    const int grid=GLU?tiles:ggml_cuda_mmq_pq2_j128_grid(args,48);
    require(grid==tiles || grid==48,"production grid");
    if constexpr (GLU) {
        for (auto kernel:{mul_mat_q_pq2_j128<true,ADD,GLU,false>,mul_mat_q_pq2_j128<true,ADD,GLU,true>}) {
            check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ggml_cuda_pq2_j128_smem()));
            check(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
        }
    }
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto upload=[&](int phase) {
        std::mt19937 input_rng(271+phase);
        for (auto & block:activation) {
            for (auto & d:block.d4) d=phase==2?0.0f:float(1+input_rng()%127)/8192.0f;
            for (auto & q:block.qs) q=phase==2?0:int(input_rng()%255)-127;
        }
        check(cudaMemcpyAsync(dy.ptr,activation.data(),activation.size()*sizeof(activation[0]),cudaMemcpyHostToDevice,stream));
    };
    upload(0);control.reset(stream);candidate.reset(stream);scratch_a.reset(stream);scratch_b.reset(stream);
    // Configure every function attribute outside graph capture.
    launch<ADD,GLU,false>(args,grid,tiles,scratch_a.data(),stream);
    launch<ADD,GLU,true>(other,grid,tiles,scratch_b.data(),stream);
    check(cudaStreamSynchronize(stream));
    Graph graphs[2];
    for (int mode=0;mode<2;++mode) {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if (mode) launch<ADD,GLU,true>(other,grid,tiles,scratch_b.data(),stream);
        else launch<ADD,GLU,false>(args,grid,tiles,scratch_a.data(),stream);
        check(cudaStreamEndCapture(stream,&graphs[mode].graph));
        check(cudaGraphInstantiate(&graphs[mode].executable,graphs[mode].graph,nullptr,nullptr,0));
    }
    std::vector<uint32_t> first_output, changed_output;
    for (int phase:{0,1,2,1,0}) {
        upload(phase);control.reset(stream);candidate.reset(stream);scratch_a.reset(stream);scratch_b.reset(stream);
        check(cudaGraphLaunch(graphs[0].executable,stream));
        check(cudaGraphLaunch(graphs[1].executable,stream));
        check(cudaStreamSynchronize(stream));
        const auto reference=control.read(),got=candidate.read();
        require(reference==got,"bitwise output and guards");
        for (size_t i=0;i<outputs;++i) {
            float value;
            std::memcpy(&value,&got[GUARD+i],sizeof(value));
            require(got[GUARD+i]!=SENTINEL && std::isfinite(value),"every output written and finite");
        }
        if (phase==0) {
            if (first_output.empty()) first_output=reference;
            else require(reference==first_output,"A graph replay deterministic");
        } else if (phase==1) {
            require(reference!=first_output,"changed activation affects graph output");
            if (changed_output.empty()) changed_output=reference;
            else require(reference==changed_output,"B graph replay deterministic");
        }
        require(scratch_a.read()==scratch_b.read(),"all scratch writes and guards");
    }
    check(cudaStreamDestroy(stream));
    std::printf("COMPACT_EXACT M=%d N=%d K=%d ADD=%d GLU=%d grid=%d changed_graphs=5\n",rows,cols,k,ADD,GLU,grid);
}
} // namespace
int main() {
    setenv("GGML_CUDA_PREFILL_COMPACT","0",1); // explicit template compares the retained path
    ggml_cuda_set_device(0);
    require(ggml_cuda_info().devices[0].cc==GGML_CUDA_CC_DGX_SPARK && ggml_cuda_info().devices[0].nsm==48,"GB10 required");
    for (int n:{128,256,512}) {
        for (int m:{1024,4096,8192,12288}) shape<false,false>(m,n,4096);
        shape<true,false>(4096,n,4096);shape<true,false>(4096,n,12288);
        shape<false,true>(24576,n,4096);
    }
    std::puts("PREFILL_COMPACT_EQUIVALENCE_PASS shapes=21 changed_graphs=105");
    return 0;
}
