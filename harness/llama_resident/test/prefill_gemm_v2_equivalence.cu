// Production D1 kernels, explicit baseline/v2 specializations, no model needed.
// All allocations, input uploads, poison fills and L2 eviction are untimed.
#include "mmq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int N = 512, GRID = 48, GUARD = 64, SHARED = 50176;
constexpr uint32_t POISON = 0xa5a5a5a5u;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
template<class T> struct Buffer {
    T * ptr = nullptr;
    explicit Buffer(size_t n) { if (n) check(cudaMalloc(&ptr, n*sizeof(T))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
};
struct Guarded {
    size_t words;
    Buffer<uint32_t> storage;
    explicit Guarded(size_t n) : words(n), storage(n+2*GUARD) {}
    float * data() const { return reinterpret_cast<float *>(storage.ptr+GUARD); }
    void reset(cudaStream_t stream) {
        check(cudaMemsetAsync(storage.ptr, 0xa5, (words+2*GUARD)*4, stream));
    }
    std::vector<uint32_t> read() const {
        std::vector<uint32_t> result(words+2*GUARD);
        check(cudaMemcpy(result.data(), storage.ptr, result.size()*4, cudaMemcpyDeviceToHost));
        for (int i=0; i<GUARD; ++i) {
            require(result[i]==POISON && result[GUARD+words+i]==POISON, "buffer canaries");
        }
        return result;
    }
};
size_t compare(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b,
        const char * shape, const char * buffer, int copy, int phase) {
    require(a.size()==b.size(), "comparison sizes");
    size_t count=0, first=a.size();
    for (size_t i=0; i<a.size(); ++i) if (a[i]!=b[i]) {
        if (count++==0) first=i;
    }
    if (count) {
        std::fprintf(stderr, "GEMM_V2_MISMATCH shape=%s buffer=%s copy=%d phase=%d "
                     "words=%zu first_word=%zu baseline=%08x v2=%08x\n",
                     shape, buffer, copy, phase, count, first, a[first], b[first]);
    }
    return count;
}

// This separate witness localizes a bad .x4 address/register permutation before
// running any GEMM. It uses the same 144-byte record layout and load helpers.
__global__ void b_load_witness(uint32_t * out) {
    __shared__ __align__(16) uint32_t tile[32*36];
    const int lane=threadIdx.x;
    for (int i=lane; i<32*36; i+=32) tile[i]=0x9e3779b9u*unsigned(i+1);
    __syncthreads();
    const uint32_t base=(uint32_t)__cvta_generic_to_shared(tile);
    const uint32_t ya=base+(lane%8)*144+16+((lane/8)%2)*16;
    for (int sub=0; sub<4; ++sub) for (int ct=0; ct<4; ct+=2) {
        int pair[4], low[2], high[2];
        ggml_cuda_pq2_j128_ldsm_x4(pair,ya+(ct+lane/16)*8*144+sub*32);
        ggml_cuda_pq2_j128_ldsm_x2(low,ya+ct*8*144+sub*32);
        ggml_cuda_pq2_j128_ldsm_x2(high,ya+(ct+1)*8*144+sub*32);
        const int original[4]={low[0],low[1],high[0],high[1]};
        for (int r=0; r<4; ++r) {
            const int i=((sub*2+ct/2)*32+lane)*4+r;
            out[i]=unsigned(pair[r])^unsigned(original[r]);
        }
    }
}

// Touch every 32-byte sector through L2, outside the CUDA event interval. The
// pool exceeds L2 even for the 4096x4096 case where four weights alone may fit.
__global__ void evict_l2(volatile uint32_t * pool, size_t words) {
    for (size_t i=(blockIdx.x*blockDim.x+threadIdx.x)*8; i<words; i+=size_t(gridDim.x)*blockDim.x*8) {
        pool[i]=pool[i]+1u;
    }
}

template<bool PLANE, bool ADD, bool HALF, bool V2>
auto kernel_for() {
    if constexpr (PLANE) return mul_mat_q_pq2_plane<false,true,true,true,true,HALF,V2>;
    else return mul_mat_q_pq2_j128<false,ADD,false,true,true,true,V2>;
}
template<bool PLANE, bool ADD, bool HALF, bool V2>
void configure(const char * shape) {
    const auto kernel=kernel_for<PLANE,ADD,HALF,V2>();
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr,kernel));
    int ctas=0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas,kernel,256,SHARED));
    std::printf("GEMM_V2_RESOURCE shape=%s variant=%s regs=%d local_bytes=%zu smem=%d ctas_per_sm=%d\n",
                shape,V2?"v2":"baseline",attr.numRegs,attr.localSizeBytes,SHARED,ctas);
}

template<bool PLANE, bool ADD, bool HALF, bool V2>
void launch(const mmq_args & args, const unsigned char * plane, float * scratch, cudaStream_t stream) {
    const int rows=int(args.nrows_x),nkb=int(args.ncols_x/QK_PQ2_0),outrows=int(args.nrows_dst);
    const auto kernel=kernel_for<PLANE,ADD,HALF,V2>();
    if constexpr (PLANE) {
        const auto * scale=reinterpret_cast<const uint16_t *>(plane+size_t(rows)*nkb*32);
        ggml_cuda_kernel_launch(kernel,
            ggml_cuda_kernel_launch_params(dim3((rows/64)*(N/128)),dim3(32,8),SHARED,stream),
            plane,scale,args.y,args.dst,nkb,N,outrows,N/128,(const float *)nullptr);
    } else {
        // Exactly the production 48-way stream-K partition (96 half-row CTAs).
        ggml_cuda_kernel_launch(kernel,
            ggml_cuda_kernel_launch_params(dim3(2*GRID),dim3(32,8),SHARED,stream),
            reinterpret_cast<const char *>(plane),args.y,args.dst,scratch,rows,nkb,nkb,N,outrows,N,N/128,
            args.add_res,args.add_dst);
        const uint3 blocks=init_fastdiv_values(nkb),ntx=init_fastdiv_values(N/128);
        if constexpr (ADD) {
            ggml_cuda_kernel_launch(mul_mat_q_pq2_j128_fixup_add<GGML_TYPE_PQ2_0>,
                ggml_cuda_kernel_launch_params(dim3(GRID,4),dim3(32,4),0,stream),
                (const float *)args.dst,(const float *)scratch,args.add_res,args.add_dst,blocks,rows,N,outrows,ntx);
        } else {
            const uint3 one=init_fastdiv_values(1);
            ggml_cuda_kernel_launch(mul_mat_q_stream_k_fixup<GGML_TYPE_PQ2_0,128,false>,
                ggml_cuda_kernel_launch_params(dim3(GRID,4),dim3(32,4),0,stream),
                nullptr,nullptr,args.dst,scratch,blocks,rows,N,outrows,one,int(args.stride_channel_dst),
                one,int(args.stride_sample_dst),ntx);
        }
    }
    check(cudaGetLastError());
}

// Exercise the real environment switch and inspect graph function identities.
// Benchmarking below uses explicit templates so both modes share one process.
template<bool PLANE, bool ADD, bool HALF>
void check_route(const mmq_args & args, float * scratch, cudaStream_t stream) {
    auto route=[&]() {
        if constexpr (PLANE) {
            bool half=false;
            require(ggml_cuda_prefill_plane_glu_run<GGML_TYPE_PQ2_0>(args,stream,&half), "plane route");
            require(half==HALF, "production GLU store format");
        } else ggml_cuda_mmq_pq2_j128_launch<GGML_TYPE_PQ2_0,false,ADD>(args,GRID,scratch,stream);
    };
    route(); // initialize function attributes outside capture
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph=nullptr;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    route();
    check(cudaStreamEndCapture(stream,&graph));
    size_t count=0;
    check(cudaGraphGetNodes(graph,nullptr,&count));
    require(count==size_t(PLANE?1:2), "production kernel/fixup node count");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph,nodes.data(),&count));
    const void * expected=ggml_cuda_prefill_gemm_v2_enabled()
        ? (const void *)kernel_for<PLANE,ADD,HALF,true>() : (const void *)kernel_for<PLANE,ADD,HALF,false>();
    int found=0;
    for (auto node:nodes) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node,&type));
        require(type==cudaGraphNodeTypeKernel, "kernel graph node");
        cudaKernelNodeParams params{};
        check(cudaGraphKernelNodeGetParams(node,&params));
        if (params.func==expected) {
            ++found;
            require(params.blockDim.x==32 && params.blockDim.y==8 && params.sharedMemBytes==SHARED,
                    "production launch geometry");
            require(params.gridDim.x==unsigned(PLANE?(args.nrows_x/64)*(N/128):2*GRID), "production grid");
        }
    }
    require(found==1, "environment selects expected baseline/v2 function");
    check(cudaGraphDestroy(graph));
}

struct Options { bool cold=false; int launches=24; int enabled=-1; };

template<bool PLANE, bool ADD=false, bool HALF=false>
void shape(const char * name, int rows, int k, const Options & options, cudaStream_t stream,
        uint32_t * eviction, size_t eviction_words) {
    const int nkb=k/QK_PQ2_0,outrows=PLANE?rows/2:rows,copies=options.cold?4:1;
    const size_t nblocks=size_t(rows)*nkb,wbytes=nblocks*sizeof(block_pq2_0);
    const size_t outputs=size_t(outrows)*N,words=HALF?outputs/2:outputs;
    Buffer<unsigned char> weights(wbytes*copies);
    Buffer<int> activation(size_t(nkb)*N*sizeof(block_q8_1_mmq)/sizeof(int));
    Buffer<float> residual(ADD?outputs:0);
    Guarded baseline(words),candidate(words),scratch_a(size_t(GRID)*128*128),scratch_b(size_t(GRID)*128*128);
    std::vector<const unsigned char *> planes;
    std::mt19937 rng(1729+rows+k);
    std::vector<block_pq2_0> host_weight(nblocks);
    for (int copy=0; copy<copies; ++copy) {
        for (auto & block:host_weight) {
            const uint16_t scale=ggml_fp32_to_fp16(float(1+rng()%127)/8192.0f);
            std::memcpy(&block.d,&scale,2);
            for (auto & q:block.qs) {
                unsigned trits=rng()%81;
                q=0;
                for (int i=0; i<4; ++i) { q|=(trits%3)<<(2*i); trits/=3; }
            }
        }
        const uint16_t edges[]={0x0000,0x8000,0x0001,0x03ff,0x0400,0x8400,0x3c00,0xbc00,0x7bff};
        // Spread scale edges over row tiles and K blocks, including split K.
        for (size_t i=0; i<sizeof(edges)/sizeof(edges[0]); ++i) {
            const size_t index=(i*size_t(67)*nkb+i*7)%nblocks;
            std::memcpy(&host_weight[index].d,&edges[i],2);
        }
        auto * source=weights.ptr+size_t(copy)*wbytes;
        check(cudaMemcpy(source,host_weight.data(),wbytes,cudaMemcpyHostToDevice));
        require(ggml_cuda_prefill_plane_register_weight(source,rows,k,GGML_TYPE_PQ2_0)==wbytes,
                "production plane registration");
        const auto * entry=ggml_cuda_prefill_plane_find(source);
        require(entry && entry->bytes==wbytes, "production plane layout");
        planes.push_back(entry->data);
    }
    std::vector<float> host_residual(ADD?outputs:0);
    for (auto & value:host_residual) value=float(int(rng()%2001)-1000)/1024.0f;
    if (ADD) check(cudaMemcpyAsync(residual.ptr,host_residual.data(),outputs*4,cudaMemcpyHostToDevice,stream));
    mmq_args args={};
    args.type_x=GGML_TYPE_PQ2_0; args.x=reinterpret_cast<const char *>(weights.ptr); args.y=activation.ptr;
    args.dst=baseline.data(); args.ncols_x=k; args.nrows_x=rows; args.stride_row_x=nkb;
    args.ncols_y=args.ncols_max=args.ncols_dst=N; args.nrows_dst=outrows;
    args.nchannels_x=args.nchannels_y=args.nsamples_x=args.nsamples_y=1;
    args.stride_channel_dst=args.stride_sample_dst=outputs;
    args.d1_y=true; args.glu_f16=HALF;
    args.add_res=ADD?residual.ptr:nullptr; args.add_dst=ADD?baseline.data():nullptr;
    auto other=args; other.dst=candidate.data(); other.add_dst=ADD?candidate.data():nullptr;
    if (!PLANE) {
        require(ggml_cuda_mmq_pq2_j128_grid(args,GRID)==GRID && ggml_cuda_mmq_pq2_j128_needs_fixup(args,GRID),
                "production shape really uses stream-K and fixup");
    }
    configure<PLANE,ADD,HALF,false>(name);
    configure<PLANE,ADD,HALF,true>(name);
    std::vector<block_q8_1_mmq> host_y(size_t(nkb)*N);
    auto upload=[&](int phase) {
        std::mt19937 input_rng(271+phase);
        for (auto & block:host_y) {
            const float scale=phase==2?0.0f:float(1+input_rng()%127)/8192.0f;
            for (auto & d:block.d4) d=scale; // D1: four identical F32 fields
            for (auto & q:block.qs) q=phase==2?0:int(input_rng()%256)-128;
        }
        check(cudaMemcpyAsync(activation.ptr,host_y.data(),host_y.size()*sizeof(host_y[0]),cudaMemcpyHostToDevice,stream));
    };
    upload(0);
    baseline.reset(stream); scratch_a.reset(stream);
    check_route<PLANE,ADD,HALF>(args,scratch_a.data(),stream);
    // Record events inside each graph so the measured interval contains no
    // host enqueue gap. These same graphs get changed-input equality replays.
    cudaEvent_t begin,end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<cudaGraph_t> graphs(2*copies,nullptr);
    std::vector<cudaGraphExec_t> executables(2*copies,nullptr);
    for (int copy=0; copy<copies; ++copy) for (int mode=0; mode<2; ++mode) {
        const int index=2*copy+mode;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        check(cudaEventRecordWithFlags(begin,stream,cudaEventRecordExternal));
        if (mode) launch<PLANE,ADD,HALF,true>(other,planes[copy],scratch_b.data(),stream);
        else launch<PLANE,ADD,HALF,false>(args,planes[copy],scratch_a.data(),stream);
        check(cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
        check(cudaStreamEndCapture(stream,&graphs[index]));
        check(cudaGraphInstantiate(&executables[index],graphs[index],nullptr,nullptr,0));
    }
    for (int copy=0; copy<copies; ++copy) {
        std::vector<uint32_t> first;
        for (int phase:{0,1,2,0}) {
            upload(phase);
            baseline.reset(stream); candidate.reset(stream); scratch_a.reset(stream); scratch_b.reset(stream);
            check(cudaGraphLaunch(executables[2*copy],stream));
            check(cudaGraphLaunch(executables[2*copy+1],stream));
            check(cudaStreamSynchronize(stream));
            const auto a=baseline.read(),b=candidate.read();
            require(compare(a,b,name,"output",copy,phase)==0, "bitwise outputs");
            const auto sa=scratch_a.read(),sb=scratch_b.read();
            require(compare(sa,sb,name,"scratch",copy,phase)==0, "bitwise stream-K scratch");
            if constexpr (!PLANE) {
                require(std::any_of(sa.begin()+GUARD,sa.end()-GUARD,[](uint32_t word) { return word!=POISON; }),
                        "stream-K partials were written");
            }
            for (size_t i=0; i<words; ++i) {
                require(b[GUARD+i]!=POISON, "every output word overwritten");
                if constexpr (!HALF) {
                    float value; std::memcpy(&value,&b[GUARD+i],4);
                    require(std::isfinite(value), "finite F32 output");
                } else {
                    require((b[GUARD+i]&0x7c00u)!=0x7c00u && (b[GUARD+i]&0x7c000000u)!=0x7c000000u, "finite F16 output");
                }
            }
            if (phase==0) {
                if (first.empty()) first=a;
                else require(first==a, "A/B/zero/A exact replay");
            } else if (phase==1) require(first!=a, "changed input affects output");
        }
    }

    // Events cover the entire production operation, including stream-K fixup.
    upload(1);
    check(cudaStreamSynchronize(stream));
    std::vector<float> times[2];
    for (int i=0; i<4; ++i) {
        check(cudaGraphLaunch(executables[2*(i%copies)],stream));
        check(cudaGraphLaunch(executables[2*(i%copies)+1],stream));
    }
    check(cudaStreamSynchronize(stream));
    for (int i=0; i<options.launches; ++i) for (int order=0; order<2; ++order) {
        const int mode=(i+order)&1,copy=i%copies; // alternate AB / BA, common weight copy
        if (options.cold) evict_l2<<<256,256,0,stream>>>(eviction,eviction_words);
        check(cudaGetLastError());
        check(cudaGraphLaunch(executables[2*copy+mode],stream));
        check(cudaStreamSynchronize(stream));
        float ms=0; check(cudaEventElapsedTime(&ms,begin,end)); times[mode].push_back(ms*1000.0f);
    }
    for (auto & samples:times) std::sort(samples.begin(),samples.end());
    auto median=[](const std::vector<float> & samples) {
        const size_t n=samples.size(); return (samples[(n-1)/2]+samples[n/2])*0.5f;
    };
    const float old_us=median(times[0]),new_us=median(times[1]);
    std::printf("GEMM_V2_EXACT shape=%s M=%d N=%d K=%d copies=%d output_words=%zu scratch_words=%zu mismatch_words=0\n",
                name,rows,N,k,copies,words,PLANE?size_t(0):scratch_a.words);
    std::printf("GEMM_V2_BENCH shape=%s cache=%s launches=%d copies=%d baseline_us=%.3f v2_us=%.3f speedup=%.5f saved_us=%.3f\n",
                name,options.cold?"cold-l2":"warm",options.launches,copies,old_us,new_us,old_us/new_us,old_us-new_us);
    for (auto executable:executables) check(cudaGraphExecDestroy(executable));
    for (auto graph:graphs) check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(end)); check(cudaEventDestroy(begin));
    for (int copy=0; copy<copies; ++copy) ggml_cuda_dl_unregister_range(weights.ptr+size_t(copy)*wbytes,wbytes);
}
} // namespace

int main(int argc, char ** argv) {
    Options options;
    for (int i=1; i<argc; ++i) {
        if (std::strcmp(argv[i],"--cold-l2")==0) options.cold=true;
        else if (std::strcmp(argv[i],"--enabled")==0) options.enabled=1;
        else if (std::strcmp(argv[i],"--disabled")==0) options.enabled=0;
        else if (std::strcmp(argv[i],"--launches")==0 && i+1<argc) options.launches=std::atoi(argv[++i]);
        else {
            std::fprintf(stderr,"usage: %s [--cold-l2] [--launches N>=20] [--enabled|--disabled]\n",argv[0]);
            return 2;
        }
    }
    require(options.launches>=20,"at least 20 launches per variant");
    if (options.enabled<0) unsetenv("GGML_CUDA_PREFILL_GEMM_V2");
    else setenv("GGML_CUDA_PREFILL_GEMM_V2",options.enabled?"1":"0",1);
    // Pin the production routes under test, independently of inherited lab knobs.
    setenv("GGML_CUDA_PREFILL_PLANE","1",1);
    setenv("GGML_CUDA_PREFILL_PLANE_D4_ALIAS","1",1);
    setenv("GGML_CUDA_PREFILL_PLANE_OWNER","1",1);
    setenv("GGML_CUDA_PREFILL_J128_TMAX","1",1);
    require(ggml_cuda_prefill_gemm_v2_enabled()==(options.enabled!=0),"default on / explicit switch");
    ggml_cuda_set_device(0);
    require(ggml_cuda_info().devices[0].cc==GGML_CUDA_CC_DGX_SPARK && ggml_cuda_info().devices[0].nsm==GRID,"GB10 required");
    cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,0));
    const size_t eviction_words=options.cold?std::max(size_t(64)<<20,size_t(props.l2CacheSize)*4)/4:0;
    Buffer<uint32_t> eviction(eviction_words),witness(4*2*32*4);
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    if (options.cold) check(cudaMemsetAsync(eviction.ptr,0,eviction_words*4,stream));
    b_load_witness<<<1,32,0,stream>>>(witness.ptr);
    check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
    std::vector<uint32_t> differences(4*2*32*4);
    check(cudaMemcpy(differences.data(),witness.ptr,differences.size()*4,cudaMemcpyDeviceToHost));
    for (auto value:differences) require(value==0,"B x4 fragments equal the two original x2 fragments");
    std::printf("GEMM_V2_MODE switch=%s cold_l2=%d eviction_bytes=%zu B_fragment_words=%zu\n",
                options.enabled<0?"absent(default-on)":options.enabled?"1":"0",int(options.cold),eviction_words*4,differences.size());
    shape<true,false,true>("plane-gu-f16",24576,4096,options,stream,eviction.ptr,eviction_words);
    shape<false>("j128-qkv",8192,4096,options,stream,eviction.ptr,eviction_words);
    shape<false>("j128-square",4096,4096,options,stream,eviction.ptr,eviction_words);
    shape<false>("j128-down",4096,12288,options,stream,eviction.ptr,eviction_words);
    shape<false,true>("j128-square-add",4096,4096,options,stream,eviction.ptr,eviction_words);
    shape<false,true>("j128-down-add",4096,12288,options,stream,eviction.ptr,eviction_words);
    shape<true>("plane-gu-f32",24576,4096,options,stream,eviction.ptr,eviction_words);
    check(cudaStreamDestroy(stream));
    std::puts("PREFILL_GEMM_V2_EQUIVALENCE_PASS shapes=7 mismatch_words=0");
    return 0;
}
