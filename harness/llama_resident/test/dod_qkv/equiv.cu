// Spark-only D1 equivalence + PDL microbenchmark. Includes production kernels.
#include "common.cuh"
#include "mmq.cuh"
#include "rope.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

int ggml_cuda_get_device() { int d = 0; cudaGetDevice(&d); return d; }
void ggml_cuda_set_device(int d) { cudaSetDevice(d); }
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "%s %s %s:%d %s\n", stmt, func, file, line, msg); std::exit(2);
}
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    std::fprintf(stderr, "%s:%d %s\n", file, line, fmt); std::exit(3);
}
const ggml_cuda_device_info & ggml_cuda_info() {
    static const auto info = [] {
        ggml_cuda_device_info i = {};
        cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
        i.device_count = i.physical_device_count = 1;
        i.devices[0].cc = p.major*100 + p.minor*10;
        i.devices[0].nsm = p.multiProcessorCount;
        i.devices[0].smpb = p.sharedMemPerBlock;
        i.devices[0].smpbo = p.sharedMemPerBlockOptin;
        i.devices[0].warp_size = p.warpSize;
        return i;
    }();
    return info;
}
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(4); } }
#define CK(x) ck(x)
static void require(bool b, const char * s) { if (!b) { std::fprintf(stderr, "FAIL %s\n", s); std::exit(5); } }
template <typename T> static T * alloc(size_t n) { T * p; CK(cudaMalloc(&p, n*sizeof(T))); return p; }
template <typename T> static T * upload(const std::vector<T> & v) {
    auto * p = alloc<T>(v.size()); CK(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template <typename T> static std::vector<T> download(const T * p, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static std::mt19937_64 rng(0xD1A771);
static float random_float(float s = 1.0f) { return float(int(rng()%20001) - 10000)*s/10000.0f; }
static constexpr int K = 4096, NMAX = 18, ROWS[3] = {8192, 1024, 1024}, OFF[3] = {0, 8192, 9216};
static constexpr size_t RB = (K/128)*sizeof(block_ptq1_0), BYTES = 10240*RB;
static constexpr unsigned POISON = 0xa5a5a5a5u;

static mmq_args separate_args(int rows, int stride, int n, const int * y, float * dst) {
    mmq_args a = {};
    a.type_x = GGML_TYPE_PQ2_0; a.y = y; a.dst = dst;
    a.ncols_x = K; a.nrows_x = rows; a.ncols_dst = a.ncols_y = a.ncols_max = n;
    a.stride_row_x = K/128; a.nrows_dst = stride;
    a.nchannels_x = a.nchannels_y = a.nsamples_x = a.nsamples_y = 1;
    a.stride_channel_dst = a.stride_sample_dst = stride*n;
    return a;
}
static void separate(char * dl, const int * y, int n, float ** dst, const int * stride, float * tmp, cudaStream_t stream) {
    for (int s = 0; s < 3; ++s) {
        auto a = separate_args(ROWS[s], stride[s], n, y, dst[s]);
        ggml_cuda_dl_entry e{nullptr, dl + OFF[s]*RB, ROWS[s], K, GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0};
        require(ggml_cuda_mmq_pq2_j16_grid(a, 48) == 48, "original grid changed");
        ggml_cuda_mmq_dl_run<GGML_TYPE_PQ2_0>(a, e, 48, 48, tmp, stream);
    }
}
static void errors(const std::vector<float> & ref, const std::vector<float> & got,
                   int rows, int stride, int n, bool exact, const char * label) {
    double absmax = 0, relmax = 0, peak = 0;
    size_t mismatches = 0;
    for (int col = 0; col < NMAX; ++col) for (int row = 0; row < stride; ++row) {
        const size_t i = size_t(col)*stride + row;
        if (col >= n || row >= rows) {
            unsigned b; std::memcpy(&b, &got[i], 4); require(b == POISON, "poisoned column/stride overwritten");
        } else {
            require(std::isfinite(ref[i]) && std::isfinite(got[i]), "nonfinite projection");
            double d = std::abs(double(got[i]) - ref[i]);
            absmax = std::max(absmax, d); peak = std::max(peak, std::abs(double(ref[i])));
            relmax = std::max(relmax, d/std::max(1e-6, std::abs(double(ref[i]))));
            mismatches += std::memcmp(&got[i], &ref[i], 4) != 0;
        }
    }
    std::printf("%s max_abs=%.9g max_rel=%.9g max_ref=%.9g normalized_abs=%.9g bit_mismatches=%zu\n",
                label, absmax, relmax, peak, absmax/std::max(1.0, peak), mismatches);
    require(absmax <= 2e-5*std::max(1.0, peak), "projection tolerance");
    if (exact) { require(mismatches == 0, "grouped must be bit exact"); }
}
__global__ static void previous(float * p) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const long long t = clock64(); while (clock64() - t < 2500) {}
    p[blockIdx.x*blockDim.x + threadIdx.x] = 1;
}
static float timing(const std::function<void(int)> & fn, int reps, cudaStream_t stream) {
    std::vector<float> times;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int pass = 0; pass < 7; ++pass) {
        for (int j = 0; j < 8; ++j) fn(j);
        CK(cudaEventRecord(a,stream));
        for (int j = 0; j < reps; ++j) fn(j%8);
        CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b)); times.push_back(ms*1000/reps);
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    std::sort(times.begin(),times.end()); return times[times.size()/2];
}

static void projection(cudaStream_t stream, int n, float scale, bool bench) {
    // Three independent production repacks, then the exact memcpy concatenation
    // used by registration. Check every repacked byte before the output tests.
    std::vector<char> host(size_t(10240)*(K/128)*sizeof(block_pq2_0));
    for (size_t i = 0; i < host.size(); i += 34) {
        half h = __float2half(random_float(0.06f)); std::memcpy(host.data()+i, &h, 2);
        for (int j = 0; j < 32; ++j) {
            unsigned q = 0; for (int t = 0; t < 4; ++t) q |= unsigned(rng()%3)<<(2*t);
            host[i+2+j] = char(q);
        }
    }
    char * source = upload(host), * dl[3];
    unsigned * fail = alloc<unsigned>(1); CK(cudaMemsetAsync(fail,0,4,stream));
    char * copies = alloc<char>(8*BYTES);
    for (int s = 0; s < 3; ++s) {
        dl[s] = alloc<char>(ROWS[s]*RB);
        ggml_cuda_dl_repack_ptq1(source + size_t(OFF[s])*(K/128)*34,dl[s],ROWS[s],K/128,GGML_CUDA_DL_TM,fail,stream);
        CK(cudaMemcpyAsync(copies + OFF[s]*RB,dl[s],ROWS[s]*RB,cudaMemcpyDeviceToDevice,stream));
    }
    CK(cudaStreamSynchronize(stream)); require(download(fail,1)[0] == 0,"PTQ repack round trip");
    for (int s = 0; s < 3; ++s) {
        require(download(dl[s],ROWS[s]*RB) == download(copies + OFF[s]*RB,ROWS[s]*RB),"verbatim segment");
        CK(cudaFree(dl[s]));
    }
    for (int c = 1; c < 8; ++c) CK(cudaMemcpy(copies + c*BYTES,copies,BYTES,cudaMemcpyDeviceToDevice));
    std::vector<block_q8_1_mmq> hy(size_t(K/128)*n + 16);
    for (auto & y : hy) {
        for (int j = 0; j < 4; ++j) y.d4[j] = random_float(scale);
        for (int j = 0; j < 128; ++j) y.qs[j] = int8_t(int(rng()%255)-127);
    }
    auto * y = upload(hy);
    const int stride[3] = {8192+8,1024+16,1024+24}; // all different, even
    float * ref[3], * got[3];
    for (int s = 0; s < 3; ++s) { ref[s] = alloc<float>(NMAX*stride[s]); got[s] = alloc<float>(NMAX*stride[s]); }
    auto * tmp = alloc<float>(144*2048);
    const auto poison = [&](float ** dst) {
        for (int s = 0; s < 3; ++s) CK(cudaMemsetAsync(dst[s],0xa5,NMAX*stride[s]*4,stream));
    };
    poison(ref); poison(got);
    separate(copies,(int *)y,n,ref,stride,tmp,stream);
    ggml_cuda_dod_qkv_args a{copies,(int *)y,got[0],got[1],got[2],{stride[0],stride[1],stride[2]},n};
    for (bool grouped : {true,false}) {
        poison(got);
        if (grouped) ggml_cuda_dod_qkv_launch<true>(a,tmp,stream);
        else         ggml_cuda_dod_qkv_launch<false>(a,tmp,stream);
        CK(cudaStreamSynchronize(stream));
        for (int s = 0; s < 3; ++s) {
            char label[100]; std::snprintf(label,sizeof(label),"QKV N=%d scale=%g %s segment=%d",n,scale,grouped?"grouped":"combined",s);
            errors(download(ref[s],NMAX*stride[s]),download(got[s],NMAX*stride[s]),ROWS[s],stride[s],n,grouped,label);
        }
    }
    if (bench) {
        auto * prev = alloc<float>(4096);
        const auto run = [&](int mode, int c) {
            ggml_cuda_kernel_launch(previous,ggml_cuda_kernel_launch_params(dim3(16),dim3(256),0,stream),prev);
            a.x = copies + c*BYTES;
            if (mode == 0) separate(copies + c*BYTES,(int *)y,n,ref,stride,tmp,stream);
            if (mode == 1) ggml_cuda_dod_qkv_launch<true>(a,tmp,stream);
            if (mode == 2) ggml_cuda_dod_qkv_launch<false>(a,tmp,stream);
        };
        float us[3];
        for (int mode = 0; mode < 3; ++mode) us[mode] = timing([&](int c){run(mode,c);},40,stream);
        std::printf("MICRO N=%d separate_us=%.3f grouped_us=%.3f combined_us=%.3f saved_grouped=%.3f saved_combined=%.3f (includes common stand-in)\n",
                    n,us[0],us[1],us[2],us[0]-us[1],us[0]-us[2]);
        CK(cudaFree(prev));
        // Recheck the actual timed outputs, including untouched columns.
        for (int mode : {1,2}) {
            poison(ref); poison(got); run(0,0); run(mode,0); CK(cudaStreamSynchronize(stream));
            for (int s=0;s<3;++s) errors(download(ref[s],NMAX*stride[s]),download(got[s],NMAX*stride[s]),ROWS[s],stride[s],n,mode==1,"post-timing");
        }
    }
    for (int s=0;s<3;++s) { CK(cudaFree(ref[s])); CK(cudaFree(got[s])); }
    CK(cudaFree(source)); CK(cudaFree(copies)); CK(cudaFree(y)); CK(cudaFree(fail)); CK(cudaFree(tmp));
}

template <bool IM, typename D>
static void mrope(cudaStream_t stream, int n, int dims) {
    const int cache_stride=1040, cache_rows=2*n+4;
    std::vector<float> hq(8192*n), hk(1024*n), hv(1024*n), hw(512);
    for (auto * v : {&hq,&hk,&hv}) for (float & x : *v) x=random_float(4);
    for (float & w : hw) w=1+random_float(0.5);
    std::vector<int32_t> pos(4*n); for (auto & p : pos) p=int32_t(rng()%8192);
    std::vector<int64_t> ki(n),vi(n); for(int t=0;t<n;++t){ki[t]=2*t+3;vi[t]=2*(n-1-t)+1;}
    auto * q=upload(hq), * k=upload(hk), * v=upload(hv), * w=upload(hw);
    auto * p=upload(pos); auto * kid=upload(ki); auto * vid=upload(vi);
    float * qr=alloc<float>(4096*n), * qg=alloc<float>(4096*n);
    D * kr=alloc<D>(cache_stride*cache_rows), * kg=alloc<D>(cache_stride*cache_rows);
    D * vr=alloc<D>(cache_stride*cache_rows), * vg=alloc<D>(cache_stride*cache_rows);
    for (auto * dst : {kr,kg,vr,vg}) CK(cudaMemsetAsync(dst,0xa5,cache_stride*cache_rows*sizeof(D),stream));
    rope_corr_dims corr{{1,32}}; mrope_sections sections{{16,8,8,0}};
    ggml_cuda_mrope_args qa{q,qg,256,512,8192,256,4096,1e-6f,w,dims,p,0.75f,0.25f,1.1f,corr,
        powf(1000000.f,-2.f/dims),sections,nullptr,0,nullptr,0,nullptr,nullptr,0};
    ggml_cuda_mrope_args ka{k,kg,256,256,1024,256,1024,1e-5f,w+256,dims,p,0.75f,0.25f,1.1f,corr,
        powf(1000000.f,-2.f/dims),sections,kid,cache_stride,v,1024,vg,vid,cache_stride};
    // Reference is the untouched original C62 kernel and launch helper.
    rms_norm_mul_mrope_cuda(q,qr,256,16,n,512,8192,256,4096,qa.eps,w,dims,p,
        qa.freq_scale,1000000.f,qa.ext_factor,qa.attn_factor,corr,sections,IM,nullptr,0,
        nullptr,0,(float *)nullptr,nullptr,0,stream);
    rms_norm_mul_mrope_cuda(k,kr,256,4,n,256,1024,256,1024,ka.eps,w+256,dims,p,
        ka.freq_scale,1000000.f,ka.ext_factor,ka.attn_factor,corr,sections,IM,kid,cache_stride,
        v,1024,vr,vid,cache_stride,stream);
    ggml_cuda_kernel_launch((rms_norm_mul_mrope_qk<256,IM,D>),
        ggml_cuda_kernel_launch_params(dim3(20,n),dim3(256),(32+256)*4,stream),qa,ka,16);
    CK(cudaStreamSynchronize(stream));
    const auto hqr=download(qr,4096*n), hqg=download(qg,4096*n);
    require(std::memcmp(hqr.data(),hqg.data(),hqr.size()*sizeof(float))==0,"Q norm/mrope exact");
    const auto hr=download(kr,cache_stride*cache_rows), hg=download(kg,cache_stride*cache_rows);
    const auto hvr=download(vr,cache_stride*cache_rows), hvg=download(vg,cache_stride*cache_rows);
    require(std::memcmp(hr.data(),hg.data(),hr.size()*sizeof(D))==0,"K norm/mrope/cache exact incl poison");
    require(std::memcmp(hvr.data(),hvg.data(),hvr.size()*sizeof(D))==0,"V cache exact incl poison");
    std::printf("MROPE N=%d dims=%d imrope=%d cache_bytes=%zu Q/K/V max_abs=0 max_rel=0 exact incl holes\n",n,dims,int(IM),sizeof(D));
    CK(cudaFree(q));CK(cudaFree(k));CK(cudaFree(v));CK(cudaFree(w));CK(cudaFree(p));CK(cudaFree(kid));CK(cudaFree(vid));
    CK(cudaFree(qr));CK(cudaFree(qg));CK(cudaFree(kr));CK(cudaFree(kg));CK(cudaFree(vr));CK(cudaFree(vg));
}
int main(int argc,char **argv) {
    const bool quick=argc>1 && std::strcmp(argv[1],"--quick")==0;
    require(ggml_cuda_info().devices[0].cc==GGML_CUDA_CC_DGX_SPARK && ggml_cuda_info().devices[0].nsm==48,"GB10/48 SM required");
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    for (int n : {9,13,16}) {
        projection(st,n,0.02f,!quick && n==16);
        if(!quick) projection(st,n,40.f,false);
        mrope<true,half>(st,n,256); mrope<false,float>(st,n,64);
    }
    CK(cudaStreamDestroy(st)); std::puts("DOD_QKV_ALL_PASS");
}
