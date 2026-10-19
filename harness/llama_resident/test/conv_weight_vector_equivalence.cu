// Frozen promoted203 kernel vs guarded candidate216, same executable/allocations.
#include "ssm-conv.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do {auto ce=(x);if(ce!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",#x,__LINE__,cudaGetErrorString(ce));exit(2);}}while(0)
int ggml_cuda_get_device(){int d=-1;CK(cudaGetDevice(&d));return d;}
[[noreturn]] void ggml_cuda_error(const char*s,const char*f,const char*p,int l,const char*m){fprintf(stderr,"%s %s %s %s:%d\n",s,m,f,p,l);abort();}
template <bool warp_rows>
static __global__ void __launch_bounds__(128) reference216_short_f32(
        const float * states, const int32_t * gather_ids, const int64_t gather_row, const float * w,
        float * y, float * qk, const __grid_constant__ ssm_conv_front_short_snaps plan,
        const int64_t C, const int64_t c_v0, const int n_chunks, const int64_t n_t,
        const __grid_constant__ ssm_conv_front_x xs,
        const int64_t w_c, const int64_t y_t, const int hold, const float eps,
        const float * zero_row, const float zero_scale, const float zero_bias) {
    constexpr int tile = 4, head = 128, prev = 3, threads = 128;
    union storage {
        struct { float state[32 * prev]; float held[4][GGML_CUDA_SSM_CONV_MAX_HOLD][32]; } v;
        struct { float input[tile + prev][head]; float alias[prev][head]; float out[warp_rows ? 1 : tile][head]; } q;
    };
    __shared__ storage sm;
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (gather_ids) states += (int64_t) gather_ids[0] * gather_row;
    const bool zeroed = zero_row && states == zero_row;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    // Like ggml_cuda_gather_rows_kernel: indexed parameter tables must not become a private stack copy.
    const auto & snaps = plan.snaps;
    const int n_qkb = (int) (c_v0 / head) * n_chunks;

    if ((int) blockIdx.x >= n_qkb) {
        // The existing v algorithm: one owner per 32 channels, staged state/snapshots and the same alias holds.
        const int64_t c0 = c_v0 + ((int) blockIdx.x - n_qkb) * 32;
        const int64_t c = c0 + lane, seg_len = (n_t + 3) / 4;
        const int64_t t0 = warp * seg_len, t1 = min(n_t, t0 + seg_len);
        const bool run = c < C && t0 < t1;
        for (int e = tid; e < 32 * prev; e += threads) {
            if (c0 + e / prev < C) {
                const float a = states[c0 * prev + e];
                sm.v.state[e] = zeroed ? zero_scale * a + zero_bias : a;
            }
        }
        __syncthreads();
        for (int e = tid; e < snaps.n * 32 * prev; e += threads) {
            const int k = e / (32 * prev), r = e % (32 * prev);
            const int64_t cs = c0 + r / prev;
            if (cs < C) snaps.dst[k][cs * prev + r % prev] =
                ssm_conv_front_input(sm.v.state, c0, xs, cs, snaps.col[k] + r % prev, prev);
        }
        float wr[4], win[prev];
        if (run) {
#pragma unroll
            for (int j = 0; j < 4; ++j) wr[j] = w[c * w_c + j];
#pragma unroll
            for (int j = 0; j < prev; ++j)
                win[j] = ssm_conv_front_input(sm.v.state, c0, xs, c, t0 + j, prev);
        }
        __syncthreads();
        const auto step = [&](const int64_t t, const float xt) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < prev; ++j) sumf += win[j] * wr[j];
            sumf += xt * wr[prev];
            sumf += 0.0f;
#pragma unroll
            for (int j = 0; j < prev - 1; ++j) win[j] = win[j + 1];
            win[prev - 1] = xt;
            const float a = ggml_cuda_op_silu_single(sumf);
            if (t - t0 < hold) sm.v.held[warp][t - t0][lane] = a;
            else y[t * y_t + c] = a;
        };
        if (run) {
            int64_t t = t0;
            for (; t + 8 <= t1; t += 8) {
                float xu[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) xu[u] = xs(t + u, c);
#pragma unroll
                for (int u = 0; u < 8; ++u) step(t + u, xu[u]);
            }
            for (; t < t1; ++t) step(t, xs(t, c));
        }
        if (hold > 0) {
            __syncthreads();
            if (run) for (int h = 0; h < hold && t0 + h < t1; ++h)
                y[(t0 + h) * y_t + c] = sm.v.held[warp][h][lane];
        }
        return;
    }

    const int h = blockIdx.x / n_chunks, ch = blockIdx.x % n_chunks;
    const int64_t c0 = (int64_t) h * head, c = c0 + tid, tc0 = ch * tile;
    const int nt = (int) min((int64_t) tile, n_t - tc0);
    // Every later chunk starts at conv-input column >=4: it never reads cached state, even for snapshots.
    float xin[tile + prev];
#pragma unroll
    for (int j = 0; j < tile + prev; ++j) {
        const int64_t col = tc0 + j;
        float a = 0.0f;
        if (col < prev) {
            const float v = states[c * prev + col];
            a = zeroed ? zero_scale * v + zero_bias : v;
        } else if (col < n_t + prev) a = xs(col - prev, c);
        xin[j] = a;
    }
    // Only chunk 0 needs the extra alias slot. Ballot distributes the pointer comparisons over a warp;
    // no block barrier or global counter is needed to discover which cache row is being overwritten.
    uint32_t alias_mask = 0;
    int alias_idx = -1, alias_col = -1;
    if (ch == 0) {
        alias_mask = __ballot_sync(0xffffffff, lane < snaps.n && snaps.dst[lane] == states);
        alias_idx = __ffs(alias_mask) - 1;
        if (alias_idx >= 0) alias_col = snaps.col[alias_idx];
        if (alias_col > tile) {
#pragma unroll
            for (int j = 0; j < prev; ++j) sm.q.alias[j][tid] = xs(alias_col + j - prev, c);
        }
    }
#pragma unroll
    for (int j = 0; j < tile + prev; ++j) sm.q.input[j][tid] = xin[j];

    // All global input/weight loads are issued before the first convolution FMA.
    float wr[warp_rows ? 4 : 1][4];
    if constexpr (warp_rows) {
        if (warp < nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wr[i][j] = w[(c0 + i * 32 + lane) * w_c + j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) wr[0][j] = w[c * w_c + j];
#pragma unroll
        for (int t = 0; t < tile; ++t) {
            if (t < nt) {
                float sumf = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) sumf += xin[t + j] * wr[0][j];
                sumf += 0.0f;
                sm.q.out[t][tid] = ggml_cuda_op_silu_single(sumf);
            }
        }
    }
    // One rendezvous for the q/k CTA: publish input/SiLU tiles and finish every cached-state read before
    // any alias snapshot store. There is no block-wide barrier between the L2 and snapshot phases.
    __syncthreads();
    if (warp < nt) {
        float values[4];
        if constexpr (warp_rows) {
            float xrow[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) xrow[i][j] = sm.q.input[warp + j][i * 32 + lane];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float sumf = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) sumf += xrow[i][j] * wr[i][j];
                sumf += 0.0f;
                values[i] = ggml_cuda_op_silu_single(sumf);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) values[i] = sm.q.out[warp][i * 32 + lane];
        }
        float tmp = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float xi = values[i]; tmp += xi * xi; }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
        for (int i = 0; i < 4; ++i)
            qk[(tc0 + warp) * c_v0 + c0 + i * 32 + lane] = scale * values[i];
    }

    uint32_t todo = plan.mask[ch] | alias_mask;
    int last_col = -1;
    float sv[prev];
    while (todo) {
        const int k = __ffs(todo) - 1;
        todo &= todo - 1;
        if (ch != 0 && snaps.dst[k] == states) continue;  // chunk 0 owns the alias slot
        const int col = snaps.col[k];
        if (col != last_col) {
#pragma unroll
            for (int u = 0; u < prev; ++u) {
                const int e = tid + u * threads, cs = e / prev, j = e % prev;
                sv[u] = ch == 0 && col > tile ? sm.q.alias[j][cs] : sm.q.input[col - tc0 + j][cs];
            }
            last_col = col;
        }
        // Coalesced stores, and reuse the three loaded values for repeated columns (15 identical slots at T=1).
#pragma unroll
        for (int u = 0; u < prev; ++u) snaps.dst[k][c0 * prev + tid + u * threads] = sv[u];
    }
}

static cudaStream_t stream;
static void launch(bool old,bool rows,const float*state,const int32_t*id,const float*w,float*y,float*qk,ssm_conv_front_short_snaps plan,int C,int cv,int T,ssm_conv_front_x xs,int wc,int hold,const float*zero){
 int chunks=(T+3)/4,blocks=(C-cv+31)/32+cv/128*chunks;
 const ggml_cuda_kernel_launch_params lp(dim3(blocks),dim3(32,4),0,stream);
#define RUN(K) ggml_cuda_kernel_launch(K,lp,state,id,int64_t(C*3),w,y,qk,plan,int64_t(C),int64_t(cv),chunks,int64_t(T),xs,int64_t(wc),int64_t(C),hold,1.e-6f,zero,0.f,0.125f)
 if(old){if(rows){RUN(reference216_short_f32<true>);}else{RUN(reference216_short_f32<false>);}}
 else if(wc==4&&(uintptr_t)w%16==0){if(rows){RUN((ssm_conv_front_l2_short_f32<true,true>));}else{RUN((ssm_conv_front_l2_short_f32<false,true>));}}
 else {if(rows){RUN((ssm_conv_front_l2_short_f32<true,false>));}else{RUN((ssm_conv_front_l2_short_f32<false,false>));}}
#undef RUN
 CK(cudaGetLastError());
}
static void equal(const void*a,const void*b,size_t n,int test,const char*what){if(memcmp(a,b,n)){auto x=(const unsigned char*)a;auto y=(const unsigned char*)b;size_t i=0;while(i<n&&x[i]==y[i])++i;fprintf(stderr,"FAIL case%d %s byte%zu %02x!=%02x\n",test,what,i,x[i],y[i]);exit(1);}}
static float time_graph(cudaGraphExec_t graph){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int i=0;i<8;++i)CK(cudaGraphLaunch(graph,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/128;}
static void check(int T,int rows,int variant,int datamode,int&cases,bool timing){
 const int C=timing?8192:512,cv=timing?2048:256,n=(variant&1)?32:4,hold=(variant&2)?1:0,alias=(variant&4)?1:0,wc=(variant&8)?5:4,woff=(variant&16)?1:0;
 const size_t X=(T+8)*C+64,S=6*C+64,Y=(T+8)*C+64,Q=T*cv+64,P=n*3*C+64,W=wc*C+64;
 std::mt19937 rng(216+cases);std::vector<float>hx(X),hs(S),hw(W);for(auto&v:hx)v=float(int(rng()%8193)-4096)/2048;for(auto&v:hs)v=float(int(rng()%8193)-4096)/2048;for(auto&v:hw)v=float(int(rng()%513)-256)/256;
 if(datamode==1){uint32_t vals[]={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0x7f812345};for(auto&v:hx){uint32_t u=vals[rng()%8];memcpy(&v,&u,4);}for(auto&v:hs){uint32_t u=vals[rng()%8];memcpy(&v,&u,4);}}
 float*x[2],*s[2],*y[2],*q[2],*p[2],*w;int32_t*id;int32_t index=1;CK(cudaMalloc(&id,4));CK(cudaMemcpy(id,&index,4,cudaMemcpyHostToDevice));CK(cudaMalloc(&w,W*4));CK(cudaMemcpy(w,hw.data(),W*4,cudaMemcpyHostToDevice));ssm_conv_front_short_snaps plan[2]={};
 for(int i=0;i<2;++i){CK(cudaMalloc(&x[i],X*4));CK(cudaMalloc(&s[i],S*4));CK(cudaMalloc(&y[i],Y*4));CK(cudaMalloc(&q[i],Q*4));CK(cudaMalloc(&p[i],P*4));CK(cudaMemcpy(x[i],hx.data(),X*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(s[i],hs.data(),S*4,cudaMemcpyHostToDevice));CK(cudaMemset(y[i],0xa5,Y*4));CK(cudaMemset(q[i],0xa5,Q*4));CK(cudaMemset(p[i],0xa5,P*4));plan[i].snaps.n=n;plan[i].snaps.width=3;
  for(int k=0;k<n;++k){int col=k*T/(n-1);if(variant&1)col=T-col;plan[i].snaps.col[k]=col;plan[i].snaps.dst[k]=(alias&&k==1)?s[i]+C*3:p[i]+k*C*3;int owner=col>0?(col-1)/4:0;plan[i].mask[owner]|=uint32_t(1)<<k;}}
 auto run=[&](int i){ssm_conv_front_x xs={x[i]+4*C,nullptr,T,int64_t(C),1,int64_t(C)};launch(i==0,rows,s[i],id,w+woff,hold?x[i]+(4-hold)*C:y[i],q[i],plan[i],C,cv,T,xs,wc,hold,datamode==2?s[i]+C*3:nullptr);};
 CK(cudaDeviceSynchronize()); // Default-stream initialization must finish before the nonblocking test stream.
 for(int i=0;i<2;++i)run(i);CK(cudaStreamSynchronize(stream));
 for(int j=0;j<5;++j){size_t count=j==0?X:j==1?S:j==2?Y:j==3?Q:P;float*a=j==0?x[0]:j==1?s[0]:j==2?y[0]:j==3?q[0]:p[0];float*b=j==0?x[1]:j==1?s[1]:j==2?y[1]:j==3?q[1]:p[1];std::vector<float>u(count),v(count);CK(cudaMemcpy(u.data(),a,count*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(v.data(),b,count*4,cudaMemcpyDeviceToHost));if(memcmp(u.data(),v.data(),count*4))fprintf(stderr,"CONTEXT T=%d rows=%d variant=%d mode=%d buffer=%d\n",T,rows,variant,datamode,j);equal(u.data(),v.data(),count*4,cases,"all buffers and guards");}
 if(timing){cudaGraphExec_t g[2];for(int i=0;i<2;++i){cudaGraph_t h;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int j=0;j<16;++j)run(i);CK(cudaStreamEndCapture(stream,&h));CK(cudaGraphInstantiate(&g[i],h,nullptr,nullptr,0));CK(cudaGraphDestroy(h));}
 for(int rep=0;rep<10;++rep){float a,b;if(rep%2){b=time_graph(g[1]);a=time_graph(g[0]);}else{a=time_graph(g[0]);b=time_graph(g[1]);}if(rep>=2)printf("GRAPH T=%d rows=%d rep=%d old_us=%.6f new_us=%.6f gain=%.6f\n",T,rows,rep,a,b,a/b);}for(auto h:g)CK(cudaGraphExecDestroy(h));}
 ++cases;for(int i=0;i<2;++i){CK(cudaFree(x[i]));CK(cudaFree(s[i]));CK(cudaFree(y[i]));CK(cudaFree(q[i]));CK(cudaFree(p[i]));}CK(cudaFree(w));CK(cudaFree(id));
}
int main(){CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));int cases=0;for(int T:{1,4,9,16,31,63})for(int rows=0;rows<2;++rows)for(int variant:{0,1,2,4,5,6,8,16})for(int data=0;data<3;++data)check(T,rows,variant,data,cases,false);printf("CONV_WEIGHT_VECTOR_EQUIVALENCE_PASS cases=%d all_buffers_exact=true\n",cases);for(int T:{1,9,16,31})for(int rows=0;rows<2;++rows)check(T,rows,0,0,cases,true);CK(cudaStreamDestroy(stream));}
