#pragma once
// GB10 C16 long prefill. Original external tensors and persistent state
// remain F32; TF32 Gram and explicitly rounded F16 MMA views change internal
// arithmetic. Derived from tested f1b386e packed/early-V research implementation.
// The original scalar final16 kernel remains the sole cache-record writer.
#include <type_traits>
namespace gdn_half_c16 {
constexpr int C=16,D=128,H=32,HK=16,VC=16,QK_STRIDE=132;
struct Frame { const float *q,*k,*v,*g,*b,*bias,*a; int nt; bool head_major; int64_t sq1; int64_t sq2; int64_t sv1; int64_t sv2; };
__host__ __device__ constexpr int packed_row(int d) {
    return (d/32)*32+(d%8)*4+(d%32)/8;
}
__device__ __forceinline__ int qk_offset(Frame f,int t,int h,int d) {
    return h*f.sq1+t*f.sq2+(f.head_major?packed_row(d):d);
}
// Four committed groups, eight independent four-byte copies per group.
// The caller waits before its existing CTA publication barrier.
__device__ __forceinline__ void stage_qk_async(Frame f,int first,int h,int tid,float *q,float *k) {
#pragma unroll
    for(int row=0;row<C;++row) {
        const int t=first+row;const bool valid=t<f.nt;
        const int off=valid?qk_offset(f,t,h,tid):0;
        const unsigned dq=unsigned(__cvta_generic_to_shared(q+row*QK_STRIDE+tid));
        const unsigned dk=unsigned(__cvta_generic_to_shared(k+row*QK_STRIDE+tid));
        asm volatile("{ .reg .pred skip; setp.eq.u32 skip, %2, 0; cp.async.ca.shared.global [%0], [%1], 4, skip; }" :: "r"(dq),"l"(f.q+off),"r"(int(valid)) : "memory");
        asm volatile("{ .reg .pred skip; setp.eq.u32 skip, %2, 0; cp.async.ca.shared.global [%0], [%1], 4, skip; }" :: "r"(dk),"l"(f.k+off),"r"(int(valid)) : "memory");
        if(row%4==3)asm volatile("cp.async.commit_group;" ::: "memory");
    }
}
struct Coeff { float a[C*C], m[C*C], p[C], last[C]; };
// These coefficients are rounded to half by every carry CTA in the baseline.
// Round once at their producer, preserving the exact same conversion and bits.
struct HalfCoeff { __half a[C*C], m[C*C]; float p[C], last[C]; };
static_assert(sizeof(HalfCoeff)==1152);
__device__ __forceinline__ unsigned tf(float x) {
    unsigned y; asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(y) : "f"(x)); return y;
}
__device__ __forceinline__ void mma(float *c,const unsigned *a,const unsigned *b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
// PTX m16n8k8 TF32 fragments: A rows g,g+8,g,g+8 and columns
// t,t,t+4,t+4; B rows t,t+4, column g; C rows g,g,g+8,g+8.
template<bool X3,int K>
__device__ __forceinline__ void dot(float *c,const float *a,int ar,int ac,
                                    const float *b,int br,int bc,int mr=0,int nc=0) {
    const int l=threadIdx.x,g=l/4,t=l%4;
#pragma unroll
    for(int kk=0;kk<K;kk+=8) {
        unsigned ah[4],bh[2],al[4],bl[2];
#pragma unroll
        for(int j=0;j<4;++j) {
            const float x=a[(mr+g+(j%2)*8)*ar+(kk+t+(j/2)*4)*ac];
            ah[j]=tf(x); if constexpr(X3) al[j]=tf(x-__uint_as_float(ah[j]));
        }
#pragma unroll
        for(int j=0;j<2;++j) {
            const float x=b[(kk+t+j*4)*br+(nc+g)*bc];
            bh[j]=tf(x); if constexpr(X3) bl[j]=tf(x-__uint_as_float(bh[j]));
        }
        if constexpr(X3) { mma(c,ah,bl); mma(c,al,bh); }
        mma(c,ah,bh);
    }
}
__device__ __forceinline__ int cr(int j) { return threadIdx.x/4+(j/2)*8; }
__device__ __forceinline__ int cc(int j) { return (threadIdx.x%4)*2+j%2; }
__device__ __forceinline__ Frame select(Frame old,Frame cur,int ch,int &first) {
    if(old.nt && ch==0) { first=0; return old; }
    first=(ch-(old.nt?1:0))*C; return cur;
}
template<bool RAW>
__device__ __forceinline__ void gates(Frame f,int t,int h,float &g,float &b) {
    if(t>=f.nt) {g=1.0f;b=0.0f;return;}
    g=f.g[t*H+h];b=f.b[t*H+h];
    if constexpr(RAW) {
        b=1.0f/(1.0f+expf(-b));
        const float x=g+f.bias[h];
        g=expf(f.a[h]*(x>20.0f?x:logf(1.0f+expf(x))));
    }
}

} // namespace gdn_half_c16
namespace gdn_half_prefill {
using gdn_half_c16::C;using gdn_half_c16::D;using gdn_half_c16::H;using gdn_half_c16::HK;
using gdn_half_c16::VC;using gdn_half_c16::Frame;using gdn_half_c16::Coeff;using gdn_half_c16::HalfCoeff;
constexpr int QK=136,SMALL=24;

template<bool TRANS> __device__ __forceinline__ void load_a(unsigned *r,const __half *p,int ld,int m,int k) {
    const int lane=threadIdx.x;
    const int row=TRANS?k+lane%8+(lane/16)*8:m+lane%16;
    const int col=TRANS?m+((lane/8)%2)*8:k+(lane/16)*8;
    const unsigned a=unsigned(__cvta_generic_to_shared(p+row*ld+col));
    if constexpr(TRANS) asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
    else asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
template<bool TRANS> __device__ __forceinline__ void load_b(unsigned *r,const __half *p,int ld,int k,int n) {
    const int lane=threadIdx.x;
    const int row=TRANS?n+lane%8:k+lane%16;
    const int col=TRANS?k+((lane/8)%2)*8:n;
    const unsigned a=unsigned(__cvta_generic_to_shared(p+row*ld+col));
    if constexpr(TRANS) asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]),"=r"(r[1]):"r"(a));
    else asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]),"=r"(r[1]):"r"(a));
}
template<bool AT,bool BT,int K> __device__ __forceinline__ void dot(float *c,const __half *a,int al,const __half*b,int bl,int m=0,int n=0) {
#pragma unroll
    for(int k=0;k<K;k+=16) {
        unsigned av[4],bv[2];load_a<AT>(av,a,al,m,k);load_b<BT>(bv,b,bl,k,n);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
            : "r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),"r"(bv[0]),"r"(bv[1]));
    }
}

template<bool RAW,bool OPT_INVERSE=false,class CoeffType=Coeff> __global__ __launch_bounds__(128) void prepare_packed(Frame f,CoeffType*out,__half*packed) {
    ggml_cuda_pdl_sync();
    const int h=blockIdx.x,ch=blockIdx.y,warp=threadIdx.y,tid=warp*32+threadIdx.x,first=ch*C;
    __shared__ float q[C*gdn_half_c16::QK_STRIDE],k[C*gdn_half_c16::QK_STRIDE],kk[C*C],qk[C*C],ai[C*C],gg[C],bb[C];
    gdn_half_c16::stage_qk_async(f,first,h%HK,tid,q,k);
    if(tid<C)gdn_half_c16::gates<RAW>(f,first+tid,h,gg[tid],bb[tid]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");__syncthreads();
    // Only one of the two value-head users writes each canonical Q/K slab.
    // This is inside the measured preparation kernel, never a warmup cache.
    if(h<HK) {
        __half *pq=packed+(ch*HK+h)*2*C*D,*pk=pq+C*D;
#pragma unroll
        for(int i=2*tid;i<C*D;i+=256) {
            const int a=(i/D)*gdn_half_c16::QK_STRIDE+i%D;
            *reinterpret_cast<__half2*>(pq+i)=__floats2half2_rn(q[a],q[a+1]);
            *reinterpret_cast<__half2*>(pk+i)=__floats2half2_rn(k[a],k[a+1]);
        }
    }
    float z[4]={0,0,0,0};
    gdn_half_c16::dot<false,D>(z,warp<2?k:q,gdn_half_c16::QK_STRIDE,1,k,1,gdn_half_c16::QK_STRIDE,0,(warp%2)*8);
#pragma unroll
    for(int j=0;j<4;++j) {
        const int row=gdn_half_c16::cr(j),col=(warp%2)*8+gdn_half_c16::cc(j);
        if constexpr(OPT_INVERSE) {
            // Exactly the original rounded product, computed once per Gram
            // cell instead of once for every inverse column that consumes it.
            (warp<2?kk:qk)[row*C+col]=warp<2?__fmul_rn(-bb[row],z[j]):z[j];
        } else (warp<2?kk:qk)[row*C+col]=z[j];
    }
    __syncthreads();
    // Each inverse column belongs to one lane. Preserve original ascending
    // FMA order but keep its previous rows private instead of shared+syncwarp.
    if(warp==0&&threadIdx.x<C) {
        const int j=threadIdx.x;float inv[C];
#pragma unroll
        for(int r=0;r<C;++r)if constexpr(!OPT_INVERSE)inv[r]=r==j?1.0f:0.0f;
        if constexpr(OPT_INVERSE)inv[0]=j==0?1.0f:0.0f;
#pragma unroll
        for(int i=1;i<C;++i) {
            float s=0;
#pragma unroll
            for(int r=0;r<C;++r)if(j<=r&&r<i) {
                if constexpr(OPT_INVERSE)s=fmaf(kk[i*C+r],inv[r],s);
                else s=fmaf(-bb[i]*kk[i*C+r],inv[r],s);
            }
            if constexpr(OPT_INVERSE)inv[i]=j<i?s:j==i?1.0f:0.0f;
            else if(j<i)inv[i]=s;
        }
#pragma unroll
        for(int r=0;r<C;++r)ai[r*C+j]=inv[r];
    }
    __syncthreads();CoeffType*c=out+ch*H+h;
    for(int e=tid;e<C*C;e+=128){const int i=e/C,j=e%C;float decay=1;
        if(j<=i)for(int r=j+1;r<=i;++r)decay*=gg[r];else decay=0;
        if constexpr(std::is_same<CoeffType,HalfCoeff>::value) {
            c->a[e]=__float2half_rn((ai[e]*decay)*bb[j]);c->m[e]=__float2half_rn(qk[e]*decay);
        } else { c->a[e]=(ai[e]*decay)*bb[j];c->m[e]=qk[e]*decay; }
    }
    if(tid<C){float p=1,last=1;for(int i=0;i<=tid;++i)p*=gg[i];for(int i=tid+1;i<C;++i)last*=gg[i];c->p[tid]=p;c->last[tid]=last;}
}

template<bool EARLY_V=false,bool WARP_R=false,class CoeffType=Coeff>
__global__ __launch_bounds__(128) void carry_packed(Frame f,const CoeffType*coef,const __half*packed,
        const float*initial,float*output,float*final_state,float*boundary,float scale) {
    ggml_cuda_pdl_sync();
    const int h=blockIdx.x,v0=blockIdx.y*VC,w=threadIdx.y,tid=w*32+threadIdx.x;
    __shared__ __align__(16) __half q[C*QK],k[C*QK],hs[D*SMALL],a[C*SMALL],m[C*SMALL],r[C*SMALL],delta[C*SMALL];
    __shared__ float oo[C*VC],p[C],last[C];
    float s[4][4];
#pragma unroll
    for(int t=0;t<4;++t){const int tile=w+t*4,mr=(tile/2)*16,n=(tile%2)*8;
#pragma unroll
        for(int j=0;j<4;++j)s[t][j]=initial[(h*D+v0+n+gdn_half_c16::cc(j))*D+mr+gdn_half_c16::cr(j)];
    }
    for(int ch=0;ch<(f.nt+C-1)/C;++ch){const int first=ch*C;const CoeffType*c=coef+ch*H+h;
        const __half*pq=packed+(ch*HK+h%HK)*2*C*D,*pk=pq+C*D;
#pragma unroll
        for(int i=tid*8;i<C*D;i+=128*8){const int dst=(i/D)*QK+i%D;
            const unsigned aq=unsigned(__cvta_generic_to_shared(q+dst)),ak=unsigned(__cvta_generic_to_shared(k+dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(aq),"l"(pq+i):"memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(ak),"l"(pk+i):"memory");
        }
        asm volatile("cp.async.commit_group;":::"memory");
        float early_v[4];
        if constexpr(EARLY_V)if(w<2) {
#pragma unroll
            for(int j=0;j<4;j+=2){const int token=first+gdn_half_c16::cr(j),vc=w*8+gdn_half_c16::cc(j);
                early_v[j]=early_v[j+1]=0.0f;
                if(token<f.nt){const float*vp=f.v+token*f.sv2+h*f.sv1+v0+vc;
                    // Keep ordinary global-load semantics and exact F32 bits.
                    // Volatile PTX prevents sinking these loads back to the
                    // residual use, after the KS dependency chain.
                    // Adjacent C-fragment values share token and an aligned
                    // float2 address. One vector load preserves both F32 bits.
                    asm volatile("ld.global.v2.f32 {%0,%1}, [%2];":"=f"(early_v[j]),"=f"(early_v[j+1]):"l"(vp):"memory");}
            }
        }
        for(int i=tid;i<C*C;i+=128){
            if constexpr(std::is_same<CoeffType,HalfCoeff>::value) {
                a[(i/C)*SMALL+i%C]=c->a[i];m[(i/C)*SMALL+i%C]=c->m[i];
            } else {a[(i/C)*SMALL+i%C]=__float2half_rn(c->a[i]);m[(i/C)*SMALL+i%C]=__float2half_rn(c->m[i]);}
        }
        if(tid<C){p[tid]=c->p[tid];last[tid]=c->last[tid];}
#pragma unroll
        for(int t=0;t<4;++t){const int tile=w+t*4,mr=(tile/2)*16,n=(tile%2)*8;
#pragma unroll
            for(int j=0;j<4;++j){const int kr=mr+gdn_half_c16::cr(j),vc=n+gdn_half_c16::cc(j);hs[kr*SMALL+vc]=__float2half_rn(s[t][j]);if(boundary && ch==(f.nt-C)/C)boundary[(h*D+v0+vc)*D+kr]=s[t][j];}
        }
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        float z[4]={0,0,0,0};dot<false,false,D>(z,w<2?k:q,QK,hs,SMALL,0,(w%2)*8);
#pragma unroll
        for(int j=0;j<4;++j){const int tr=gdn_half_c16::cr(j),vc=(w%2)*8+gdn_half_c16::cc(j),token=first+tr;
            if(w<2){float vv;if constexpr(EARLY_V)vv=early_v[j];else vv=token<f.nt?f.v[token*f.sv2+h*f.sv1+v0+vc]:0.0f;r[tr*SMALL+vc]=__float2half_rn(vv-p[tr]*z[j]);}
            else oo[tr*VC+vc]=p[tr]*z[j];
        }
        if constexpr(WARP_R){if(w<2)__syncwarp();}else __syncthreads();
        if(w<2){float dd[4]={0,0,0,0};dot<false,false,C>(dd,a,SMALL,r,SMALL,0,w*8);
#pragma unroll
            for(int j=0;j<4;++j){const int tr=gdn_half_c16::cr(j),vc=w*8+gdn_half_c16::cc(j);delta[tr*SMALL+vc]=__float2half_rn(dd[j]);
                // Each warp finished reading only its own eight R columns.
                // Reuse them for F32(last*Delta) rounded once to half. The
                // existing publication barrier below protects both readers.
                r[tr*SMALL+vc]=__float2half_rn(last[tr]*dd[j]);}
        }
        __syncthreads();
        if(w<2){float o[4];
#pragma unroll
            for(int j=0;j<4;++j)o[j]=oo[gdn_half_c16::cr(j)*VC+w*8+gdn_half_c16::cc(j)];
            dot<false,false,C>(o,m,SMALL,delta,SMALL,0,w*8);
#pragma unroll
            for(int j=0;j<4;++j)if(first+gdn_half_c16::cr(j)<f.nt)output[((first+gdn_half_c16::cr(j))*H+h)*D+v0+w*8+gdn_half_c16::cc(j)]=o[j]*scale;
        }
#pragma unroll
        for(int t=0;t<4;++t){const int tile=w+t*4,mr=(tile/2)*16,n=(tile%2)*8;
#pragma unroll
            for(int j=0;j<4;++j)s[t][j]*=p[C-1];
            dot<true,false,C>(s[t],k,QK,r,SMALL,mr,n);
        }
        __syncthreads();
    }
#pragma unroll
    for(int t=0;t<4;++t){const int tile=w+t*4,mr=(tile/2)*16,n=(tile%2)*8;
#pragma unroll
        for(int j=0;j<4;++j)final_state[(h*D+v0+n+gdn_half_c16::cc(j))*D+mr+gdn_half_c16::cr(j)]=s[t][j];
    }
}
} // namespace gdn_half_prefill

// Materialize the exact selected initial state, including GD1/GD2 NOFINAL
// replay and the existing deferred SCALE, before changing its numerical method.
static __global__ void gdn_half_initial_state(
        const float * current, const float * state, float * initial, int64_t slot_stride,
        const int32_t * ids, int64_t row_stride, int lazy,
        const float * zero_row, float zero_scale, float zero_bias) {
    ggml_cuda_pdl_sync();
    const int h=blockIdx.x,lane=threadIdx.x;
    const int col=(blockIdx.z*4+threadIdx.y)*4;
    const int64_t selected=ids?int64_t(ids[0])*row_stride:0;
    const bool zeroed=zero_row && current+selected==zero_row;
    current+=selected+int64_t(h)*128*128;
    state+=int64_t(h)*128*128;
    float shard[4][4];
    const bool replayed=lazy && !zeroed &&
        gdn_lazy_replay<128,32,4,4>(shard,current,state,slot_stride,16,col,lane);
#pragma unroll
    for(int c=0;c<4;++c) {
#pragma unroll
        for(int r=0;r<4;++r) {
            const int row=r*32+lane;
            float value=replayed?shard[c][r]:current[(col+c)*128+row];
            if(zeroed)value=zero_scale*value+zero_bias;
            initial[(int64_t(h)*128+col+c)*128+row]=value;
        }
    }
}

template<bool RAW,bool PRE,bool PACKED>
static void gdn_half_prefill_launch(
        ggml_backend_cuda_context & ctx,const float * q,const float * k,const float * v,
        const float * g,const float * beta,const float * bias,const float * a,
        const float * current,float * output,float * state,int T,
        int64_t sq1,int64_t sq2,int64_t sq3,int64_t sv1,int64_t sv2,int64_t sv3,
        int64_t sb1,int64_t sb2,int64_t sb3,float scale,int64_t slot_stride,
        const int32_t * ids,int64_t row_stride,int lazy,
        const float * zero_row,float zero_scale,float zero_bias,cudaStream_t stream) {
    static_assert(!RAW && PRE,"first production experiment requires precomputed gates");
    const int prefix=T-16;
    ggml_cuda_pool_alloc<float> initial(ctx.pool(),size_t(32)*128*128);
    ggml_cuda_pool_alloc<gdn_half_c16::HalfCoeff> coefficients(ctx.pool(),size_t(32)*(prefix/16));
    ggml_cuda_pool_alloc<__half> packed(ctx.pool(),size_t(prefix)*16*128*2);
    const ggml_cuda_kernel_launch_params mat(dim3(32,1,8),dim3(32,4),0,stream);
    ggml_cuda_kernel_launch(gdn_half_initial_state,mat,current,state,initial.ptr,slot_stride,
        ids,row_stride,lazy,zero_row,zero_scale,zero_bias);
    const gdn_half_c16::Frame frame{q,k,v,g,beta,bias,a,prefix,PACKED,sq1,sq2,sv1,sv2};
    const ggml_cuda_kernel_launch_params prep(dim3(32,prefix/16),dim3(32,4),0,stream);
    ggml_cuda_kernel_launch(gdn_half_prefill::prepare_packed<false,false,gdn_half_c16::HalfCoeff>,prep,frame,coefficients.ptr,packed.ptr);
    const ggml_cuda_kernel_launch_params carry(dim3(32,8),dim3(32,4),0,stream);
    ggml_cuda_kernel_launch(gdn_half_prefill::carry_packed<true,false,gdn_half_c16::HalfCoeff>,carry,frame,coefficients.ptr,packed.ptr,
        initial.ptr,output,state,(float *)nullptr,scale);
    // Same scalar tail as N1. It creates every snapshot/header/checksum from
    // the new rounded boundary; later rollback readers retain their protocol.
    ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<false,true,PACKED,true>,mat,
        q+prefix*sq2,k+prefix*sq2,v+prefix*sv2,g+prefix*sb2,beta+prefix*sb2,
        bias,a,state,output+int64_t(prefix)*32*128,state,
        int64_t(32),int64_t(16),int64_t(1),sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,
        init_fastdiv_values(16),init_fastdiv_values(1),scale,slot_stride,16,
        (const int32_t *)nullptr,int64_t(0),lazy,(const float *)nullptr,0.0f,0.0f);
}
