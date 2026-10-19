#pragma once
// Research C32: F16 Gram operands, F32 diagonal inverses and TF32x3 lower
// inverse products. Persistent state/output and the scalar cache tail stay F32.
namespace gdn_half_c32 {
using gdn_half_c16::Frame;
constexpr int C=32,D=128,H=32,HK=16,VC=32,QK=136,SMALL=40,FS=36,TS=20;
struct Coeff { float a[C*C],m[C*C],p[C],last[C]; };
using gdn_half_c16::cr;using gdn_half_c16::cc;
using gdn_half_prefill::dot;

static __global__ __launch_bounds__(256,3) void prepare(Frame f,Coeff*out,__half*packed) {
    ggml_cuda_pdl_sync();
    const int h=blockIdx.x,ch=blockIdx.y,w=threadIdx.y,tid=w*32+threadIdx.x,first=ch*C;
    __shared__ __align__(16) __half q[C*QK],k[C*QK];
    __shared__ float kk[C*FS],qk[C*FS],inv[C*FS],temp[16*TS],g[C],b[C];
    for(int i=tid;i<C*D;i+=256) {
        const int row=i/D,d=i%D,t=first+row;
        const int64_t at=t<f.nt?gdn_half_c16::qk_offset(f,t,h%HK,d):0;
        q[row*QK+d]=__float2half_rn(t<f.nt?f.q[at]:0.0f);
        k[row*QK+d]=__float2half_rn(t<f.nt?f.k[at]:0.0f);
    }
    if(tid<C)gdn_half_c16::gates<false>(f,first+tid,h,g[tid],b[tid]);
    for(int i=tid;i<C*FS;i+=256)inv[i]=0.0f;
    __syncthreads();
    if(h<HK) {
        __half *pq=packed+(ch*HK+h)*2*C*D,*pk=pq+C*D;
        for(int i=2*tid;i<C*D;i+=512) {
            const int at=(i/D)*QK+i%D;
            *reinterpret_cast<__half2*>(pq+i)=*reinterpret_cast<const __half2*>(q+at);
            *reinterpret_cast<__half2*>(pk+i)=*reinterpret_cast<const __half2*>(k+at);
        }
    }
    const int mr=(w/4)*16,nc=(w%4)*8;
    float z[4]={},y[4]={};
    dot<false,true,D>(z,k,QK,k,QK,mr,nc);
    dot<false,true,D>(y,q,QK,k,QK,mr,nc);
#pragma unroll
    for(int j=0;j<4;++j) {
        const int row=mr+cr(j),col=nc+cc(j);
        kk[row*FS+col]=row>col?__fmul_rn(-b[row],z[j]):0.0f;
        qk[row*FS+col]=y[j];
    }
    __syncthreads();
    // Two independent original-order 16x16 diagonal inverses.
    if(w<2&&threadIdx.x<16) {
        const int base=w*16,col=threadIdx.x;float v[16];
#pragma unroll
        for(int i=0;i<16;++i)v[i]=i==col?1.0f:0.0f;
#pragma unroll
        for(int i=1;i<16;++i) {
            float s=0.0f;
#pragma unroll
            for(int r=0;r<16;++r)if(col<=r&&r<i)s=fmaf(kk[(base+i)*FS+base+r],v[r],s);
            if(col<i)v[i]=s;
        }
#pragma unroll
        for(int i=0;i<16;++i)inv[(base+i)*FS+base+col]=v[i];
    }
    __syncthreads();
    // inv10 = inv11 * L10 * inv00. Keep the cross product F32 via TF32x3.
    if(w<2) {
        float t[4]={};
        gdn_half_c16::dot<true,16>(t,kk+16*FS,FS,1,inv,FS,1,0,w*8);
#pragma unroll
        for(int j=0;j<4;++j)temp[cr(j)*TS+w*8+cc(j)]=t[j];
    }
    __syncthreads();
    if(w<2) {
        float t[4]={};
        gdn_half_c16::dot<true,16>(t,inv+16*FS+16,FS,1,temp,TS,1,0,w*8);
#pragma unroll
        for(int j=0;j<4;++j)inv[(16+cr(j))*FS+w*8+cc(j)]=t[j];
    }
    __syncthreads();
    Coeff*c=out+ch*H+h;
    for(int i=tid;i<C*C;i+=256) {
        const int row=i/C,col=i%C;float decay=1.0f;
        if(col<=row)for(int r=col+1;r<=row;++r)decay*=g[r];else decay=0.0f;
        c->a[i]=(inv[row*FS+col]*decay)*b[col];
        c->m[i]=qk[row*FS+col]*decay;
    }
    if(tid<C) {
        float p=1.0f,last=1.0f;
        for(int i=0;i<=tid;++i)p*=g[i];
        for(int i=tid+1;i<C;++i)last*=g[i];
        c->p[tid]=p;c->last[tid]=last;
    }
}

union __align__(16) OperandStorage {
    struct {__half q[C*QK],k[C*QK];} pair;
    float initial[VC*D];
};
static_assert(sizeof(OperandStorage)==2*C*QK*sizeof(__half),"C32 import reuses Q/K storage");
template<bool FUSED_INITIAL=false>
__global__ __launch_bounds__(256,2) void carry(Frame f,const Coeff*coef,const __half*packed,
        const float*initial,float*output,float*state,float scale,gdn_half_prefill::InitialState init) {
    ggml_cuda_pdl_sync();
    const int h=FUSED_INITIAL?blockIdx.x/4:blockIdx.x,v0=(FUSED_INITIAL?blockIdx.x%4:blockIdx.y)*VC,w=threadIdx.y,tid=w*32+threadIdx.x;
    __shared__ OperandStorage operands;
    __half* q=operands.pair.q;__half* k=operands.pair.k;
    __shared__ __align__(16) __half hs[D*SMALL];
    __shared__ __align__(16) __half a[C*SMALL],m[C*SMALL],r[C*SMALL],delta[C*SMALL],last_delta[C*SMALL];
    __shared__ float oo[C*VC],p[C],last[C];
    if constexpr(FUSED_INITIAL){gdn_half_prefill::read_initial(init,operands.initial,h,v0);__syncthreads();}
    float s[4][4];
#pragma unroll
    for(int t=0;t<4;++t) {
        const int tile=w+t*8,mr=(tile/4)*16,n=(tile%4)*8;
#pragma unroll
        for(int j=0;j<4;++j) {
            if constexpr(FUSED_INITIAL)s[t][j]=operands.initial[(n+cc(j))*D+mr+cr(j)];
            else s[t][j]=initial[(h*D+v0+n+cc(j))*D+mr+cr(j)];
        }
    }
    if constexpr(FUSED_INITIAL)__syncthreads();
    for(int ch=0;ch<(f.nt+C-1)/C;++ch) {
        const int first=ch*C,mr=(w/4)*16,nc=(w%4)*8;
        const Coeff*c=coef+ch*H+h;
        const __half*pq=packed+(ch*HK+h%HK)*2*C*D,*pk=pq+C*D;
        for(int i=tid*8;i<C*D;i+=256*8) {
            const int at=(i/D)*QK+i%D;
            const unsigned aq=unsigned(__cvta_generic_to_shared(q+at));
            const unsigned ak=unsigned(__cvta_generic_to_shared(k+at));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(aq),"l"(pq+i):"memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(ak),"l"(pk+i):"memory");
        }
        asm volatile("cp.async.commit_group;":::"memory");
        float vv[4]={};
#pragma unroll
        for(int j=0;j<4;j+=2) {
            const int token=first+mr+cr(j),vc=nc+cc(j);
            if(token<f.nt) {
                const float*vp=f.v+token*f.sv2+h*f.sv1+v0+vc;
                asm volatile("ld.global.v2.f32 {%0,%1}, [%2];":"=f"(vv[j]),"=f"(vv[j+1]):"l"(vp):"memory");
            }
        }
        for(int i=tid;i<C*C;i+=256) {
            a[(i/C)*SMALL+i%C]=__float2half_rn(c->a[i]);
            m[(i/C)*SMALL+i%C]=__float2half_rn(c->m[i]);
        }
        if(tid<C){p[tid]=c->p[tid];last[tid]=c->last[tid];}
#pragma unroll
        for(int t=0;t<4;++t) {
            const int tile=w+t*8,kr=(tile/4)*16,n=(tile%4)*8;
#pragma unroll
            for(int j=0;j<4;++j)hs[(kr+cr(j))*SMALL+n+cc(j)]=__float2half_rn(s[t][j]);
        }
        asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
        float ks[4]={},qs[4]={};
        dot<false,false,D>(ks,k,QK,hs,SMALL,mr,nc);
        dot<false,false,D>(qs,q,QK,hs,SMALL,mr,nc);
#pragma unroll
        for(int j=0;j<4;++j) {
            const int row=mr+cr(j),col=nc+cc(j);
            r[row*SMALL+col]=__float2half_rn(vv[j]-p[row]*ks[j]);
            oo[row*VC+col]=p[row]*qs[j];
        }
        __syncthreads();
        float dd[4]={};dot<false,false,C>(dd,a,SMALL,r,SMALL,mr,nc);
#pragma unroll
        for(int j=0;j<4;++j) {
            const int row=mr+cr(j),col=nc+cc(j);
            delta[row*SMALL+col]=__float2half_rn(dd[j]);
            last_delta[row*SMALL+col]=__float2half_rn(last[row]*dd[j]);
        }
        __syncthreads();
        float o[4];
#pragma unroll
        for(int j=0;j<4;++j)o[j]=oo[(mr+cr(j))*VC+nc+cc(j)];
        dot<false,false,C>(o,m,SMALL,delta,SMALL,mr,nc);
#pragma unroll
        for(int j=0;j<4;++j)if(first+mr+cr(j)<f.nt)
            output[((first+mr+cr(j))*H+h)*D+v0+nc+cc(j)]=o[j]*scale;
#pragma unroll
        for(int t=0;t<4;++t) {
            const int tile=w+t*8,kr=(tile/4)*16,n=(tile%4)*8;
#pragma unroll
            for(int j=0;j<4;++j)s[t][j]*=p[C-1];
            dot<true,false,C>(s[t],k,QK,last_delta,SMALL,kr,n);
        }
        __syncthreads();
    }
#pragma unroll
    for(int t=0;t<4;++t) {
        const int tile=w+t*8,mr=(tile/4)*16,n=(tile%4)*8;
#pragma unroll
        for(int j=0;j<4;++j)state[(h*D+v0+n+cc(j))*D+mr+cr(j)]=s[t][j];
    }
}
} // namespace gdn_half_c32

template<bool PACKED>
static void gdn_half_prefill32_launch(
        ggml_backend_cuda_context &ctx,const float*q,const float*k,const float*v,
        const float*g,const float*beta,const float*bias,const float*a,
        const float*current,float*output,float*state,int T,
        int64_t sq1,int64_t sq2,int64_t sq3,int64_t sv1,int64_t sv2,int64_t sv3,
        int64_t sb1,int64_t sb2,int64_t sb3,float scale,int64_t slot_stride,
        const int32_t*ids,int64_t row_stride,int lazy,
        const float*zero_row,float zero_scale,float zero_bias,cudaStream_t stream) {
    const int prefix=T-16,chunks=(prefix+31)/32;
    static const bool fused_initial=getenv("GGML_CUDA_GDN_FUSED_INITIAL")==nullptr||atoi(getenv("GGML_CUDA_GDN_FUSED_INITIAL"))!=0;
    ggml_cuda_pool_alloc<float> initial(ctx.pool());
    if(!fused_initial)initial.alloc(size_t(32)*128*128);
    ggml_cuda_pool_alloc<gdn_half_c32::Coeff> coef(ctx.pool(),size_t(32)*chunks);
    ggml_cuda_pool_alloc<__half> packed(ctx.pool(),size_t(chunks)*32*16*128*2);
    const ggml_cuda_kernel_launch_params mat(dim3(32,1,8),dim3(32,4),0,stream);
    if(!fused_initial)ggml_cuda_kernel_launch(gdn_half_initial_state,mat,current,state,initial.ptr,slot_stride,
        ids,row_stride,lazy,zero_row,zero_scale,zero_bias);
    const gdn_half_c16::Frame frame{q,k,v,g,beta,bias,a,prefix,PACKED,sq1,sq2,sv1,sv2};
    const ggml_cuda_kernel_launch_params prep(dim3(32,chunks),dim3(32,8),0,stream);
    ggml_cuda_kernel_launch(gdn_half_c32::prepare,prep,frame,coef.ptr,packed.ptr);
    const ggml_cuda_kernel_launch_params carry(dim3(32,4),dim3(32,8),0,stream);
    if(fused_initial) {
        static bool logged=false;
        if(!logged){GGML_LOG_INFO("gdn_c32_fused_initial: logical8x4 reader, flat128 carry\n");logged=true;}
        const ggml_cuda_kernel_launch_params flat(dim3(128,1),dim3(32,8),0,stream);
        const gdn_half_prefill::InitialState init{current,state,ids,slot_stride,row_stride,lazy,zero_row,zero_scale,zero_bias};
        ggml_cuda_kernel_launch(gdn_half_c32::carry<true>,flat,frame,coef.ptr,packed.ptr,
            (const float*)nullptr,output,state,scale,init);
    } else {
        ggml_cuda_kernel_launch(gdn_half_c32::carry<false>,carry,frame,coef.ptr,packed.ptr,initial.ptr,output,state,scale,
            gdn_half_prefill::InitialState{});
    }
    ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<false,true,PACKED,true>,mat,
        q+prefix*sq2,k+prefix*sq2,v+prefix*sv2,g+prefix*sb2,beta+prefix*sb2,
        bias,a,state,output+int64_t(prefix)*32*128,state,
        int64_t(32),int64_t(16),int64_t(1),sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,
        init_fastdiv_values(16),init_fastdiv_values(1),scale,slot_stride,16,
        (const int32_t*)nullptr,int64_t(0),lazy,(const float*)nullptr,0.0f,0.0f);
}
