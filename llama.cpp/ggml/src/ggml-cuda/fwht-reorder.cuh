#pragma once
// One shared owner exchange. F32 operations, different stage order.
namespace fwht_reorder {
__device__ __forceinline__ int column(int reg, int tid) {
    return (reg & 8)*256 + (tid/32)*256 + (reg & 7)*32 + tid%32;
}
__device__ __forceinline__ void transform(float *v, float *shared, int tid) {
    constexpr int NT=256, NE=16;
    const int lane=tid%32;
    // Complete the high register bits before they become warp-owner bits.
#pragma unroll
    for(int step=1;step<NE;step*=2) {
#pragma unroll
        for(int j=0;j<NE;j+=2*step) {
#pragma unroll
            for(int b=0;b<step;++b) {
                const float x=v[j+b],y=v[j+b+step];
                v[j+b]=x+y;v[j+b+step]=x-y;
            }
        }
    }
#pragma unroll
    for(int h=1;h<32;h*=2) {
#pragma unroll
        for(int j=0;j<NE;++j) {
            const float x=v[j],y=__shfl_xor_sync(0xffffffff,x,h,32);
            v[j]=(lane&h)==0?x+y:y-x;
        }
    }
#pragma unroll
    for(int j=0;j<NE;++j)shared[j*NT+tid]=v[j];
    __syncthreads();
#pragma unroll
    for(int j=0;j<NE;++j)v[j]=shared[column(j,tid)];
    // Logical bits5..7 are now register bits0..2; lanes still own bits0..4.
#pragma unroll
    for(int step=1;step<8;step*=2) {
#pragma unroll
        for(int j=0;j<NE;j+=2*step) {
#pragma unroll
            for(int b=0;b<step;++b) {
                const float x=v[j+b],y=v[j+b+step];
                v[j+b]=x+y;v[j+b+step]=x-y;
            }
        }
    }
    __syncthreads(); // caller may reuse shared after all exchange loads
}
} // namespace fwht_reorder
