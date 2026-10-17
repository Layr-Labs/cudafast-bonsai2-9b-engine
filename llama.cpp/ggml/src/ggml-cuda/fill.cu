#include "fill.cuh"
#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    // programmatic stream serialization on GB10: an earlier kernel may still read the old dst
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    dst[i] = value;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_cuda_kernel_launch_params launch_params((dim3) num_blocks, (dim3) CUDA_FILL_BLOCK_SIZE, 0, stream);

    switch (dst->type) {
        case GGML_TYPE_F32:
            ggml_cuda_kernel_launch_gb10(cc, fill_kernel<float>, launch_params, (float *)dst_d, k, value);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_kernel_launch_gb10(cc, fill_kernel<half>, launch_params, (half *)dst_d, k, ggml_cuda_cast<half>(value));
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
