#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GB10: device-to-device copy of nbytes as a kernel on the programmatic launch path (a memcpy node ends the
// PDL chain of a CUDA graph). Returns false, and copies nothing, on other devices or unaligned/odd sizes.
bool ggml_cuda_memcpy_d2d_gb10(void * dst, const void * src, size_t nbytes, cudaStream_t stream);
