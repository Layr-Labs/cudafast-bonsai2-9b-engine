#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// layout: see ggml_cuda_q8_1_layout_for() in common.cuh. The caller must pass the same layout the
// consuming mat-vec kernel expects for (type_src0, ncols_dst, ids).
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_cuda_q8_1_layout layout, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// Hadamard transform (width n, optional per-element signs, scale 1/sqrt(n)) + q8_1 quantization in one
// kernel (ggml_cuda_fwht_q8). x holds ncols contiguous rows of ne00 elements; ne0 is the padded row
// width the consuming mat-vec expects (pad blocks are written as zeros).
bool ggml_cuda_fwht_quantize_supported(int n, int64_t ne00);
void fwht_quantize_row_q8_1_cuda(
        const void * x, ggml_type x_type, const float * signs, int n, void * vy,
        ggml_cuda_q8_1_layout layout, int64_t ne00, int64_t ne0, int64_t ncols, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// Plain Q8_0 D4 activation quantization of a contiguous row concatenation.
// No intermediate F32 concat is needed; all quantization math/layout is unchanged.
void quantize_mmq_q8_1_concat_cuda(
        const float * first, const float * second, void * out,
        int64_t width, int first_rows, int second_rows, cudaStream_t stream);

void quantize_mmq_q8_1_swiglu_cuda(
        const float * x, const float * gate, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_rms_cuda(
        const float * x, const float * weight, const float * scale, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);

// Exact signed/unsigned FWHT4096 followed by PQ2 MMQ D4 activation quantization.
// With g, the input is SWIGLU(x, g) as ggml_cuda_op_swiglu computes it (row strides o0/o1 in elements).
void fwht_quantize_mmq_d4_cuda(const void * x, ggml_type type, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, cudaStream_t stream,
    const void * g = nullptr, int64_t o0 = 0, int64_t o1 = 0);

// Shared opt-in for all FWHT producers and metadata-validated prefill consumers.
bool ggml_cuda_prefill_d1_enabled(int64_t rows);

// Private native D4-versus-D1 complete-chain comparison; no model dispatch uses this.
void fwht_quantize_mmq_d1_test_cuda(const void * x, ggml_type type, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, cudaStream_t stream);

// The same transform of SWIGLU(x, g) (f32) read through a permuted view of it: transform column
// i0 + n0*(i1 + n1*i2) of row r reads offset i0 + s1*i1 + s2*i2 of operand rows r (strides o0/o1, elements).
void fwht_quantize_mmq_d4_glu_perm_cuda(const float * x, const float * g, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1, int n0, int n1, int64_t s1, int64_t s2,
    cudaStream_t stream);

// The same with RMS_NORM(x, eps) * w as the input (k == 4096, one row of x per transform row, x rows
// stride_row floats apart); norm, when set, receives the normalized product as ggml_cuda_op_rms_norm_fused
// writes it.
void fwht_quantize_mmq_d4_rms_cuda(const float * x, int64_t stride_row, const float * w, float eps,
    const float * signs, float * norm, void * y, int64_t k, int64_t rows, cudaStream_t stream);

// The same with RMS_NORM(a + b, eps) * w as the input, formed as add_rms_norm_f32<1024> forms it (one 1024-thread
// block per 4096-wide contiguous row); sum receives a + b, norm (when set) the normalized product.
void fwht_quantize_mmq_d4_add_rms_cuda(const float * a, const float * b, float * sum, const float * w, float eps,
    const float * signs, float * norm, void * y, int64_t k, int64_t rows, cudaStream_t stream);

// GDN output projection input (fwht_quantize_mmq_d4_gnorm): x and z hold contiguous 128-wide head rows (z rows
// z_stride floats apart), w the 128 norm weights; feature a0 + 128*(a1 + ne1*a2) of transform row r is head row
// r*g3 + a1*g1 + a2*g2, column a0, of SWIGLU(z, RMS_NORM(x, eps) * w).
void fwht_quantize_mmq_d4_gnorm_cuda(const float * x, const float * z, int64_t z_stride, const float * w, float eps,
    int64_t ne1, int64_t g1, int64_t g2, int64_t g3, const float * signs, void * y, int64_t rows, cudaStream_t stream);

// C62: the transform of MUL(attn, SIGMOID(gate)) with the gate read through a strided view: transform column
// i0 + n0*i1 of row r reads gate[r*gt + i1*g1 + i0] and attn[r*at + i0 + n0*i1] (strides in elements).
void fwht_quantize_mmq_d4_sigmoid_gate_cuda(const float * gate, const float * attn, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, int n0, int64_t g1, int64_t gt, int64_t at, cudaStream_t stream);

// Only for the guarded M4096/N16/grid48 PTQ producer with K4096 or K12288.
void fwht_quantize_mmq_d4_fixup_add_rms_cuda(const float * a, const float * b, float * sum,
    const float * w, float eps, const float * signs, float * norm, void * y,
    const float * partials, cudaStream_t stream);
