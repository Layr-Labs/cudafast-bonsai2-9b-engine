#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_q_case<GGML_TYPE_PQ2_0>(ctx, args, stream);
            break;
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_PTQ1_0:
            mul_mat_q_case<GGML_TYPE_PTQ1_0>(ctx, args, stream);
            break;
#endif
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static size_t ggml_cuda_mmq_q8_buffer_size(
        ggml_type type, bool fallback, int cc,
        int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13);

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * gate,
        const ggml_tensor * norm_weight, const ggml_tensor * norm_scale,
        void * external_q8, bool quantize_external, const float * external_norm_scale,
        const float * add_res, float * add_dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT(!gate || (!ids && gate->type == GGML_TYPE_F32));
    GGML_ASSERT((norm_weight == nullptr) == (norm_scale == nullptr && external_norm_scale == nullptr));
    GGML_ASSERT(!norm_weight || (!ids && !gate && norm_weight->type == GGML_TYPE_F32 &&
                ((norm_scale && norm_scale->type == GGML_TYPE_F32) || external_norm_scale)));

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = use_native_fp4 ?
            ne13*ne12*ne11*ne10_padded*y_block_size/y_values_per_block +
                ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11)*sizeof(block_q8_1_mmq) :
            ggml_cuda_mmq_q8_buffer_size(src0->type, fallback, cc, ne10_padded, ne11, ne12, ne13);
        GGML_ASSERT(!external_q8 || (!gate && !use_native_fp4));
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        void * shared_q8 = nullptr;
        bool shared_ready = false;
        auto & reuse = ctx.fwht_q8();
        const auto users = reuse.plain_q8_users.find(src1);
        if (!external_q8 && !gate && !norm_weight && !use_native_fp4 &&
            src0->type == GGML_TYPE_Q8_0 && !fallback && ctx.curr_stream_no == 0 &&
            users != reuse.plain_q8_users.end() && users->second > 1) {
            const auto found = reuse.plain_q8.find(src1);
            if (found != reuse.plain_q8.end()) {
                shared_q8 = found->second;
                shared_ready = true;
            } else {
                auto buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes_src1_q8_1);
                shared_q8 = buf->get();
                reuse.held.push_back(std::move(buf));
                reuse.plain_q8[src1] = shared_q8;
            }
        }
        char * src1_q8_ptr = external_q8 ? (char *) external_q8 :
            shared_q8 ? (char *) shared_q8 : src1_q8_1.alloc(nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        if ((!external_q8 || quantize_external) && !shared_ready) {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                    quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_ptr, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                if (norm_weight) {
                    GGML_ASSERT(norm_weight->ne[0] == src1->ne[0] && ggml_nrows(norm_weight) == 1);
                    GGML_ASSERT(ggml_is_contiguous(norm_weight));
                    GGML_ASSERT(external_norm_scale || (ggml_is_contiguous(norm_scale) &&
                                ggml_nelements(norm_scale) >= ggml_nrows(src1)));
                    quantize_mmq_q8_1_rms_cuda(src1_d, (const float *) norm_weight->data,
                                               external_norm_scale ? external_norm_scale : (const float *) norm_scale->data,
                                               src1_q8_ptr, src0->type,
                                               ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                } else if (gate) {
                    GGML_ASSERT(ggml_are_same_shape(src1, gate));
                    GGML_ASSERT(ggml_is_contiguous(gate));
                    quantize_mmq_q8_1_swiglu_cuda(src1_d, (const float *) gate->data, src1_q8_ptr, src0->type,
                                                  ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                } else {
                    quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_ptr, src0->type, ne10, s11, s12, s13, ne10_padded,
                                           ne11, ne12, ne13, stream);
                }
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, add_res, add_dst};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }
    GGML_ASSERT(add_res == nullptr && add_dst == nullptr);

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

static size_t ggml_cuda_mmq_q8_buffer_size(
        ggml_type type, bool fallback, int cc,
        int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13) {
    return ne13*ne12*ne11*ne10_padded*sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type, fallback, cc, ne11)*sizeof(block_q8_1_mmq);
}

size_t ggml_cuda_mul_mat_q_q8_size(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_ASSERT(src0 && src1 && src1->type == GGML_TYPE_F32);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return ggml_cuda_mmq_q8_buffer_size(src0->type, src0->ne[1] % 128 != 0, cc,
        GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING), src1->ne[1], src1->ne[2], src1->ne[3]);
}

void ggml_cuda_mul_mat_q_fused_two(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a, const ggml_tensor * src0_b, const ggml_tensor * src1,
        ggml_tensor * dst_a, ggml_tensor * dst_b,
        const ggml_tensor * norm_weight, const float * norm_scale) {
    GGML_ASSERT(src0_a && src0_b && src1 && dst_a && dst_b && norm_weight && norm_scale);
    GGML_ASSERT(src0_a->type == src0_b->type && src1->type == GGML_TYPE_F32 &&
                dst_a->type == GGML_TYPE_F32 && dst_b->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_a->ne[0] == src1->ne[0] && src0_b->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0_a->ne[2] == src0_b->ne[2] && src0_a->ne[3] == src0_b->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(norm_weight));
    GGML_ASSERT(norm_weight->type == GGML_TYPE_F32 && norm_weight->ne[0] == src1->ne[0] &&
                ggml_nrows(norm_weight) == 1);

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const bool fallback = src0_a->ne[1] % 128 != 0;
    GGML_ASSERT(fallback == (src0_b->ne[1] % 128 != 0));

    const size_t nbytes_src1_q8_1 = ggml_cuda_mmq_q8_buffer_size(
        src0_a->type, fallback, cc, ne10_padded, ne11, ne12, ne13);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12_src = src1->nb[2] / sizeof(float);
    const int64_t s13_src = src1->nb[3] / sizeof(float);
    quantize_mmq_q8_1_rms_cuda((const float *) src1->data, (const float *) norm_weight->data,
                               norm_scale, src1_q8_1.get(), src0_a->type,
                               ne10, s11, s12_src, s13_src, ne10_padded, ne11, ne12, ne13, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t stride_q_channel = ne11*ne10_padded*sizeof(block_q8_1_mmq)/(QK8_1_MMQ*sizeof(int));
    const int64_t stride_q_sample = ne12*stride_q_channel;
    auto launch_one = [&](const ggml_tensor * src0, ggml_tensor * dst) {
        GGML_ASSERT(src0->type == src0_a->type && src0->ne[0] == ne10);
        const size_t ts0 = ggml_type_size(src0->type);
        const mmq_args args = {
            (const char *) src0->data, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, (float *) dst->data,
            nullptr,
            src0->ne[0], src0->ne[1], dst->ne[1], (int64_t)(src0->nb[1]/ts0), ne11, (int64_t)(dst->nb[1]/sizeof(float)),
            src0->ne[2], ne12, (int64_t)(src0->nb[2]/ts0), stride_q_channel, (int64_t)(dst->nb[2]/sizeof(float)),
            src0->ne[3], ne13, (int64_t)(src0->nb[3]/ts0), stride_q_sample, (int64_t)(dst->nb[3]/sizeof(float)),
            dst->ne[1]};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    };
    launch_one(src0_a, dst_a);
    launch_one(src0_b, dst_b);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_PTQ1_0:
            mmq_supported = turing_mma_available(cc);
            break;
#endif
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

#if !defined(GGML_USE_HIP)
    if (type == GGML_TYPE_PTQ1_0) {
        // the fp16 dequantize + cuBLAS fallback is the source of PTQ1_0's extra error on CUDA, so
        // the MMQ tile path runs at every batch by default; the env var is the A/B knob for
        // deployments that prefer cuBLAS's ~7% at pp512 over the accuracy
        static const int64_t max_batch = [] {
            const char * s = getenv("GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH");
            return s ? (int64_t) atoll(s) : (int64_t) MMQ_PTQ1_0_MAX_BATCH_SIZE;
        }();
        return ne11 <= max_batch;
    }
#endif

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

// -----------------------------------------------------------------------
// GB10 decode layout registry (mmq-j16-dl.cuh). Written at model load, read when launch_mul_mat_q runs
// (graph capture or eager), erased when the weight buffer is freed.

static std::mutex & ggml_cuda_dl_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<const void *, ggml_cuda_dl_entry> & ggml_cuda_dl_map() {
    static std::unordered_map<const void *, ggml_cuda_dl_entry> m;
    return m;
}

const ggml_cuda_dl_entry * ggml_cuda_dl_find(const void * src) {
    std::lock_guard<std::mutex> lock(ggml_cuda_dl_mutex());
    auto & m = ggml_cuda_dl_map();
    const auto it = m.find(src);
    return it == m.end() ? nullptr : &it->second;
}

// Copy of a contiguous [nrows][ncols] PQ2_0 / Q8_0 weight in the decode layout (TM), on the current
// device. Returns the bytes allocated (0: not eligible, disabled, or out of memory: the weight then keeps
// the regular kernels). GGML_CUDA_DECODE_LAYOUT=0 disables it. A PQ2_0 weight gets PTQ1_0 groups (C51,
// value-exact re-encoding) when every block is a trit block; GGML_CUDA_DECODE_PTQ1=0 keeps PQ2_0 groups.
size_t ggml_cuda_dl_register(const void * src, const int64_t nrows, const int64_t ncols, const ggml_type type) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_DECODE_LAYOUT");
        return env == nullptr || atoi(env) != 0;
    }();
    static const bool ptq1_enabled = [] {
        const char * env = getenv("GGML_CUDA_DECODE_PTQ1");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || src == nullptr || (type != GGML_TYPE_PQ2_0 && type != GGML_TYPE_Q8_0)) {
        return 0;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_DGX_SPARK) {
        return 0;
    }
    const int64_t qk = type == GGML_TYPE_PQ2_0 ? QK_PQ2_0 : QK8_0;
    if (nrows < 1024 || nrows % 128 != 0 || ncols % MMQ_ITER_K != 0 || nrows > INT_MAX || ncols/qk > INT_MAX) {
        return 0;
    }
    static_assert(sizeof(block_pq2_0) == 34 && sizeof(block_q8_0) == 34, "34-byte blocks");
    const int64_t nkb   = ncols/qk;
    const size_t  bytes = (size_t) (nrows*nkb*34);

    std::lock_guard<std::mutex> lock(ggml_cuda_dl_mutex());
    auto & m = ggml_cuda_dl_map();
    if (m.count(src)) {
        return 0;
    }
    // The weight upload (possibly asynchronous, other streams) must be complete.
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (type == GGML_TYPE_PQ2_0 && ptq1_enabled) {
        static_assert(sizeof(block_ptq1_0) == 28, "28-byte PTQ1_0 groups");
        const size_t bytes_p = (size_t) (nrows*nkb)*sizeof(block_ptq1_0);
        char * dl = nullptr;
        unsigned int * fail_d = nullptr;
        if (cudaMalloc(&dl, bytes_p) == cudaSuccess && cudaMalloc(&fail_d, sizeof(unsigned int)) == cudaSuccess) {
            CUDA_CHECK(cudaMemsetAsync(fail_d, 0, sizeof(unsigned int), stream));
            ggml_cuda_dl_repack_ptq1((const char *) src, dl, (int) nrows, (int) nkb, GGML_CUDA_DL_TM, fail_d, stream);
            CUDA_CHECK(cudaGetLastError());
            unsigned int fail = 0;
            CUDA_CHECK(cudaMemcpyAsync(&fail, fail_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(fail_d));
            if (fail == 0) {
                CUDA_CHECK(cudaStreamDestroy(stream));
                m.emplace(src, ggml_cuda_dl_entry{ (const char *) src, dl, nrows, ncols, type, GGML_TYPE_PTQ1_0 });
                return bytes_p;
            }
            // a block with a code 3 (+2d) or one that does not round-trip: keep the PQ2_0 groups
            GGML_LOG_WARN("%s: PQ2_0 weight %p has a block PTQ1_0 cannot hold; it keeps PQ2_0 decode groups\n", __func__, src);
            CUDA_CHECK(cudaFree(dl));
        } else {
            (void) cudaGetLastError();
            if (dl != nullptr) {
                CUDA_CHECK(cudaFree(dl));
            }
        }
    }
    char * dl = nullptr;
    if (cudaMalloc(&dl, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }
    if (type == GGML_TYPE_PQ2_0) {
        ggml_cuda_dl_repack<1, 34>((const char *) src, dl, (int) nrows, (int) nkb, GGML_CUDA_DL_TM, stream);
    } else {
        ggml_cuda_dl_repack<4, 34>((const char *) src, dl, (int) nrows, (int) nkb, GGML_CUDA_DL_TM, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    m.emplace(src, ggml_cuda_dl_entry{ (const char *) src, dl, nrows, ncols, type, type });
    return bytes;
}

bool ggml_cuda_mul_mat_q_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * src1,
                             ggml_tensor * gate_dst, const void * stacked, void * external_q8) {
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if (cc != GGML_CUDA_CC_DGX_SPARK || gate_w->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 ||
            gate_dst->type != GGML_TYPE_F32 || external_q8 == nullptr || stacked == nullptr ||
            gate_w->ne[2] != 1 || gate_w->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            gate_dst->ne[2] != 1 || gate_dst->ne[3] != 1 || gate_w->nb[0] != ggml_type_size(gate_w->type) ||
            gate_dst->nb[0] != sizeof(float) || gate_w->ne[1] % 128 != 0 || gate_dst->ne[0] != gate_w->ne[1] ||
            gate_dst->ne[1] != src1->ne[1] || src1->ne[0] != gate_w->ne[0]) {
        return false;
    }
    // the J that mul_mat_q_switch_J<PQ2_0, false> picks for these columns
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    int J_best = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(GGML_TYPE_PQ2_0, J, false, cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles_x = (int) ((src1->ne[1] + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    if (J_best != 128) {
        return false;
    }
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const int64_t s01 = gate_w->nb[1] / ggml_type_size(gate_w->type);
    const int64_t s1  = gate_dst->nb[1] / sizeof(float);
    const int64_t s12 = src1->ne[1]*ne10_padded*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    mmq_args args = {
        (const char *) gate_w->data, GGML_TYPE_PQ2_0, (const int *) external_q8, nullptr, nullptr, (float *) gate_dst->data,
        nullptr,
        gate_w->ne[0], gate_w->ne[1], gate_dst->ne[1], s01, src1->ne[1], s1,
        1, 1, s01*gate_w->ne[1], s12, s1*gate_dst->ne[1],
        1, 1, s01*gate_w->ne[1], s12, s1*gate_dst->ne[1],
        gate_dst->ne[1], nullptr, nullptr};
    const int ntiles = (int) (args.nrows_x/128) * (int) ((args.ncols_max + 127)/128);
    if (!ggml_cuda_mmq_pq2_j128_use(args, nsm) || ggml_cuda_mmq_pq2_j128_grid(args, nsm) != ntiles) {
        return false; // the separate launch would not be pure tiling in the J=128 kernel
    }
    args.x       = (const char *) stacked;
    args.nrows_x = 2*gate_w->ne[1];
    if (!ggml_cuda_mmq_pq2_j128_supported(args)) {
        return false;
    }
    ggml_cuda_mmq_pq2_j128_glu_launch<GGML_TYPE_PQ2_0>(args, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static std::unordered_map<const void *, ggml_cuda_glu_pair> & ggml_cuda_glu_map() {
    static std::unordered_map<const void *, ggml_cuda_glu_pair> m;
    return m;
}

const ggml_cuda_glu_pair * ggml_cuda_glu_pair_find(const void * gate) {
    std::lock_guard<std::mutex> lock(ggml_cuda_dl_mutex());
    auto & m = ggml_cuda_glu_map();
    const auto it = m.find(gate);
    return it == m.end() ? nullptr : &it->second;
}

// Returns the bytes allocated (0: not eligible or out of memory; the pair then keeps the separate launches).
size_t ggml_cuda_glu_pair_register(const void * gate, const void * up, const int64_t nrows, const int64_t ncols,
                                   const ggml_type type) {
    if (gate == nullptr || up == nullptr || type != GGML_TYPE_PQ2_0 || nrows % 128 != 0 || ncols % QK_PQ2_0 != 0 ||
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_DGX_SPARK) {
        return 0;
    }
    const size_t row_bytes = (size_t) (ncols/QK_PQ2_0)*sizeof(block_pq2_0);
    const size_t bytes     = 2*(size_t) nrows*row_bytes;
    std::lock_guard<std::mutex> lock(ggml_cuda_dl_mutex());
    auto & m = ggml_cuda_glu_map();
    if (m.count(gate)) {
        return 0;
    }
    char * stacked = nullptr;
    if (cudaMalloc(&stacked, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    // The weight upload (possibly asynchronous, other streams) must be complete.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy2D(stacked,             16*row_bytes, gate, 8*row_bytes, 8*row_bytes, (size_t) nrows/8, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(stacked + 8*row_bytes, 16*row_bytes, up, 8*row_bytes, 8*row_bytes, (size_t) nrows/8, cudaMemcpyDeviceToDevice));
    m[gate] = { gate, up, stacked, nrows, ncols };
    return bytes;
}

void ggml_cuda_dl_unregister_range(const void * base, const size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_dl_mutex());
    {
        auto & g = ggml_cuda_glu_map();
        const auto in_range = [&](const void * p) {
            return (const char *) p >= (const char *) base && (const char *) p < (const char *) base + size;
        };
        for (auto it = g.begin(); it != g.end();) {
            if (in_range(it->second.gate) || in_range(it->second.up)) {
                CUDA_CHECK(cudaFree(it->second.stacked));
                it = g.erase(it);
            } else {
                ++it;
            }
        }
    }
    auto & m = ggml_cuda_dl_map();
    for (auto it = m.begin(); it != m.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            CUDA_CHECK(cudaFree(it->second.dl));
            it = m.erase(it);
        } else {
            ++it;
        }
    }
}
