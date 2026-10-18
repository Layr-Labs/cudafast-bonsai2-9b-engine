#pragma once
#include "common.cuh"

// One counter belongs to one private recurrent cache, not to a graph instance.
// prepare runs outside capture; contexts retain the allocation until their
// graphs are destroyed. Serialized contexts/clones of the same cache share it.
struct ggml_cuda_gdn_dedup_entry;
bool ggml_cuda_gdn_dedup_enabled();
bool ggml_cuda_gdn_dedup_prepare(ggml_backend_cuda_context & ctx, float * cache,
        int64_t stride, int64_t heads, int slots, ggml_backend_buffer_t buffer);
uint32_t * ggml_cuda_gdn_dedup_counter(ggml_backend_cuda_context & ctx, float * cache,
        int64_t stride, int64_t heads, int slots);
// Every external write/import and non-GDN graph write marks the next invocation
// as a compatibility pass. The operation is ordered on the writer's stream.
void ggml_cuda_gdn_dedup_written(const void * p, size_t size, cudaStream_t stream);
void ggml_cuda_gdn_dedup_forget(const void * p, size_t size);

void ggml_cuda_gdn_dedup_collect_writes(ggml_backend_cuda_context & ctx, const std::vector<std::pair<const void *, size_t>> & outputs,
        std::vector<std::pair<const void *, size_t>> & caches);
void ggml_cuda_gdn_dedup_written_ranges(const std::vector<std::pair<const void *, size_t>> & ranges, cudaStream_t stream);
void ggml_cuda_gdn_dedup_prune();

uint64_t ggml_cuda_gdn_dedup_generation();
