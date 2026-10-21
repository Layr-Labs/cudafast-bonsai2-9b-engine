#include "gdn-dedup.cuh"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>

struct ggml_cuda_gdn_dedup_entry {
    const uintptr_t base;
    const size_t bytes;
    const int64_t stride, heads;
    const int slots, device;
    uint32_t * counter = nullptr;
    ggml_cuda_gdn_dedup_entry(uintptr_t base, size_t bytes, int64_t stride,
            int64_t heads, int slots, int device) :
        base(base), bytes(bytes), stride(stride), heads(heads), slots(slots), device(device) {}
    ~ggml_cuda_gdn_dedup_entry() {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(counter));
    }
};
namespace {
struct registry {
    std::mutex mutex;
    std::vector<std::shared_ptr<ggml_cuda_gdn_dedup_entry>> entries;
    std::atomic<uintptr_t> lo { UINTPTR_MAX }, hi { 0 };
    std::atomic<uint64_t> generation { 0 };
};
registry & state() { static registry r; return r; }
bool overlaps(uintptr_t a, size_t an, uintptr_t b, size_t bn) {
    return an && bn && a < b + bn && b < a + an;
}
// The high-bit compatibility pass preserves imported shared-record parity, then
// the eighth CTA resets to zero. No host callback reads data-dependent headers.
__global__ void rehydrate(uint32_t * p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0x80000000u;
}
struct rehydrate_batch { uint32_t * p[64]; int n[64]; };
__global__ void rehydrate_many(rehydrate_batch b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < b.n[blockIdx.y]) b.p[blockIdx.y][i] = 0x80000000u;
}
}
uint64_t ggml_cuda_gdn_dedup_generation() { return state().generation.load(std::memory_order_acquire); }
bool ggml_cuda_gdn_dedup_enabled() {
    const char * e = getenv("GGML_CUDA_GDN_DEDUP");
    return !e || atoi(e) == 1;
}
bool ggml_cuda_gdn_dedup_prepare(ggml_backend_cuda_context & ctx, float * cache,
        int64_t stride, int64_t heads, int slots, ggml_backend_buffer_t buffer) {
    if (!ggml_cuda_gdn_dedup_enabled() || !cache || !buffer || heads <= 0 || heads > 4096 ||
            slots < 12 || slots > 16 || stride < 128 * 128 * heads || ctx.curr_stream_no != 0) return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(cache);
    const size_t bytes = size_t((slots - 1) * stride + 128 * 128 * heads) * sizeof(float);
    const uintptr_t buffer_base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    if (base < buffer_base || bytes > ggml_backend_buffer_get_size(buffer) ||
            base - buffer_base > ggml_backend_buffer_get_size(buffer) - bytes) return false;
    cudaStreamCaptureStatus capturing;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capturing));
    if (capturing != cudaStreamCaptureStatusNone) return false;
    auto & r = state();
    std::lock_guard<std::mutex> lock(r.mutex);
    std::shared_ptr<ggml_cuda_gdn_dedup_entry> entry;
    for (const auto & e : r.entries) {
        if (!overlaps(base, bytes, e->base, e->bytes)) continue;
        if (base != e->base || stride != e->stride || heads != e->heads || slots != e->slots || ctx.device != e->device) return false;
        entry = e;
        break;
    }
    if (!entry) {
        entry = std::make_shared<ggml_cuda_gdn_dedup_entry>(base, bytes, stride, heads, slots, ctx.device);
        CUDA_CHECK(cudaMalloc(&entry->counter, size_t(heads) * sizeof(uint32_t)));
        rehydrate<<<(heads + 127) / 128, 128, 0, ctx.stream()>>>(entry->counter, int(heads));
        // Creation is a one-time pre-capture operation. Completing initialization
        // also makes a later serialized context safe without hidden event state.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        r.entries.push_back(entry);
        r.lo.store(std::min(r.lo.load(std::memory_order_relaxed), base), std::memory_order_relaxed);
        r.hi.store(std::max(r.hi.load(std::memory_order_relaxed), base + bytes), std::memory_order_relaxed);
        r.generation.fetch_add(1, std::memory_order_release);
    }
    if (std::find(ctx.gdn_dedup_owners.begin(), ctx.gdn_dedup_owners.end(), entry) == ctx.gdn_dedup_owners.end()) {
        ctx.gdn_dedup_owners.push_back(entry);
    }
    ctx.gdn_dedup_caches[cache] = entry;
    return true;
}
uint32_t * ggml_cuda_gdn_dedup_counter(ggml_backend_cuda_context & ctx, float * cache,
        int64_t stride, int64_t heads, int slots) {
    if (!ggml_cuda_gdn_dedup_enabled() || ctx.curr_stream_no != 0) return nullptr;
    const auto it = ctx.gdn_dedup_caches.find(cache);
    if (it == ctx.gdn_dedup_caches.end()) return nullptr;
    const auto & e = it->second;
    return e->stride == stride && e->heads == heads && e->slots == slots ? e->counter : nullptr;
}
void ggml_cuda_gdn_dedup_collect_writes(ggml_backend_cuda_context & ctx, const std::vector<std::pair<const void *, size_t>> & outputs,
        std::vector<std::pair<const void *, size_t>> & caches) {
    caches.clear();
    auto & r = state();
    if (r.hi.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto & e : r.entries) {
        for (const auto & out : outputs) {
            if (overlaps(reinterpret_cast<uintptr_t>(out.first), out.second, e->base, e->bytes)) {
                caches.emplace_back(reinterpret_cast<const void *>(e->base), e->bytes);
                // Reset nodes also embed the allocation. Writer-only graph
                // contexts must keep it alive after the original GDN owner dies.
                if (std::find(ctx.gdn_dedup_owners.begin(), ctx.gdn_dedup_owners.end(), e) == ctx.gdn_dedup_owners.end()) {
                    ctx.gdn_dedup_owners.push_back(e);
                }
                break;
            }
        }
    }
}
void ggml_cuda_gdn_dedup_written_ranges(const std::vector<std::pair<const void *, size_t>> & ranges, cudaStream_t stream) {
    if (ranges.empty()) return;
    auto & r = state();
    std::lock_guard<std::mutex> lock(r.mutex);
    rehydrate_batch batch = {};
    int count = 0, max_heads = 0;
    auto flush = [&] {
        if (!count) return;
        rehydrate_many<<<dim3((max_heads + 127)/128, count), 128, 0, stream>>>(batch);
        CUDA_CHECK(cudaGetLastError());
        count = max_heads = 0;
    };
    for (const auto & e : r.entries) {
        bool touched = false;
        for (const auto & range : ranges) {
            touched |= overlaps(reinterpret_cast<uintptr_t>(range.first), range.second, e->base, e->bytes);
        }
        if (!touched) continue;
        batch.p[count] = e->counter;
        batch.n[count++] = int(e->heads);
        max_heads = std::max(max_heads, int(e->heads));
        if (count == 64) flush();
    }
    flush();
}
void ggml_cuda_gdn_dedup_written(const void * p, size_t size, cudaStream_t stream) {
    auto & r = state();
    const uintptr_t base = reinterpret_cast<uintptr_t>(p);
    if (!p || !size || base + size <= r.lo.load(std::memory_order_relaxed) || base >= r.hi.load(std::memory_order_relaxed)) return;
    ggml_cuda_gdn_dedup_written_ranges({{p,size}},stream);
}
void ggml_cuda_gdn_dedup_prune() {
    auto & r = state();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.entries.erase(std::remove_if(r.entries.begin(), r.entries.end(), [](const auto & e) {
        return e.use_count() == 1; // only registry; no context or captured graph can refer to it
    }), r.entries.end());
}
void ggml_cuda_gdn_dedup_forget(const void * p, size_t size) {
    // A buffer VA can be reused while an executable graph still embeds that
    // address. Keep owned entries in the import-invalidation registry, so such
    // an old graph sees the same counter and a rehydrate flag on the new write.
    // Incompatible new geometry falls back until the old contexts are gone.
    GGML_UNUSED(p); GGML_UNUSED(size);
    ggml_cuda_gdn_dedup_prune();
}
