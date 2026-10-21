#pragma once

// Per-request memory clear that zeroes only what can have been written since the previous clear (C68).
//
// llama_memory_clear(mem, true) used to zero every byte of the memory buffers (KV cache, recurrent state) at
// each request. A buffer is zero after allocation and after every clear, so after a clear only the bytes
// written since the previous one can be nonzero: zeroing exactly those gives the same all-zero buffer.
//  * KV cache: the cells the cache assigned since the last clear (a high-water mark per stream).
//  * recurrent state: the R (conv) tensors in full; the S tensors as the backend recorded their writes.
// The ranges are zeroed by one asynchronous kernel per buffer (ggml-cuda, GB10). Without the backend entry
// points, or with LLAMA_STATE_CLEAR_RANGES=0, the whole buffer is cleared as before.
//
// Debug check (LLAMA_STATE_CLEAR_CHECK=1, default LLAMA_STATE_CLEAR_CHECK_DEFAULT): after each such clear the
// whole buffer is read back on the device and must be all zero, which is what the whole-buffer clear
// guaranteed; the log line also carries a hash of the buffer before the clear (the state the previous request
// left), so two runs can compare the state a request leaves.

#include "llama-impl.h"

#include "ggml-backend.h"

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef LLAMA_STATE_CLEAR_CHECK_DEFAULT
#define LLAMA_STATE_CLEAR_CHECK_DEFAULT 0
#endif

struct llama_state_clear_api {
    bool (*clear_ranges)(ggml_backend_buffer_t, const size_t *, int)     = nullptr; // (offset, width, pitch, height)
    bool (*track_writes)(const ggml_tensor *)                           = nullptr;
    int  (*dirty_ranges)(const ggml_tensor *, size_t *, int, bool)       = nullptr;
    bool (*digest)(ggml_backend_buffer_t, uint64_t *, uint64_t *)        = nullptr;
    bool enabled = false;
    bool check   = false;

    static const llama_state_clear_api & get() {
        static const llama_state_clear_api api = [] {
            llama_state_clear_api a;
            const char * env = getenv("LLAMA_STATE_CLEAR_RANGES");
            const char * chk = getenv("LLAMA_STATE_CLEAR_CHECK");
            a.check = chk ? atoi(chk) != 0 : LLAMA_STATE_CLEAR_CHECK_DEFAULT != 0;
            if (env && atoi(env) == 0) {
                return a;
            }
            ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
            if (!reg) {
                return a;
            }
            a.clear_ranges = (decltype(a.clear_ranges)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_clear_ranges");
            a.track_writes = (decltype(a.track_writes)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_track_writes");
            a.dirty_ranges = (decltype(a.dirty_ranges)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_dirty_ranges");
            a.digest       = (decltype(a.digest))       ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_digest");
            a.enabled      = a.clear_ranges != nullptr;
            return a;
        }();
        return api;
    }
};

// Zero the given ranges of buf (offsets relative to the buffer base), or the whole buffer when the ranged clear
// is not available. what: a label for the debug check.
static inline void llama_state_clear_buffer(ggml_backend_buffer_t buf, const std::vector<size_t> & ranges, bool ranged,
                                            const char * what) {
    const llama_state_clear_api & api = llama_state_clear_api::get();
    uint64_t pre_hash = 0, pre_nz = 0;
    const bool check = api.check && api.digest != nullptr;
    if (check) {
        api.digest(buf, &pre_hash, &pre_nz);
    }
    bool done = false;
    if (ranged && api.enabled) {
        done = api.clear_ranges(buf, ranges.data(), (int) (ranges.size() / 4));
    }
    if (!done) {
        ggml_backend_buffer_clear(buf, 0);
    }
    if (check) {
        uint64_t post_hash = 0, post_nz = 0;
        api.digest(buf, &post_hash, &post_nz);
        size_t bytes = 0;
        for (size_t i = 0; i + 3 < ranges.size(); i += 4) {
            bytes += ranges[i + 1]*ranges[i + 3];
        }
        LLAMA_LOG_INFO("%s: state clear check %s: %s, %zu bytes in %zu ranges; before: hash %016" PRIx64 " nonzero %" PRIu64
                       " bytes; after: nonzero %" PRIu64 " bytes of %zu\n", __func__, what, done ? "ranged" : "whole buffer",
                       done ? bytes : ggml_backend_buffer_get_size(buf), done ? ranges.size()/4 : (size_t) 1, pre_hash, pre_nz,
                       post_nz, ggml_backend_buffer_get_size(buf));
        GGML_ASSERT(post_nz == 0 && "state clear left nonzero bytes");
    }
}
