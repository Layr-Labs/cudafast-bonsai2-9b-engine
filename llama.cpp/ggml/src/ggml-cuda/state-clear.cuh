#pragma once

// GB10 per-request state clear (C68).
//
// llama zeroes its whole memory buffers at every clear (each request): the KV cache and the recurrent-state
// buffer. To zero only what the previous request can have written, llama tracks the KV cells it assigned and
// asks this backend for the bytes written into the tensors it registers for write tracking (the recurrent
// S caches), then zeroes those ranges (ggml_backend_cuda_clear_ranges, asynchronous like the whole-buffer
// clear). Everything else in the buffer was zero after the previous clear and has not been written since.
//
// Tracking granularity: rows of GGML_CUDA_DIRTY_ROW bytes from the start of the tensor; per row the length of
// the dirty prefix (a write [a, b) inside a row marks the prefix up to b, a superset). Writers:
//  * every graph node whose output lies in a tracked tensor, marked before the graph is evaluated
//    (ggml_cuda_dirty_mark_graph) and again when ggml_cuda_compute_forward runs it;
//  * the GATED_DELTA_NET kernel that writes its state snapshots straight into the cache (the CPY it absorbs is
//    not marked in the pre-pass): its exact footprint (gated_delta_net.cu);
//  * host writes through the buffer interface (set / memset / cpy tensor).
// Marks made while a stream is being captured into a CUDA graph also go to a sticky set that no clear resets:
// the graph may be replayed later without any host-side evaluation, and each replay rewrites that footprint.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#define GGML_CUDA_DIRTY_ROW (64*1024)

// any tensor registered for write tracking (cheap, lock-free)
bool ggml_cuda_dirty_tracking();

// mark [p, p + n) written; stream: the stream the write is issued on (nullptr for host writes), only used to
// tell whether it is being captured into a CUDA graph
void ggml_cuda_dirty_mark(const void * p, size_t n, cudaStream_t stream);

// mark height rows of width bytes, pitch bytes apart, written
void ggml_cuda_dirty_mark_2d(const void * p, size_t width, size_t pitch, size_t height, cudaStream_t stream);
