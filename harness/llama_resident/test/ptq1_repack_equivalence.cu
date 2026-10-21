// Exact check of the GB10 decode layout copies of PQ2_0 weights with PTQ1_0 groups (ggml-cuda/mmq-j16-dl.cuh,
// mmq-ptq1-j16.cuh). For every PQ2_0 tensor of the model, and for random PQ2_0 tensors of the decode shapes,
// the J=16 matmul through ggml_cuda_mul_mat_q (the entry point of every decode MMQ) with the copy registered
// must have the bits of the result without it (the PQ2_0 kernels on the original), at 16, 13, 9 and 15
// columns. The copy is made by the function the model load calls (ggml_backend_cuda_register_decode_layout)
// and must hold PTQ1_0 groups; the original tensor is cleared before the copy runs, so the result can only
// come from the copy. A tensor with a code-3 block (not a trit) must get PQ2_0 groups instead, same bits.
//
// usage: ptq1-repack-equivalence MODEL.gguf [max_model_tensors]
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"
#include "common.cuh"
#include "mmq.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(2);
    }
}

using register_fn_t = size_t (*)(const ggml_tensor *);

static const int NCOLS[4] = {16, 13, 9, 15};

struct io {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * x[4] = {};
};

// activations [K][ncols] for the four column counts (one buffer per K)
static io make_io(ggml_backend_t backend, const int64_t K, std::mt19937 & rng) {
    io r;
    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
    r.ctx = ggml_init(ip);
    for (int i = 0; i < 4; ++i) {
        r.x[i] = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, K, NCOLS[i]);
    }
    r.buf = ggml_backend_alloc_ctx_tensors(r.ctx, backend);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < 4; ++i) {
        std::vector<float> v((size_t) K*NCOLS[i]);
        for (size_t j = 0; j < v.size(); ++j) {
            const float s = (j % 97 == 0) ? 40.0f : 1.0f; // a few outliers per row, like real activations
            v[j] = j % 31 == 0 ? 0.0f : s*nd(rng);
        }
        ggml_backend_tensor_set(r.x[i], v.data(), 0, v.size()*sizeof(float));
    }
    return r;
}

struct result { int64_t compared = 0, mismatched = 0; };

// Runs the four column counts of weight w through ggml_cuda_mul_mat_q into host vectors.
static void run_all(ggml_backend_cuda_context & cuda, ggml_backend_t backend, ggml_tensor * w, const io & in,
                    std::vector<std::vector<float>> & out) {
    const int64_t M = w->ne[1];
    out.resize(4);
    for (int i = 0; i < 4; ++i) {
        ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, NCOLS[i]);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, backend);
        // on the context's (non-blocking) stream: a legacy-stream memset would not be ordered with the matmul
        check(cudaMemsetAsync(dst->data, 0x5A, ggml_nbytes(dst), cuda.stream()));
        ggml_cuda_mul_mat_q(cuda, w, in.x[i], nullptr, dst);
        check(cudaStreamSynchronize(cuda.stream()));
        check(cudaGetLastError());
        out[i].resize((size_t) M*NCOLS[i]);
        check(cudaMemcpy(out[i].data(), dst->data, out[i].size()*sizeof(float), cudaMemcpyDeviceToHost));
        ggml_backend_buffer_free(b);
        ggml_free(ctx);
    }
}

// One weight: reference run, register the copy, clear the original, copy run, compare. Returns false on failure.
static bool check_weight(ggml_backend_cuda_context & cuda, ggml_backend_t backend, register_fn_t reg, const std::string & name,
                         const int64_t K, const int64_t M, const void * data, const io & in, const bool expect_ptq1,
                         result & res, size_t & copy_bytes) {
    ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_PQ2_0, K, M);
    ggml_set_name(w, name.c_str());
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, data, 0, ggml_nbytes(w));

    std::vector<std::vector<float>> ref, got;
    run_all(cuda, backend, w, in, ref);

    const size_t bytes = reg(w);
    copy_bytes += bytes;
    bool ok = true;
    const ggml_cuda_dl_entry * e = ggml_cuda_dl_find(w->data);
    const ggml_type want = expect_ptq1 ? GGML_TYPE_PTQ1_0 : GGML_TYPE_PQ2_0;
    const size_t want_bytes = (size_t) (M*K/QK_PQ2_0)*(expect_ptq1 ? sizeof(block_ptq1_0) : sizeof(block_pq2_0));
    if (e == nullptr || e->dl_type != want || bytes != want_bytes) {
        std::fprintf(stderr, "FAIL %s: copy bytes %zu type %s (expected %zu %s)\n", name.c_str(), bytes,
                     e ? ggml_type_name(e->dl_type) : "none", want_bytes, ggml_type_name(want));
        ok = false;
    }
    if (e != nullptr) {
        // the copy alone must give the same bits: clear the original
        check(cudaMemsetAsync(w->data, 0, ggml_nbytes(w), cuda.stream()));
        check(cudaStreamSynchronize(cuda.stream()));
    }
    run_all(cuda, backend, w, in, got);
    int64_t mism = 0, n = 0;
    int64_t per[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        for (size_t j = 0; j < ref[i].size(); ++j) {
            const bool d = std::memcmp(&ref[i][j], &got[i][j], sizeof(float)) != 0;
            if (d && per[i] == 0) {
                std::fprintf(stderr, "  %s ncols=%d first mismatch row %lld col %lld: %a vs %a\n", name.c_str(), NCOLS[i],
                             (long long) (j % M), (long long) (j / M), ref[i][j], got[i][j]);
            }
            per[i] += d;
        }
        n += (int64_t) ref[i].size();
    }
    mism = per[0] + per[1] + per[2] + per[3];
    if (mism != 0) {
        std::fprintf(stderr, "FAIL %s: %lld of %lld outputs differ (ncols 16/13/9/15: %lld/%lld/%lld/%lld)\n", name.c_str(),
                     (long long) mism, (long long) n, (long long) per[0], (long long) per[1], (long long) per[2], (long long) per[3]);
        ok = false;
    }
    res.compared += n;
    res.mismatched += mism;
    ggml_backend_buffer_free(wb); // unregisters the copy
    ggml_free(ctx);
    return ok;
}

static uint16_t f16_bits(float f) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [max_model_tensors]\n", argv[0]);
        return 2;
    }
    const int max_tensors = argc > 2 ? std::atoi(argv[2]) : 1 << 30;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    if (ggml_cuda_info().devices[0].cc != GGML_CUDA_CC_DGX_SPARK) {
        std::printf("ptq1-repack-equivalence: not GB10, nothing to check (the copies are GB10 only)\n");
        return 0;
    }
    ggml_backend_reg_t creg = ggml_backend_reg_by_name("CUDA");
    register_fn_t reg = creg ? (register_fn_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_cuda_register_decode_layout") : nullptr;
    if (!reg) {
        std::fprintf(stderr, "no ggml_backend_cuda_register_decode_layout\n");
        return 2;
    }

    std::mt19937 rng(20261016);
    io in4096   = make_io(backend, 4096, rng);
    io in12288  = make_io(backend, 12288, rng);
    auto input_for = [&](int64_t K) -> const io * { return K == 4096 ? &in4096 : K == 12288 ? &in12288 : nullptr; };

    bool ok = true;
    result model_res, rand_res;
    size_t model_copy_bytes = 0, rand_copy_bytes = 0;
    int n_model = 0, n_rand = 0;

    // every PQ2_0 tensor of the model file
    {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
        gguf_context * gguf = gguf_init_from_file(argv[1], gp);
        if (!gguf) {
            std::fprintf(stderr, "cannot read %s\n", argv[1]);
            return 2;
        }
        FILE * f = std::fopen(argv[1], "rb");
        if (!f) return 2;
        std::vector<uint8_t> data;
        for (int64_t t = 0; t < gguf_get_n_tensors(gguf) && n_model < max_tensors; ++t) {
            const char * name = gguf_get_tensor_name(gguf, t);
            ggml_tensor * m = ggml_get_tensor(meta, name);
            if (!m || m->type != GGML_TYPE_PQ2_0 || ggml_n_dims(m) != 2) continue;
            const io * in = input_for(m->ne[0]);
            if (!in) {
                std::fprintf(stderr, "FAIL %s: unexpected K %lld\n", name, (long long) m->ne[0]);
                ok = false;
                continue;
            }
            data.resize(ggml_nbytes(m));
            if (fseeko(f, (off_t) (gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, t)), SEEK_SET) != 0 ||
                    std::fread(data.data(), 1, data.size(), f) != data.size()) {
                std::fprintf(stderr, "cannot read %s\n", name);
                return 2;
            }
            ok = check_weight(cuda, backend, reg, name, m->ne[0], m->ne[1], data.data(), *in, true, model_res, model_copy_bytes) && ok;
            ++n_model;
        }
        std::fclose(f);
        gguf_free(gguf);
        ggml_free(meta);
    }

    // random tensors: codes in {0, 1, 2}, scales incl. zero, negative, subnormal and large
    {
        struct rs { int64_t M, K; };
        const rs shapes[] = { {12288, 4096}, {4096, 12288}, {4096, 4096}, {8192, 4096}, {1024, 4096}, {248320, 4096} };
        std::uniform_real_distribution<float> du(-0.08f, 0.08f);
        for (const rs & s : shapes) {
            const int64_t nb = s.M*s.K/QK_PQ2_0;
            std::vector<block_pq2_0> w((size_t) nb);
            for (int64_t b = 0; b < nb; ++b) {
                const uint32_t r = rng();
                float d = du(rng);
                switch (r % 64) {
                    case 0: d = 0.0f; break;
                    case 1: d = -0.0f; break;
                    case 2: d = 3.0e-7f; break;      // fp16 subnormal
                    case 3: d = -1000.0f; break;
                    case 4: d = 60000.0f; break;
                    default: break;
                }
                const uint16_t bits = f16_bits(d);
                std::memcpy(&w[b].d, &bits, 2);
                const int pattern = (r >> 8) % 8; // some blocks constant (all -1, all 0, all +1)
                for (int k = 0; k < QK_PQ2_0/4; ++k) {
                    uint8_t q = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int code = pattern < 3 ? pattern : (int) (rng() % 3);
                        q |= (uint8_t) (code << (2*j));
                    }
                    w[b].qs[k] = q;
                }
            }
            const std::string name = "random." + std::to_string(s.M) + "x" + std::to_string(s.K);
            ok = check_weight(cuda, backend, reg, name, s.K, s.M, w.data(), *input_for(s.K), true, rand_res, rand_copy_bytes) && ok;
            ++n_rand;
        }
        // one code 3 in one block: no PTQ1_0 groups, the copy keeps PQ2_0 groups
        {
            const int64_t M = 4096, K = 4096, nb = M*K/QK_PQ2_0;
            std::vector<block_pq2_0> w((size_t) nb);
            for (int64_t b = 0; b < nb; ++b) {
                const uint16_t bits = f16_bits(du(rng));
                std::memcpy(&w[b].d, &bits, 2);
                for (int k = 0; k < QK_PQ2_0/4; ++k) {
                    uint8_t q = 0;
                    for (int j = 0; j < 4; ++j) q |= (uint8_t) ((rng() % 3) << (2*j));
                    w[b].qs[k] = q;
                }
            }
            w[nb/3].qs[7] |= 0x3 << 4;
            ok = check_weight(cuda, backend, reg, "random.code3.4096x4096", K, M, w.data(), in4096, false, rand_res, rand_copy_bytes) && ok;
            ++n_rand;
        }
    }

    std::printf("ptq1-repack-equivalence: model %d PQ2_0 tensors (PTQ1_0 decode copies %.2f MiB): %lld outputs, %lld mismatches; "
                "random %d tensors: %lld outputs, %lld mismatches; columns 16/13/9/15\n",
                n_model, model_copy_bytes/1024.0/1024.0, (long long) model_res.compared, (long long) model_res.mismatched,
                n_rand, (long long) rand_res.compared, (long long) rand_res.mismatched);

    ggml_backend_buffer_free(in4096.buf);
    ggml_backend_buffer_free(in12288.buf);
    ggml_free(in4096.ctx);
    ggml_free(in12288.ctx);
    ggml_backend_free(backend);
    return ok && n_model > 0 ? 0 : 1;
}
