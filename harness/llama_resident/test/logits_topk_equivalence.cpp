// Exact check of the device logits top-k (llama_set_logits_topk) against the host scan it
// replaces. The device side runs the two ops build_logits_topk adds to the graph (ggml_top_k and
// the row gather of the logits at those ids) on the CUDA backend; the host side is
// top_from_logits. Every row whose selection top_from_selection accepts must give the scan's
// token, ids and logit bits; the rows it refuses (a NaN reaches the scan) go to the scan in the
// engine, and only rows holding a NaN may be refused.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "logits_rank.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

struct Device {
    ggml_backend_t backend = nullptr;

    Device() {
        backend = ggml_backend_init_by_name("CUDA0", nullptr);
        if (!backend) {
            fprintf(stderr, "no CUDA backend\n");
            exit(2);
        }
    }
    ~Device() { ggml_backend_free(backend); }

    // ids and logits of the device top-k of every row, as build_logits_topk computes them
    void select(const std::vector<float> &x, int ncols, int nrows, std::vector<int32_t> &ids,
                std::vector<float> &vals) const {
        ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
        ggml_context *ctx = ggml_init(ip);
        ggml_tensor *logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
        ggml_tensor *t_ids = ggml_top_k(ctx, logits, kTopK);
        ggml_tensor *t_vals = ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits, 1, ncols, nrows), t_ids);
        ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_vals);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(logits, x.data(), 0, x.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(2);
        }
        ids.resize((size_t) nrows * kTopK);
        vals.resize((size_t) nrows * kTopK);
        ggml_backend_tensor_get(t_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_get(t_vals, vals.data(), 0, vals.size() * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

int n_checked = 0;
int n_refused = 0;
int n_failed = 0;

void check(const Device &dev, const char *name, const std::vector<float> &x, int ncols, int nrows) {
    std::vector<int32_t> ids;
    std::vector<float> vals;
    dev.select(x, ncols, nrows, ids, vals);
    for (int r = 0; r < nrows; ++r) {
        const float *row = x.data() + (size_t) r * ncols;
        const Emit want = top_from_logits(row, ncols);
        bool has_nan = false;
        for (int c = 0; c < ncols; ++c) {
            has_nan |= std::isnan(row[c]);
        }
        Emit got;
        if (!top_from_selection(ids.data() + (size_t) r * kTopK, vals.data() + (size_t) r * kTopK, row, got)) {
            ++n_refused;
            if (!has_nan) {
                ++n_failed;
                fprintf(stderr, "FAIL %s row %d: selection refused a row without NaN\n", name, r);
            }
            continue;
        }
        ++n_checked;
        bool ok = got.token == want.token && got.top.size() == want.top.size();
        for (size_t i = 0; ok && i < want.top.size(); ++i) {
            ok = got.top[i].first == want.top[i].first && same_bits(got.top[i].second, want.top[i].second);
        }
        if (!ok) {
            ++n_failed;
            fprintf(stderr, "FAIL %s row %d:", name, r);
            for (size_t i = 0; i < want.top.size(); ++i) {
                fprintf(stderr, " want %d/%a got %d/%a", want.top[i].first, want.top[i].second,
                        i < got.top.size() ? got.top[i].first : -1, i < got.top.size() ? got.top[i].second : 0.0f);
            }
            fprintf(stderr, "\n");
        }
    }
}

}  // namespace

int main() {
    const Device dev;
    std::mt19937 rng(20261016);
    const float nan_pos = std::numeric_limits<float>::quiet_NaN();
    const float nan_neg = -nan_pos;
    const float inf = std::numeric_limits<float>::infinity();

    for (const int ncols : {248320, 151936, 8192 * 3 + 1, 1025}) {
        for (const int nrows : {16, 1, 3}) {
            std::vector<float> x((size_t) ncols * nrows);
            auto rows = [&](auto fill) {
                for (int r = 0; r < nrows; ++r) {
                    fill(x.data() + (size_t) r * ncols, r);
                }
            };
            char name[96];

            // logits-like rows at several scales
            for (const float scale : {1.0f, 8.0f, 1e-3f, 1e30f}) {
                std::normal_distribution<float> nd(0.0f, scale);
                rows([&](float *row, int) {
                    for (int c = 0; c < ncols; ++c) {
                        row[c] = nd(rng);
                    }
                });
                snprintf(name, sizeof(name), "normal%g/%dx%d", scale, ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // heavy ties: a handful of distinct values, so the top-k is decided by ids
            for (const int levels : {1, 2, 5}) {
                std::uniform_int_distribution<int> ud(0, levels - 1);
                rows([&](float *row, int) {
                    for (int c = 0; c < ncols; ++c) {
                        row[c] = (float) ud(rng) * 0.25f - 1.0f;
                    }
                });
                snprintf(name, sizeof(name), "ties%d/%dx%d", levels, ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // -0.0 and +0.0 mixed: equal logits, ranked by id; the value bits must survive
            {
                std::uniform_int_distribution<int> ud(0, 3);
                rows([&](float *row, int r) {
                    for (int c = 0; c < ncols; ++c) {
                        const int v = ud(rng);
                        row[c] = v == 0 ? -0.0f : v == 1 ? 0.0f : -1.0f;
                    }
                    if (r % 2) {
                        row[0] = -0.0f;
                        row[ncols - 1] = 0.0f;
                    }
                });
                snprintf(name, sizeof(name), "signed-zero/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // denormals of both signs around zero
            {
                std::uniform_int_distribution<uint32_t> mant(1, 0x7fffffu);
                std::uniform_int_distribution<int> ud(0, 4);
                rows([&](float *row, int) {
                    for (int c = 0; c < ncols; ++c) {
                        const int v = ud(rng);
                        const uint32_t m = v == 4 ? 1u : mant(rng);
                        row[c] = v == 0 ? 0.0f : v == 1 ? -0.0f : from_bits((v == 2 ? 0x80000000u : 0u) | m);
                    }
                });
                snprintf(name, sizeof(name), "denormal/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // infinities, tied at the top and at the bottom
            {
                std::uniform_int_distribution<int> ud(0, 99);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                rows([&](float *row, int r) {
                    for (int c = 0; c < ncols; ++c) {
                        const int v = ud(rng);
                        row[c] = v < 2 ? inf : v < 4 ? -inf : nd(rng);
                    }
                    if (r == 0) {
                        for (int c = 0; c < ncols; ++c) {
                            row[c] = -inf;
                        }
                    }
                });
                snprintf(name, sizeof(name), "inf/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // the winners in the scan's seed, at chunk edges (8192 columns), and at the end
            {
                std::normal_distribution<float> nd(0.0f, 1.0f);
                rows([&](float *row, int r) {
                    for (int c = 0; c < ncols; ++c) {
                        row[c] = nd(rng);
                    }
                    for (int i = 0; i < kTopK; ++i) {
                        const int col = r % 3 == 0 ? i
                                      : r % 3 == 1 ? std::min(ncols - 1, 8192 * (i + 1) - (i % 2))
                                                   : ncols - 1 - i;
                        row[col] = 100.0f + (float) (i % 3);
                    }
                });
                snprintf(name, sizeof(name), "placed/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // monotonic rows (every block of the scan inserts, or none does)
            {
                rows([&](float *row, int r) {
                    for (int c = 0; c < ncols; ++c) {
                        row[c] = r % 2 ? (float) c : (float) (ncols - c);
                    }
                });
                snprintf(name, sizeof(name), "monotonic/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }

            // NaN: +NaN anywhere, -NaN past the seed (the selection must match), NaN in the seed,
            // rows that are all NaN
            {
                std::normal_distribution<float> nd(0.0f, 1.0f);
                std::uniform_int_distribution<int> col(0, ncols - 1);
                std::uniform_int_distribution<int> past_seed(kTopK, ncols - 1);
                std::uniform_int_distribution<int> seed(0, kTopK - 1);
                rows([&](float *row, int r) {
                    for (int c = 0; c < ncols; ++c) {
                        row[c] = nd(rng);
                    }
                    switch (r % 6) {
                    case 0:
                        row[col(rng)] = nan_pos;
                        break;
                    case 1:
                        for (int i = 0; i < 64; ++i) {
                            row[past_seed(rng)] = nan_neg;
                        }
                        break;
                    case 2:
                        row[seed(rng)] = nan_neg;
                        break;
                    case 3:
                        row[seed(rng)] = nan_pos;
                        break;
                    case 4:
                        for (int c = 0; c < ncols; ++c) {
                            row[c] = c % 2 ? nan_pos : nan_neg;
                        }
                        break;
                    default:
                        break;
                    }
                });
                snprintf(name, sizeof(name), "nan/%dx%d", ncols, nrows);
                check(dev, name, x, ncols, nrows);
            }
        }
    }

    printf("logits-topk-equivalence: %d rows exact, %d rows to the scan (NaN), %d failures\n", n_checked, n_refused,
           n_failed);
    return n_failed == 0 ? 0 : 1;
}
