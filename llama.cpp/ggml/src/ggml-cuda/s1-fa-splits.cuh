#pragma once
#include "common.cuh"

// No measured policy yet. Arming and per-family overrides are separate so that
// a master kill always restores the parent, including its nonuniform grid.
#ifndef GGML_CUDA_S1_FA_SPLITS_DEFAULT
#define GGML_CUDA_S1_FA_SPLITS_DEFAULT 1
#endif
#ifndef GGML_CUDA_S1_FA_TARGET_DEFAULT
#define GGML_CUDA_S1_FA_TARGET_DEFAULT 6
#endif
#ifndef GGML_CUDA_S1_FA_DRAFTER_DEFAULT
#define GGML_CUDA_S1_FA_DRAFTER_DEFAULT 0
#endif
// [D128,D256][T9,T15,T16][KV512,KV768,KV1024]. Every cell has a
// numeric default for the runner's header-only sed overlays; 0 is the parent.
#ifndef GGML_CUDA_S1_FA_D128_T9_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D128_T9_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T9_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D128_T9_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T9_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D128_T9_KV1024_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T15_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D128_T15_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T15_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D128_T15_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T15_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D128_T15_KV1024_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T16_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D128_T16_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T16_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D128_T16_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D128_T16_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D128_T16_KV1024_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T9_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D256_T9_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T9_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D256_T9_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T9_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D256_T9_KV1024_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T15_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D256_T15_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T15_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D256_T15_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T15_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D256_T15_KV1024_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T16_KV512_DEFAULT
#define GGML_CUDA_S1_FA_D256_T16_KV512_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T16_KV768_DEFAULT
#define GGML_CUDA_S1_FA_D256_T16_KV768_DEFAULT 0
#endif
#ifndef GGML_CUDA_S1_FA_D256_T16_KV1024_DEFAULT
#define GGML_CUDA_S1_FA_D256_T16_KV1024_DEFAULT 0
#endif

static inline bool ggml_cuda_s1_fa_split_valid(int splits) {
    switch (splits) {
        case 1: case 2: case 4: case 6: case 8: case 12:
        case 16: case 24: case 32: case 48: return true;
        default: return false;
    }
}

static inline int ggml_cuda_s1_fa_env(const char * name, int fallback) {
    const char * s = getenv(name);
    return s ? atoi(s) : fallback;
}

static inline int ggml_cuda_s1_fa_splits(const ggml_tensor * dst, int cc, int max_splits) {
    if (!ggml_cuda_s1_fa_env("GGML_CUDA_S1_FA_SPLITS", GGML_CUDA_S1_FA_SPLITS_DEFAULT)) return 0;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const int64_t t = q->ne[1], kv = k->ne[1];
    if (cc != GGML_CUDA_CC_DGX_SPARK || q->ne[3] != 1 ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        k->ne[2] <= 0 || q->ne[2] != 4*k->ne[2] ||
        (t != 9 && t != 15 && t != 16) ||
        (kv != 512 && kv != 768 && kv != 1024) || v->ne[0] != q->ne[0]) return 0;
    if ((q->ne[0] == 256 && q->ne[2] != 16) || (q->ne[0] == 128 && q->ne[2] != 32)) return 0;
    int splits = 0;
    if (q->ne[0] == 256) splits = ggml_cuda_s1_fa_env("GGML_CUDA_S1_FA_TARGET", GGML_CUDA_S1_FA_TARGET_DEFAULT);
    if (q->ne[0] == 128) splits = ggml_cuda_s1_fa_env("GGML_CUDA_S1_FA_DRAFTER", GGML_CUDA_S1_FA_DRAFTER_DEFAULT);
    if (splits == -1 && (q->ne[0] == 128 || q->ne[0] == 256)) {
        static const int policy[2][3][3] = {
            {
                {GGML_CUDA_S1_FA_D128_T9_KV512_DEFAULT,GGML_CUDA_S1_FA_D128_T9_KV768_DEFAULT,GGML_CUDA_S1_FA_D128_T9_KV1024_DEFAULT},
                {GGML_CUDA_S1_FA_D128_T15_KV512_DEFAULT,GGML_CUDA_S1_FA_D128_T15_KV768_DEFAULT,GGML_CUDA_S1_FA_D128_T15_KV1024_DEFAULT},
                {GGML_CUDA_S1_FA_D128_T16_KV512_DEFAULT,GGML_CUDA_S1_FA_D128_T16_KV768_DEFAULT,GGML_CUDA_S1_FA_D128_T16_KV1024_DEFAULT},
            },
            {
                {GGML_CUDA_S1_FA_D256_T9_KV512_DEFAULT,GGML_CUDA_S1_FA_D256_T9_KV768_DEFAULT,GGML_CUDA_S1_FA_D256_T9_KV1024_DEFAULT},
                {GGML_CUDA_S1_FA_D256_T15_KV512_DEFAULT,GGML_CUDA_S1_FA_D256_T15_KV768_DEFAULT,GGML_CUDA_S1_FA_D256_T15_KV1024_DEFAULT},
                {GGML_CUDA_S1_FA_D256_T16_KV512_DEFAULT,GGML_CUDA_S1_FA_D256_T16_KV768_DEFAULT,GGML_CUDA_S1_FA_D256_T16_KV1024_DEFAULT},
            },
        };
        splits = policy[q->ne[0] == 256][t == 9 ? 0 : t == 15 ? 1 : 2][(kv-512)/256];
    }
    // Invalid/unsupported experiments fall back, rather than silently clamping.
    // More pieces than KV work tiles would create CTAs which return without
    // initializing metadata; the inherited uniform merge cannot consume them.
    return ggml_cuda_s1_fa_split_valid(splits) && splits <= max_splits ? splits : 0;
}
