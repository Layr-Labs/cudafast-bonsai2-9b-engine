#pragma once

#include "engine_iface.hpp"

#include <algorithm>
#include <array>
#include <cmath>

inline bool logit_id_before(float la, int32_t a, float lb, int32_t b) {
    if (la != lb) {
        return la > lb;
    }
    return a < b;
}

inline Emit top_from_logits(const float *logits, int n_vocab) {
    const int k = std::min<int>(kTopK, n_vocab);
    std::array<std::pair<float, int32_t>, kTopK> best{};
    for (int i = 0; i < k; ++i) {
        best[i] = {logits[i], i};
    }
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return logit_id_before(a.first, a.second, b.first, b.second);
    };
    std::sort(best.begin(), best.begin() + k, better);

    const auto insert = [&](int i) {
        const float li = logits[i];
        if (!logit_id_before(li, i, best[k - 1].first, best[k - 1].second)) {
            return;
        }
        int pos = k - 1;
        while (pos > 0 && logit_id_before(li, i, best[pos - 1].first, best[pos - 1].second)) {
            best[pos] = best[pos - 1];
            --pos;
        }
        best[pos] = {li, i};
    };
    // Ids rise with i, so a logit enters only if it is above the k-th best: a block with no such
    // logit is skipped whole (the k-th best only rises inside a block). Same result as one by one.
    // The count is a plain reduction, so the compiler vectorizes the block test.
    constexpr int kBlock = 64;
    int i = k;
    for (; i + kBlock <= n_vocab; i += kBlock) {
        const float thr = best[k - 1].first;
        int above = 0;
        for (int j = 0; j < kBlock; ++j) {
            above += logits[i + j] > thr;
        }
        if (above != 0) {
            for (int j = 0; j < kBlock; ++j) {
                insert(i + j);
            }
        }
    }
    for (; i < n_vocab; ++i) {
        insert(i);
    }

    Emit e;
    e.token = best[0].second;
    for (int i = 0; i < k; ++i) {
        e.top.push_back({best[i].second, best[i].first});
    }
    return e;
}

// The device selection of a row (llama_get_logits_topk_ith: the kTopK largest logits and the
// first kTopK logits of the row) gives the scan's result unless a NaN reaches the scan. A +NaN
// would top the selection, a NaN among the first kTopK logits would seed the scan. Without
// either, the row's NaNs are -NaN, which the device ranks below every other logit and the scan
// never inserts, so both keep the kTopK largest of the other logits in logit_id_before order.
// Returns false when the row needs the scan.
inline bool top_from_selection(const int32_t *ids, const float *logits, const float *head, Emit &e) {
    for (int i = 0; i < kTopK; ++i) {
        if (std::isnan(logits[i]) || std::isnan(head[i])) {
            return false;
        }
    }
    e.token = ids[0];
    for (int i = 0; i < kTopK; ++i) {
        e.top.push_back({ids[i], logits[i]});
    }
    return true;
}
