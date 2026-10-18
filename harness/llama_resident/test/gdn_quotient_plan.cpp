// Integer-only edge proof for the production dedup replay quotient reuse.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {
struct Result { bool aligned; int64_t rb; };
Result old_index(int64_t diff, int64_t stride) {
    if (stride > 0 && diff >= 0 && diff % stride == 0) return {true, diff / stride};
    return {false, -1};
}
template<bool REUSE_QUOTIENT>
Result new_index(int64_t diff, int64_t stride) {
    int64_t slot_q = 0;
    if constexpr (REUSE_QUOTIENT) {
        if (stride > 0 && diff >= 0) slot_q = diff / stride;
    }
    if (stride > 0 && diff >= 0 &&
            (REUSE_QUOTIENT ? slot_q * stride == diff : diff % stride == 0)) {
        const int64_t rb = REUSE_QUOTIENT ? slot_q : diff / stride;
        return {true, rb};
    }
    return {false, -1};
}
void require(bool value) { if (!value) std::abort(); }
uint64_t pairs = 0, predicates = 0, unaligned = 0, negatives = 0, boundaries = 0;
void check(int64_t diff, int64_t stride) {
    const auto old = old_index(diff, stride), now = new_index<true>(diff, stride);
    const auto off = new_index<false>(diff, stride);
    require(old.aligned == now.aligned && old.rb == now.rb);
    require(old.aligned == off.aligned && old.rb == off.rb);
    if (stride > 0 && diff >= 0) {
        const int64_t q = diff / stride;
        const __int128 product = __int128(q) * stride;
        // Independent widened proof of the multiplication's signed-64 range.
        require(product >= 0 && product <= diff && product <= INT64_MAX);
        require((product == diff) == (diff % stride == 0));
        unaligned += product != diff;
    } else ++negatives;
    for (int w = 0; w <= 16; ++w) {
        const bool a = old.aligned && w >= 1 && old.rb <= w;
        const bool b = now.aligned && w >= 1 && now.rb <= w;
        require(a == b); ++predicates;
    }
    ++pairs;
}
}
int main() {
    std::vector<int64_t> values = {INT64_MIN, INT64_MIN+1, -1048576, -524288, -17, -1,
        0, 1, 2, 7, 8, 15, 16, 17, 127, 128, 255, 256, 4096, 16384, 524288,
        INT32_MAX-1, INT32_MAX, int64_t(INT32_MAX)+1, int64_t(UINT32_MAX),
        int64_t(UINT32_MAX)+1, INT64_MAX-17, INT64_MAX-1, INT64_MAX};
    for (int64_t x : values) for (int64_t s : values) check(x, s);
    for (int64_t stride : values) if (stride > 0) {
        for (int64_t rb : values) if (rb >= 0) for (int delta = -1; delta <= 1; ++delta) {
            const __int128 x = __int128(rb)*stride + delta;
            if (x >= INT64_MIN && x <= INT64_MAX) { check(int64_t(x), stride); ++boundaries; }
        }
        for (int64_t x : {INT64_MAX, INT64_MAX-1, INT64_MAX-stride}) check(x, stride);
    }
    // Dense small domain includes every invalid stride/sign/alignment branch.
    for (int64_t s = -32; s <= 256; ++s) for (int64_t d = -33; d <= 4096; ++d) check(d, s);
    std::mt19937_64 rng(0x8ca5fa172973e021ull);
    for (int n = 0; n < 200000; ++n) {
        const int64_t x = int64_t(rng()), s = int64_t(rng());
        check(x, s); check(int64_t(uint64_t(x) & INT64_MAX), int64_t(uint64_t(s) & INT64_MAX));
    }
    std::printf("GDN_QUOTIENT_PLAN PASS pairs=%llu window_predicates=%llu unaligned=%llu invalid_sign_or_stride=%llu boundary_cases=%llu no_signed_product_overflow=1 both_specializations=1\n",
        (unsigned long long)pairs, (unsigned long long)predicates, (unsigned long long)unaligned,
        (unsigned long long)negatives, (unsigned long long)boundaries);
}
