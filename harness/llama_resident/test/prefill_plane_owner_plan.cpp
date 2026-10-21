// CPU ownership/bank witness for the one test-only producer permutation.
// Explicit checks remain active in Release and do not depend on assert().
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <vector>

static void need(bool x, const char * message) {
    if (!x) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
static int owner(int tid) { return tid ^ (((tid >> 2) & 3) << 6); }
static int bank(int byte) { return (byte / 4) % 32; }
static std::pair<int, int> conflicts(const std::vector<int> & bytes) {
    std::array<std::set<int>, 32> words;
    for (int a : bytes) words[bank(a)].insert(a / 4);
    int worst = 0, active = 0;
    for (const auto & w : words) { worst = std::max(worst, int(w.size())); active += !w.empty(); }
    return {worst, active};
}
int main() {
    std::array<int, 256> seen = {};
    for (int tid = 0; tid < 256; ++tid) {
        const int p = owner(tid); ++seen[p];
        need(owner(p) == tid && p % 64 == tid % 64, "involution and unchanged low six bits");
        for (int off = 0; off < 32; off += 4)
            need(bank((p / 4) * 144 + (p % 4) * 32 + off) ==
                 bank((tid / 4) * 144 + (tid % 4) * 32 + off), "all X-store banks unchanged");
        if (tid % 4 == 0)
            need(bank((p / 4) * 144 + 128) == bank((tid / 4) * 144 + 128), "scale-store bank unchanged");
    }
    need(std::all_of(seen.begin(), seen.end(), [](int x) { return x == 1; }), "256 unique owners");
    int old_scale_requests = 0, new_scale_requests = 0;
    for (int warp = 0; warp < 8; ++warp) {
        for (int offset : {0, 144}) {
            std::vector<int> old, next;
            for (int lane = 0; lane < 32; ++lane) {
                const int t = warp * 32 + lane, p = owner(t);
                old.push_back((2 * t % 128) * 144 + (t / 64) * 4 + offset);
                next.push_back((2 * p % 128) * 144 + (p / 64) * 4 + offset);
            }
            need(conflicts(old) == std::make_pair(8, 4), "original scalar D4 loads eight-way");
            need(conflicts(next) == std::make_pair(2, 16), "candidate scalar D4 loads two-way");
        }
        for (int part = 0; part < 2; ++part) for (bool permuted : {false, true}) {
            std::vector<int> words;
            for (int lane = part * 16; lane < (part + 1) * 16; ++lane) {
                const int t = warp * 32 + lane, p = permuted ? owner(t) : t;
                words.push_back(p * 8); words.push_back(p * 8 + 4);
            }
            need(conflicts(words) == std::make_pair(1, 32), "code64/alias-store64 128B groups balanced");
        }
        std::set<int> old_sectors, next_sectors;
        for (int lane = 0; lane < 32; lane += 4) {
            const int t = warp * 32 + lane;
            old_sectors.insert(2 * (t / 4) / 32); next_sectors.insert(2 * (owner(t) / 4) / 32);
        }
        old_scale_requests += int(old_sectors.size()); new_scale_requests += int(next_sectors.size());
    }
    need(old_scale_requests == 8 && new_scale_requests == 32, "global scale tradeoff charged");
    uint64_t steps = 0, code_bytes = 0, x_bytes = 0, d4_words = 0, consumer_words = 0;
    std::mt19937 rng(107);
    std::array<int, 256> order; std::iota(order.begin(), order.end(), 0);
    // Reordered producer execution cannot overwrite another thread's unread
    // code. The existing CTA barrier publishes the identical full X/D4 tiles.
    for (int nkb : {1, 2, 3, 5, 31, 32, 33}) for (int trial = 0; trial < 16; ++trial) {
        for (int kb = 0; kb < nkb; ++kb) {
            const int buffer = kb & 1, phase = (kb / 2) & 1;
            need(buffer == kb % 2 && phase == (kb / 2) % 2, "original TMA slot/phase");
            std::shuffle(order.begin(), order.end(), rng);
            std::array<int, 2048> code_uses = {};
            std::array<int, 64 * 144> x_uses = {};
            std::array<int, 64> scale_uses = {};
            std::array<int, 512> d4_source; d4_source.fill(-1);
            for (int tid : order) {
                const int p = owner(tid), row = p / 4, quarter = p % 4;
                for (int j = 0; j < 8; ++j) ++code_uses[p * 8 + j];
                for (int j = 0; j < 32; ++j) ++x_uses[row * 144 + quarter * 32 + j];
                if (quarter == 0) ++scale_uses[row];
                for (int j : {2 * p, 2 * p + 1}) {
                    need(d4_source[j] == -1, "one D4 writer");
                    d4_source[j] = (j % 128) * 144 + (j / 128) * 4;
                    need(j * 4 >= p * 8 && j * 4 + 3 < p * 8 + 8, "alias only own consumed code bytes");
                }
            }
            need(std::all_of(code_uses.begin(), code_uses.end(), [](int x) { return x == 1; }), "all code bytes once");
            for (int row = 0; row < 64; ++row) {
                need(scale_uses[row] == 1, "scale row follows original code row");
                for (int j = 0; j < 144; ++j)
                    need(x_uses[row * 144 + j] == int(j < 128), "exact X rows; scale/barrier padding untouched");
            }
            for (int wc = 0; wc < 4; ++wc) for (int s = 0; s < 4; ++s)
                for (int ct = 0; ct < 4; ++ct) for (int c = 0; c < 4; ++c) for (int h = 0; h < 2; ++h) {
                    const int column = wc * 32 + ct * 8 + 2 * c + h;
                    need(d4_source[s * 128 + column] == column * 144 + s * 4, "unchanged compute_compact DT address");
                    ++consumer_words;
                }
            ++steps; code_bytes += 2048; x_bytes += 8192; d4_words += 512;
        }
    }
    std::printf("PLANE_OWNER_PLAN_PASS owners=256 steps=%llu code_bytes=%llu X_bytes=%llu D4_words=%llu consumer_words=%llu D4_banks=8to2 scale_sector_requests=8to32 canonical_global_layout=1\n",
        (unsigned long long)steps, (unsigned long long)code_bytes, (unsigned long long)x_bytes,
        (unsigned long long)d4_words, (unsigned long long)consumer_words);
}
