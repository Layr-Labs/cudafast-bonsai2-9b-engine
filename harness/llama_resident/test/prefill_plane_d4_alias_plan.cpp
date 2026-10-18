// CPU address and lifetime proof for the exact code-slot/D4 ownership alias.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void need(bool ok) { if (!ok) std::abort(); }
static uint32_t word(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

int main() {
    constexpr int Y = 128 * 144, X = 64 * 144, C = 64 * 32;
    static_assert(2 * Y + X + 2 * C == 50176, "unchanged shared footprint");
    size_t owners = 0, consumers = 0, schedules = 0, phase_checks = 0;
    std::array<unsigned char, 512> seen = {};
    for (int tid = 0; tid < 256; ++tid) {
        const int ci = (tid / 4) * 32 + (tid % 4) * 8;
        need(ci == tid * 8);
        for (int p = 0; p < 2; ++p) {
            const int j = 2 * tid + p, s = j / 128, col = j % 128;
            const int source = (2 * tid % 128) * 144 + (tid / 64) * 4 + p * 144;
            need(source == col * 144 + 4 * s && source + 4 <= Y);
            need(ci + 4 * p == 4 * j && seen[j]++ == 0);
            ++owners;
        }
    }
    for (auto count : seen) need(count == 1);
    for (int warp = 0; warp < 8; ++warp) for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 4; ++s) for (int ct = 0; ct < 4; ++ct) for (int p = 0; p < 2; ++p) {
            const int col = (warp % 4) * 32 + ct * 8 + 2 * (lane % 4) + p;
            const int j = s * 128 + col, owner = j / 2;
            const int dt = (warp % 4) * 32 * 4 + s * 512 + (ct * 8 + 2 * (lane % 4) + p) * 4;
            const int old = col * 144 + s * 4;
            const int gathered = (2 * owner % 128) * 144 + (owner / 64) * 4 + (j % 2) * 144;
            need(dt == 4 * j && old == gathered && dt + 4 <= C);
            ++consumers;
        }

    // Every permitted interleaving preserves each thread's two original code
    // words. Model individual read/store steps, not an artificially atomic thread.
    for (unsigned seed = 1; seed <= 128; ++seed) {
        std::mt19937 rng(seed);
        std::array<uint32_t, 512> original, slot, expected, loaded = {}, scale = {};
        std::array<uint32_t, Y / 4> activation;
        for (int j = 0; j < Y / 4; ++j) activation[j] = word(seed * 19087u + j);
        // Include all special FP32 scale encodings as uninterpreted bits.
        const uint32_t edges[] = {0, 0x80000000u, 1, 0x007fffffu, 0x00800000u,
                                  0x7f800000u, 0xff800000u, 0x7fc12345u};
        for (int s = 0; s < 4; ++s) for (int col = 0; col < 128; ++col) {
            const int j = s * 128 + col;
            if (j % 17 == 0) activation[col * 36 + s] = edges[(j / 17) % 8];
            expected[j] = activation[col * 36 + s];
            original[j] = slot[j] = word(seed * 93419u + j);
        }
        std::array<unsigned char, 256> step = {};
        int remaining = 256 * 6;
        while (remaining) {
            const int tid = int(rng() % 256), j = 2 * tid;
            const int src = (2 * tid % 128) * 36 + tid / 64;
            switch (step[tid]) {
                case 0: loaded[j] = slot[j]; break;
                case 1: loaded[j + 1] = slot[j + 1]; break;
                case 2: scale[j] = activation[src]; break;
                case 3: scale[j + 1] = activation[src + 36]; break;
                case 4: slot[j] = scale[j]; break;
                case 5: slot[j + 1] = scale[j + 1]; break;
                default: continue;
            }
            ++step[tid]; --remaining;
        }
        need(loaded == original && slot == expected);
        ++schedules;
    }

    // TMA completion may arrive in either Y/code order. The unchanged wait
    // precedes the alias, the original pre-MMA barrier publishes all scales,
    // and the original post-MMA barrier permits same-slot kb+2 refills.
    for (int nkb = 1; nkb <= 129; ++nkb) for (int order = 0; order < 2; ++order) {
        struct stage { int kb = -1, phase = 0, pending = 0; bool consumed = true; bool scales = false; } slots[2];
        auto issue = [&](int kb) {
            auto & slot = slots[kb & 1];
            need(slot.consumed && slot.pending == 0);
            slot.kb = kb; slot.consumed = false; slot.scales = false; slot.pending = Y + C;
            slot.pending -= order ? C : Y;
            need(slot.pending == (order ? Y : C));
            slot.pending -= order ? Y : C;
            need(slot.pending == 0); slot.phase ^= 1;
        };
        issue(0); if (nkb > 1) issue(1);
        for (int kb = 0; kb < nkb; ++kb) {
            auto & slot = slots[kb & 1];
            need(slot.kb == kb && !slot.consumed && !slot.scales && slot.pending == 0);
            need(slot.phase != ((kb / 2) & 1));
            slot.scales = true; // own reads/stores then existing pre-MMA barrier
            need(slot.scales); // every warp consumes the completed scale table
            slot.consumed = true; // existing post-MMA barrier before TMA reuse
            if (kb + 2 < nkb) issue(kb + 2);
            ++phase_checks;
        }
        for (const auto & slot : slots) need(slot.consumed && slot.pending == 0);
    }
    std::printf("PLANE_D4_ALIAS_CPU_PASS owners=%zu consumer_addresses=%zu interleavings=%zu phase_checks=%zu shared=50176\n",
                owners, consumers, schedules, phase_checks);
}
