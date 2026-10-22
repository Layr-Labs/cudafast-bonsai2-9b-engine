#include "ngram-cache.h"
#include <cassert>
#include <cstdio>
#include <vector>
using V = std::vector<llama_token>;
int main() {
    assert(common_history_suffix_draft({}, 8, 2).empty());
    assert(common_history_suffix_draft({1,2,3,4,5,6,99,1,2,3}, 4, 2).empty());
    assert(common_history_suffix_draft({1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7}, 8, 2) == V({9,10}));
    assert(common_history_suffix_draft({1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7}, 8, 0).empty());
    assert(common_history_suffix_draft({1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7}, 8, 20).empty());
    assert(common_history_suffix_draft({1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7,8,11,12,98,1,2,3,4,5,6,7}, 8, 2).empty());
    assert(common_history_suffix_draft({1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7,8,9,10,98,1,2,3,4,5,6,7}, 8, 2) == V({9,10}));
    assert(common_history_suffix_draft({0,1,2,3,4,5,6,7,8,9,10,99,1,2,3,4,5,6,7,8,11,12,98,0,1,2,3,4,5,6,7}, 8, 2) == V({9,10}));
    assert(common_history_suffix_draft(V(16, 1), 1, 2) == V({1,1}));
    std::puts("HISTORY_SUFFIX_CONFIDENCE_PASS cases=9 min_match=8 target_verification_required=1");
}
