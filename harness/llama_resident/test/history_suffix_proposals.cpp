#include "ngram-cache.h"
#include <cassert>
#include <cstdio>
#include <vector>
using V=std::vector<llama_token>;
int main() {
 assert(common_history_suffix_draft({},1,3).empty());
 assert(common_history_suffix_draft({1,2,3,4,5},4,0).empty());
 // The query is [1,2,3,anchor=4], continuation only from committed earlier history.
 assert(common_history_suffix_draft({1,2,3,4,5,6,99,1,2,3},4,2)==V({5,6}));
 assert(common_history_suffix_draft({1,2,3,4,5,6,99,1,2},3,2).empty());
 // Two longest matches with different futures must abstain.
 assert(common_history_suffix_draft({1,2,3,4,5,6,8,1,2,3,4,7,8,9,1,2,3},4,2).empty());
 assert(common_history_suffix_draft({1,2,3,4,5,6,9,1,2,3,4,5,6,9,1,2,3},4,2)==V({5,6}));
 // A longer suffix supersedes conflicting shorter matches.
 assert(common_history_suffix_draft({0,1,2,3,4,5,6,9,8,1,2,3,4,7,8,9,0,1,2,3},4,2)==V({5,6}));
 // Never propose ids beyond the committed history tail.
 assert(common_history_suffix_draft({1,2,3,4,5,99,1,2,3},4,6).empty());
 assert(common_history_suffix_draft({1,1,1,1,1,1,1,1},1,2)==V({1,1}));
 std::puts("HISTORY_SUFFIX_PROPOSALS_PASS cases=9 target_verification_required=1");
}
