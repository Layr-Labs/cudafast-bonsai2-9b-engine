#include "llama.h"
#include "common.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
static void require(bool ok) { if (!ok) { std::fprintf(stderr,"VERIFY_BATCH_FAIL\n");std::exit(1); } }
static void fill(llama_batch &b,int draft,int pos) {
    common_batch_clear(b);
    common_batch_add(b,42,pos,{0},true);
    for(int i=0;i<draft;++i) common_batch_add(b,100+i,pos+i+1,{0},true);
}
static double bench(int capacity) {
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<4000;++i) { auto b=llama_batch_init(capacity,0,1);fill(b,15,512);llama_batch_free(b); }
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/4000;
}
int main() {
    int cases=0;
    for(int capacity:{16,128,512,2048}) for(int draft=0;draft<=15;++draft) for(int pos:{0,512,8175}) {
        auto a=llama_batch_init(capacity,0,1),b=llama_batch_init(draft+1,0,1);fill(a,draft,pos);fill(b,draft,pos);
        require(a.n_tokens==b.n_tokens && b.n_tokens==draft+1 && b.seq_id[draft+1]==nullptr);
        for(int i=0;i<b.n_tokens;++i) require(a.token[i]==b.token[i] && a.pos[i]==b.pos[i] && a.n_seq_id[i]==b.n_seq_id[i] && a.seq_id[i][0]==b.seq_id[i][0] && a.logits[i]==b.logits[i]);
        llama_batch_free(a);llama_batch_free(b);++cases;
    }
    bench(2048);bench(16);
    std::vector<double>a,b;
    for(int i=0;i<3;++i) { if(i%2) {b.push_back(bench(16));a.push_back(bench(2048));} else {a.push_back(bench(2048));b.push_back(bench(16));} }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    std::printf("VERIFY_BATCH_PASS cases=%d exact_fields=true old_us=%.4f new_us=%.4f gain=%.4f old_range=%.4f..%.4f new_range=%.4f..%.4f\n",cases,a[1],b[1],a[1]/b[1],a.front(),a.back(),b.front(),b.back());
}
