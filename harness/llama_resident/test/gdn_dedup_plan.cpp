// CPU proof of the private GD2L byte layout and per-head ticket protocol.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
static void verify(bool ok){if(!ok)std::abort();}
int main(){
    constexpr uint32_t flag=0x80000000u;uint64_t schedules=0,words=0;
    std::array<int,8> order{};std::iota(order.begin(),order.end(),0);
    do{
        for(uint32_t initial:{0u,8u,0x7ffffff8u,flag,flag+0x108u}){
            uint32_t counter=initial;std::array<uint32_t,8> tickets{};
            for(int cta:order){tickets[cta]=counter++;if((tickets[cta]&flag)&&(tickets[cta]&7)==7)counter=0;}
            for(auto t:tickets){verify((t&flag)==(initial&flag));verify(((t&~flag)/8)==((initial&~flag)/8));}
            verify(counter==((initial&flag)?0:initial+8));++schedules;
        }
    }while(std::next_permutation(order.begin(),order.end()));
    for(int W=1;W<=16;++W){
        std::vector<int> source(W*146),canonical(W*146),v(W*16),shared(W*130);
        std::iota(source.begin(),source.end(),0);
        for(int t=0;t<W;++t){for(int c=0;c<128;++c)shared[t*130+c]=source[t*146+c];for(int c=0;c<16;++c)v[t*16+c]=source[t*146+128+c];for(int j=0;j<2;++j)shared[t*130+128+j]=source[t*146+144+j];}
        for(int t=0;t<W;++t){for(int c=0;c<128;++c)canonical[t*146+c]=shared[t*130+c];for(int c=0;c<16;++c)canonical[t*146+128+c]=v[t*16+c];for(int j=0;j<2;++j)canonical[t*146+144+j]=shared[t*130+128+j];}
        verify(canonical==source);words+=canonical.size();
        verify(8+W*130<=128*128&&8+W*16<=128*128);
        for(uint32_t gen:{1u,2u,17u,0x0fffffffu})verify(((gen-1)&1)!=(gen&1));
    }
    // Global slot regions: per-CTA records2..9, shared10/11, base1 and
    // marked current0 are disjoint. Compact expansion only reads10/11.
    for(int z=0;z<8;++z){verify(2+z<10);verify((2+z)*128*128+8+16*16<(3+z)*128*128);}
    std::printf("GD2L CPU PASS schedules=%llu payload_words=%llu all_ticket_permutations=40320 import_one_pass=1 wrap=1\n",(unsigned long long)schedules,(unsigned long long)words);
}
