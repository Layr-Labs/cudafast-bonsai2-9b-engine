// Compare the same C16 arithmetic through the actual natural/packed producers.
#define main unused_gdn_prefill_layout_main
#include "gdn_prefill_layout_equivalence.cu"
#undef main

int main(int argc,char ** argv) {
    // This fixture compares natural and permuted F32 producers with the same Gram arithmetic.
    setenv("GGML_CUDA_CONV_C16_PACK","0",1);
    for(int i=1;i<argc;++i)if(std::strcmp(argv[i],"--legacy-control")==0) {
        // Diagnostic only: reproduce the inherited H4/C32 fixture failure with
        // the new method explicitly disabled. This is not a tolerance change.
        setenv("GGML_CUDA_GDN_HALF_PREFILL","0",1);
        correctness();return 0;
    }
    unsetenv("GGML_CUDA_GDN_HALF_PREFILL");
    int checks=0;
    for(int tokens:{64,128,512})for(bool raw:{false,true})for(bool zero:{false,true}) {
        Shape s{32,tokens,K,raw,zero,false,MATCH,"C16-natural-packed"};
        auto ps=pipelines(s);
        for(int generation=0;generation<3;++generation) {
            Result reference;
            for(int mode=0;mode<2;++mode) {
                auto& p=*ps[mode];p.upload(701+generation*37);p.run();p.assert_route();
                auto result=p.fetch();
                if(mode==0)reference=std::move(result);
                else {compare(s,p.mode,generation,0,reference,result);++checks;}
            }
        }
    }
    for(int tokens:{16,512}) {
        Shape seed{32,tokens,K,true,false,false,MATCH,"C16-lazy-seed"};
        auto seeds=pipelines(seed);std::array<Result,2> saved;
        for(int mode=0;mode<2;++mode) {
            seeds[mode]->upload(911);seeds[mode]->run();seeds[mode]->assert_route();
            saved[mode]=seeds[mode]->fetch();
        }
        Shape next{32,64,K,true,false,false,MATCH,"C16-lazy-rollback"};
        auto ps=pipelines(next);
        for(int rollback=0;rollback<=K;++rollback) {
            Result reference;
            for(int mode=0;mode<2;++mode) {
                auto& p=*ps[mode];p.upload(1301+rollback,&saved[mode],rollback);p.run();p.assert_route();
                auto result=p.fetch();
                if(mode==0)reference=std::move(result);
                else {compare(next,p.mode,rollback,rollback,reference,result);++checks;}
            }
        }
    }
    std::printf("GDN_HALF_PREFILL_LAYOUT_PASS checks=%d natural_packed=1 raw_normal=1 zero=1 all_rb=0..16\n",checks);
}
