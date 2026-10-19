// Fused carry import at T0 against the independently materialized eager states.
#define main unused_gdn_dedup_route_main
#include "gdn_dedup_route.cu"
#undef main
#include "gated_delta_net_kernel.cuh"
#include "gdn-half-prefill.cuh"

int main() {
    Backend backend;
    Inputs inputs(backend);
    float * output=nullptr;
    check(cudaMalloc(&output,(ROW+2*G)*sizeof(float)));
    std::vector<uint32_t> poison(ROW+2*G,CANARY),actual(ROW+2*G);
    int cases=0;
    for(int mode:{0,1})for(bool gd2:{false,true}) {
        inputs.fill(12345+mode,mode);inputs.row(0);
        State eager(backend),lazy(backend);
        const auto start=State::initial(24680+mode);
        eager.put(start);lazy.put(start);
        Graph reference(backend,inputs,eager,16,mode,false,false,false,true);
        Graph writer(backend,inputs,lazy,16,mode,gd2);
        for(int generation=0;generation<3;++generation) {
            inputs.row(0);
            const auto previous=eager.get();
            reference.run();writer.run();
            const auto expected=eager.get(),before=lazy.get();
            const auto counts=writer.counts();
            for(int rollback=0;rollback<=16;++rollback) {
                inputs.row(rollback);
                check(cudaMemcpy(output,poison.data(),poison.size()*4,cudaMemcpyHostToDevice));
                const ggml_cuda_kernel_launch_params lp(dim3(H*8,1,1),dim3(32,4),0,backend.cuda().stream());
                const gdn_half_prefill::InitialState init{lazy.data(),lazy.data(),(const int32_t*)inputs.ids->data,
                    int64_t(ROW),int64_t(ROW),1,nullptr,0.0f,0.0f};
                ggml_cuda_kernel_launch(gdn_half_prefill::carry_packed<true,false,true>,lp,gdn_half_c16::Frame{},
                    (const gdn_half_c16::Coeff*)nullptr,(const __half*)nullptr,(const float*)nullptr,
                    (float*)nullptr,output+G,(float*)nullptr,1.0f,init);
                check(cudaStreamSynchronize(backend.cuda().stream()));
                check(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
                const auto& selected=rollback==16?previous:expected;
                const size_t offset=G+(rollback==16?0:size_t(rollback)*ROW);
                for(int i=0;i<G;++i)need(actual[i]==CANARY&&actual[G+ROW+i]==CANARY,"materializer output guards");
                for(size_t i=0;i<ROW;++i)if(actual[G+i]!=selected[offset+i]) {
                    std::fprintf(stderr,"initial mismatch mode=%d gd2=%d generation=%d rb=%d word=%zu got=%08x want=%08x\n",
                        mode,int(gd2),generation,rollback,i,actual[G+i],selected[offset+i]);
                    need(false,"exact GD1/GD2 materialized state");
                }
                ++cases;
            }
            equal("reader leaves full cache unchanged",before,lazy.get());
            equal("reader leaves counter unchanged",counts,writer.counts());
        }
    }
    // The deferred zero path bypasses stale lazy headers and zeros every word.
    {
        State state(backend);state.put(State::initial(999));inputs.row(0);
        const ggml_cuda_kernel_launch_params lp(dim3(H*8,1,1),dim3(32,4),0,backend.cuda().stream());
        const gdn_half_prefill::InitialState init{state.data(),state.data(),(const int32_t*)inputs.ids->data,
            int64_t(ROW),int64_t(ROW),1,state.data(),0.0f,0.0f};
        ggml_cuda_kernel_launch(gdn_half_prefill::carry_packed<true,false,true>,lp,gdn_half_c16::Frame{},
            (const gdn_half_c16::Coeff*)nullptr,(const __half*)nullptr,(const float*)nullptr,
            (float*)nullptr,output+G,(float*)nullptr,1.0f,init);
        check(cudaStreamSynchronize(backend.cuda().stream()));
        check(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<ROW;++i)need(actual[G+i]==0,"exact deferred positive zero");
        ++cases;
    }
    check(cudaFree(output));
    std::printf("GDN_FUSED_INITIAL_STATE_PASS cases=%d all_rb=0..16 GD1_GD2=1 repeated_generations=3 cache_counter_immutable=1\n",cases);
}
