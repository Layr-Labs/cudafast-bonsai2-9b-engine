// Real backend graphs: original projection made observable as the control;
// the candidate may omit only its exclusive fixup, retaining sum/norm/D4.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "common.cuh"
#include "mmq.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static void require(bool value,const char * message) {
    if (!value) {std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
static void check(cudaError_t error) {
    if (error!=cudaSuccess) {std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(error));std::exit(2);}
}
static std::vector<uint32_t> read(const ggml_tensor * t) {
    std::vector<uint32_t> result(ggml_nbytes(t)/4);
    ggml_backend_tensor_get(t,result.data(),0,result.size()*4);return result;
}
static void equal(const std::vector<uint32_t> & a,const std::vector<uint32_t> & b,const char * label) {
    require(a.size()==b.size(),"equal output sizes");
    for (size_t i=0;i<a.size();++i) if (a[i]!=b[i]) {
        std::fprintf(stderr,"%s word=%zu expected=%08x actual=%08x\n",label,i,a[i],b[i]);
        require(false,"all output bytes equal");
    }
}
struct spec {int k,n,views,chains;bool reverse=false,alias=false,extra=false,registered=true,norm=true;};
struct weights_and_inputs {
    static constexpr int GUARD=16;
    static constexpr uint32_t POISON=0x4a5a39c7;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * w,*w2,*probe,*rotation,*nw,*signs,*input_signs,*xraw,*rraw,*x,*res;
    spec s;
    weights_and_inputs(ggml_backend_t backend,spec s):s(s) {
        ctx=ggml_init({ggml_tensor_overhead()*32,nullptr,true});
        w=ggml_new_tensor_2d(ctx,GGML_TYPE_PQ2_0,s.k,4096);
        w2=ggml_new_tensor_2d(ctx,GGML_TYPE_PQ2_0,4096,4096);
        probe=ggml_new_tensor_2d(ctx,GGML_TYPE_PQ2_0,4096,128);
        rotation=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,4096);
        nw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4096);
        signs=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4096);
        input_signs=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,s.k);
        xraw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(s.k)*s.n+2*GUARD);
        rraw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(4096)*s.n+2*GUARD);
        x=ggml_view_2d(ctx,xraw,s.k,s.n,s.k*4,GUARD*4);
        res=ggml_view_2d(ctx,rraw,4096,s.n,4096*4,GUARD*4);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"weights/input allocation");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 random(20261018);
        auto reg=(size_t (*)(const ggml_tensor *))ggml_backend_reg_get_proc_address(
            ggml_backend_reg_by_name("CUDA"),"ggml_backend_cuda_register_decode_layout");
        require(reg,"real decode layout registration API");
        for (auto * t:{w,w2,probe}) {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            for (size_t block=0;block<bytes.size()/34;++block) {
                float scale=float(1+random()%100)/5000;
                if (block%127==0) scale=0;
                if (block%131==0) scale=-scale;
                if (block%137==0) scale=3e-7f;
                const auto bits=ggml_fp32_to_fp16(scale);std::memcpy(bytes.data()+34*block,&bits,2);
                for (int q=0;q<32;++q) for (int j=0;j<4;++j) bytes[34*block+2+q]|=uint8_t(random()%3)<<(2*j);
            }
            ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());
            if (s.registered && t!=probe) require(reg(t)==size_t(t->ne[0]/128)*t->ne[1]*28,"PTQ registration bytes");
        }
        std::vector<float> h(size_t(4096)*4096);
        for (int i=0;i<4096;++i) for (int j=0;j<4096;++j)
            h[size_t(i)*4096+j]=__builtin_parity(unsigned(i&j))?-1.0f/64:1.0f/64;
        ggml_backend_tensor_set(rotation,h.data(),0,h.size()*4);
    }
    ~weights_and_inputs(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void upload(unsigned seed,bool zero=false,bool tiny=false) {
        std::mt19937 random(seed);
        std::uniform_real_distribution<float> dist(-0.5f,0.5f);
        for (auto * t:{xraw,rraw}) {
            std::vector<uint32_t> words(ggml_nelements(t),POISON);
            for (size_t i=GUARD;i<words.size()-GUARD;++i) {
                float value=zero?0:dist(random)*(tiny?1e-30f:1.0f);
                if (i%97==0) value=-0.0f;
                std::memcpy(&words[i],&value,4);
            }
            ggml_backend_tensor_set(t,words.data(),0,words.size()*4);
        }
        for (auto * t:{nw,signs,input_signs}) {
            std::vector<float> values(ggml_nelements(t));
            for (float & v:values) v=t==nw?1.0f+dist(random):(random()&1?1.0f:-1.0f);
            ggml_backend_tensor_set(t,values.data(),0,values.size()*4);
        }
    }
    void guards() {
        for (auto * t:{xraw,rraw}) {
            const auto words=read(t);
            for (int i=0;i<GUARD;++i) require(words[i]==POISON && words[words.size()-1-i]==POISON,"input canaries");
        }
    }
};
struct graph_case {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_gallocr_t allocator;
    ggml_backend_t backend;
    weights_and_inputs & in;
    std::vector<ggml_tensor *> projections,sums,norms;
    ggml_tensor * output;
    bool baseline;
    graph_case(ggml_backend_t backend,weights_and_inputs & in,bool baseline):backend(backend),in(in),baseline(baseline) {
        ctx=ggml_init({ggml_tensor_overhead()*128+ggml_graph_overhead(),nullptr,true});
        graph=ggml_new_graph(ctx);
        // Distinct view objects retain identical input addresses but give both
        // graphs distinct real backend cache keys (first-node identity).
        auto * graph_input=ggml_view_2d(ctx,in.xraw,in.s.k,in.s.n,in.s.k*4,in.GUARD*4);
        auto * graph_res=ggml_view_2d(ctx,in.rraw,4096,in.s.n,4096*4,in.GUARD*4);
        ggml_build_forward_expand(graph,graph_input);ggml_build_forward_expand(graph,graph_res);
        auto transform=[&](ggml_tensor * x,ggml_tensor * signs) {
            auto * mul=ggml_mul(ctx,x,signs);
            auto * reshape=ggml_reshape_2d(ctx,mul,4096,ggml_nelements(x)/4096);
            auto * mm=ggml_mul_mat(ctx,in.rotation,reshape);
            ggml_set_op_params_i32(mm,1,GGML_HINT_SRC0_IS_HADAMARD);
            return ggml_reshape_2d(ctx,mm,x->ne[0],x->ne[1]);
        };
        auto * x=transform(graph_input,in.input_signs);
        auto * residual=graph_res;
        for (int chain=0;chain<in.s.chains;++chain) {
            auto * mm=ggml_mul_mat(ctx,chain?in.w2:in.w,x);projections.push_back(mm);
            if (baseline || in.s.extra) ggml_set_output(mm);
            auto * view=mm;
            for (int i=0;i<in.s.views;++i) view=ggml_reshape_2d(ctx,view,4096,in.s.n);
            auto * r=in.s.alias?view:residual;
            auto * sum=in.s.reverse?ggml_add(ctx,r,view):ggml_add(ctx,view,r);
            ggml_set_output(sum);sums.push_back(sum);
            auto * norm=ggml_mul(ctx,ggml_rms_norm(ctx,sum,1e-6f),in.nw);
            if (in.s.norm) ggml_set_output(norm);
            norms.push_back(norm);x=transform(norm,in.signs);residual=sum;
        }
        output=ggml_mul_mat(ctx,in.probe,x);ggml_set_output(output);ggml_build_forward_expand(graph,output);
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator,graph),"real graph allocation");
    }
    ~graph_case(){ggml_gallocr_free(allocator);ggml_free(ctx);}
    void run(){require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"real backend graph compute");}
    cudaGraph_t captured() {
#ifdef USE_CUDA_GRAPH
        auto & cuda=*(ggml_backend_cuda_context *)backend->context;
        for (auto & e:cuda.cuda_graphs) {
            const auto & cached=*e.second;
            if (cached.instance && cached.warmup_complete && !cached.node_props.empty() &&
                    cached.node_props.back().node.data==output->data && cached.node_props.size()==size_t(graph->n_nodes))
                return cached.graph;
        }
#endif
        require(false,"actual captured graph exists");return nullptr;
    }
    std::vector<std::vector<uint32_t>> inspect(int expected_fused) {
        cudaGraph_t g=captured();size_t count=0;check(cudaGraphGetNodes(g,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(g,nodes.data(),&count));
        std::vector<std::vector<uint32_t>> q8(sums.size());
        std::vector<void *> produced(sums.size(),nullptr),consumed(sums.size(),nullptr);
        int fused=0;
        for (auto node:nodes) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if (type!=cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));
            const char * name=nullptr;check(cudaFuncGetName(&name,p.func));
            const bool is_fused=std::strstr(name,"fwht_quantize_mmq_d4_fixup_add_rms");
            if (is_fused || std::strstr(name,"fwht_quantize_mmq_d4_add_rms")) {
                for (size_t i=0;i<sums.size();++i) if (*(void **)p.kernelParams[2]==sums[i]->data) {
                    const void * data=*(void **)p.kernelParams[8];
                    q8[i].resize(size_t(4096/128)*in.s.n*sizeof(block_q8_1_mmq)/4);
                    check(cudaMemcpy(q8[i].data(),data,q8[i].size()*4,cudaMemcpyDeviceToHost));
                    if (is_fused) {
                        ++fused;consumed[i]=*(void **)p.kernelParams[10];
                        require(p.gridDim.y==16 && p.blockDim.x==1024,"fused consumer geometry");
                        cudaFuncAttributes a;check(cudaFuncGetAttributes(&a,p.func));
                        require(a.localSizeBytes==0 && a.numRegs<=64,"fused resource bound");
                    }
                }
            }
            if (std::strstr(name,"mul_mat_q_j16_dl_ring")) {
                for (size_t i=0;i<projections.size();++i) if (*(void **)p.kernelParams[2]==projections[i]->data) {
                    produced[i]=*(void **)p.kernelParams[3];
                    require(p.gridDim.x==48 && p.blockDim.x==32 && p.blockDim.y==16 &&
                            *(int *)p.kernelParams[9]==0,"original producer partition and mode");
                }
            }
        }
        require(fused==expected_fused,"positive fused-kernel count or explicit fallback");
        if (fused) for (size_t i=0;i<sums.size();++i) {
            require(produced[i] && produced[i]==consumed[i],"captured exact producer/consumer scratch identity");
            require(!q8[i].empty(),"actual quantizer bytes observed");
        }
        return q8;
    }
};
int main(int argc,char ** argv) {
    const bool disabled=argc==2 && !std::strcmp(argv[1],"--disabled");
    const bool ring_off=argc==2 && !std::strcmp(argv[1],"--ring-off");
    const bool unsigned_mma=argc==2 && !std::strcmp(argv[1],"--unsigned");
    require(argc==1 || disabled || ring_off || unsigned_mma,"supported mode");
    if (disabled) setenv("GGML_CUDA_DECODE_FIXUP_ADD_RMS","0",1);
    else setenv("GGML_CUDA_DECODE_FIXUP_ADD_RMS","1",1);
    setenv("GGML_CUDA_DL_RING_PT",ring_off?"0":"1",1);
    setenv("GGML_CUDA_PTQ1_SIGNED",unsigned_mma?"0":"1",1);
    setenv("GGML_CUDA_DECODE_LAYOUT","1",1);setenv("GGML_CUDA_DECODE_PTQ1","1",1);
    setenv("GGML_CUDA_PDL","1",1);unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    const std::vector<spec> cases={
        {4096,16,0,2},{12288,16,2,2},{4096,16,1,1,false,false,false,true,false},
        {4096,1,0,1},{4096,7,0,1},{12288,15,0,1},{8192,16,0,1},
        {4096,16,0,1,true},{4096,16,0,1,false,true},
        {4096,16,0,1,false,false,true},{4096,16,0,1,false,false,false,false}};
    int checks=0;
    for (const auto s:cases) {
        ggml_backend_t backend=ggml_backend_cuda_init(0);require(backend,"CUDA backend");
        {
        weights_and_inputs inputs(backend,s);
        {
            graph_case original(backend,inputs,true),candidate(backend,inputs,false);
            const bool eligible=s.n==16 && s.k!=8192 && !s.reverse && !s.alias && !s.extra && s.registered;
            const int expected=!disabled && !ring_off && !unsigned_mma && eligible?s.chains:0;
            inputs.upload(1);
            for (int i=0;i<3;++i) {original.run();candidate.run();}
            std::vector<uint32_t> previous;
            for (int phase:{0,1,1,0,2,3,0}) {
                inputs.upload(701+phase,phase==2,phase==3);
                const auto x_before=read(inputs.xraw),r_before=read(inputs.rraw);
                original.run();const auto out=read(original.output);const auto qref=original.inspect(0);
                std::vector<std::vector<uint32_t>> sum,norm;
                for (auto * t:original.sums) sum.push_back(read(t));
                if (s.norm) for (auto * t:original.norms) norm.push_back(read(t));
                candidate.run();equal(out,read(candidate.output),"downstream GEMM");
                const auto qgot=candidate.inspect(expected);
                for (size_t i=0;i<sum.size();++i) {
                    equal(sum[i],read(candidate.sums[i]),"residual sum");
                    if (s.norm) equal(norm[i],read(candidate.norms[i]),"normalized product");
                    equal(qref[i],qgot[i],"all D4 scale/quantizer bytes");
                }
                equal(x_before,read(inputs.xraw),"input preserved");equal(r_before,read(inputs.rraw),"residual input preserved");
                inputs.guards();
                if (phase==1 && !previous.empty()) require(out!=previous,"changed input affects output");
                previous=phase==0?out:std::vector<uint32_t>{};++checks;
            }
            // Repeated capture replay and allocator reuse across two ordinary
            // graphs must not leave partials from the previous invocation.
            const auto expected_out=read(candidate.output);
            for (int i=0;i<64;++i) {original.run();candidate.run();}
            equal(expected_out,read(candidate.output),"reused scratch across 128 graph launches");
            candidate.inspect(expected);inputs.guards();
            std::printf("PASS K=%d N=%d views=%d chains=%d fused=%d reverse=%d alias=%d extra=%d registered=%d\n",
                s.k,s.n,s.views,s.chains,expected,s.reverse,s.alias,s.extra,s.registered);
        }
        }
        ggml_backend_free(backend);
    }
    std::printf("DECODE_FIXUP_ADD_RMS_ROUTE_PASS checks=%d cases=%zu disabled=%d\n",checks,cases.size(),disabled);
}
