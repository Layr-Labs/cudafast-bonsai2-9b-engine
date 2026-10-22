"""Source-bound CPU graph check; call only inside the root resource lease."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
import time

BASE = '759629bd402c90fac450c406615607e9b40e0ca2'


def selector(source, name):
    start = source.index('static void build_dflash2_selector(')
    end = source.index('// DFlash decoder, dual-mode', start)
    return source[start:end].replace('build_dflash2_selector', name, 1)


PRE = r'''
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
struct HParams { int64_t dflash_selector_top_k, dflash_selector_rank, dflash_block_size; };
struct Result { ggml_tensor * t_logits, * t_embd, * t_h_nextn = nullptr, * t_dflash_walk = nullptr; };
struct llama_model { ggml_tensor * dflash_selector_hidden, * dflash_selector_next, * dflash_selector_prev; };
struct llm_graph_context {
    ggml_context * ctx0; Result * res; HParams hparams; int64_t n_tokens, n_embd;
    struct { int64_t n_seqs_unq; } ubatch;
    struct { bool dflash_device_walk; } cparams;
    ggml_cgraph * gf;
    ggml_tensor * build_lora_mm(ggml_tensor * a, ggml_tensor * b) { return ggml_mul_mat(ctx0, a, b); }
    void cb(ggml_tensor *, const char *, int) {}
};
'''

POST = r'''
struct Answer { std::vector<float> lattice; std::vector<int> walk; };
static Answer run(bool old, int first, int length, int blocks, int k, bool strided, int seed) {
    int embd = ((k+k*k+7)/8)*8, rank=8, vocab=128, total=length*blocks;
    auto * ctx = ggml_init({8*1024*1024, nullptr, true});
    auto * backend = ggml_backend_cpu_init();
    auto * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,total);
    auto * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,embd,total);
    llama_model model {ggml_new_tensor_2d(ctx,GGML_TYPE_F32,embd,rank),
                       ggml_new_tensor_2d(ctx,GGML_TYPE_F32,rank,vocab),
                       ggml_new_tensor_2d(ctx,GGML_TYPE_F32,rank,vocab)};
    // Padding exercises noncontiguous column views, including the last live row.
    int width=k+(strided?3:0);
    auto * cand_src=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,width,total);
    auto * unary_src=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,total);
    auto * cand=ggml_view_2d(ctx,cand_src,k,total-first,cand_src->nb[1],first*cand_src->nb[1]);
    auto * unary=ggml_view_2d(ctx,unary_src,k,total-first,unary_src->nb[1],first*unary_src->nb[1]);
    // Production top-k/gather outputs are contiguous; test the source offset using
    // an explicit pack so the selector receives the same documented layout.
    cand=ggml_cont(ctx,cand); unary=ggml_cont(ctx,unary);
    auto * logits=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,total-first);
    Result result{logits,hidden};
    llm_graph_context g{ctx,&result,{k,rank,length},total,embd,{blocks},{false},ggml_new_graph(ctx)};
    if(old) reference_selector(g,model,tokens); // reference receives complete rows
    else candidate_selector(g,model,tokens,0,0,cand,unary,first);
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    auto setfloat=[&](ggml_tensor*t,int salt){std::vector<float>v(ggml_nelements(t));
      for(size_t i=0;i<v.size();++i)v[i]=float((int(i*7+salt+seed)%17)-8)/16.f;
      ggml_backend_tensor_set(t,v.data(),0,v.size()*4);};
    setfloat(hidden,1);setfloat(model.dflash_selector_hidden,2);
    setfloat(model.dflash_selector_next,3);setfloat(model.dflash_selector_prev,4);
    std::vector<int32_t>tok(total),ids(width*total);std::vector<float> vals(width*total);
    for(int r=0;r<total;++r){tok[r]=(r*3+seed)%vocab;
      for(int c=0;c<width;++c){ids[r*width+c]=(r*13+c+seed)%vocab;
        vals[r*width+c]=float((r+c+seed)%5)/8.f;
        if(seed==13 && r==0) vals[r*width+c]=std::numeric_limits<float>::quiet_NaN();
        if(seed==17) vals[r*width+c]=(c%2)?-0.0f:0.0f;}}
    ggml_backend_tensor_set(tokens,tok.data(),0,tok.size()*4);
    ggml_backend_tensor_set(cand_src,ids.data(),0,ids.size()*4);
    ggml_backend_tensor_set(unary_src,vals.data(),0,vals.size()*4);
    if(ggml_backend_graph_compute(backend,g.gf)!=GGML_STATUS_SUCCESS)std::abort();
    Answer a; a.lattice.resize(ggml_nelements(result.t_h_nextn));
    ggml_backend_tensor_get(result.t_h_nextn,a.lattice.data(),0,a.lattice.size()*4);
    for(int b=0;b<blocks;++b){int pred=0;
      for(int pos=1;pos<length;++pos){auto*row=a.lattice.data()+(b*length+pos)*embd;
        int best=0; for(int j=1;j<k;++j)if(row[k+pred*k+j]>row[k+pred*k+best])best=j;
        a.walk.push_back(int(row[best]));pred=best;}}
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);return a;
}
static bool same(const Answer&a,const Answer&b){return a.lattice.size()==b.lattice.size() &&
  std::memcmp(a.lattice.data(),b.lattice.data(),a.lattice.size()*4)==0 && a.walk==b.walk;}
int main(){int checks=0;size_t words=0;
  for(int k:{1,2,8,32})for(int len:{1,2,3,8,16})for(int blocks:{1,2})
    for(bool strided:{false,true})for(int seed:{0,7,13,17}){
      auto ref=run(true,0,len,blocks,k,strided,seed);
      auto fallback=run(false,0,len,blocks,k,strided,seed);
      if(!same(ref,fallback)){puts("FAIL fallback");return 1;}
      ++checks;words+=ref.lattice.size();
      if(blocks==1&&len>1){auto skip=run(false,1,len,blocks,k,strided,seed);
        if(!same(ref,skip)){puts("FAIL anchor crop");return 1;}
        ++checks;words+=ref.lattice.size();}}
  printf("PASS %d graph comparisons / %zu lattice words and host walks\n",checks,words);
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=pathlib.Path)
    ap.add_argument('--scratch', type=pathlib.Path)
    ap.add_argument('--output', type=pathlib.Path)
    args = ap.parse_args()
    root = args.source.resolve()
    scratch = args.scratch or pathlib.Path(tempfile.mkdtemp(prefix='yukon-cuda-anchor-'))
    source = (root/'llama.cpp/src/models/dflash.cpp').read_text()
    base = subprocess.check_output(['git','show',BASE+':llama.cpp/src/models/dflash.cpp'],cwd=root,text=True)
    post = POST.replace('reference_selector(g,model,tokens);','reference_selector(g,model,tokens,0,0,cand,unary);')
    (scratch/'check.cpp').write_text(PRE+selector(base,'reference_selector')+selector(source,'candidate_selector')+post)
    (scratch/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(anchorcheck C CXX)
set(CMAKE_CXX_STANDARD 17)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("'''+str(root/'llama.cpp/ggml')+'''" ggml)
add_executable(anchorcheck check.cpp)
target_link_libraries(anchorcheck PRIVATE ggml ggml-cpu)
''')
    commands=[['cmake','-S',str(scratch),'-B',str(scratch/'build'),'-DCMAKE_BUILD_TYPE=Release'],
              ['cmake','--build',str(scratch/'build'),'-j','2','--target','anchorcheck'],
              [str(scratch/'build/anchorcheck')],
              ['clang++','-std=c++17','-fsyntax-only','-Illama.cpp/include','-Illama.cpp/src',
               '-Illama.cpp/ggml/include','-Illama.cpp/ggml/src','llama.cpp/src/models/dflash.cpp']]
    receipt={'source':str(root),'base':BASE,'scratch':str(scratch),
             'sha256':hashlib.sha256(source.encode()).hexdigest(),'checks':[],'gpu_executed':False}
    output=args.output or scratch/'validation.json'
    for i,cmd in enumerate(commands):
        start=time.time();p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=scratch/f'check-{i}.log';log.write_text(p.stdout)
        receipt['checks'].append({'command':cmd,'exit_code':p.returncode,'seconds':time.time()-start,'log':str(log)})
        output.write_text(json.dumps(receipt,indent=2)+'\n')
        print(i,p.returncode,p.stdout[-1800:],flush=True)
        if p.returncode:
            raise SystemExit(p.returncode)
    print("receipt", output, flush=True)


if __name__=='__main__':
    main()
