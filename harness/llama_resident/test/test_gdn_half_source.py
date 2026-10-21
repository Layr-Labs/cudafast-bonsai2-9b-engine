#!/usr/bin/env python3
"""Whitelist C16 additions while reconstructing the independently pinned239 source.

These are source/provenance checks, not numerical equivalence claims for C16.
"""
from hashlib import sha256
from pathlib import Path

root = Path(__file__).resolve().parents[3]
HEADER_SHA = '0301cfaa8255510ed87cd0bae46fbabebf41cabac82481c60fb03694cd4076e0'
BRANCH_SHA = '6e875f735d256de3fe0302edf69b0eb9179d430b7476891791f2d80d090d5bd9'


def without_half_launcher(source):
    include = '#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n#include "gdn-half-prefill.cuh"\n#endif\n'
    assert source.count(include) == 1
    start = source.index('        static const bool half_prefill =')
    end = source.index('        ggml_cuda_pool_alloc<gdn_chunk32>', start)
    assert sha256(source[start:end].encode()).hexdigest() == BRANCH_SHA
    assert sha256((root / 'llama.cpp/ggml/src/ggml-cuda/gdn-half-prefill.cuh').read_bytes()).hexdigest() == HEADER_SHA
    return (source[:start] + source[end:]).replace(include, '', 1)


def without_half_route_test(source):
    replacements = [
        ('int gdns=0,prepares=0,scans=0,resets=0,half_prepares=0,half_scans=0,materializers=0;',
         'int gdns=0,prepares=0,scans=0,resets=0;'),
        ('half_prepares+=std::strstr(name,"prepare_packed")!=nullptr;half_scans+=std::strstr(name,"carry_packed")!=nullptr;materializers+=std::strstr(name,"gdn_half_initial_state")!=nullptr;', ''),
        ('const char* half_env=std::getenv("GGML_CUDA_GDN_HALF_PREFILL");const bool half=n1()&&(!half_env||std::atoi(half_env)!=0);need(prepares==int(n1()&&!half)&&scans==int(n1()&&!half),"captured C32 prepare/scan selection");need(half_prepares==int(half)&&half_scans==int(half)&&materializers==int(half),"captured C16 prepare/carry/materializer selection");', 'need(prepares==int(n1())&&scans==int(n1()),"captured N1 prepare/scan selection");'),
    ]
    for new, old in replacements:
        assert source.count(new) == 1
        source = source.replace(new, old, 1)
    return source
