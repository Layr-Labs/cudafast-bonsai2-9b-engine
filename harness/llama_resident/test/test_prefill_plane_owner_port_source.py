#!/usr/bin/env python3
"""Pin owner ON to passed27f0589 and owner OFF to full original b954 header."""
from hashlib import sha256
from pathlib import Path

FLAG = '''// Reduce D4-gather bank conflicts by permuting only producer ownership.
// Canonical weights/shared layout and warp compute mapping remain unchanged.
static bool ggml_cuda_prefill_plane_owner_enabled() {
    static const bool enabled = ggml_cuda_dl_env_int("GGML_CUDA_PREFILL_PLANE_OWNER", 1) != 0;
    return enabled;
}

'''
OLD_TEMPLATE = 'template <bool ADD,bool GLU,bool D4_ALIAS=false>'
NEW_TEMPLATE = 'template <bool ADD,bool GLU,bool D4_ALIAS=false,bool OWNER=false>'
OLD_PRODUCER = 'const int tid=threadIdx.y*32+threadIdx.x,quarter=tid%4,row=tid/4;'
NEW_PRODUCER = '''const int tid=threadIdx.y*32+threadIdx.x;
    const int owner=OWNER?(tid^(((tid>>2)&3)<<6)):tid,quarter=owner%4,row=owner/4;'''
OLD_Y = '(2*tid%128)*144+(tid/64)*4;'
NEW_Y = '(2*owner%128)*144+(owner/64)*4;'
OLD_DISPATCH = '''    const bool alias=ggml_cuda_prefill_plane_d4_alias_enabled();
    const auto kernel=alias?mul_mat_q_pq2_plane<false,true,true>:mul_mat_q_pq2_plane<false,true,false>;
    static bool attr[2][GGML_CUDA_MAX_DEVICES]={};
    if (!attr[alias][device]) {'''
NEW_DISPATCH = '''    const bool alias=ggml_cuda_prefill_plane_d4_alias_enabled();
    const bool owner=alias && ggml_cuda_prefill_plane_owner_enabled();
    const auto kernel=owner?mul_mat_q_pq2_plane<false,true,true,true>:
                      alias?mul_mat_q_pq2_plane<false,true,true,false>:mul_mat_q_pq2_plane<false,true,false,false>;
    static bool attr[2][2][GGML_CUDA_MAX_DEVICES]={};
    if (!attr[alias][owner][device]) {'''


def normalize_owner_header(source):
    """Undo only the enumerated owner addition, then require exact parent hash."""
    for text in (FLAG, NEW_TEMPLATE, NEW_PRODUCER, NEW_Y, NEW_DISPATCH,
                 'attr[alias][owner][device]=true;'):
        assert source.count(text) == 1, text
    source = source.replace(FLAG, '').replace(NEW_TEMPLATE, OLD_TEMPLATE)
    source = source.replace(NEW_PRODUCER, OLD_PRODUCER).replace(NEW_Y, OLD_Y)
    source = source.replace(NEW_DISPATCH, OLD_DISPATCH)
    source = source.replace('attr[alias][owner][device]=true;', 'attr[alias][device]=true;')
    assert sha256(source.encode()).hexdigest() == 'ffd2383949340c29b7f18ecdade486b6a4d769ef68d0e0532286b44feeb71c15'
    return source


def main():
    root = Path(__file__).resolve().parents[3]
    test = Path(__file__).resolve().parent
    source = (root / 'llama.cpp/ggml/src/ggml-cuda/mmq-pq2-plane.cuh').read_text()
    original = normalize_owner_header(source)
    start = source.index(NEW_TEMPLATE)
    enabled = source[start:source.index('// Only complete, contiguous GLU tiles', start)].strip()
    enabled = enabled.replace(NEW_TEMPLATE, OLD_TEMPLATE)
    enabled = enabled.replace('OWNER?(tid^(((tid>>2)&3)<<6)):tid', 'tid^(((tid>>2)&3)<<6)')
    enabled = enabled.replace('mul_mat_q_pq2_plane(', 'prefill_plane_owner(')
    assert sha256(enabled.encode()).hexdigest() == 'f732ad469aedac4fb65528f953c7d4af5757aaa3e5ee4b9cafc65634a50d89bb'
    reference = (test / 'prefill_plane_owner_reference.cuh').read_text()
    reference = reference[reference.index(OLD_TEMPLATE):].strip()
    assert sha256(reference.encode()).hexdigest() == '4d473c0cf86ec770b571566d9ac7a4278c781e8f83d0e268034484199c96ebce'
    start = original.index(OLD_TEMPLATE)
    original_kernel = original[start:original.index('// Only complete, contiguous GLU tiles', start)].strip()
    assert reference.replace('prefill_plane_owner_b954_reference(', 'mul_mat_q_pq2_plane(') == original_kernel
    pinned = {
        'llama.cpp/ggml/src/ggml-cuda/mmq-pq2-j128.cuh': 'a4725ab1c0d0200a71cd8347626659590b6a549e67d36dac29d97c5136f15074',
        'llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu': '84d806261ae0547fee98f5a590f145ab357390148fb1edde23c30434ff883d91',
        'llama.cpp/ggml/src/ggml-cuda/gated_delta_net_dedup.cuh': '046575952e427bfefb5b65a3e15ce5c670e70f059a47cfd2c87063f743d3eb95',
        'harness/llama_resident/test/prefill_plane_owner_plan.cpp': 'e2f6081daa8c6e22d36ce3714a78a59088e82e817c4ad9c42ba8c581c21c0977',
    }
    for path, digest in pinned.items():
        data = (root / path).read_bytes()
        if path == 'llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu':
            from test_gdn_half_source import without_half_launcher
            data = without_half_launcher(data.decode()).encode()
        assert sha256(data).hexdigest() == digest, path
    route = (test / 'prefill_plane_equivalence.cu').read_text()
    assert 'owner environment genuinely absent' in route
    assert 'actual production owner/alias/original function identity' in route
    assert 'prefill_plane_owner_b954_reference<false,true,true>' in route
    assert 'all nine production kernel arguments' in route
    assert 'cudaGraphKernelNodeSetParams' not in route
    assert 'cudaGraphDestroy(graph)); // all graphs die before registration/storage teardown' in route
    assert '--owner' in route and '--owner-off' in route and '--alias-off' in route and '--disabled' in route
    assert 'for (int cycle=0;cycle<2;++cycle)' in route and 'for(int replay=0;replay<4;++replay)' in route
    print('PLANE_OWNER_PORT_SOURCE_PASS passed_enabled_kernel=1 frozen_parent_reference=1 original_off_header=1 canonical_format=1 default_on=1 explicit_zero=1 quotient_parent_unchanged=1')


if __name__ == '__main__':
    main()
