#!/usr/bin/env python3
"""Pin the exact quotient guard and preserve the accepted239 frontier changes."""
from hashlib import sha256
from pathlib import Path

test = Path(__file__).resolve().parent
root = test.parents[2]
reference = (test / 'gdn_quotient_reference.cuh').read_text()
prefix = ('#include "gated_delta_net_kernel.cuh"\n\n'
          '// Frozen b9549c7 control: retain the two-division alignment guard independently.\n'
          'namespace gdn_quotient_reference {\n')
suffix = '\n} // namespace gdn_quotient_reference\n'
assert reference.count(prefix) == 1 and reference.endswith(suffix)
original = reference.replace(prefix, '#include "gated_delta_net_kernel.cuh"\n', 1)[:-len(suffix)]
assert sha256(original.encode()).hexdigest() == '6863514955cd22250ddd40d93ad22c6f0a7e309e21719c92ee65e4335a0c3e40'
assert sha256((test / 'gdn_dedup_route.cu').read_bytes()).hexdigest() == '546e7448b905818957853a3fe97a32ef1e99dd8fd579883713553458435a843e'
old = '''        if (state_slot_stride > 0 && diff >= 0 && diff % state_slot_stride == 0) {
            const int64_t rb  = diff / state_slot_stride;
'''
new = '''        int64_t slot_q = 0;
        if constexpr (REUSE_QUOTIENT) {
            if (state_slot_stride > 0 && diff >= 0) {
                slot_q = diff / state_slot_stride;
            }
        }
        // Positive stride and nonnegative diff give slot_q*stride <= diff <= INT64_MAX.
        // The product check is exact and reuses the quotient needed by rollback.
        if (state_slot_stride > 0 && diff >= 0 &&
                (REUSE_QUOTIENT ? slot_q * state_slot_stride == diff : diff % state_slot_stride == 0)) {
            const int64_t rb = REUSE_QUOTIENT ? slot_q : diff / state_slot_stride;
'''
assert original.count(old) == 1
expected = original.replace(old, new).replace('template <bool RAW, bool G_PRECOMPUTED, bool LAZY>',
    'template <bool RAW, bool G_PRECOMPUTED, bool LAZY, bool REUSE_QUOTIENT = false>', 1)
assert (root / 'llama.cpp/ggml/src/ggml-cuda/gated_delta_net_dedup.cuh').read_text() == expected
# Constant-folding OFF selects the literal old guard, quotient and complete body.
assert new.split('        if (state_slot_stride > 0 && diff >= 0 &&\n')[1].split('\n')[0].endswith(': diff % state_slot_stride == 0)) {')
launcher = (root / 'llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu').read_text()
new_launch = '''                        // Exact integer-only guard rewrite. Set 0 before graph construction
                        // to retain the original remainder-then-quotient path for A/B runs.
                        static const bool reuse_quotient = getenv("GGML_CUDA_GDN_QUOTIENT_REUSE") == nullptr ||
                                                          atoi(getenv("GGML_CUDA_GDN_QUOTIENT_REUSE")) != 0;
                        const auto kernel = reuse_quotient
                            ? gated_delta_net_dedup_l8v_cuda<RAW, G_PRECOMPUTED, true, true>
                            : gated_delta_net_dedup_l8v_cuda<RAW, G_PRECOMPUTED, true, false>;
                        ggml_cuda_kernel_launch(kernel, launch_params,
'''
old_launch = '''                        ggml_cuda_kernel_launch(gated_delta_net_dedup_l8v_cuda<RAW, G_PRECOMPUTED, true>, launch_params,
'''
assert launcher.count(new_launch) == 1
assert sha256(launcher.replace(new_launch, old_launch).encode()).hexdigest() == '5e4282f65708a145094ff7990c73cd3ba2ac5ec6db9fa7b448c4d91062889866'
# Accepted239 N1/cache/runtime changes are carried byte-for-byte; the launcher
# above permits only our enumerated quotient selector in that one file.
frontier_pins = {
    'harness/llama_resident/test/gdn_dedup_route.cu': '546e7448b905818957853a3fe97a32ef1e99dd8fd579883713553458435a843e',
    'harness/llama_resident/test/gdn_lazy_rollback_equivalence.cu': '85cca64e963b20558671fd30a1f5934d2ebcff3cb749c2c4d868480d9203d2a4',
    'llama.cpp/ggml/src/ggml-backend.cpp': '03f343acd6e9a199bb91a176e5606479a2f0fb9a95b2c030fae06d93f1dd342f',
    'llama.cpp/ggml/src/ggml-cuda/gated_delta_net_kernel.cuh': 'c2d2045986b5bda6c95bac740e91adf380723cdaba045bb8a1d63420d6741872',
    'llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu': '79aaf6e3bba48e1c1385c9bcdda6da4d274f133d8e36293d82e3faae0e4784fe',
    'llama.cpp/src/llama-context.cpp': '51e407ab79db65e15dc293ba4ebd5d0772b71107ee78bd0a494884e07105fb1a',
    'llama.cpp/src/llama-context.h': '5325fe7e8602a3edabd6b6ee5031221b3f12132b3aef6367ec36d61b9443498c',
    'llama.cpp/src/models/dflash.cpp': '8578e9f86841f268262f374a16d031aa71ded41038c023550b42d15e94de818c',
}
for path, digest in frontier_pins.items():
    data = (root / path).read_bytes()
    if path == 'llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu':
        # Permit only the independently hash-pinned default-OFF concat helper;
        # the accepted239 frontier digest itself remains unchanged.
        from test_dflash_concat_q8_source import without_concat_backend
        data = without_concat_backend(data.decode()).encode()
    assert sha256(data).hexdigest() == digest, path

b = (test / 'gdn_quotient_route.cu').read_text()
assert 'cudaGraphClone(&graph,src)' in b and 'edge_identity(src,graph)' in b
assert 'if(frozen_reference)' in b and b.count('cudaGraphKernelNodeSetParams(node,&p)') == 1
assert 'candidate keeps actual backend dispatch unmodified' in b
assert 'actual RAW/PRE/LAZY/QUOTIENT specialization' in b
assert 'default route genuinely has absent environment' in b
assert '"--enabled"' in b and '"--disabled"' in b
assert b.count('Arm original(g,true),self(g,true),candidate(g,false);') == 2
assert 'for(int j=0;j<31;++j)' in b and 'values[j].data()' in b
assert 'constexpr int QUARTETS=21' in b and 'saved.restore(g)' in b
assert 'warm_steady_replays=64 cold_snapshot_replays=1' in b
print('GDN_QUOTIENT_SOURCE PASS only_guard_rewrite=1 default_on=1 explicit_off_original=1 frozen_b954_reference=1 accepted239_source_preserved=1 actual_production_dispatch=1 all_FP_protocol_source_identical=1')
