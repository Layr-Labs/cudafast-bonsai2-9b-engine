#!/usr/bin/env python3
"""Exact original FP bodies plus the GPU-audited private-format recurrence."""
from pathlib import Path
import hashlib
root = Path(__file__).resolve().parents[3]
a = (root / 'llama.cpp/ggml/src/ggml-cuda/gated_delta_net_kernel.cuh').read_text()
b = (root / 'llama.cpp/ggml/src/ggml-cuda/gated_delta_net_dedup.cuh').read_text()
def pieces(text):
 generic=text[text.index('gated_delta_net_cuda(const'):text.index('// GB10, n_tokens >= GDN_L8_MIN_TOKENS')]
 long=text[text.index('gated_delta_net_l8_cuda(const'):text.index('// GB10 verify-sized ubatches')]
 longbody=long[long.index('    if (!replayed)'):]
 verify=text[text.index('gated_delta_net_l8v_cuda(const'):]
 verifybody=verify[verify.index('    __shared__ float attn_sm'):];verifybody=verifybody[:verifybody.index('\n    if (lazy) {')]
 return generic,longbody,verifybody

expected = ['a136ab7a6307b90ea2b8b833eca1fb514d6fdbd4ca353487ba20295c08626b1a', 'c0e9c6d13ef594026deda7c1ef4bbcddfed3c275331be475dd7c91671347f025', '84abfa7996102b09ff5468c963070f2e0735f8f26a33e9c5f71d6c4b7f8599a4']
assert [hashlib.sha256(x.encode()).hexdigest() for x in pieces(a)] == expected, 'original generic/long/verify FP bodies changed'
x = pieces(a)[2]
y = b[b.index('    __shared__ float attn_sm'):]
y = y[:y.index('\n    if (lazy) {')].replace('if (!lazy)', 'if constexpr (!LAZY)')
assert x == y, 'dedup verify FP body changed'
start = '                    float kr[GDN_L8_ROWS];'
end = '                replayed = true;'
x = a[a.index('gated_delta_net_l8v_cuda(const'):]
x = x[x.index(start):x.index(end)]
y = b[b.index(start):b.index(end)]
assert x == y, 'dedup replay FP body changed'
assert 'ggml_cuda_pdl_lc()' not in b, 'counter/cache ownership requires completion before next invocation'
print('GD2L production source PASS: original generic/long/verify bodies and dedup replay/FP order')
