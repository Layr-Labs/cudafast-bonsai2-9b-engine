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

# Independent bit-layout/hash property checks for the new memory-only path.
# This is a CPU mapping check, not CUDA compilation or performance evidence.
import random

def transpose(reg):
    out=[lane[:] for lane in reg]
    for r in range(4):
        x=[[reg[lane][m*4+r] for m in range(4)] for lane in range(32)]
        for mask, pairs in ((8, ((0,1),(2,3))), (16, ((0,2),(1,3)))):
            old=[row[:] for row in x]
            for lo, hi in pairs:
                selected=[row[lo if lane&mask else hi] for lane,row in enumerate(old)]
                for lane in range(32):
                    x[lane][lo if lane&mask else hi]=selected[lane^mask]
        for lane in range(32):
            for m in range(4): out[lane][m*4+r]=x[lane][m]
    return out

MASK=(1<<32)-1
def mix(h,x):
    return (h^((x+0x9e3779b9+((h<<6)&MASK)+(h>>2))&MASK))&MASK

def hashes(reg):
    h=[0x811c9dc5^lane for lane in range(32)]
    for c in range(4):
        for r in range(4):
            h=[mix(x,reg[lane][c*4+r]) for lane,x in enumerate(h)]
    for offset in (16,8,4,2,1):
        old=h[:]
        h=[mix(old[lane]^offset,old[lane^offset])^mix(old[lane^offset]^offset,old[lane]) for lane in range(32)]
    return h

special=(0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321)
for seed in range(8):
    rng=random.Random(seed)
    base=[special[i%8] if i%17==0 else rng.getrandbits(32) for i in range(512)]
    natural=[[base[c*128+r*32+lane] for c in range(4) for r in range(4)] for lane in range(32)]
    l8=transpose(natural)
    expected=[[base[(lane//8)*128+r*32+lane%8+8*m] for m in range(4) for r in range(4)] for lane in range(32)]
    assert l8==expected, 'natural->l8 register transpose'
    restored=transpose(l8)
    assert restored==natural and hashes(restored)==hashes(natural), 'self-inverse transpose and exact c/r hash'
    # Every staged GD1/GD2 K coordinate, V column and activated gate retains
    # the same raw word, including parity-token strides that are not 16B aligned.
    for v2 in (False,True):
        stride=130 if v2 else 146
        record=[rng.getrandbits(32) for _ in range(16*stride)]
        values=[rng.getrandbits(32) for _ in range(16*16)]
        for warp in range(4):
            for t in range(16):
                stage=[0]*136
                for lane in range(32):
                    for r in range(4):
                        stage[(r*8+lane%8)*4+lane//8]=record[t*stride+r*32+lane]
                for c in range(4): stage[128+c]=values[t*16+warp*4+c] if v2 else record[t*stride+128+warp*4+c]
                for j in range(2): stage[132+j]=record[t*stride+(128 if v2 else 144)+j]
                for lane in range(32):
                    for m in range(4):
                        for r in range(4):
                            assert stage[(r*8+lane%8)*4+m]==record[t*stride+r*32+lane%8+8*m]
                assert stage[132:134]==record[t*stride+(128 if v2 else 144):t*stride+(130 if v2 else 146)]
print('G1_CPU_LAYOUT PASS seeds=8 words=4096 raw_bits=1 transpose_roundtrip=1 checksum_order=1 GD1_GD2_staging=1')

# The new replay reads different storage but retains exactly these scalar FP
# expressions and the old tree. No pair algebra or coefficient compute remains.
compact=''.join(b.split())
for expr in ('p[m]=0.0f;', 'p[m]+=s[m*4+r]*kr[m*4+r];',
             'constfloatkv_col=gdn_l8_reduce(p);',
             'constfloatdelta_col=(tok[128+grp]-g_val*kv_col)*beta_val;',
             's[j]=g_val*s[j]+kr[j]*delta_col;'):
    assert expr in compact, expr
assert 'gdn_g1_verify_pairs' not in b
print('G1_SOURCE PASS original replay/verify FP bodies retained; staged replay expressions unchanged')

# Accepted 91eb311 quotient arithmetic remains the fourth bool, independently
# of G1's fifth bool. Pin the full guard, not just one product-check substring.
guard_start=b.index('        int64_t slot_q = 0;')
guard_end=b.index('            const float * rec =',guard_start)
assert hashlib.sha256(b[guard_start:guard_end].encode()).hexdigest() == '0778636ed1dfdd2d0b6300ae527a6a34f28f6383763a184a1861abef2575472b'
assert 'bool REUSE_QUOTIENT = false, bool G1 = false>' in b
launcher=(root/'llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu').read_text()
for quotient in ('true','false'):
    for g1 in ('true','false'):
        assert f'gated_delta_net_dedup_l8v_cuda<RAW, G_PRECOMPUTED, true, {quotient}, {g1}>' in launcher
assert 'getenv("GGML_CUDA_GDN_QUOTIENT_REUSE")' in launcher
assert 'const bool g1 = g1_enabled && n_tokens == 16;' in launcher
# This accepted helper independently pins the half-prefill branch/header.
from test_gdn_half_source import without_half_launcher
without_half_launcher(launcher)
print('G1_PORT PASS quotient_guard_91_exact=1 all_four_G1_quotient_routes=1 half_prefill_branch_header_unchanged=1')
