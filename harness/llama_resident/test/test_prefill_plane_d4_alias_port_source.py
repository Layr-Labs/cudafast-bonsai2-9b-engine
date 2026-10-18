#!/usr/bin/env python3
"""Prove disabled header == accepted478 and enabled kernel == GPU-passed 6f0a07c."""
from pathlib import Path
import hashlib
import re
import sys
sys.dont_write_bytecode = True
from test_prefill_plane_owner_port_source import normalize_owner_header
root = Path(__file__).resolve().parents[3]
source = (root / 'llama.cpp/ggml/src/ggml-cuda/mmq-pq2-plane.cuh').read_text()
source = normalize_owner_header(source)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
start = source.index('// Each thread owns')
end = source.index('// Original 64x128', start)
helper = source[start:end].strip().replace('ggml_cuda_prefill_plane_d4_alias_load', 'prefill_plane_d4_alias_load')
assert sha(helper) == '457e2f6c11c295f1373ec404ed681162e19d144161a9035c21a135d4f6699a06', 'GPU-passed ordered PTX helper changed'
start = source.index('template <bool ADD,bool GLU,bool D4_ALIAS=false>')
end = source.index('// Only complete, contiguous GLU tiles', start)
kernel = source[start:end].strip()
pattern = r'        if constexpr \(D4_ALIAS\) \{\n(.*?)        \} else \{\n(.*?)        \}'
assert len(re.findall(pattern, kernel, re.S)) == 2, 'exactly two compile-time choices'
def select(enabled):
    body = re.sub(pattern, lambda m: '\n'.join(line[4:] for line in m.group(1 if enabled else 2).rstrip('\n').split('\n')), kernel, flags=re.S)
    return body.replace('template <bool ADD,bool GLU,bool D4_ALIAS=false>', 'template <bool ADD,bool GLU>')
enabled = select(True).replace('mul_mat_q_pq2_plane(', 'prefill_plane_d4_alias(').replace('ggml_cuda_prefill_plane_d4_alias_load', 'prefill_plane_d4_alias_load')
assert sha(enabled) == '858e9157be33c7deb94269cbd161b76d5a358687e21a623ac378a7393518fc87', 'enabled kernel differs from GPU-passed test body'
reference = (root / 'harness/llama_resident/test/prefill_plane_accepted_478_reference.cuh').read_text()
reference = reference[reference.index('template <bool ADD,bool GLU>'):].strip().replace('prefill_plane_accepted_478_reference(', 'mul_mat_q_pq2_plane(')
assert sha(reference) == '7f51df99073b17988ff375fc391422be385555f4ddcade5a0b189089bec902d8', 'independent accepted478 reference kernel changed'
assert reference == select(False), 'disabled production kernel differs from independent accepted478 reference'
disabled = source[:start] + select(False) + '\n\n' + source[end:]
a = disabled.index('// Each thread owns')
b = disabled.index('// Original 64x128', a)
disabled = disabled[:a] + disabled[b:]
a = disabled.index('// D4 alias route:')
b = disabled.index('static __global__ void ggml_cuda_prefill_plane_pack', a)
assert 'ggml_cuda_dl_env_int("GGML_CUDA_PREFILL_PLANE_D4_ALIAS", 1)' in disabled[a:b], 'default ON with explicit-zero fallback'
disabled = disabled[:a] + disabled[b:]
new = """    const bool alias=ggml_cuda_prefill_plane_d4_alias_enabled();
    const auto kernel=alias?mul_mat_q_pq2_plane<false,true,true>:mul_mat_q_pq2_plane<false,true,false>;
    static bool attr[2][GGML_CUDA_MAX_DEVICES]={};
    if (!attr[alias][device]) {"""
old = """    const auto kernel=mul_mat_q_pq2_plane<false,true>;
    static bool attr[GGML_CUDA_MAX_DEVICES]={};
    if (!attr[device]) {"""
assert disabled.count(new) == 1
assert disabled.count('attr[alias][device]=true;') == 1
disabled = disabled.replace(new, old).replace('attr[alias][device]=true;', 'attr[device]=true;')
assert sha(disabled) == '6e88c10089ad3521a4408fa4cfce6899961e3384d75a86f52ceafde2e63a3f7e', 'disabled kernel/host/guards/repack differ from accepted478'
helpers = (root / 'llama.cpp/ggml/src/ggml-cuda/mmq-pq2-j128.cuh').read_bytes()
assert hashlib.sha256(helpers).hexdigest() == 'a4725ab1c0d0200a71cd8347626659590b6a549e67d36dac29d97c5136f15074', 'original DT/math/store helper changed'
print('PLANE_D4_ALIAS_PORT_SOURCE_PASS original_disabled_header=1 passed_enabled_kernel=1 accepted_reference=1 unchanged_math_store=1 default_on=1 explicit_zero_fallback=1')
