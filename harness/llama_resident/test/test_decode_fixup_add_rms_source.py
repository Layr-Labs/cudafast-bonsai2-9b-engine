#!/usr/bin/env python3
"""Prove that the production consumer changes only its corrected input load."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = (ROOT / "llama.cpp/ggml/src/ggml-cuda/quantize.cu").read_text()
start = "template <int NT, bool SIGNED>\n__launch_bounds__(NT, 1)"
original = src[src.index(start):src.index("\nvoid fwht_quantize_mmq_d4_add_rms_cuda")]
begin = src.index(start, src.index("// Exact grid48 PTQ stream-K fixup"))
candidate = src[begin:src.index("\nvoid fwht_quantize_mmq_d4_fixup_add_rms_cuda", begin)]
candidate = candidate.replace("fwht_quantize_mmq_d4_fixup_add_rms", "fwht_quantize_mmq_d4_add_rms", 1)
candidate = candidate.replace("const int64_t rows, const float * partials)", "const int64_t rows)", 1)
begin = candidate.index("        // M4096 / grid48")
end = candidate.index("        sum[col]=xi;", begin)
candidate = candidate[:begin] + "        const float xi=__fadd_rn(a[col], b[col]);\n" + candidate[end:]
assert candidate == original, "retained RMS/FWHT/D4/PDL code differs"
print("DECODE_FIXUP_ADD_RMS_SOURCE_PASS rms_order=1 fwht_order=1 quant_bytes=1 pdl=1")
