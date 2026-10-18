from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[3]
source=(root/'llama.cpp/ggml/src/ggml-cuda/quantize.cu').read_text()
branch='    if constexpr (row_concat) {\n        static_assert(!scatter && !swiglu && !rms_scale && ds_layout == MMQ_Q8_1_DS_LAYOUT_D4,\n                      "two-source concat keeps the plain Q8_0 activation path");\n        // Every float4 belongs to one token row. Source selection changes only\n        // the address; the reduction, rounding and k-major output layout below\n        // remain the original quantizer\'s, including N=32 row interleaving.\n        // In this mutually exclusive mode the otherwise unused gate and\n        // n_expert_used arguments hold the second slab and first row count.\n        if (blockIdx.x >= (unsigned) n_expert_used) {\n            x = gate;\n            base_idx -= (int64_t) n_expert_used*s01;\n        }\n    }\n'
wrapper='\nvoid quantize_mmq_q8_1_concat_cuda(\n        const float * first, const float * second, void * out,\n        int64_t width, int first_rows, int second_rows, cudaStream_t stream) {\n    GGML_ASSERT(first && second && out && width > 0 && width % 512 == 0 && first_rows > 0 && second_rows > 0);\n    const int rows = first_rows + second_rows;\n    const dim3 grid(rows, width/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ), 1);\n    const dim3 block(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);\n    const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);\n    ggml_cuda_kernel_launch(quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, false, false, true>, lp,\n        first, (const int32_t *) nullptr, out, width, width, width*rows, width*rows,\n        width, rows, 1, first_rows, second, (const float *) nullptr, (const float *) nullptr);\n}\n'
assert source.count(branch)==1 and source.count(wrapper)==1
restored=source.replace(branch,'',1).replace(wrapper,'',1).replace('bool rms_scale = false, bool row_concat = false>','bool rms_scale = false>',1)
assert hashlib.sha256(restored.encode()).hexdigest()=='5d964829940878fa9e12a601e1d46a10066246cdaca4cdf7aab23022f586e098', 'all disabled quantization source must equal pinned b954'
frozen=(Path(__file__).parent/'dflash_concat_q8_reference.cuh').read_text()
frozen=frozen[frozen.index('template <mmq_q8_1_ds_layout'):].replace('dflash_concat_q8_reference(', 'quantize_mmq_q8_1(')
assert hashlib.sha256(frozen.encode()).hexdigest()=='fd670999fde5dff20426ef879f44e396a6c71268c22d355f22d07f52868608ab', 'independent original quantizer hash'

def without_concat_backend(source):
    include = '#include "ggml-cuda/dflash-concat-q8.h"\n'
    begin = source.index('// Avoid the six DFlash2 internal [context;noise] F32 copies.')
    end = source.index('static void ggml_cuda_graph_evaluate_and_capture', begin)
    helper = source[begin:end]
    assert hashlib.sha256(helper.encode()).hexdigest() == '205f1ecb445fc3a776181173d92a0f47da55fc33970416f8c6cf48130d61b45b', 'exact reviewed concat backend helper'
    call = '                if (node->op == GGML_OP_CONCAT && !is_concurrent_event_active &&\n                        stream_ctx.concurrent_events.empty() &&\n                        ggml_cuda_try_dflash_concat_q8(*cuda_ctx, cgraph, i)) {\n                    continue;\n                }\n\n'
    assert source.count(include) == 1 and source.count(call) == 1
    return (source[:begin] + source[end:]).replace(include, '', 1).replace(call, '', 1)

backend = (root/'llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
assert hashlib.sha256(without_concat_backend(backend).encode()).hexdigest() == '79aaf6e3bba48e1c1385c9bcdda6da4d274f133d8e36293d82e3faae0e4784fe', 'accepted239 complete backend after removing exact concat change'


assert 'return !value || std::atoi(value) == 1;' in backend, 'default ON, explicit zero OFF'
route=(Path(__file__).parent/'dflash_concat_q8_route.cu').read_text()
assert 'const bool enabled=!disabled;' in route and 'const bool fused=enabled&&i==0;' in route
assert 'default route genuinely has absent environment' in route and 'else unsetenv("GGML_CUDA_DFLASH_CONCAT_Q8")' in route
assert 'if(explicit_enabled)direct_gate(be,in,func);' in route, 'timing only in explicit enabled gate'

print('DFLASH_CONCAT_Q8_SOURCE_PASS original_quantizer_file_and_frozen_kernel=true')
