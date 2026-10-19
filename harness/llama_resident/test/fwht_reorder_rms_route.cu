// Retain every original fixture assertion, with an independent reordered stage oracle.
#include "fwht_reorder_test_oracle.cuh"
#define main inherited_fixture
#include "rms_fwht_equivalence.cu"
#undef main
#undef cudaStreamEndCapture
#undef ggml_cuda_op_fwht
#undef ggml_cuda_op_fwht_signed
int main(int argc,char **argv) {
    if(argc>1 && !strcmp(argv[1],"--enabled"))setenv("GGML_CUDA_FWHT_REORDER","1",1);
    else if(argc>1 && !strcmp(argv[1],"--disabled"))setenv("GGML_CUDA_FWHT_REORDER","0",1);
    else unsetenv("GGML_CUDA_FWHT_REORDER");
    const int result=inherited_fixture();if(result)return result;
    if(!reorder_seen_old || (reorder_test_enabled() && !reorder_seen_new))return 7;
    printf("REORDER_RMS_ROUTE_PASS new=%d old=%d independent_stage_oracle=true\n",reorder_seen_new,reorder_seen_old);return 0;
}
