# CMake generated Testfile for 
# Source directory: /Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident
# Build directory: /Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(prefill-plane-owner-plan "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/prefill-plane-owner-plan")
set_tests_properties(prefill-plane-owner-plan PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;59;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(gdn-quotient-plan "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/gdn-quotient-plan")
set_tests_properties(gdn-quotient-plan PROPERTIES  _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;63;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(prefill-plane-d4-alias-plan "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/prefill-plane-d4-alias-plan")
set_tests_properties(prefill-plane-d4-alias-plan PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;66;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(gdn-dedup-plan "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/gdn-dedup-plan")
set_tests_properties(gdn-dedup-plan PROPERTIES  _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;70;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(prefill-gates-early-cpu "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/prefill-gates-early-cpu")
set_tests_properties(prefill-gates-early-cpu PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;80;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(prefill-last-attn-rows-cpu "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/prefill-last-attn-rows-cpu")
set_tests_properties(prefill-last-attn-rows-cpu PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;80;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(decode-gates-side-cpu "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/decode-gates-side-cpu")
set_tests_properties(decode-gates-side-cpu PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;80;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(dflash-concat-q8-plan "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/dflash-concat-q8-plan")
set_tests_properties(dflash-concat-q8-plan PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;87;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(decode-fixup-add-rms-match "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/decode-fixup-add-rms-match")
set_tests_properties(decode-fixup-add-rms-match PROPERTIES  LABELS "retained;cpu" _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;94;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(llama-resident-tests "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama-resident-tests")
set_tests_properties(llama-resident-tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;312;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
add_test(engine-boot-tests "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/engine-boot-tests")
set_tests_properties(engine-boot-tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;324;add_test;/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/CMakeLists.txt;0;")
subdirs("llama.cpp")
