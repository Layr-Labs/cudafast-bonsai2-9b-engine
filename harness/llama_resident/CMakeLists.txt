cmake_minimum_required(VERSION 3.16)
project(llama-resident CXX)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp" CACHE PATH "Path to vendored llama.cpp")

option(LLAMA_RESIDENT_BUILD_TESTS "Build llama-resident unit tests" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
option(LLAMA_RESIDENT_CUDA "Build with the CUDA ggml backend" ON)
set(GGML_CUDA ${LLAMA_RESIDENT_CUDA} CACHE BOOL "Build CUDA backend" FORCE)
if(LLAMA_RESIDENT_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

set(RESIDENT_SOURCES
    engine_fake.cpp
    engine_llama.cpp
    speculative_round.cpp
    session.cpp
    wire.cpp
    main.cpp
)

add_executable(llama-resident ${RESIDENT_SOURCES})
target_compile_features(llama-resident PRIVATE cxx_std_17)
target_include_directories(llama-resident PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/include
)
target_link_libraries(llama-resident PRIVATE llama llama-common)
set_target_properties(llama-resident PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(llama-resident-min test/min_sync.cpp)
target_compile_features(llama-resident-min PRIVATE cxx_std_17)
target_include_directories(llama-resident-min PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/include
)
target_link_libraries(llama-resident-min PRIVATE llama llama-common)
set_target_properties(llama-resident-min PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(LLAMA_RESIDENT_BUILD_TESTS)
    enable_testing()
    add_executable(prefill-plane-owner-plan test/prefill_plane_owner_plan.cpp)
    target_compile_features(prefill-plane-owner-plan PRIVATE cxx_std_17)
    add_test(NAME prefill-plane-owner-plan COMMAND prefill-plane-owner-plan)
    set_tests_properties(prefill-plane-owner-plan PROPERTIES LABELS "retained;cpu")
    add_executable(gdn-quotient-plan test/gdn_quotient_plan.cpp)
    target_compile_features(gdn-quotient-plan PRIVATE cxx_std_17)
    add_test(NAME gdn-quotient-plan COMMAND gdn-quotient-plan)
    add_executable(prefill-plane-d4-alias-plan test/prefill_plane_d4_alias_plan.cpp)
    target_compile_features(prefill-plane-d4-alias-plan PRIVATE cxx_std_17)
    add_test(NAME prefill-plane-d4-alias-plan COMMAND prefill-plane-d4-alias-plan)
    set_tests_properties(prefill-plane-d4-alias-plan PROPERTIES LABELS "retained;cpu")
    add_executable(gdn-dedup-plan test/gdn_dedup_plan.cpp)
    target_compile_features(gdn-dedup-plan PRIVATE cxx_std_17)
    add_test(NAME gdn-dedup-plan COMMAND gdn-dedup-plan)

    # Retained candidates are default-off; these gates exercise their on/off
    # routes independently of model data and the benchmark measurement protocol.
    foreach(name prefill-gates-early prefill-last-attn-rows decode-gates-side)
        string(REPLACE "-" "_" source_name "${name}")
        add_executable(${name}-cpu test/${source_name}.cpp)
        target_compile_features(${name}-cpu PRIVATE cxx_std_17)
        target_include_directories(${name}-cpu PRIVATE ${LLAMA_CPP_DIR}/src)
        target_link_libraries(${name}-cpu PRIVATE ggml)
        add_test(NAME ${name}-cpu COMMAND ${name}-cpu)
        set_tests_properties(${name}-cpu PROPERTIES LABELS "retained;cpu")
    endforeach()
    add_executable(dflash-concat-q8-plan test/dflash_concat_q8_plan.cpp)
    target_compile_features(dflash-concat-q8-plan PRIVATE cxx_std_17)
    target_include_directories(dflash-concat-q8-plan PRIVATE ${LLAMA_CPP_DIR}/ggml/src ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(dflash-concat-q8-plan PRIVATE ggml-base)
    add_test(NAME dflash-concat-q8-plan COMMAND dflash-concat-q8-plan)
    set_tests_properties(dflash-concat-q8-plan PROPERTIES LABELS "retained;cpu")
    add_executable(decode-fixup-add-rms-match test/decode_fixup_add_rms_match.cpp)
    target_compile_features(decode-fixup-add-rms-match PRIVATE cxx_std_17)
    target_include_directories(decode-fixup-add-rms-match PRIVATE
        ${LLAMA_CPP_DIR}/ggml/src ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
    target_link_libraries(decode-fixup-add-rms-match PRIVATE ggml-base)
    add_test(NAME decode-fixup-add-rms-match COMMAND decode-fixup-add-rms-match)
    set_tests_properties(decode-fixup-add-rms-match PROPERTIES LABELS "retained;cpu")
    if(LLAMA_RESIDENT_CUDA)
        function(add_retained_cuda_target name source)
            add_executable(${name} test/${source})
            target_include_directories(${name} PRIVATE
                ${LLAMA_CPP_DIR}/src ${LLAMA_CPP_DIR}/ggml/src ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
            target_compile_definitions(${name} PRIVATE $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
            target_compile_options(${name} PRIVATE $<TARGET_PROPERTY:ggml-cuda,COMPILE_OPTIONS>)
            target_link_libraries(${name} PRIVATE ggml ggml-cuda ggml-base CUDA::cudart)
        endfunction()
        function(add_retained_cuda_test name target)
            add_test(NAME ${name} COMMAND ${target} ${ARGN})
            set_tests_properties(${name} PROPERTIES LABELS "retained;cuda")
            if(NOT "${name}" STREQUAL "${target}")
                add_custom_target(${name} DEPENDS ${target})
            endif()
        endfunction()
        add_retained_cuda_target(fattn-half-partials-equivalence fattn_half_partials_equivalence.cu)
        add_retained_cuda_test(fattn-half-partials-equivalence fattn-half-partials-equivalence)
        add_retained_cuda_target(dflash-concat-q8-route dflash_concat_q8_route.cu)
        add_retained_cuda_test(dflash-concat-q8-route dflash-concat-q8-route)
        add_retained_cuda_test(dflash-concat-q8-enabled dflash-concat-q8-route --enabled)
        add_retained_cuda_test(dflash-concat-q8-disabled dflash-concat-q8-route --disabled)
        add_retained_cuda_target(dod-qkv-registry dod_qkv/registry.cu)
        add_retained_cuda_test(dod-qkv-registry dod-qkv-registry)
        add_retained_cuda_test(dod-qkv-registry-exact dod-qkv-registry --exact)
        add_retained_cuda_test(dod-qkv-registry-off dod-qkv-registry --off)
        add_retained_cuda_target(prefill-gates-early-cuda prefill_gates_early_cuda.cu)
        add_retained_cuda_test(prefill-gates-early-cuda prefill-gates-early-cuda)
        add_retained_cuda_target(decode-gates-side-cuda decode_gates_side_cuda.cu)
        add_retained_cuda_test(decode-gates-side-cuda decode-gates-side-cuda)
        add_retained_cuda_target(prefill-last-attn-rows-cuda prefill_last_attn_rows_cuda.cu)
        add_retained_cuda_test(prefill-last-attn-rows-cuda prefill-last-attn-rows-cuda)
        add_retained_cuda_target(d4-store-address-paired d4_store_address_paired.cu)
        add_retained_cuda_test(d4-store-address-paired d4-store-address-paired)
        add_retained_cuda_target(prefill-plane-equivalence prefill_plane_equivalence.cu)
        add_retained_cuda_test(prefill-plane-equivalence prefill-plane-equivalence --small)
        add_retained_cuda_test(prefill-plane-alias-default prefill-plane-equivalence)
        add_retained_cuda_test(prefill-plane-owner-default prefill-plane-equivalence)
        add_retained_cuda_test(prefill-plane-owner-on prefill-plane-equivalence --owner)
        add_retained_cuda_test(prefill-plane-owner-off prefill-plane-equivalence --owner-off)
        add_retained_cuda_test(prefill-plane-owner-alias-off prefill-plane-equivalence --owner --alias-off)
        add_retained_cuda_test(prefill-plane-owner-disabled prefill-plane-equivalence --owner --disabled)
        add_retained_cuda_test(prefill-plane-disabled prefill-plane-equivalence --disabled)
        add_retained_cuda_test(prefill-plane-alias-on prefill-plane-equivalence --alias)
        add_retained_cuda_test(prefill-plane-alias-off prefill-plane-equivalence --alias-off)
        add_retained_cuda_test(prefill-plane-alias-disabled prefill-plane-equivalence --alias --disabled)
        add_retained_cuda_target(prefill-compact-equivalence prefill_compact_equivalence.cu)
        add_retained_cuda_test(prefill-compact-equivalence prefill-compact-equivalence)
        add_retained_cuda_target(prefill-gemm-v2-equivalence prefill_gemm_v2_equivalence.cu)
        add_retained_cuda_test(prefill-gemm-v2-equivalence prefill-gemm-v2-equivalence)
        add_retained_cuda_test(prefill-gemm-v2-enabled prefill-gemm-v2-equivalence --enabled)
        add_retained_cuda_test(prefill-gemm-v2-disabled prefill-gemm-v2-equivalence --disabled)
        set_tests_properties(prefill-gemm-v2-equivalence prefill-gemm-v2-enabled
            prefill-gemm-v2-disabled PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE)
        add_retained_cuda_target(gdn-dedup-route gdn_dedup_route.cu)
        add_retained_cuda_test(gdn-dedup-route gdn-dedup-route)
        add_retained_cuda_test(gdn-g1-route gdn-dedup-route --g1)
        set_tests_properties(gdn-g1-route PROPERTIES LABELS "retained;cuda;g1")
        add_retained_cuda_test(gdn-dedup-c32-route gdn-dedup-route)
        set_tests_properties(gdn-dedup-c32-route PROPERTIES ENVIRONMENT "GGML_CUDA_GDN_HALF_PREFILL=0")
        add_retained_cuda_target(gdn-half-initial-state-route gdn_half_initial_state_route.cu)
        add_retained_cuda_test(gdn-half-initial-state-route gdn-half-initial-state-route)
        add_retained_cuda_target(gdn-half-prefill-layout-route gdn_half_prefill_layout_route.cu)
        add_retained_cuda_test(gdn-half-prefill-layout-route gdn-half-prefill-layout-route)
        add_retained_cuda_target(gdn-quotient-route gdn_quotient_route.cu)
        add_retained_cuda_test(gdn-quotient-route gdn-quotient-route)
        add_retained_cuda_test(gdn-quotient-route-enabled gdn-quotient-route --enabled --timing)
        add_retained_cuda_test(gdn-quotient-route-disabled gdn-quotient-route --disabled)
        set_tests_properties(gdn-quotient-route gdn-quotient-route-enabled gdn-quotient-route-disabled
            PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
        add_retained_cuda_target(gdn-prefill-layout-equivalence gdn_prefill_layout_equivalence.cu)
        add_retained_cuda_test(gdn-prefill-layout-equivalence gdn-prefill-layout-equivalence)
        add_retained_cuda_target(q8-fixup-pair-equivalence q8_fixup_pair_equivalence.cu)
        add_retained_cuda_test(q8-fixup-pair-equivalence q8-fixup-pair-equivalence)
        add_retained_cuda_target(decode-q8-tm64-registration decode_q8_tm64_registration.cu)
        add_retained_cuda_test(decode-q8-tm64-registration-off decode-q8-tm64-registration)
        add_retained_cuda_test(decode-q8-tm64-registration-on decode-q8-tm64-registration --enabled)
        add_retained_cuda_test(decode-q8-tm64-registration-ring-off decode-q8-tm64-registration --ring-off)
        add_retained_cuda_target(decode-ptq-ffn-route decode_ptq_ffn_route.cu)
        add_retained_cuda_target(decode-fixup-add-rms-route decode_fixup_add_rms_route.cu)
        add_retained_cuda_test(decode-fixup-add-rms-route decode-fixup-add-rms-route)
        foreach(mode disabled ring-off unsigned)
            add_retained_cuda_test(decode-fixup-add-rms-route-${mode} decode-fixup-add-rms-route --${mode})
        endforeach()
        add_retained_cuda_target(decode-gu-epilogue-route decode_gu_epilogue_route.cu)
        add_retained_cuda_test(decode-gu-epilogue-route-on decode-gu-epilogue-route)
        foreach(mode disabled ring-off jit-off unsigned)
            add_retained_cuda_test(decode-gu-epilogue-route-${mode} decode-gu-epilogue-route --${mode})
        endforeach()
        add_retained_cuda_test(decode-ptq-ffn-route-off decode-ptq-ffn-route)
        add_retained_cuda_test(decode-ptq-ffn-route-on decode-ptq-ffn-route --enabled)
        add_retained_cuda_test(decode-ptq-ffn-route-ring-off decode-ptq-ffn-route --ring-off)
        add_retained_cuda_test(decode-ptq-ffn-route-unsigned decode-ptq-ffn-route --unsigned)
        add_retained_cuda_target(fixup-static-grid-equivalence fixup_static_grid_equivalence.cu)
        add_retained_cuda_test(fixup-static-grid-equivalence fixup-static-grid-equivalence)
    endif()
    if(LLAMA_RESIDENT_CUDA)
        add_executable(fwht-mmq-equivalence test/fwht_mmq_equivalence.cu)
        target_include_directories(fwht-mmq-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        # This test uses the internal CUDA context; graph-related macros affect its ABI.
        target_compile_definitions(fwht-mmq-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(fwht-mmq-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME fwht-mmq-equivalence COMMAND fwht-mmq-equivalence)
        # compares the D4 activation path; D1 (one scale per 128) is a deliberate numerical change
        set_tests_properties(fwht-mmq-equivalence PROPERTIES ENVIRONMENT "GGML_CUDA_PREFILL_D1=0")

        add_executable(logits-topk-equivalence test/logits_topk_equivalence.cpp)
        target_compile_features(logits-topk-equivalence PRIVATE cxx_std_17)
        target_include_directories(logits-topk-equivalence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(logits-topk-equivalence PRIVATE ggml)
        add_test(NAME logits-topk-equivalence COMMAND logits-topk-equivalence)

        add_executable(dflash2-conv-equivalence test/dflash2_conv_equivalence.cpp)
        target_compile_features(dflash2-conv-equivalence PRIVATE cxx_std_17)
        target_link_libraries(dflash2-conv-equivalence PRIVATE ggml)
        add_test(NAME dflash2-conv-equivalence COMMAND dflash2-conv-equivalence)

        add_executable(conv-front-equivalence test/conv_front_equivalence.cpp)
        target_compile_features(conv-front-equivalence PRIVATE cxx_std_17)
        target_link_libraries(conv-front-equivalence PRIVATE ggml)
        add_test(NAME conv-front-equivalence COMMAND conv-front-equivalence)
        add_test(NAME conv-front-l2-equivalence COMMAND conv-front-equivalence --split-l2-reference --short-only --replays 3)
        add_test(NAME conv-front-l2-off-equivalence COMMAND conv-front-equivalence --split-l2-reference --replays 3)
        set_tests_properties(conv-front-l2-off-equivalence PROPERTIES ENVIRONMENT "GGML_CUDA_DOD_CONVL2=0")
        add_test(NAME conv-front-l2-warp-rows-equivalence COMMAND conv-front-equivalence --split-l2-reference --short-only --replays 3)
        set_tests_properties(conv-front-l2-warp-rows-equivalence PROPERTIES ENVIRONMENT "GGML_CUDA_DOD_CONVL2_MODE=1")
        add_test(NAME conv-front-l2-reverse-equivalence COMMAND conv-front-equivalence --split-l2-reference --short-only --reverse-snapshots --replays 3)

        add_executable(token-embd-equivalence test/token_embd_equivalence.cpp)
        target_compile_features(token-embd-equivalence PRIVATE cxx_std_17)
        target_link_libraries(token-embd-equivalence PRIVATE ggml)
        add_test(NAME token-embd-equivalence COMMAND token-embd-equivalence
            ${CMAKE_CURRENT_SOURCE_DIR}/../../reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf)
        add_executable(mmf-pair-equivalence test/mmf_pair_equivalence.cu)
        target_include_directories(mmf-pair-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(mmf-pair-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(mmf-pair-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME mmf-pair-equivalence COMMAND mmf-pair-equivalence)
        add_executable(swiglu-fwht-equivalence test/swiglu_fwht_equivalence.cu)
        target_include_directories(swiglu-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(swiglu-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(swiglu-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME swiglu-fwht-equivalence COMMAND swiglu-fwht-equivalence)
        add_executable(add-rms-fwht-equivalence test/add_rms_fwht_equivalence.cu)
        target_include_directories(add-rms-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(add-rms-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(add-rms-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME add-rms-fwht-equivalence COMMAND add-rms-fwht-equivalence)
        add_executable(rms-fwht-equivalence test/rms_fwht_equivalence.cu)
        target_include_directories(rms-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(rms-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(rms-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME rms-fwht-equivalence COMMAND rms-fwht-equivalence)
        # compares the D4 activation path; D1 (one scale per 128) is a deliberate numerical change
        set_tests_properties(rms-fwht-equivalence PROPERTIES ENVIRONMENT "GGML_CUDA_PREFILL_D1=0")
        add_executable(gnorm-fwht-equivalence test/gnorm_fwht_equivalence.cu)
        target_include_directories(gnorm-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(gnorm-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(gnorm-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME gnorm-fwht-equivalence COMMAND gnorm-fwht-equivalence)
        # compares the D4 activation path; D1 (one scale per 128) is a deliberate numerical change
        set_tests_properties(gnorm-fwht-equivalence PROPERTIES ENVIRONMENT "GGML_CUDA_PREFILL_D1=0")
        add_executable(attn-glue-equivalence test/attn_glue_equivalence.cu)
        target_include_directories(attn-glue-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(attn-glue-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(attn-glue-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME attn-glue-equivalence COMMAND attn-glue-equivalence)
        add_executable(glu-perm-fwht-equivalence test/glu_perm_fwht_equivalence.cu)
        target_include_directories(glu-perm-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(glu-perm-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(glu-perm-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME glu-perm-fwht-equivalence COMMAND glu-perm-fwht-equivalence)
        add_executable(ptq1-repack-equivalence test/ptq1_repack_equivalence.cu)
        target_include_directories(ptq1-repack-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(ptq1-repack-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(ptq1-repack-equivalence PRIVATE ggml ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME ptq1-repack-equivalence COMMAND ptq1-repack-equivalence
            ${CMAKE_CURRENT_SOURCE_DIR}/../../reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf)
    endif()
    add_executable(llama-resident-tests
        test/test_session.cpp
        engine_fake.cpp
        engine_spec_round_test.cpp
        speculative_round.cpp
        session.cpp
        wire.cpp
    )
    target_compile_features(llama-resident-tests PRIVATE cxx_std_17)
    target_include_directories(llama-resident-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLAMA_CPP_DIR}/vendor
    )
    target_compile_definitions(llama-resident-tests PRIVATE
        ADAPTER_SESSION_BACKEND="${CMAKE_CURRENT_SOURCE_DIR}/../protocol-adapter/src/session_backend.rs"
        LLAMA_RESIDENT_BIN="${CMAKE_BINARY_DIR}/bin/llama-resident")
    add_test(NAME llama-resident-tests COMMAND llama-resident-tests)

    add_executable(engine-boot-tests test/test_engine_boot.cpp engine_llama.cpp speculative_round.cpp)
    target_compile_features(engine-boot-tests PRIVATE cxx_std_17)
    target_include_directories(engine-boot-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLAMA_CPP_DIR}/vendor
        ${LLAMA_CPP_DIR}/common
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/src
    )
    target_link_libraries(engine-boot-tests PRIVATE llama llama-common)
    add_test(NAME engine-boot-tests COMMAND engine-boot-tests)
endif()
