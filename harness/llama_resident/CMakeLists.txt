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
        add_executable(gnorm-fwht-equivalence test/gnorm_fwht_equivalence.cu)
        target_include_directories(gnorm-fwht-equivalence PRIVATE
            ${LLAMA_CPP_DIR}/ggml/src
            ${LLAMA_CPP_DIR}/ggml/src/ggml-cuda)
        target_compile_definitions(gnorm-fwht-equivalence PRIVATE
            $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
        target_link_libraries(gnorm-fwht-equivalence PRIVATE ggml-cuda ggml-base CUDA::cudart)
        add_test(NAME gnorm-fwht-equivalence COMMAND gnorm-fwht-equivalence)
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

    add_executable(verify-batch-equivalence test/verify_batch_equivalence.cpp)
    target_compile_features(verify-batch-equivalence PRIVATE cxx_std_17)
    target_include_directories(verify-batch-equivalence PRIVATE ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/include)
    target_link_libraries(verify-batch-equivalence PRIVATE llama llama-common)
    add_test(NAME verify-batch-equivalence COMMAND verify-batch-equivalence)

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
