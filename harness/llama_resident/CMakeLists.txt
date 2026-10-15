cmake_minimum_required(VERSION 3.16)
project(llama-resident CXX)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp" CACHE PATH "Path to vendored llama.cpp")

option(LLAMA_RESIDENT_BUILD_TESTS "Build llama-resident unit tests" ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
option(LLAMA_RESIDENT_CUDA "Build with the CUDA ggml backend" ON)
set(GGML_CUDA ${LLAMA_RESIDENT_CUDA} CACHE BOOL "Build CUDA backend" FORCE)
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
