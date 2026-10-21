# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    # ponytail: trimmed vendor — llama-bench only
    add_subdirectory(llama-bench)
endif()
