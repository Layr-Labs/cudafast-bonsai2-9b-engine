# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    # ponytail: trimmed vendor — only speculative-simple for Bonsai resident reference loop
    add_subdirectory(speculative-simple)
endif()
