include(FetchContent)

# Link everything into the gemmalive binaries.
set(BUILD_SHARED_LIBS OFF)

# Metal on Apple devices; off in the iOS Simulator, whose Metal can't take
# the models' buffers. Its shaders are built into the library: an app can't
# load them from beside it.
if(APPLE AND NOT CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
    set(_gemmalive_metal ON)
else()
    set(_gemmalive_metal OFF)
endif()
option(GEMMALIVE_METAL "Run the models on Metal" ${_gemmalive_metal})
set(GGML_METAL               ${GEMMALIVE_METAL})
set(GGML_METAL_EMBED_LIBRARY ${GEMMALIVE_METAL})
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(GGML_METAL_TARGET_OS ios)
endif()

# Pinned dependencies. For local development, point FetchContent at your own
# checkouts via FETCHCONTENT_SOURCE_DIR_<NAME> (see CMakeUserPresets.json.example).

set(GEMMALIVE_GGML_COMMIT      facdc44a67da17df4433adcc468b424f48e9ded2) # v0.28.0
set(GEMMALIVE_LIVEVOICE_COMMIT 7f66740d6adf25cd7c30fbbd2fdeee9a31f81e39)

FetchContent_Declare(ggml
    URL https://github.com/ggml-org/ggml/archive/${GEMMALIVE_GGML_COMMIT}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(livevoice
    URL https://github.com/nodcai/livevoice.cpp/archive/${GEMMALIVE_LIVEVOICE_COMMIT}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Header only; its own build is not needed.
FetchContent_Declare(miniaudio
    URL https://github.com/mackron/miniaudio/archive/refs/tags/0.11.25.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR none
)

set(GGML_BUILD_TESTS    OFF)
set(GGML_BUILD_EXAMPLES OFF)

# livevoice: library only, on the ggml above.
set(LIVEVOICE_BUILD_EXAMPLES OFF)
set(LIVEVOICE_BUILD_TESTS    OFF)

# Order matters: ggml first, so that livevoice uses it.
FetchContent_MakeAvailable(ggml)
FetchContent_MakeAvailable(livevoice miniaudio)
