cmake_minimum_required(VERSION 3.24)
project(gemmalive VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GEMMALIVE_BUILD_PROGRAMS "Build the gemmalive program" ${PROJECT_IS_TOP_LEVEL})
option(GEMMALIVE_BUILD_TOOLS    "Build the benchmark and training tools" ${PROJECT_IS_TOP_LEVEL})

include(cmake/dependencies.cmake)

if(MSVC)
    set(GEMMALIVE_WARNINGS /W4)
else()
    set(GEMMALIVE_WARNINGS -Wall -Wextra -Wpedantic)
endif()

# The voice pipeline, for the program, the tools and apps that embed it.
add_library(gemmalive STATIC
    src/audio_encoder.cpp
    src/audio_io.cpp
    src/backend.cpp
    src/device.cpp
    src/gemma4.cpp
    src/llm.cpp
    src/logging.cpp
    src/mel.cpp
    src/model_file.cpp
    src/sampler.cpp
    src/session.cpp
    src/tokenizer.cpp
    src/tts.cpp
    src/turn_head.cpp
    src/vad.cpp
)
target_include_directories(gemmalive PUBLIC src)
target_include_directories(gemmalive PRIVATE ${miniaudio_SOURCE_DIR})
target_link_libraries(gemmalive PUBLIC livevoice ggml)
if(APPLE)
    target_link_libraries(gemmalive PUBLIC "-framework Accelerate")
endif()
target_compile_options(gemmalive PRIVATE ${GEMMALIVE_WARNINGS})
if(NOT MSVC)
    # miniaudio's decoder leaves helpers it does not need unused.
    set_source_files_properties(src/audio_io.cpp PROPERTIES COMPILE_OPTIONS -Wno-unused-function)
endif()

if(GEMMALIVE_BUILD_PROGRAMS)
    add_executable(gemmalive-cli src/main.cpp)
    set_target_properties(gemmalive-cli PROPERTIES OUTPUT_NAME gemmalive)
    target_link_libraries(gemmalive-cli PRIVATE gemmalive)
    target_compile_options(gemmalive-cli PRIVATE ${GEMMALIVE_WARNINGS})
endif()

if(GEMMALIVE_BUILD_TOOLS)
    # Feature extraction for the end-of-turn and addressee heads (see training/).
    add_executable(gemmalive-turn-features tools/turn_features.cpp)
    target_link_libraries(gemmalive-turn-features PRIVATE gemmalive)
endif()
