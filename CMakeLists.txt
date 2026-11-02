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
