cmake_minimum_required(VERSION 3.24)
project(CoA_Engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /MP /W3)
endif()

include(FetchContent)

# 1. 自动拉取 SDL3 (静态编译，免去拷贝 SDL3.dll)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        release-3.2.4
    GIT_SHALLOW    TRUE
)

# 2. 自动拉取 GLM 数学库
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)

# 3. 自动拉取 LZ4 压缩库
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG        v1.10.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  build/cmake
)

# 4. 自动拉取 Zstd 压缩库
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        v1.5.6
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  build/cmake
)

FetchContent_MakeAvailable(SDL3 glm lz4 zstd)

find_package(OpenGL REQUIRED)

add_executable(CoA_Engine
    src/main.cpp
    third_party/glad/src/glad.c
)

target_include_directories(CoA_Engine PRIVATE
    src
    third_party/glad/include
    ${glm_SOURCE_DIR}
    ${lz4_SOURCE_DIR}/lib
    ${zstd_SOURCE_DIR}/lib
)

target_compile_definitions(CoA_Engine PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

target_link_libraries(CoA_Engine PRIVATE
    SDL3::SDL3-static
    OpenGL::GL
    lz4_static
    libzstd_static
)
# 5. 自动拉取 StormLib 并编译为 storm.dll 供 Python 离线解包脚本调用
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    StormLib
    GIT_REPOSITORY https://github.com/ladislav-zezula/StormLib.git
    GIT_TAG        v9.25
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(StormLib)
target_compile_definitions(storm PRIVATE UNICODE _UNICODE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)