cmake_minimum_required(VERSION 3.22)
project(CoA_Engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(CoA_Engine
    src/main.cpp
    third_party/glad/src/glad.c
)

target_include_directories(CoA_Engine PRIVATE
    src
    third_party/glad/include
    third_party/glm
    third_party/StormLib/src
)

# 强制 GLM 使用 [0, 1] 深度范围以配合 glClipControl 与 Reverse-Z
target_compile_definitions(CoA_Engine PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

target_link_libraries(CoA_Engine PRIVATE
    SDL3::SDL3
    OpenGL::GL
    storm # StormLib
)