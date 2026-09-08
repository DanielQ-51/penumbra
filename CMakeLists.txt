cmake_minimum_required(VERSION 3.24)

project(NovumRender LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)


find_package(CUDAToolkit REQUIRED)

add_executable(NovumRender
    src/main.cu
    src/deviceCode.cu
    src/imageUtil.cu
    src/rng.cu
    src/fastIntegrators.cu
    src/volumeRendering.cu
    src/tinyexr_impl.cpp
    src/stb_image_impl.cpp
    #src/realTime.cu
    thirdparty/tinyexr/miniz.c
    thirdparty/tinygltf/tiny_gltf_v3.c
)

target_include_directories(NovumRender PRIVATE include)

target_include_directories(NovumRender PRIVATE thirdparty)

target_include_directories(NovumRender PRIVATE thirdparty/tinyexr)


target_compile_definitions(NovumRender PRIVATE
    ROOT_DIR=\"${CMAKE_SOURCE_DIR}\"
    TINYGLTF3_ENABLE_FS
)

target_compile_options(NovumRender PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    #$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

target_compile_options(NovumRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Fix the fatal LNK2019 errors by linking the CUDA Driver and Runtime
target_link_libraries(NovumRender PRIVATE 
    CUDA::curand 
    CUDA::cuda_driver 
    CUDA::cudart
)

# Fix the LNK4098 warning by ignoring the conflicting static C runtime library
if(MSVC)
    target_link_options(NovumRender PRIVATE /NODEFAULTLIB:LIBCMT)
endif()