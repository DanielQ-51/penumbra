cmake_minimum_required(VERSION 3.27)  # CUDA_OPTIX_COMPILATION was added in CMake 3.27

project(OptixExperiment LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# Select the host GPU by default for local builds, while allowing callers to
# choose explicit architectures for cross-compilation or distribution builds.
set(NOVUM_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures, e.g. native, 86, or 86;89")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${NOVUM_CUDA_ARCHITECTURES}")
endif()

# CUDA 13.3's bundled CCCL rejects MSVC's legacy preprocessor.  Pass the
# conforming mode through nvcc to the host compiler on Windows.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

find_package(CUDAToolkit REQUIRED)

set(OPTIX_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include")

# ==============================================================================
# 0. DEVICE MODULE FORMAT: OptiX-IR vs PTX
# ==============================================================================
# ON  -> compile device modules to OptiX-IR (native CMake CUDA_OPTIX_COMPILATION;
#        no -rdc, no hand-rolled nvcc, so the util.cuh __constant__ globals do NOT
#        need to be touched). Also defines OPTIX_IR_DIAGNOSTIC (validation mode on).
# OFF -> original PTX path, unchanged.
option(USE_OPTIX_IR "Compile device modules to OptiX-IR instead of PTX" ON)
# Validation mode + detailed create/link logs. Defaults ON while bringing OptiX-IR up;
# turn OFF (-DOPTIX_VALIDATION=OFF) once the pipeline links, to profile without overhead.
option(OPTIX_VALIDATION "Enable OptiX validation mode + verbose module/pipeline logs" ${USE_OPTIX_IR})

if(USE_OPTIX_IR)
    set(OPTIX_DEVICE_COMPILE_PROP CUDA_OPTIX_COMPILATION)
    set(OPTIX_DEVICE_MODULE_EXT   "optixir")
else()
    set(OPTIX_DEVICE_COMPILE_PROP CUDA_PTX_COMPILATION)
    set(OPTIX_DEVICE_MODULE_EXT   "ptx")
endif()
message(STATUS "Device modules: ${OPTIX_DEVICE_MODULE_EXT} (USE_OPTIX_IR=${USE_OPTIX_IR})")

# ==============================================================================
# 1. COMPILE THE DEVICE CODE (PTX or OptiX-IR per USE_OPTIX_IR)
# ==============================================================================

# Treat the megakernel as an "OBJECT" library and tell CMake to emit the device module
add_library(optix_ptx_renderer OBJECT renderer.cu)
set_property(TARGET optix_ptx_renderer PROPERTY ${OPTIX_DEVICE_COMPILE_PROP} ON)

target_include_directories(optix_ptx_renderer PRIVATE
    ../include
    ../thirdparty
    ${OPTIX_INCLUDE_DIR}
)

target_compile_options(optix_ptx_renderer PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    
)


# --- DEVICE MODULE 2: ReSTIR Enhanced Shaders ---
add_library(optix_ptx_restir OBJECT restirPTenhancedShaders.cu)
set_property(TARGET optix_ptx_restir PROPERTY ${OPTIX_DEVICE_COMPILE_PROP} ON)

target_include_directories(optix_ptx_restir PRIVATE
    ../include
    ../thirdparty
    ${OPTIX_INCLUDE_DIR}
)
target_compile_options(optix_ptx_restir PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# ==============================================================================
# 2. COMPILE THE HOST CODE (EXECUTABLE)
# ==============================================================================

add_executable(renderer_optix
    main.cu
    restirPTenhanced_kernels.cu
    ../src/tinyexr_impl.cpp
    ../src/stb_image_impl.cpp
    ../thirdparty/tinyexr/miniz.c
    ../src/imageUtil.cu
    ../src/rng.cu
    ../thirdparty/tinygltf/tiny_gltf_v3.c
)

# The device-module object targets are consumed by the post-build copy below,
# so make their build order explicit for Ninja and other parallel generators.
add_dependencies(renderer_optix optix_ptx_renderer optix_ptx_restir)

# Reach UP one directory to link your shared headers to the host code
target_include_directories(renderer_optix PRIVATE 
    ../include
    ../thirdparty
    ../thirdparty/tinyexr
    ${OPTIX_INCLUDE_DIR}
)

# ==============================================================================
# 3. FIXING THE ASSET_PATH MACRO
# ==============================================================================

# Get the absolute path to the parent directory (the root of NovumRender)
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PARENT_DIR)

# Inject ROOT_DIR so your C++ ASSET_PATH macro works exactly like it did before
target_compile_definitions(renderer_optix PRIVATE
    ROOT_DIR=\"${PARENT_DIR}\"
    PTX_DIR=\"$<TARGET_FILE_DIR:renderer_optix>\"
    OPTIX_MODULE_EXT=\"${OPTIX_DEVICE_MODULE_EXT}\"
    TINYGLTF3_ENABLE_FS
)

if(OPTIX_VALIDATION)
    # Enables OptiX device-context validation mode + detailed create/link logs (hostSetup.cuh).
    # Diagnostic: adds per-launch overhead -- turn OPTIX_VALIDATION OFF before profiling.
    target_compile_definitions(renderer_optix PRIVATE OPTIX_IR_DIAGNOSTIC)
endif()


# ==============================================================================
# 4. LINKING LIBRARIES AND COPYING THE PTX
# ==============================================================================

# Link CUDA, the CUDA driver, and Windows API (Required by OptiX)
target_link_libraries(renderer_optix PRIVATE 
    CUDA::cuda_driver
    CUDA::cudart
    advapi32
)

# CMake buries .ptx files deep in build folders. 
# This automatically copies the generated PTX file and places it right next 
# to renderer_optix.exe so read_file_to_string("renderer.ptx") finds it instantly!
add_custom_command(TARGET renderer_optix POST_BUILD
    # Copy the renderer device module (renderer.optixir or renderer.ptx)
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_OBJECTS:optix_ptx_renderer>
    $<TARGET_FILE_DIR:renderer_optix>/renderer.${OPTIX_DEVICE_MODULE_EXT}

    # Copy the ReSTIR device module
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_OBJECTS:optix_ptx_restir>
    $<TARGET_FILE_DIR:renderer_optix>/restirPTenhancedShaders.${OPTIX_DEVICE_MODULE_EXT}
)