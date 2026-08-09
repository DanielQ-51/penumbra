#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <exception>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "restirPTObjects.cuh"
#include "settings.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>
#include <iomanip>


struct OptixEngineState {
    OptixDeviceContext context = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt_unidirectional = {};
    OptixShaderBindingTable sbt_restirCandidate = {};
    OptixShaderBindingTable sbt_restirSpatial = {};
    OptixShaderBindingTable sbt_restirTemporal = {};
    OptixProgramGroup raygenUnidirectionalProgramGroup = nullptr;
    OptixProgramGroup raygenRestirCandidateProgramGroup = nullptr;
    OptixProgramGroup raygenRestirSpatialProgramGroup = nullptr;
    OptixProgramGroup raygenRestirTemporalProgramGroup = nullptr;
    OptixProgramGroup missProgramGroup = nullptr;
    OptixProgramGroup hitgroupProgramGroup = nullptr;
    OptixModule module = nullptr;
    OptixModule restirModule = nullptr;

    CUdeviceptr d_rgRecord = 0;
    CUdeviceptr d_msRecord = 0;
    CUdeviceptr d_hgRecord = 0;
};


struct CommonParams {
    Camera camera;
    ShadeContext shadeContext; // multiple pointers
    OptixTraversableHandle bvh_handle; // long long
    float4* __restrict__ accum_buffer;
    float4* __restrict__ overlay_buffer;
    uint32_t w;
    uint32_t h;
    uint32_t frame_index;
    uint32_t max_depth;
};

struct RestirCommonParams {
    Reservoir lastFrameReservoir;
    Reservoir reservoir;
    GBuffer gbuffer;
    GBuffer prevGbuffer;
    Camera lastFrameCamera;
    uint8_t* __restrict__ duplication_map;
    short2* reuseTextures[NUM_REUSE_TEXTURES];
    uint32_t reuseTextureSizes[NUM_REUSE_TEXTURES];
    ShiftResultBuffer shiftResultBuffer[NUM_REUSE_TEXTURES];
    DenoiserGuides denoiserGuides;
};

struct CandidateGenParams {
    
};

struct SpatialReuseParams {
    
};

struct TemporalReuseParams {

};

struct PipelineParams {
    CommonParams common;
    RestirCommonParams restir;
    CandidateGenParams candidateGen;
    SpatialReuseParams spatial;
    TemporalReuseParams temporal;
};