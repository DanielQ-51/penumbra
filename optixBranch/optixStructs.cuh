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
    OptixShaderBindingTable sbt_restirTemporalFwd = {};
    OptixShaderBindingTable sbt_restirTemporalBwd = {};
    OptixProgramGroup raygenUnidirectionalProgramGroup = nullptr;
    OptixProgramGroup raygenRestirCandidateProgramGroup = nullptr;
    OptixProgramGroup raygenRestirSpatialProgramGroup = nullptr;
    OptixProgramGroup raygenRestirTemporalProgramGroup = nullptr;
    OptixProgramGroup raygenRestirTemporalFwdProgramGroup = nullptr;
    OptixProgramGroup raygenRestirTemporalBwdProgramGroup = nullptr;
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

    // Runtime technique-variant selector for the profiling suite. The raygen
    // shaders branch on this (if (params.common.debugVersion == 0/1/2 ...)) to
    // pick between optimization variants of the SAME integrator. launchProfile
    // sweeps it across the PROFILE_ARMS table; normal render paths leave it 0
    // (zero-initialized via `CommonParams params = {}` at every call site).
    uint32_t debugVersion;
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
    TemporalFwdBuffer temporalFwd; // inter-launch record for the two-launch temporal split
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