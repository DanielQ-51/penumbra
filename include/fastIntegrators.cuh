#pragma once

#include "objects.cuh"
#include "wavefrontHelper.cuh"

__host__ void launch_wavefrontUnidirectional(
    Camera camera, 
    const SceneContext sceneContext,
    int numSample, int maxDepth,
    int h_w, int h_h, 
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess, 
    float exposure
);