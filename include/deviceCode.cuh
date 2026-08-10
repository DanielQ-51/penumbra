#pragma once

#include "sceneContexts.cuh"
#include "configParser.cuh"

__host__ void updateConstants(const RenderConfig& config);

__host__ void launch_unidirectional(
    int maxDepth, 
    Camera camera, 
    const Material* __restrict__ materials, 
    TextureView textures, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, 
    int numSample, 
    bool useMIS, 
    int w, int h, 
    float4* __restrict__ colors
);

__host__ void launch_naive_unidirectional(
    int maxDepth, 
    Camera camera, 
    const Material* __restrict__ materials, 
    TextureView textures, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, 
    int numSample, 
    bool useMIS, 
    int w, int h, 
    float4* __restrict__ colors
);

__host__ void launch_bidirectional(
    int eyeDepth, 
    int lightDepth, 
    Camera camera, 
    PathVertices* eyePath, 
    PathVertices* lightPath, 
    const Material* __restrict__ materials, 
    TextureView textures, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, int numSample, int w, int h, 
    float3 h_sceneCenter, float h_sceneRadius, 
    float4* __restrict__ colors, float4* __restrict__ overlay, 
    bool postProcess
);

__host__ void launch_VCM(
    int eyeDepth, 
    int lightDepth, 
    Camera camera, 
    VCMPathVertices* lightPath, 
    Photons* photons, 
    Photons* photons_sorted, 
    const Material* __restrict__ materials, 
    TextureView textures, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, int numSample, 
    int w, int h, 
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess, 
    float mergeRadiusPower, 
    float initialRadiusMultiplier
);

__host__ void launch_SPPM(
    int eyeDepth, 
    int lightDepth, 
    Camera camera, 
    Photons* photons, 
    Photons* photons_sorted, 
    const Material* __restrict__ materials, 
    TextureView textures, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, int numSample, 
    int w, int h, 
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess, 
    float mergeRadiusPower, 
    float initialRadiusMultiplier
);
