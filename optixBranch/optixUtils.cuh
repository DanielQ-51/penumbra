#pragma once
#include "objects.cuh"
#include "util.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
#include "helpers.cuh"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#ifndef USE_SER
#define USE_SER 1
#endif

struct SurfaceHit {
    bool isHit;
    float t;
    uint32_t primId;
    uint32_t instanceId;
    float2 barycentrics;
};

__device__ __forceinline__ SurfaceHit traceClosestSER(
    const CommonParams& params,
    const Ray& r,
    float tmax = 1e30
) {
    optixTraverse(
        params.bvh_handle,
        r.origin, r.direction,
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0
    );

    uint32_t reorderKey;
    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();
        hit.instanceId = optixHitObjectGetInstanceId();

        // although this is better than invoking, this is still not optimal.
        // This requires an whole triangle intersection sequence, but it is faster than the context switch
        // that comes with a non-empty closest hit shader
        hit.barycentrics = getBarycentrics(params.shadeContext, hit.primId, r, hit.instanceId);

        //hit.barycentrics = optixHitObjectGetTriangleBarycentrics();

        reorderKey = 1;
    } else {
        reorderKey = 0;
    }

    optixReorder(reorderKey, 1);

    return hit;
}

__device__ __forceinline__ SurfaceHit traceClosestHitObjSER(
    const CommonParams& params,
    const Ray& r,
    float tmax = 1e30
) {
    optixTraverse(
        params.bvh_handle,
        r.origin, r.direction,
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0
    );

    uint32_t reorderKey;
    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();
        hit.instanceId = optixHitObjectGetInstanceId();

        // although this is better than invoking, this is still not optimal.
        // This requires an whole triangle intersection sequence, but it is faster than the context switch
        // that comes with a non-empty closest hit shader
        hit.barycentrics = getBarycentrics(params.shadeContext, hit.primId, r, hit.instanceId);

        //hit.barycentrics = optixHitObjectGetTriangleBarycentrics();
    }

    optixReorder();

    return hit;
}

__device__ __forceinline__ SurfaceHit traceClosestStreamCompactSER(
    const CommonParams& params,
    const Ray& r,
    float tmax = 1e30
) {
    optixTraverse(
        params.bvh_handle,
        r.origin, r.direction,
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0
    );

    uint32_t reorderKey;
    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();
        hit.instanceId = optixHitObjectGetInstanceId();

        // although this is better than invoking, this is still not optimal.
        // This requires an whole triangle intersection sequence, but it is faster than the context switch
        // that comes with a non-empty closest hit shader
        hit.barycentrics = getBarycentrics(params.shadeContext, hit.primId, r, hit.instanceId);

        //hit.barycentrics = optixHitObjectGetTriangleBarycentrics();
    }

    optixReorder();

    return hit;
}

__device__ __forceinline__ SurfaceHit traceClosestNoSER(
    const CommonParams& params,
    Ray r,
    float tmax = 999999999.0f) 
{
    uint32_t p0 = 0, p1 = 0;
    optixTraverse(
        params.bvh_handle,
        r.origin, r.direction,
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1
    );

    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();
        hit.instanceId = optixHitObjectGetInstanceId();

        // although this is better than invoking, this is still not optimal.
        hit.barycentrics = getBarycentrics(params.shadeContext, hit.primId, r, hit.instanceId);
    }

    return hit;
}

__device__ __forceinline__ SurfaceHit traceClosest(
    const CommonParams& params,
    Ray r,
    float tmax = 999999999.0f) 
{
#if USE_SER == 1
    return traceClosestSER(params, r, tmax);
#else 
    return traceClosestNoSER(params, r, tmax);
#endif
}

__device__ __forceinline__ bool traceVisibility(
    const CommonParams& params, 
    Ray r,
    float targetDistance) 
{
    optixTraverse(
        params.bvh_handle,
        r.origin, r.direction,
        EPSILON, targetDistance, 0.0f, 
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 
        0, 1, 0
    );

    return optixHitObjectIsHit();
}