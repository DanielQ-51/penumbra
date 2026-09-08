#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include "wavefrontHelper.cuh"
#include "sceneContexts.cuh"
#include "helpers.cuh"
#include <cub/cub.cuh>

__device__ __constant__ bool SAMPLE_ENVIRONMENT = false;
__device__ __constant__ float sceneRadius;
__device__ __constant__ float3 sceneCenter;
__device__ __constant__ float3 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

#ifndef USE_MORTON_CODE_SORT
#define USE_MORTON_CODE_SORT 1
#endif

#ifndef USE_MATERIAL_SORT
#define USE_MATERIAL_SORT 1
#endif




__device__ bool approxEq(float a, float b, float tol = 1e-3f) {
    return fabsf(a - b) <= tol;
}
__device__ bool approxEqF3(float3 a, float3 b, float tol = 1e-3f) {
    return approxEq(a.x, b.x, tol) &&
           approxEq(a.y, b.y, tol) &&
           approxEq(a.z, b.z, tol);
}
__global__ void testWriteKernel(RayQueue rq, HitBuffer hb, ShadowQueue sq) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) return;

    printf("--- Generating Data ---\n");

    // RayQueue Data
    float3 inOrigin = make_float3(1.5f, -2.5f, 3.14f);
    float3 inDir = make_float3(0.57735f, 0.57735f, 0.57735f);
    bool inPrevDelta = true;
    float3 inThroughput = make_float3(0.25f, 0.5f, 0.75f);
    int inPixelIdx = 1234567;
    int inDepth = 14;
    uint32_t inStack = 0xABCD;
    float inLastPDF = 3.14159f;

    rq.setAll(0, inOrigin, inDir, inPrevDelta, inThroughput, inPixelIdx, inDepth, inStack, inLastPDF);

    // HitBuffer Data
    hb.setHit(0, 42.5f, 0.333f, 0.666f, 888999);

    // ShadowQueue Data
    float3 inShadowOrigin = make_float3(10.0f, 20.0f, 30.0f);
    float3 inShadowDir = make_float3(0.0f, 1.0f, 0.0f);
    float3 inL = make_float3(10.0f, 5.0f, 2.5f);

    sq.setShadowRay(0, inShadowOrigin, inShadowDir, 100.0f, inL, 654321);
}
__global__ void testReadKernel(RayQueue rq, HitBuffer hb, ShadowQueue sq, int* passed, int* failed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) return;

    // Expected Data (Hardcoded to match Kernel 1)
    float3 inOrigin = make_float3(1.5f, -2.5f, 3.14f);
    float3 inDir = make_float3(0.57735f, 0.57735f, 0.57735f);
    bool inPrevDelta = true;
    float3 inThroughput = make_float3(0.25f, 0.5f, 0.75f);

    float3 outOrigin, outDir, outThroughput;
    bool outPrevDelta;
    int outPixelIdx, outDepth;
    uint32_t outStack;
    float outLastPDF;

    // --- TEST: RayQueue ---
    printf("\n[Testing RayQueue]\n");
    rq.getAll(0, outOrigin, outDir, outPrevDelta, outThroughput, outPixelIdx, outDepth, outStack, outLastPDF);

    if (!approxEqF3(inOrigin, outOrigin)) { printf("FAIL: RayQueue Origin.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (!approxEqF3(inDir, outDir, 0.01f)) { printf("FAIL: RayQueue Direction.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (inPrevDelta != outPrevDelta) { printf("FAIL: RayQueue PrevDelta.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (!approxEqF3(inThroughput, outThroughput, 0.05f)) { printf("FAIL: RayQueue Throughput.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (1234567 != outPixelIdx) { printf("FAIL: RayQueue PixelIdx.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (14 != outDepth) { printf("FAIL: RayQueue Depth.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (0xABCD != outStack) { printf("FAIL: RayQueue Stack.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (3.14159f != outLastPDF) { printf("FAIL: RayQueue LastPDF.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);

    // --- TEST: HitBuffer ---
    printf("\n[Testing HitBuffer]\n");
    float outT, outU, outV;
    int outTriID;
    hb.getAllInfo(0, outT, outU, outV, outTriID);

    if (42.5f != outT) { printf("FAIL: HitBuffer T.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (0.333f != outU || 0.666f != outV) { printf("FAIL: HitBuffer UV.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (888999 != outTriID) { printf("FAIL: HitBuffer TriID.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);

    // Overwrite HitBuffer for Kernel 3
    hb.setMiss(0);

    // --- TEST: ShadowQueue ---
    printf("\n[Testing ShadowQueue]\n");
    Ray outShadowRay;
    float outMaxT;
    sq.getAnyHitData(0, outShadowRay, outMaxT);

    float3 inShadowOrigin = make_float3(10.0f, 20.0f, 30.0f);
    float3 inShadowDir = make_float3(0.0f, 1.0f, 0.0f);
    float3 inL = make_float3(10.0f, 5.0f, 2.5f);

    if (!approxEqF3(inShadowOrigin, outShadowRay.origin)) { printf("FAIL: ShadowQueue Origin.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (!approxEqF3(inShadowDir, outShadowRay.direction, 0.01f)) { printf("FAIL: ShadowQueue Direction.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (100.0f != outMaxT) { printf("FAIL: ShadowQueue MaxT.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);

    float3 outL;
    int outShadowPIdx;
    bool validData = sq.getAccumulateData(0, outL, outShadowPIdx);

    if (!validData) { printf("FAIL: ShadowQueue valid accumulate data flagged as false.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (!approxEqF3(inL, outL, 0.5f)) { printf("FAIL: ShadowQueue L (RGB9E5).\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);
    if (654321 != outShadowPIdx) { printf("FAIL: ShadowQueue PIdx.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);

    // Write sentinel for Kernel 3 to check
    sq.setAnyHitResultNoAlphaTest(0, false);
}
__global__ void testSentinelKernel(HitBuffer hb, ShadowQueue sq, int* passed, int* failed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx != 0) return;

    if (hb.isHit(0)) { printf("FAIL: HitBuffer setMiss() failed.\n"); atomicAdd(failed, 1); } else atomicAdd(passed, 1);

    float3 outL;
    int outShadowPIdx;
    if (sq.getAccumulateData(0, outL, outShadowPIdx)) {
        printf("FAIL: ShadowQueue sentinel failed. Should return false.\n");
        atomicAdd(failed, 1);
    } else atomicAdd(passed, 1);
}
void runDataStructureTests() {
    int maxRays = 1;

    RayQueue rq;
    HitBuffer hb;
    ShadowQueue sq;

    // Allocate queues
    cudaMalloc(&rq.origin_plus_dir, maxRays * sizeof(float4));
    cudaMalloc(&rq.payload, maxRays * sizeof(uint4));
    cudaMalloc(&hb.data, maxRays * sizeof(float4));
    cudaMalloc(&sq.origin_plus_dist, maxRays * sizeof(float4));
    cudaMalloc(&sq.direction, maxRays * sizeof(uint32_t));
    cudaMalloc(&sq.payload, maxRays * sizeof(uint2));

    // Allocate host/device tracking ints
    int *d_passed, *d_failed;
    cudaMalloc(&d_passed, sizeof(int));
    cudaMalloc(&d_failed, sizeof(int));
    cudaMemset(d_passed, 0, sizeof(int));
    cudaMemset(d_failed, 0, sizeof(int));

    printf("--- Starting Data Structure Tests ---\n");

    // Launch Pipeline
    testWriteKernel<<<1, 1>>>(rq, hb, sq);
    cudaDeviceSynchronize(); // FLUSH L1/L2 CACHE

    testReadKernel<<<1, 1>>>(rq, hb, sq, d_passed, d_failed);
    cudaDeviceSynchronize(); // FLUSH L1/L2 CACHE

    testSentinelKernel<<<1, 1>>>(hb, sq, d_passed, d_failed);
    cudaDeviceSynchronize(); // WAIT FOR DEVICE PRINTF

    // Fetch Results
    int h_passed = 0;
    int h_failed = 0;
    cudaMemcpy(&h_passed, d_passed, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n--- Test Summary ---\n");
    printf("Passed: %d\n", h_passed);
    printf("Failed: %d\n", h_failed);
    if (h_failed == 0) printf("SUCCESS: All data structures working as expected!\n");

    // Free memory
    cudaFree(rq.origin_plus_dir);
    cudaFree(rq.payload);
    cudaFree(hb.data);
    cudaFree(sq.origin_plus_dist);
    cudaFree(sq.direction);
    cudaFree(sq.payload);
    cudaFree(d_passed);
    cudaFree(d_failed);
}

__host__ void allocateBuffers(
    RayQueue& q1,
    RayQueue& q2,
    HitBuffer& hb,
    ShadowQueue& sq,
    uint32_t*& pr,
    uint32_t*& si,
    float4*& out,
    uint32_t*& skeys,
    uint32_t*& svals,
    uint32_t*& skeys1,
    uint32_t*& svals1,
    int maxRays
)
{
    size_t f4Size    = maxRays * sizeof(float4);
    size_t uIntSize  = maxRays * sizeof(uint32_t);
    size_t uInt2Size = maxRays * sizeof(uint2);
    size_t uInt4Size = maxRays * sizeof(uint4);

    void* temp_ptr;

    // Queue 1
    cudaMalloc(&temp_ptr, f4Size);
    q1.origin_plus_dir = (float4*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    q1.payload = (uint4*)temp_ptr;

    // Queue 2
    cudaMalloc(&temp_ptr, f4Size);
    q2.origin_plus_dir = (float4*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    q2.payload = (uint4*)temp_ptr;

    // Hit Buffer
    cudaMalloc(&temp_ptr, f4Size);
    hb.data = (float4*)temp_ptr;

    // Shadow Queue
    cudaMalloc(&temp_ptr, f4Size);
    sq.origin_plus_dist = (float4*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    sq.direction = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt2Size);
    sq.payload = (uint2*)temp_ptr;

    // Compaction Buffers
    cudaMalloc(&temp_ptr, uIntSize);
    pr = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    si = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    skeys = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    svals = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    skeys1 = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    svals1 = (uint32_t*)temp_ptr;

    // Output Buffer
    cudaMalloc(&temp_ptr, f4Size);
    out = (float4*)temp_ptr;
}

void freeBuffers(
    RayQueue& q1,
    RayQueue& q2,
    HitBuffer& hb,
    ShadowQueue& sq,
    uint32_t* pr,
    uint32_t* si,
    uint32_t* skeys,
    uint32_t* svals,
    uint32_t* skeys1,
    uint32_t* svals1,
    float4* out
) {
    cudaFree(q1.origin_plus_dir);
    cudaFree(q1.payload);

    cudaFree(q2.origin_plus_dir);
    cudaFree(q2.payload);

    cudaFree(hb.data);

    cudaFree(sq.origin_plus_dist);
    cudaFree(sq.direction);
    cudaFree(sq.payload);

    cudaFree(pr);
    cudaFree(si);

    cudaFree(skeys);
    cudaFree(svals);
    cudaFree(skeys1);
    cudaFree(svals1);

    cudaFree(out);
}

__global__ void generateInitialRays(
    RNGState* rngStates,
    Camera camera,
    RayQueue queue,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int rayID = y*w + x;

    RNGState localState = load_rng(rayID, frameNum, 0, rngStates);
    Ray r = camera.generateCameraRay(localState, x, y);

    queue.setAll(
        rayID,          // idx
        r.origin,       // inOrigin
        r.direction,    // inDir
        true,           // inPrevDelta (camera rays are treated as delta paths)
        f3(1.0f),       // inThroughput
        rayID,          // inPixelIdx
        0,              // inDepth
        0u,             // inStack
        0.0f            // inLastPDF
    );

    save_rng(rayID, &localState, rngStates);
}

__global__ void closestHit(
    const BVHContext bvhContext,
    const RayQueue rayQueue,
    HitBuffer hitBuffer,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= activeRays) return;

    float3 barycentric;
    float t;
    int triID;
    BVHSceneIntersect_lightweight(
        rayQueue.getRay(idx),
        bvhContext,
        barycentric,
        t,
        triID
    );

    hitBuffer.setHit(
        idx,
        t,
        barycentric.x,
        barycentric.y,
        triID
    );
}

__global__ void shade(
    RNGState* rngStates,

    const ShadeContext shadeContext,
    const RayQueue readQueue,
    const HitBuffer hits,
    RayQueue writeQueue,
    ShadowQueue shadowRays,

    uint32_t* predicate,
    int activeRays,

    int frameNum,
    int maxDepth,

    uint32_t* shadowRayIndex, // ?

    float4* output,

    uint32_t* sortValuesOut
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

#if USE_MATERIAL_SORT == 1
    int readIndex = __ldg(&sortValuesOut[idx]);
#else
    int readIndex = idx;
#endif

    float3 prevPos;
    float3 incomingDir;
    bool prevDelta;
    float3 throughput;
    int pixelIdx;
    int depth;
    uint32_t mediumStack;
    float lastPDF;

    readQueue.getAll(
        readIndex,
        prevPos,
        incomingDir,
        prevDelta,
        throughput,
        pixelIdx,
        depth,
        mediumStack,
        lastPDF
    );

    // handle medium true/false hit (unimplemented)
    int triID;
    int materialID;
    float2 uv;
    bool backface;
    float3 normal;
    float3 shadingPos;

    RNGState localState = load_rng(pixelIdx, frameNum, depth, rngStates);
    // handle current interactiion
    {
        float u, v;
        float t;
        hits.getAllInfo(readIndex, t, u, v, triID);

        if (t >= 1e30f) // indicates a miss
        {
            predicate[idx] = 0;
            //float3 contribution = sampleSky(incomingDir) * throughput;

            float3 contribution = shadeContext.lightSampler.envMap.sampleDir(incomingDir) * throughput;

            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                shadeContext.lightSampler.evaluateEnvPdf(incomingDir) // alternate strategy
            );

            accumulateOutput(output, contribution * misWeight, pixelIdx);
            shadowRays.setIgnore(idx);
            return;
        }

        // triangle is heavy on registers so we dont keep it around too long
        const Triangle& tri = shadeContext.scene[triID];
        materialID = tri.materialID;

        uv = __ldg(&shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) +
            __ldg(&shadeContext.vertices->uvs[tri.uvbInd]) * u +
            __ldg(&shadeContext.vertices->uvs[tri.uvcInd]) * v;

        float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
        float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
        float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

        shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

        float3 a_n = f3(__ldg(&shadeContext.vertices->normals[tri.naInd]));
        float3 b_n = f3(__ldg(&shadeContext.vertices->normals[tri.nbInd]));
        float3 c_n = f3(__ldg(&shadeContext.vertices->normals[tri.ncInd]));

        normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
        backface = dot(normal, incomingDir) > 0.0f;
        normal = backface ? -normal : normal;

        float3 contribution = backface ? f3(0.0f) : f3(tri.emission) * throughput;

        toLocal(incomingDir, normal, incomingDir);

        float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
            lastPDF, // primary strategy
            (t * t * shadeContext.lightSampler.evaluateMeshPdf(tri) / (fabsf(incomingDir.z))) // alternate strategy
        );

        accumulateOutput(output, contribution * misWeight, pixelIdx);
    }

    bool currDelta = shadeContext.materials[materialID].isSpecular;
    // NEE
    if (!currDelta) {
        float3 lightNormal;
        float3 emission;
        float3 shadingPosToLightNormalized;
        float t_max;
        float pdf;

        bool sampledEnv = sample(
            shadeContext.lightSampler,
            rand(&localState), rand4(&localState),
            shadingPos,
            shadeContext.vertices,
            emission,
            shadingPosToLightNormalized,
            lightNormal,
            t_max,
            pdf
        );


        float3 shadingPosToLightLocal;
        toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

        bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
        bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
        if (surfaceBackface || lightBackface) {
            shadowRays.setIgnore(idx);
        } else {
            float bsdfPDF;

            pdf_eval(
                shadeContext.materials,
                materialID,
                shadeContext.textures,
                incomingDir,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                bsdfPDF,
                uv
            );

            float3 f_val;
            f_eval(
                shadeContext.materials,
                materialID,
                shadeContext.textures,
                incomingDir,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                f_val,
                uv
            );

            float3 contribution;
            float misWeight;

            if (sampledEnv) {
                float cosSurface = dot(normal, shadingPosToLightNormalized);
                contribution = throughput * f_val * emission * cosSurface / pdf;
                misWeight = powerHeuristicTwoStrategy(
                    pdf,
                    bsdfPDF
                );
            } else {
                float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                float cosSurface = dot(normal, shadingPosToLightNormalized);

                contribution = throughput * // throughput
                    f_val * emission * cosLight * // NEE contribution
                    cosSurface / (pdf * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                misWeight = powerHeuristicTwoStrategy(
                    (t_max * t_max) * pdf / cosLight, // convert area pdf to SA
                    bsdfPDF // alt strat
                );
            }

            shadowRays.setShadowRay(
                idx,
                shadingPos + shadingPosToLightNormalized * RAY_EPSILON,
                shadingPosToLightNormalized,
                t_max * (1.0f - EPSILON3),
                contribution * (misWeight),
                pixelIdx
            );
        }
    } else // current is specular, skip NEE
    {
        shadowRays.setIgnore(idx);
    }

    // handle russian roulette
    {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);

        if (rand(&localState) > p)   // survive with probability p
        {
            predicate[idx] = 0;
            save_rng(pixelIdx, &localState, rngStates);
            return;
        }

        throughput /= p;
    }

    if (depth > maxDepth) {
        predicate[idx] = 0;
        save_rng(pixelIdx, &localState, rngStates);
        return;
    }

    // Sample next Direction
    {
        float3 outgoing;
        float3 f_val;
        float pdf;


        sample_f_eval(
            localState,
            shadeContext.materials,
            materialID,
            shadeContext.textures,
            incomingDir,
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            backface,
            outgoing,
            f_val,
            pdf,
            uv,
            TRANSPORTMODE_RADIANCE
        );



        if (pdf < EPSILON)
        {
            predicate[idx] = 0; // mark as dead
            save_rng(pixelIdx, &localState, rngStates);
            return;
        }

        throughput *= f_val * fabsf(outgoing.z) / pdf;
        toWorld(outgoing, normal, outgoing);

        // write to next rayQueue

        writeQueue.setAll(
            idx,            // idx
            shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON,       // inOrigin
            outgoing,       // inDir
            currDelta,      // inPrevDelta
            throughput,     // inThroughput
            pixelIdx,       // inPixelIdx
            depth + 1,      // inDepth
            0u,             // inStack
            pdf             // inLastPDF
        );
    }

    predicate[idx] = 1;
    save_rng(pixelIdx, &localState, rngStates);
}

__global__ void anyHit(
    BVHContext bvhContext,
    ShadowQueue shadowRays,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    Ray r;
    float maxT;

    // how much the throughput is scaled by going through the shadow ray.
    // 0 if occluded, 1 if not occluded
    float3 throughputScale;

    shadowRays.getAnyHitData(idx, r, maxT);
    if (maxT == -1.0f) {
        shadowRays.setAnyHitResultNoAlphaTest(idx, false);
        return;
    }

    BVHShadowRay_NoAlpha(
        r,
        bvhContext.BVH,
        bvhContext.BVHindices,
        bvhContext.vertices,
        bvhContext.scene,
        throughputScale,
        maxT
    );

    shadowRays.setAnyHitResultNoAlphaTest(idx, lengthSquared(throughputScale) > 0.0f);
}

__global__ void shadeShadowRay(
    ShadowQueue shadowRays,
    int activeRays,
    float4* output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    float3 contribution;
    int pixelIdx;
    if (shadowRays.getAccumulateData(idx, contribution, pixelIdx)) {
        accumulateOutput(output, contribution, pixelIdx);
    }
}
__global__ void compactRayQueue_NOSORT(
    const RayQueue sparseQueue,
    RayQueue denseQueue,
    const uint32_t* __restrict__ predicates,
    const uint32_t* __restrict__ scanIndices,
    int activeRays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    if (predicates[idx] == 1)
    {
        int denseIdx = scanIndices[idx];

        float4 rayData = __ldg(&sparseQueue.origin_plus_dir[idx]);

        denseQueue.origin_plus_dir[denseIdx] = rayData;
        denseQueue.payload[denseIdx] = __ldg(&sparseQueue.payload[idx]);
    }
}

__global__ void compactRayQueue_SORT(
    const RayQueue sparseQueue,
    RayQueue denseQueue,
    const uint32_t* __restrict__ predicates,
    const uint32_t* __restrict__ scanIndices,
    uint32_t* sortKeysIn,
    uint32_t* sortValuesIn,
    int activeRays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    if (predicates[idx] == 1)
    {
        int denseIdx = scanIndices[idx];

        float4 rayData = __ldg(&sparseQueue.origin_plus_dir[idx]);

        denseQueue.origin_plus_dir[denseIdx] = rayData;
        denseQueue.payload[denseIdx] = __ldg(&sparseQueue.payload[idx]);

        float3 direction = getDirectionFromPacked(rayData);
        float invDiameter = 1.0f / (sceneRadius * 2.0f);
        sortKeysIn[denseIdx] = generateMortonSortKey(f3(rayData), direction, sceneMin, invDiameter);
        sortValuesIn[denseIdx] = denseIdx;
    }
}

#if USE_MATERIAL_SORT == 1
// takes a dense hitbuffer and calculates sort keys for it
__global__ void calculateMaterialSortKeys(
    const HitBuffer hitBuffer,
    const Material* __restrict__ materials,
    const Triangle* __restrict__ scene,
    uint32_t* sortKeysIn,
    uint32_t* sortValuesIn,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    // -1 if its a miss
    int triID = hitBuffer.getID(idx);
    int materialID = (triID < 0) ? -1 : __ldg(&scene[triID].materialID);

    // as a hint as to which texture its using
    int textureStartIndex = (materialID < 0) ?
        0 : (__ldg(&materials[materialID].baseColorTex) + 1); // +1 so "no texture" (-1) maps to 0

    uint32_t key = generateMaterialSortKey(materialID, textureStartIndex);
    sortKeysIn[idx] = key;
    sortValuesIn[idx] = idx;
}
#endif

__global__ void reorderRayQueue(
    const RayQueue unsortedQueue,
    RayQueue sortedQueue,
    const uint32_t* __restrict__ sortedIndices,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    uint32_t sortedIdx = sortedIndices[idx];

    sortedQueue.origin_plus_dir[idx] = __ldg(&unsortedQueue.origin_plus_dir[sortedIdx]);
    sortedQueue.payload[idx]         = __ldg(&unsortedQueue.payload[sortedIdx]);
}

__global__ void debugPrintRayOrigin(RayQueue readQueue, int currSample, int depth, int activeRays, int targetPixelIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    int pixelIdx, rayDepth;
    readQueue.getPackedState(idx, pixelIdx, rayDepth);

    // Filter to a single pixel to prevent millions of printf calls
    if (pixelIdx == targetPixelIdx) {
        float3 origin = readQueue.getOrigin(idx);
        printf("Sample: %4d | Depth: %2d | Pixel: %7d | Origin: (%8.4f, %8.4f, %8.4f)\n",
               currSample, depth, pixelIdx, origin.x, origin.y, origin.z);
    }
}

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
)
{
    runDataStructureTests();
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float3));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float3));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));
    cudaMemcpyToSymbol(w, &(h_w), sizeof(int));
    cudaMemcpyToSymbol(h, &(h_h), sizeof(int));

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    RayQueue temp_rayQueue1;
    RayQueue temp_rayQueue2;
    HitBuffer temp_hitBuffer;
    ShadowQueue temp_shadowQueue;
    uint32_t* d_predicate;
    uint32_t* d_scanIndices;
    float4* d_finalOutput;

    uint32_t* d_sortKeysIn;
    uint32_t* d_sortValuesOut;
    uint32_t* d_sortKeysOut;
    uint32_t* d_sortValuesIn;

    uint32_t* d_shadowRayIndex;
    cudaMalloc(&d_shadowRayIndex, sizeof(uint32_t));

    allocateBuffers(
        temp_rayQueue1,
        temp_rayQueue2,
        temp_hitBuffer,
        temp_shadowQueue,
        d_predicate,
        d_scanIndices,
        d_finalOutput,
        d_sortKeysIn,
        d_sortValuesOut,
        d_sortKeysOut,
        d_sortValuesIn,
        h_h * h_w
    );

    RayQueue* d_readQueue = &temp_rayQueue1;
    RayQueue* d_writeQueue = &temp_rayQueue2;
    HitBuffer* d_hits = &temp_hitBuffer;
    ShadowQueue* d_shadowQueue = &temp_shadowQueue;

    BVHContext bvhContext = getBVHContext(sceneContext);
    ShadeContext shadeContext = getShadeContext(sceneContext);

    void* d_temp_storage_exclusiveSum = nullptr;
    size_t temp_storage_bytes_exSum = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage_exclusiveSum, temp_storage_bytes_exSum,
        d_predicate, d_scanIndices, h_h * h_w
    );

    cudaMalloc(&d_temp_storage_exclusiveSum, temp_storage_bytes_exSum);

    void* d_temp_storage_sort = nullptr;
    size_t temp_storage_bytes_sort = 0;

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage_sort, temp_storage_bytes_sort,
        d_sortKeysIn, d_sortKeysOut, d_sortValuesIn, d_sortValuesOut, h_h * h_w
    );

    cudaMalloc(&d_temp_storage_sort, temp_storage_bytes_sort);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    auto lastSaveTime = std::chrono::steady_clock::now();
    int saveIntervalSamples = 20000;
    Image image = Image(h_w, h_h);
    // Post-processing (exposure/tonemap/gamma) now happens on the GPU in
    // cleanFormatAndPostProcessImage, so saveImageBMP() must not re-apply it.
    image.postProcess = false;
    std::vector<float4> h_finalOutput(h_w * h_h);

    std::cout << "Begin Render" << std::endl;

    // for initial camera raygen
    dim3 blockSize(16, 16);
    dim3 gridSize((h_w+15)/16, (h_h+15)/16);

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        int activeRays = h_h * h_w;

        generateInitialRays<<<gridSize, blockSize>>> (
            d_rngStates,
            camera,
            *d_readQueue,
            currSample
        );

        for (int depth = 0; depth <= maxDepth; depth++)
        {
            int blocks = (activeRays + 255) / 256;

#if USE_MORTON_CODE_SORT == 1

#if USE_MATERIAL_SORT == 1
            if (depth > 0)
#else
            if (depth > 3)
#endif
            {
                cub::DeviceRadixSort::SortPairs(
                    d_temp_storage_sort, temp_storage_bytes_sort,
                    d_sortKeysIn, d_sortKeysOut, d_sortValuesIn, d_sortValuesOut, activeRays
                );

                reorderRayQueue<<<blocks, 256>>>(
                    *d_readQueue, *d_writeQueue, d_sortValuesOut, activeRays
                );

                std::swap(d_readQueue, d_writeQueue);
            }
#endif
            closestHit<<<blocks, 256>>> (
                bvhContext,
                *d_readQueue,
                *d_hits,
                activeRays
            );

#if USE_MATERIAL_SORT == 1

            calculateMaterialSortKeys<<<blocks, 256>>> (
                *d_hits,
                sceneContext.materials,
                sceneContext.scene,
                d_sortKeysIn,
                d_sortValuesIn,
                activeRays
            );

            cub::DeviceRadixSort::SortPairs(
                d_temp_storage_sort, temp_storage_bytes_sort,
                d_sortKeysIn, d_sortKeysOut, d_sortValuesIn, d_sortValuesOut, activeRays
            );
#endif
            shade<<<blocks, 256>>> (
                d_rngStates,
                shadeContext,
                *d_readQueue,
                *d_hits,
                *d_writeQueue,
                *d_shadowQueue,
                d_predicate,
                activeRays,
                currSample,
                maxDepth,
                d_shadowRayIndex,
                colors,
                d_sortValuesOut
            );

            anyHit<<<blocks, 256>>> (
                bvhContext,
                *d_shadowQueue,
                activeRays
            );

            shadeShadowRay<<<blocks, 256>>> (
                *d_shadowQueue,
                activeRays,
                colors
            );

            cub::DeviceScan::ExclusiveSum(
                d_temp_storage_exclusiveSum, temp_storage_bytes_exSum,
                d_predicate, d_scanIndices, activeRays
            );

            uint32_t lastPredicate = 0;
            uint32_t lastScanIndex = 0;

            int lastIndex = activeRays - 1;

            cudaMemcpy(&lastPredicate, &d_predicate[lastIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastScanIndex, &d_scanIndices[lastIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost);

            int newActiveRays = lastScanIndex + lastPredicate;
            if (newActiveRays == 0) {
                break;
            }
#if USE_MATERIAL_SORT == 1
            if (false)
#else
            if (depth < 3)
#endif
            {
                compactRayQueue_NOSORT<<<blocks, 256>>> (
                    *d_writeQueue, // writes from writequeue to readqueue
                    *d_readQueue,
                    d_predicate,
                    d_scanIndices,
                    activeRays
                );
            } else {
                compactRayQueue_SORT<<<blocks, 256>>> (
                    *d_writeQueue, // writes from writequeue to readqueue
                    *d_readQueue,
                    d_predicate,
                    d_scanIndices,
                    d_sortKeysIn,
                    d_sortValuesIn,
                    activeRays
                );
            }


            activeRays = newActiveRays;
        }

        if ((currSample % saveIntervalSamples == 0 || currSample == numSample-1) && DO_PROGRESSIVERENDER)
        {
            if (postProcess) {
                cleanFormatAndPostProcessImage<<<gridSize, blockSize>>>(
                    colors, overlay, d_finalOutput, h_w, h_h, currSample, exposure, image.use_fitted_aces
                );
            } else {
                cleanAndFormatImage<<<gridSize, blockSize>>>(
                    colors, overlay, d_finalOutput, h_w, h_h, currSample
                );
            }

            cudaMemcpy(h_finalOutput.data(), d_finalOutput, h_w * h_h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < h_w * h_h; i++) {
                int x = i % h_w;
                int y = i / h_w;
                image.setColor(x, y, h_finalOutput[i]);
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);


            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            cudaMemset(overlay, 0, h_w * h_h * sizeof(float4));
        }
    }

    printf("\n");
    cudaFree(d_rngStates);
    cudaFree(d_shadowRayIndex);
    cudaFree(d_temp_storage_exclusiveSum);
    cudaFree(d_temp_storage_sort);

    freeBuffers(
        temp_rayQueue1,
        temp_rayQueue2,
        temp_hitBuffer,
        temp_shadowQueue,
        d_predicate,
        d_scanIndices,
        d_sortKeysIn,
        d_sortValuesOut,
        d_sortKeysOut,
        d_sortValuesIn,
        d_finalOutput
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}