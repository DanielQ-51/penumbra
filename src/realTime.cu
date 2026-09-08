#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "deviceCode.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include <cub/cub.cuh>

#include "realTimeUtils.cuh"
#include "wavefrontHelper.cuh"

__device__ __constant__ bool SAMPLE_ENVIRONMENT = false;
__device__ __constant__ float sceneRadius;
__device__ __constant__ float4 sceneCenter;
__device__ __constant__ float4 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

__global__ void processPrimary(
    RNGState* rngStates, 
    const BVHContext bvhContext,
    const ShadeContext shadeContext,
    HitBuffer hitBuffer,

    ReSTIRRayQueue writeQueue,
    ReSTIRShadowQueue shadowRays,

    Reservoir writeReservoir,
    
    uint32_t* predicate,
    int activeRays,

    GBuffer gBuffer,
    Camera camera,
    int frameNum,
    int maxDepth,

    float4* output
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

#if RNG_MODE == 3
    uint32_t seed = localState.getSeed();
#else // Restir integrator MUST use stateless RNG
    uint32_t seed = 0xFFFFFFFF;
#endif
    
    float4 barycentric;
    float4 incomingDir;
    float t;
    int triID;
{
    Ray r = camera.generateCameraRay(localState, x, y);
    BVHSceneIntersect_lightweight(
        r,
        bvhContext,
        barycentric,
        t, 
        triID
    );
    incomingDir = r.direction;
}

    float u, v;
    u = barycentric.x;
    v = barycentric.y;

    // No need to set hit buffer, since all primary hit stuff is deal with in this kernel
    int triID;
    int materialID;
    float2 uv;
    bool backface;
    float4 normal;
    float4 shadingPos;

    {
        if (t >= 1e30f) // indicates a miss
        {
            predicate[pixelIdx] = 0;
            float4 contribution = shadeContext.lightSampler.envMap.sampleDir(incomingDir);
            accumulateOutput(output, contribution, pixelIdx);
            shadowRays.setIgnore(pixelIdx);
            save_rng(pixelIdx, &localState, rngStates);
            return;
        }

        // triangle is heavy on registers so we dont keep it around too long
        const Triangle& tri = shadeContext.scene[triID];
        materialID = tri.materialID;

        uv = __ldg(&shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) + 
            __ldg(&shadeContext.vertices->uvs[tri.uvbInd]) * u + 
            __ldg(&shadeContext.vertices->uvs[tri.uvcInd]) * v;

        float4 apos = __ldg(&shadeContext.vertices->positions[tri.aInd]);
        float4 bpos = __ldg(&shadeContext.vertices->positions[tri.bInd]);
        float4 cpos = __ldg(&shadeContext.vertices->positions[tri.cInd]);

        shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

        float4 a_n = __ldg(&shadeContext.vertices->normals[tri.naInd]);
        float4 b_n = __ldg(&shadeContext.vertices->normals[tri.nbInd]);
        float4 c_n = __ldg(&shadeContext.vertices->normals[tri.ncInd]);
        
        normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
        backface = dot(normal, incomingDir) > 0.0f;
        normal = backface ? -normal : normal;

        float4 contribution = backface ? f4(0.0f) : tri.emission;

        toLocal(incomingDir, normal, incomingDir);

        accumulateOutput(output, contribution, pixelIdx);
    }

    gBuffer.setPayload(
        pixelIdx,
        t,
        f3(normal)
    );

    bool currDelta = shadeContext.materials[materialID].isSpecular;
    // NEE
    if (!currDelta) {
        float4 lightNormal;
        float4 emission;
        float4 shadingPosToLightNormalized;
        float t_max;
        float pdf;
        uint32_t primID; // 0xFFFFFFFF for env, otherwise the triangle primID
        float2 barycentrics;

        bool sampledEnv = shadeContext.lightSampler.sample_ReSTIR_rc_data(
            rand(&localState), rand4(&localState), 
            shadingPos, 
            shadeContext.vertices, 
            emission,
            shadingPosToLightNormalized, 
            lightNormal, 
            t_max, 
            pdf,
            primID,
            barycentrics
        );


        float4 shadingPosToLightLocal;
        toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

        bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
        bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
        if (surfaceBackface || lightBackface) {
            shadowRays.setIgnore(pixelIdx);
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

            float4 f_val;
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

            float4 contribution;
            float4 unweightedContribution;
            float misWeight;
            float denom;

            if (sampledEnv) {
                float cosSurface = dot(normal, shadingPosToLightNormalized);
                denom = pdf + bsdfPDF;
                unweightedContribution = f_val * emission * cosSurface;
                contribution = unweightedContribution / pdf; 
                misWeight = balanceHeuristicTwoStrategy(
                    pdf,
                    bsdfPDF
                );
            } else {
                float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                float cosSurface = dot(normal, shadingPosToLightNormalized);

                denom = bsdfPDF + (t_max * t_max) * pdf / cosLight;

                unweightedContribution =  // throughput = 1.0
                    f_val * emission * cosSurface; // NEE contribution

                contribution = unweightedContribution * (cosLight / (pdf * t_max * t_max));

                misWeight = balanceHeuristicTwoStrategy(
                    denom - bsdfPDF, 
                    bsdfPDF // alt strat
                );
            }

            writeReservoir.initialize(
                pixelIdx,
                1.0f / denom,
                f3(unweightedContribution),
                seed,
                1, // M
                2, // rc index
                2, // length
                1, // technique: NEE
                1.0f,// jacobian product
                sampledEnv ? pdf : denom - bsdfPDF, // solid angle nee pdf
                primID,
                f3(), // rcvertexwi is nonsensical here
                barycentrics,
                emission
            );

            writeReservoir.setWeightSum(pixelIdx, 1.0f / denom);
            writeReservoir.setFootprint(pixelIdx, camera.getInitialRayFootprint());

            shadowRays.setShadowRay(
                pixelIdx, 
                shadingPos + shadingPosToLightNormalized * RAY_EPSILON, 
                shadingPosToLightNormalized, 
                t_max * (1.0f - EPSILON3), 
                pixelIdx
            );
        }
    } else // current is specular, skip NEE
    {   
        shadowRays.setIgnore(pixelIdx);
        // no need to zero out reservoir, since ignored shadow ray maps to a miss, which leads to the weight being killed
    }

    // Sample next Direction
    {
        float4 outgoing;
        float4 f_val;
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
            predicate[pixelIdx] = 0; // mark as dead
            save_rng(pixelIdx, &localState, rngStates);
            return;
        }
        
        float4 throughput = f_val * fabsf(outgoing.z) / pdf;
        toWorld(outgoing, normal, outgoing);

        // write to next rayQueue

        writeQueue.setAll(
            pixelIdx,            // idx
            shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON,       // inOrigin
            outgoing,       // inDir
            currDelta,      // inPrevDelta
            false,
            throughput,     // inThroughput
            pixelIdx,       // inPixelIdx
            1,      // inDepth
            pdf,             // total accumulated pdf so far
            pdf             // inLastPDF
        );
    }
    
    predicate[pixelIdx] = 1;
    save_rng(pixelIdx, &localState, rngStates);
}

// Consider arena allocation
__host__ void allocateBuffers(
    ReSTIRRayQueue& q1, 
    ReSTIRRayQueue& q2, 
    HitBuffer& hb, 
    ReSTIRShadowQueue& sq, 
    uint32_t*& pr,
    uint32_t*& si,
    float4*& out,
    uint32_t*& skeys,
    uint32_t*& svals,
    uint32_t*& skeys1,
    uint32_t*& svals1,
    Reservoir& r1,
    Reservoir& r2,
    CandidateReservoirs& cr1,
    GBuffer& gb,
    int maxRays
)
{
    size_t f4Size    = maxRays * sizeof(float4);
    size_t floatSize = maxRays * sizeof(float);
    size_t halfSize  = maxRays * sizeof(half);
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

    cudaMalloc(&temp_ptr, uIntSize);
    sq.pixelIdx = (uint32_t*)temp_ptr;

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

    // Reservoir 1
    cudaMalloc(&temp_ptr, floatSize);
    r1.W = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    r1.pathFlags = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    r1.shiftState = (uint4*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    r1.rcGeometry = (uint4*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r1.modifiedMachedJacobianProduct = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r1.footprint = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r1.weightSum = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    r1.suffixThroughput = (uint32_t*)temp_ptr;

    // Reservoir 2
    cudaMalloc(&temp_ptr, floatSize);
    r2.W = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    r2.pathFlags = (uint32_t*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    r2.shiftState = (uint4*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    r2.rcGeometry = (uint4*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r2.modifiedMachedJacobianProduct = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r2.footprint = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    r2.weightSum = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    r2.suffixThroughput = (uint32_t*)temp_ptr;

    // Candidate Reservoirs 1
    cudaMalloc(&temp_ptr, floatSize);
    cr1.W = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uInt4Size);
    cr1.payload = (uint4*)temp_ptr;

    cudaMalloc(&temp_ptr, floatSize);
    gb.depth = (float*)temp_ptr;

    cudaMalloc(&temp_ptr, uIntSize);
    gb.normal = (uint32_t*)temp_ptr;
}

void freeBuffers(
    ReSTIRRayQueue& q1, 
    ReSTIRRayQueue& q2, 
    HitBuffer& hb, 
    ReSTIRShadowQueue& sq, 
    uint32_t* pr,
    uint32_t* si,
    float4* out,
    uint32_t* skeys,
    uint32_t* svals,
    uint32_t* skeys1,
    uint32_t* svals1,
    Reservoir& r1,
    Reservoir& r2,
    CandidateReservoirs& cr1,
    GBuffer& gb
) {
    cudaFree(q1.origin_plus_dir);
    cudaFree(q1.payload);

    cudaFree(q2.origin_plus_dir);
    cudaFree(q2.payload);

    cudaFree(hb.data);

    cudaFree(sq.origin_plus_dist);
    cudaFree(sq.direction);
    cudaFree(sq.pixelIdx);

    cudaFree(pr);
    cudaFree(si);

    cudaFree(skeys);
    cudaFree(svals);
    cudaFree(skeys1);
    cudaFree(svals1);

    cudaFree(out);

    cudaFree(r1.W);
    cudaFree(r1.pathFlags);
    cudaFree(r1.shiftState);
    cudaFree(r1.rcGeometry);
    cudaFree(r1.modifiedMachedJacobianProduct);
    cudaFree(r1.footprint);
    cudaFree(r1.weightSum);
    cudaFree(r1.suffixThroughput);

    cudaFree(r2.W);
    cudaFree(r2.pathFlags);
    cudaFree(r2.shiftState);
    cudaFree(r2.rcGeometry);
    cudaFree(r2.modifiedMachedJacobianProduct);
    cudaFree(r2.footprint);
    cudaFree(r2.weightSum);
    cudaFree(r2.suffixThroughput);

    cudaFree(cr1.W);
    cudaFree(cr1.payload);

    cudaFree(gb.depth);
    cudaFree(gb.normal);
}

__global__ void closestHit(
    const BVHContext bvhContext,
    const ReSTIRRayQueue rayQueue,
    HitBuffer hitBuffer,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= activeRays) return;

    float4 barycentric;
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
    const ReSTIRRayQueue readQueue,
    const HitBuffer hits,
    ReSTIRRayQueue writeQueue,
    ReSTIRShadowQueue shadowRays,

    Reservoir writeReservoir,
    
    uint32_t* predicate,
    int activeRays,

    int frameNum,
    int maxDepth,

    uint32_t* shadowRayIndex,
    
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

    float4 prevPos;
    float4 incomingDir;
    bool prevDelta;
    bool foundXk;
    float4 throughput;
    int pixelIdx;
    int depth;
    uint32_t totalPDF;
    float lastPDF;

    readQueue.getAll(
        readIndex,
        prevPos,
        incomingDir,
        prevDelta,
        foundXk,
        throughput,
        pixelIdx,
        depth,
        totalPDF,
        lastPDF
    );

    int triID;
    int materialID;
    float2 uv;
    bool backface;
    float4 normal;
    float4 shadingPos;

    RNGState localState = load_rng(pixelIdx, frameNum, depth, rngStates);

    float t;
    float u, v;
    // handle current interactiion
    {
        hits.getAllInfo(readIndex, t, u, v, triID);

        if (t >= 1e30f) // indicates a miss
        {
            predicate[idx] = 0;

            float4 contribution = shadeContext.lightSampler.envMap.sampleDir(incomingDir) * throughput;

            float misWeight = (prevDelta || depth == 0) ? 1.0f : balanceHeuristicTwoStrategy(
                lastPDF, // primary strategy
                shadeContext.lightSampler.evaluateEnvPdf(incomingDir) // alternate strategy
            );

            // Normal Reservoir update, use actual rc vertex
            if (foundXk) {

            } 
            // Use backup k=d rc vertex. This does NOT mean that Xk has been found
            else {
                
            }

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

        float4 apos = __ldg(&shadeContext.vertices->positions[tri.aInd]);
        float4 bpos = __ldg(&shadeContext.vertices->positions[tri.bInd]);
        float4 cpos = __ldg(&shadeContext.vertices->positions[tri.cInd]);

        shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

        float4 a_n = __ldg(&shadeContext.vertices->normals[tri.naInd]);
        float4 b_n = __ldg(&shadeContext.vertices->normals[tri.nbInd]);
        float4 c_n = __ldg(&shadeContext.vertices->normals[tri.ncInd]);
        
        normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
        backface = dot(normal, incomingDir) > 0.0f;
        normal = backface ? -normal : normal;

        float4 contribution = backface ? f4(0.0f) : tri.emission * throughput;

        toLocal(incomingDir, normal, incomingDir);

        float misWeight = (prevDelta || depth == 0) ? 1.0f : balanceHeuristicTwoStrategy(
            lastPDF, // primary strategy
            (t * t * shadeContext.lightSampler.evaluateMeshPdf(tri) / (fabsf(incomingDir.z))) // alternate strategy
        );

        accumulateOutput(output, contribution * misWeight, pixelIdx);
    }

    bool currDelta = shadeContext.materials[materialID].isSpecular;
    // NEE
    if (!currDelta) {
        float4 lightNormal;
        float4 emission;
        float4 shadingPosToLightNormalized;
        float t_max;
        float pdf;

        bool sampledEnv = shadeContext.lightSampler.sample(
            rand(&localState), rand4(&localState), 
            shadingPos, 
            shadeContext.vertices, 
            emission,
            shadingPosToLightNormalized, 
            lightNormal, 
            t_max, 
            pdf
        );


        float4 shadingPosToLightLocal;
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

            float4 f_val;
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

            float4 contribution;
            float misWeight;

            if (sampledEnv) {
                float cosSurface = dot(normal, shadingPosToLightNormalized);
                contribution = throughput * f_val * emission * cosSurface / pdf; 
                misWeight = balanceHeuristicTwoStrategy(
                    pdf,
                    bsdfPDF
                );
            } else {
                float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                float cosSurface = dot(normal, shadingPosToLightNormalized);

                contribution = throughput * // throughput
                    f_val * emission * cosLight * // NEE contribution
                    cosSurface / (pdf * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                misWeight = balanceHeuristicTwoStrategy(
                    (t_max * t_max) * pdf / cosLight, // convert area pdf to SA
                    bsdfPDF // alt strat
                );
            }

            shadowRays.setShadowRay(
                idx, 
                shadingPos + shadingPosToLightNormalized * RAY_EPSILON, 
                shadingPosToLightNormalized, 
                t_max * (1.0f - EPSILON3), 
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

    if (depth == maxDepth) {
        predicate[idx] = 0;
        save_rng(pixelIdx, &localState, rngStates);
        return;
    }

    // Sample next Direction
    {
        float4 outgoing;
        float4 f_val;
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

        if (!foundXk) {
            float primaryFootprint = writeReservoir.getFootprint(pixelIdx);
            float fwdFootprint = t * t / (lastPDF * abs(incomingDir.z));

            // Storing the previous cosine would be too expensive, so we apply a conservating estimate of the cosine
            float conservativeInvFootprint = t * t / (pdf);

            if (min(fwdFootprint, conservativeInvFootprint) > primaryFootprint * 0.0002) {
                foundXk = true;
            }

            // P(k-1->k) * G(k-1->k)
            float partialProduct = (lastPDF * abs(incomingDir.z)) / (t * t);

            //writeReservoir.setSpecialCachedJacobianProduct(
            //    pixelIdx,
            //    partialProduct * 1.0f // Special case for k=d
            //);

            writeReservoir.setRCGeometry(
                pixelIdx,
                triID,
                f3(toWorld(incomingDir, normal)),
                make_float2(u, v),
                partialProduct * pdf // the bsdf version, used for most cases where k!=d
            );

            // write the incoming throughput at the reconnection vertex
            writeReservoir.setSuffixThroughput(pixelIdx, throughput);
        }

        throughput *= f_val * fabsf(outgoing.z) / pdf;
        toWorld(outgoing, normal, outgoing);

        

        // write to next rayQueue

        writeQueue.setAll(
            idx,            // idx
            shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON,       // inOrigin
            outgoing,       // inDir
            currDelta,      // inPrevDelta
            foundXk,
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
    ReSTIRShadowQueue shadowRays,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    Ray r;
    float maxT;

    // how much the throughput is scaled by going through the shadow ray.
    // 0 if occluded, 1 if not occluded
    float4 throughputScale;

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

__global__ void processShadowRay(
    ReSTIRShadowQueue shadowRays,
    Reservoir writeReservoir,
    int activeRays,
    float4* output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    int pixelIdx;
    if (!shadowRays.getAccumulateData(idx, pixelIdx)) {
        writeReservoir.killWeight(idx);
    }
}
__global__ void compactRayQueue_NOSORT(
    const ReSTIRRayQueue sparseQueue,
    ReSTIRRayQueue denseQueue,
    const uint32_t* __restrict__ predicates,
    const uint32_t* __restrict__ scanIndices,
    int activeRays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    if (predicates[idx] == 1) 
    {
        int denseIdx = scanIndices[idx]; 

        float4 rayData = __ldcs(&sparseQueue.origin_plus_dir[idx]);

        __stcs(&denseQueue.origin_plus_dir[denseIdx], __ldcs(&sparseQueue.origin_plus_dir[idx]));
        __stcs(&denseQueue.payload[denseIdx], __ldcs(&sparseQueue.payload[idx])); 
        __stcs(&denseQueue.rayFootprint[denseIdx], __ldcs(&sparseQueue.rayFootprint[idx]));
    }
}

__global__ void compactRayQueue_SORT(
    const ReSTIRRayQueue sparseQueue,
    ReSTIRRayQueue denseQueue,
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

        float4 rayData = __ldcs(&sparseQueue.origin_plus_dir[idx]);

        __stcs(&denseQueue.origin_plus_dir[denseIdx], __ldcs(&sparseQueue.origin_plus_dir[idx]));
        __stcs(&denseQueue.payload[denseIdx], __ldcs(&sparseQueue.payload[idx])); 
        __stcs(&denseQueue.rayFootprint[denseIdx], __ldcs(&sparseQueue.rayFootprint[idx]));

        float4 direction = getDirectionFromPacked(rayData);
        float invDiameter = 1.0f / (sceneRadius * 2.0f);
        __stcs(&sortKeysIn[denseIdx], generateMortonSortKey(rayData, direction, sceneMin, invDiameter));
        __stcs(&sortValuesIn[denseIdx], denseIdx);
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
    const ReSTIRRayQueue unsortedQueue,
    ReSTIRRayQueue sortedQueue,
    const uint32_t* __restrict__ sortedIndices,
    int activeRays
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= activeRays) return;

    uint32_t sortedIdx = sortedIndices[idx];

    __stcs(&sortedQueue.origin_plus_dir[idx], __ldcs(&unsortedQueue.origin_plus_dir[sortedIdx]));
    __stcs(&sortedQueue.payload[idx]        , __ldcs(&unsortedQueue.payload[sortedIdx]));
}


__host__ void launch_ReSTIR_PT(
    Camera camera, 
    const SceneContext sceneContext,
    int numSample, int maxDepth,
    int h_w, int h_h, 
    float4 h_sceneCenter, float h_sceneRadius, float4 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess,
    float exposure
)
{
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float4));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float4));
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

    ReSTIRRayQueue readQueue;
    ReSTIRRayQueue writeQueue;
    HitBuffer hitBuffer;
    ReSTIRShadowQueue shadowQueue;
    uint32_t* d_predicate;
    uint32_t* d_scanIndices;
    float4* d_finalOutput;

    Reservoir readReservoir;
    Reservoir writeReservoir;

    CandidateReservoirs temp_candidates;
    GBuffer gBuffer;

    uint32_t* d_sortKeysIn;
    uint32_t* d_sortValuesOut;
    uint32_t* d_sortKeysOut;
    uint32_t* d_sortValuesIn;

    uint32_t* d_shadowRayIndex;
    cudaMalloc(&d_shadowRayIndex, sizeof(uint32_t));

    allocateBuffers(
        readQueue, 
        writeQueue, 
        hitBuffer,
        shadowQueue, 
        d_predicate,
        d_scanIndices,
        d_finalOutput,
        d_sortKeysIn,
        d_sortValuesOut,
        d_sortKeysOut,
        d_sortValuesIn,
        readReservoir,
        writeReservoir,
        temp_candidates,
        gBuffer,
        h_h * h_w
    );

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
    int saveIntervalSamples = 100;
    Image image = Image(h_w, h_h);
    // Post-processing (exposure/tonemap/gamma) now happens on the GPU in
    // cleanFormatAndPostProcessImage, so saveImageBMP() must not re-apply it.
    image.postProcess = false;
    std::vector<float4> h_finalOutput(h_w * h_h);

    std::cout << "Begin Render" << std::endl;

    // for initial camera raygen, use 8x4 warps to maximize spatial coherence
    dim3 blockSize(8, 32);  
    dim3 gridSize((h_w+15)/8, (h_h+15)/32);

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();
    for (int currSample = 0; currSample < numSample; currSample++)
    {   
        int activeRays = h_h * h_w;

        // Performs everything for the first bounce
        processPrimary<<<gridSize, blockSize>>> (
            d_rngStates, bvhContext, shadeContext, hitBuffer,
            writeQueue, shadowQueue, writeReservoir, d_predicate,
            activeRays, gBuffer, camera, currSample,
            maxDepth, colors
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

        activeRays = lastScanIndex + lastPredicate;
        
        if (activeRays > 0) {
            int blocks = (h_h * h_w + 255) / 256; 
            compactRayQueue_NOSORT<<<blocks, 256>>> (
                writeQueue,     // Read from primary output
                readQueue,  // Write to depth loop input
                d_predicate, d_scanIndices, h_h * h_w
            );
        }
        
        
        for (int depth = 1; depth <= maxDepth; depth++)
        {
            if (activeRays == 0) break;

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
                    *readQueue, *writeQueue, d_sortValuesOut, activeRays
                );

                std::swap(readQueue, writeQueue);
            }
#endif
            closestHit<<<blocks, 256>>> (
                bvhContext,
                readQueue,
                hitBuffer,
                activeRays
            );

#if USE_MATERIAL_SORT == 1

            calculateMaterialSortKeys<<<blocks, 256>>> (
                hitBuffer,
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
                readQueue,
                hitBuffer,
                writeQueue,
                shadowQueue,
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

            processShadowRay<<<blocks, 256>>> (
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
        d_finalOutput,
        d_sortKeysIn,
        d_sortValuesOut,
        d_sortKeysOut,
        d_sortValuesIn,
        temp_reservoir1,
        temp_reservoir2,
        temp_candidates
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