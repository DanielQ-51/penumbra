#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "settings.cuh"

/*
enum TechniqueType{
    TYPE_INTERNAL_RC = 0u,
    TYPE_BSDF = 1u
};*/

using TechniqueType = uint32_t;

#ifndef RESERVOIR_SIZE
#define RESERVOIR_SIZE 44
#endif

#ifndef GBUFFER_SIZE
#define GBUFFER_SIZE 20
#endif

// Per-pixel bytes for the OptiX denoiser guide arena: albedo (float4) + normal
// (float4) + flow (float2, temporal motion vector).
#ifndef DENOISER_GUIDES_SIZE
#define DENOISER_GUIDES_SIZE 40
#endif

// Path Types

#define SHIFT_IS_NEE           (1 << 0) // 0 = BSDF, 1 = NEE
#define SHIFT_IS_ENV           (1 << 1) // 0 = Area/Mesh, 1 = Environment
#define SHIFT_K_IS_D           (1 << 2) // 1 if k == d
#define SHIFT_K_IS_D_MINUS_1   (1 << 3) // 1 if k == d - 1
#define SHIFT_K_LESS_D_MINUS_1 (1 << 4) // 1 if k < d - 1

// --- K = D (Reconnecting directly to the light) ---
#define PATH_TYPE_NEE_AREA_K_EQ_D  (SHIFT_IS_NEE | SHIFT_K_IS_D)
#define PATH_TYPE_NEE_ENV_K_EQ_D   (SHIFT_IS_NEE | SHIFT_IS_ENV | SHIFT_K_IS_D)
#define PATH_TYPE_BSDF_AREA_K_EQ_D (SHIFT_K_IS_D) // Implicitly 0 for NEE and ENV
#define PATH_TYPE_BSDF_ENV_K_EQ_D  (SHIFT_IS_ENV | SHIFT_K_IS_D)

// --- K = D - 1 (Reconnecting one bounce before the light) ---
#define PATH_TYPE_NEE_AREA_K_EQ_D_MINUS_1  (SHIFT_IS_NEE | SHIFT_K_IS_D_MINUS_1)
#define PATH_TYPE_NEE_ENV_K_EQ_D_MINUS_1   (SHIFT_IS_NEE | SHIFT_IS_ENV | SHIFT_K_IS_D_MINUS_1)
#define PATH_TYPE_BSDF_AREA_K_EQ_D_MINUS_1 (SHIFT_K_IS_D_MINUS_1)
#define PATH_TYPE_BSDF_ENV_K_EQ_D_MINUS_1  (SHIFT_IS_ENV | SHIFT_K_IS_D_MINUS_1)

// --- K < D - 1 (Reconnecting deep inside the path) ---
#define PATH_TYPE_NEE_AREA_K_LESS_D_MINUS_1  (SHIFT_IS_NEE | SHIFT_K_LESS_D_MINUS_1)
#define PATH_TYPE_NEE_ENV_K_LESS_D_MINUS_1   (SHIFT_IS_NEE | SHIFT_IS_ENV | SHIFT_K_LESS_D_MINUS_1)
#define PATH_TYPE_BSDF_AREA_K_LESS_D_MINUS_1 (SHIFT_K_LESS_D_MINUS_1)
#define PATH_TYPE_BSDF_ENV_K_LESS_D_MINUS_1  (SHIFT_IS_ENV | SHIFT_K_LESS_D_MINUS_1)

#define FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND 0xFFFFFFFF
#define FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY 0xFF
#define FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_DIRECTION_COPY 0xFE
__device__ __forceinline__ uint32_t packPathFlags(
    uint32_t M, uint32_t pathLength, uint32_t rcVertexInd, TechniqueType type
) {
    return (M & 0xFF) |
           ((pathLength  & 0xFF) << 8) |
           ((rcVertexInd & 0xFF) << 16) |
           ((static_cast<uint32_t>(type) & 0xFF) << 24);
}

__device__ __forceinline__ uint32_t extractM(
    uint32_t packed
) {
    return (packed & 0xFF);
}

__device__ __forceinline__ uint32_t extractType(
    uint32_t packed
) {
    return (packed >> 24);
}

__device__ __forceinline__ uint32_t extractRcInd(
    uint32_t packed
) {
    return ((packed >> 16) & 0xFF);
}

__device__ __forceinline__ uint32_t extractPathLength(
    uint32_t packed
) {
    return ((packed >> 8) & 0xFF);
}

__device__ __forceinline__ uint32_t updateM(
    uint32_t packed, uint32_t M
) {
    return (M & 0xFF) | (packed & 0xFFFFFF00);
}

__device__ __forceinline__ uint4 packRcGeometry(
    uint32_t primID,
    float2 barycentrics,
    float3 rcVertexWi,
    uint32_t rcInstanceID // Changed from radiance to instanceID
) {
    uint4 data;
    data.x = primID;
    data.y = packFloat2ToUnorm16(barycentrics);
    data.z = packOct(rcVertexWi);
    data.w = rcInstanceID; // Directly storing the instance ID
    return data;
}

// Takes in a packed uint4, and just replaces the 3rd channel with what the wi should be.
__device__ __forceinline__ uint4 updateRcVertexWi(const uint4& in, float3 wi) {
    return make_uint4(in.x, in.y, packOct(wi), in.w);
}

#if RESERVOIR_LAYOUT_AOS

// 32B per-pixel shift descriptor: exactly what evaluateHybridShift consumes, so a
// shift loads it as one unit (two adjacent 16B loads landing in a single 32B-aligned
// region). See RESERVOIR_LAYOUT_AOS in settings.cuh.
struct __align__(16) ReservoirDesc {
    /**
     *  x: rcVertexPrimID
     *  y: rcVertexBarycentrics (2 x 16 bit unorm)
     *  z: rcVertexWi (octahedral 2 x 16 bit snorm)
     *  w: rcVertexInstanceID
     */
    uint4 geo;
    /**
     *  x: rcVertexRadiance (RGB9E5)
     *  y: pathFlags  (M | pathLength<<8 | rcVertexInd<<16 | technique<<24)
     *  z: cachedJacobian (float bit pattern)
     *  w: cachedNeePdf   (float bit pattern)
     */
    uint4 misc;
};

// 8B per-pixel weights, always read as a pair (F*W display, MIS math).
struct __align__(8) ReservoirWeights {
    float    W;
    uint32_t F; // RGB9E5 encoded
};

#endif // RESERVOIR_LAYOUT_AOS

struct Reservoir {
#if RESERVOIR_LAYOUT_AOS
    ReservoirDesc*    __restrict__ desc;    // 32B shift descriptor
    ReservoirWeights* __restrict__ weights; // 8B {W, F}
    uint32_t*         __restrict__ initRandomSeed; 
#else
    float* __restrict__ W;

    // RGB9E5 encoded
    uint32_t* __restrict__ F;

    uint32_t* __restrict__ initRandomSeed;

    /**
     * Bit  0 -  7: M
     * Bit  8 - 15: Path length
     * Bit 16 - 23: rc vertex index
     * Bit 24 - 31: Technique
     */
    uint32_t* __restrict__ pathFlags;

    /**
     *  x: rcVertexPrimID
     *  y: rcVertexBarycentrics (2 x 16 bit unorm)
     *  z: rcVertexWi (octahedral 2 x 16 bit snorm)
     *  w: rcVertexInstanceID
     */
    uint4* __restrict__ rcVertexGeometry;

    uint32_t* __restrict__ rcVertexRadiance;

    float* __restrict__ cachedJacobian;
    float* __restrict__ cachedNeePdf;
#endif

    __device__ __forceinline__ float3 getF_globalLoad(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return fromRGB9E5(__ldg(&weights[idx].F));
#else
        return fromRGB9E5(__ldg(&(F[idx])));
#endif
    }

    __device__ __forceinline__ float getW_globalLoad(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return __ldg(&weights[idx].W);
#else
        return __ldg(&(W[idx]));
#endif
    }

    // W and F are always read together (MIS math, shading). One 8B load in AOS;
    // the same two loads the SOA path always did otherwise.
    __device__ __forceinline__ void getWF(uint32_t idx, float& outW, float3& outF) const {
#if RESERVOIR_LAYOUT_AOS
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(&weights[idx]));
        outW = __uint_as_float(raw.x);
        outF = fromRGB9E5(raw.y);
#else
        outW = __ldg(&W[idx]);
        outF = fromRGB9E5(__ldg(&F[idx]));
#endif
    }

    // Streaming (evict-first) {W,F} read used by the standalone display kernel.
    __device__ __forceinline__ float3 getShadingColor_streaming(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return fromRGB9E5(__ldcs(&weights[idx].F)) * __ldcs(&weights[idx].W);
#else
        return fromRGB9E5(__ldcs(&F[idx])) * __ldcs(&W[idx]);
#endif
    }

    __device__ __forceinline__ float getCachedJacobian_globalLoad(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return __uint_as_float(__ldg(&desc[idx].misc.z));
#else
        return __ldg(&(cachedJacobian[idx]));
#endif
    }

    __device__ __forceinline__ float setCachedJacobian(uint32_t idx, float val) const {
#if RESERVOIR_LAYOUT_AOS
        desc[idx].misc.z = __float_as_uint(val);
        return val;
#else
        return cachedJacobian[idx] = val;
#endif
    }

    __device__ __forceinline__ float getCachedNEE_globalLoad(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return __uint_as_float(__ldg(&desc[idx].misc.w));
#else
        return __ldg(&(cachedNeePdf[idx]));
#endif
    }

    __device__ __forceinline__ void getCachedValues_globalLoad(uint32_t idx, float& jacobian, float& neePDF) const {
#if RESERVOIR_LAYOUT_AOS
        uint4 m = __ldg(&desc[idx].misc);
        jacobian = __uint_as_float(m.z);
        neePDF   = __uint_as_float(m.w);
#else
        jacobian = __ldg(&(cachedJacobian[idx]));
        neePDF = __ldg(&(cachedNeePdf[idx]));
#endif
    }

    // Updated to output instanceID and fetch radiance from the new buffer
    __device__ __forceinline__ void getRcVertexGeometry_globalLoad(
        uint32_t idx,
        uint32_t& primID,
        float2& bary,
        float3& wi,
        float3& radiance,
        uint32_t& instanceID
    ) const {
#if RESERVOIR_LAYOUT_AOS
        uint4 g = __ldg(&desc[idx].geo);
        primID = g.x;
        bary = unpackUnorm16ToFloat2(g.y);
        wi = unpackOct(g.z);
        instanceID = g.w;
        radiance = fromRGB9E5(__ldg(&desc[idx].misc.x));
#else
        uint4 data = __ldg(&rcVertexGeometry[idx]);
        primID = data.x;
        bary = unpackUnorm16ToFloat2(data.y);
        wi = unpackOct(data.z);
        instanceID = data.w;

        radiance = fromRGB9E5(__ldg(&rcVertexRadiance[idx]));
#endif
    }

    __device__ __forceinline__ void getPathFlags(
        uint32_t idx, uint32_t& M, uint32_t& pathLength, uint32_t& rcVertexInd, TechniqueType& technique) const {
#if RESERVOIR_LAYOUT_AOS
        uint32_t flags = __ldg(&desc[idx].misc.y);
#else
        uint32_t flags = __ldg(&pathFlags[idx]);
#endif
        M = flags & 0x000000FF;
        pathLength = (flags >> 8) & 0x000000FF;
        rcVertexInd = (flags >> 16) & 0x000000FF;
        technique = static_cast<TechniqueType>((flags >> 24) & 0x000000FF);
    }

    __device__ __forceinline__ uint32_t getPathFlagsRaw(uint32_t idx) const {
#if RESERVOIR_LAYOUT_AOS
        return __ldg(&desc[idx].misc.y);
#else
        return __ldg(&pathFlags[idx]);
#endif
    }

    // Grouped load of the full 32B shift descriptor. In AOS this is two adjacent 16B
    // loads; in SOA it degenerates to the same per-field __ldg calls the shift did
    // before, so the two layouts stay behaviour-identical. cachedNeePdf follows the
    // old semantics: only meaningful for needNeePDF() paths, else forced to -1.
    __device__ __forceinline__ void getShiftDescriptor(
        uint32_t idx,
        uint32_t& M, uint32_t& pathLength, uint32_t& rcVertexInd, TechniqueType& technique,
        uint32_t& rcPrimID, float2& bary, float3& wi, float3& radiance, uint32_t& instanceID,
        float& outJacobian, float& outNeePdf) const {
#if RESERVOIR_LAYOUT_AOS
        uint4 g = __ldg(&desc[idx].geo);
        uint4 m = __ldg(&desc[idx].misc);
        rcPrimID   = g.x;
        bary       = unpackUnorm16ToFloat2(g.y);
        wi         = unpackOct(g.z);
        instanceID = g.w;
        radiance   = fromRGB9E5(m.x);
        uint32_t flags = m.y;
        M           = flags & 0x000000FF;
        pathLength  = (flags >> 8) & 0x000000FF;
        rcVertexInd = (flags >> 16) & 0x000000FF;
        technique   = static_cast<TechniqueType>((flags >> 24) & 0x000000FF);
        outJacobian = __uint_as_float(m.z);
        bool nee    = (technique & (SHIFT_K_IS_D | SHIFT_K_IS_D_MINUS_1)) != 0;
        outNeePdf   = nee ? __uint_as_float(m.w) : -1.0f;
#else
        uint4 data = __ldg(&rcVertexGeometry[idx]);
        rcPrimID   = data.x;
        bary       = unpackUnorm16ToFloat2(data.y);
        wi         = unpackOct(data.z);
        instanceID = data.w;
        radiance   = fromRGB9E5(__ldg(&rcVertexRadiance[idx]));
        uint32_t flags = __ldg(&pathFlags[idx]);
        M           = flags & 0x000000FF;
        pathLength  = (flags >> 8) & 0x000000FF;
        rcVertexInd = (flags >> 16) & 0x000000FF;
        technique   = static_cast<TechniqueType>((flags >> 24) & 0x000000FF);
        outJacobian = __ldg(&cachedJacobian[idx]);
        bool nee    = (technique & (SHIFT_K_IS_D | SHIFT_K_IS_D_MINUS_1)) != 0;
        outNeePdf   = nee ? __ldg(&cachedNeePdf[idx]) : -1.0f;
#endif
    }

    __device__ __forceinline__ void setInitRandomSeed(uint32_t idx, uint32_t seed) const {
        __stcs(&initRandomSeed[idx], seed);
    }

    __device__ __forceinline__ uint32_t getInitRandomSeed(uint32_t idx) const {
        return __ldg(&initRandomSeed[idx]);
    }

    __device__ __forceinline__ uint32_t getSeed(uint32_t idx) const {
        return __ldcs(&initRandomSeed[idx]);
    }

    // prefer to keep values in cache for the dupe map generation stage
    __device__ __forceinline__ uint32_t getSeed_notstreaming(uint32_t idx) const {
        return __ldg(&initRandomSeed[idx]);
    }

    __device__ __forceinline__ void setW(uint32_t idx, float w) const {
#if RESERVOIR_LAYOUT_AOS
        __stcs(&weights[idx].W, w);
#else
        __stcs(&W[idx], w);
#endif
    }

    __device__ __forceinline__ void setW_noCS(uint32_t idx, float w) const {
#if RESERVOIR_LAYOUT_AOS
        weights[idx].W = w;
#else
        W[idx] = w;
#endif
    }

    __device__ __forceinline__ void setF(uint32_t idx, uint32_t f) const {
#if RESERVOIR_LAYOUT_AOS
        weights[idx].F = f;
#else
        F[idx] = f;
#endif
    }

    __device__ __forceinline__ void setPathFlags(uint32_t idx, uint32_t currPathFlags) const {
#if RESERVOIR_LAYOUT_AOS
        desc[idx].misc.y = currPathFlags;
#else
        pathFlags[idx] = currPathFlags;
#endif
    }

    __device__ __forceinline__ void saveReservoirFinal(
        uint32_t idx,
        float inW,
        uint32_t inF,
        uint32_t inPathFlags,
        uint4 inRcVertexGeometry,
        uint32_t inRcVertexRadiance,
        float inRcVertexJacobian,
        float inNeePDF
    ) const {
#if RESERVOIR_LAYOUT_AOS
        desc[idx].geo  = inRcVertexGeometry;
        desc[idx].misc = make_uint4(inRcVertexRadiance, inPathFlags,
                                    __float_as_uint(inRcVertexJacobian),
                                    __float_as_uint(inNeePDF));
        weights[idx].W = inW;
        weights[idx].F = inF;
#else
        W[idx] = inW;
        F[idx] = inF;
        pathFlags[idx] = inPathFlags;
        rcVertexGeometry[idx] = inRcVertexGeometry;
        rcVertexRadiance[idx] = inRcVertexRadiance;
        cachedJacobian[idx] = inRcVertexJacobian;
        cachedNeePdf[idx] = inNeePDF;
#endif
    }

    __device__ __forceinline__ void saveReservoirAll(
        uint32_t idx,
        float inW,
        float3 inF,
        uint32_t seed,
        uint32_t M,
        uint32_t pathLength,
        uint32_t rcVertexIndex,
        TechniqueType type,
        uint32_t rcInstanceID,
        uint32_t rcPrimID,
        float2 rcBarycentrics,
        float3 rcWi,
        float3 rcRadiance,
        float inRcVertexJacobian,
        float inNeePDF
    ) const {
#if RESERVOIR_LAYOUT_AOS
        desc[idx].geo  = packRcGeometry(rcPrimID, rcBarycentrics, rcWi, rcInstanceID);
        desc[idx].misc = make_uint4(toRGB9E5(rcRadiance),
                                    packPathFlags(M, pathLength, rcVertexIndex, type),
                                    __float_as_uint(inRcVertexJacobian),
                                    __float_as_uint(inNeePDF));
        weights[idx].W = inW;
        weights[idx].F = toRGB9E5(inF);
        initRandomSeed[idx] = seed;
#else
        W[idx] = inW;
        F[idx] = toRGB9E5(inF);
        initRandomSeed[idx] = seed;
        pathFlags[idx] = packPathFlags(M, pathLength, rcVertexIndex, type);
        rcVertexGeometry[idx] = packRcGeometry(rcPrimID, rcBarycentrics, rcWi, rcInstanceID); // Updated geometry pack
        rcVertexRadiance[idx] = toRGB9E5(rcRadiance); // Pack directly to the new buffer
        cachedJacobian[idx] = inRcVertexJacobian;
        cachedNeePdf[idx] = inNeePDF;
#endif
    }
};

__host__ inline void* allocateReservoir(Reservoir& r, uint32_t numPixel) {
    numPixel = (numPixel + 31) & ~31;

    void* raw;
    cudaMalloc(&raw, numPixel * RESERVOIR_SIZE);

    char* ptr = static_cast<char*>(raw);
#if RESERVOIR_LAYOUT_AOS
    r.desc = reinterpret_cast<ReservoirDesc*>(ptr); ptr += numPixel * sizeof(ReservoirDesc);          // 32B
    r.weights = reinterpret_cast<ReservoirWeights*>(ptr); ptr += numPixel * sizeof(ReservoirWeights); // 8B
    r.initRandomSeed = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);          // 4B
#else
    r.W = reinterpret_cast<float*>(ptr); ptr += numPixel * sizeof(float);
    r.F = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);

    r.initRandomSeed = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);
    r.pathFlags = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);
    r.rcVertexGeometry = reinterpret_cast<uint4*>(ptr); ptr += numPixel * sizeof(uint4);

    r.rcVertexRadiance = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);

    r.cachedJacobian = reinterpret_cast<float*>(ptr); ptr += numPixel * sizeof(float);
    r.cachedNeePdf = reinterpret_cast<float*>(ptr); ptr += numPixel * sizeof(float);
#endif

    return raw;
}

/**
 * Gbuffer for shift rejection, primary footprint calculation, and guide layers for denoising
 */
struct GBuffer {
    uint32_t* __restrict__ normals;
    half2* __restrict__ distance_matID;

    // RGB10A2
    uint32_t* __restrict__ albedos;

    half2* __restrict__ motionVector;
    half2* __restrict__ dualMotionVector;

    __device__ __forceinline__ uint32_t getMatID(uint32_t idx) const {
        return __half_as_short(__ldcs(&distance_matID[idx].y));
    }

    __device__ __forceinline__ float3 getNormal(uint32_t idx) const {
        return unpackOct(__ldcs(&normals[idx]));
    }

    __device__ __forceinline__ float getDepth(uint32_t idx) const {
        return __half2float(__ldcs(&distance_matID[idx].x));
    }

    // prefer to keep values in cache for the dual mv generation stage
    __device__ __forceinline__ float getDepth_notstreaming(uint32_t idx) const {
        return __half2float(__ldg(&distance_matID[idx].x));
    }

    // prefer to keep values in cache for the dual mv generation stage
    __device__ __forceinline__ half2 getMV_notstreaming(uint32_t idx) const {
        return __ldg(&motionVector[idx]);
    }

    __device__ __forceinline__ half2 getMV(uint32_t idx) const {
        return __ldcs(&motionVector[idx]);
    }

    __device__ __forceinline__ half2 getDualMV(uint32_t idx) const {
        return __ldcs(&dualMotionVector[idx]);
    }

    __device__ __forceinline__ void setGeometry(
        uint32_t idx,
        float3 normal,
        float dist,
        uint32_t matID,
        float3 albedo
    ) const {
        __stcs(&normals[idx], packOct(normal));
        __stcs(&distance_matID[idx], make_half2(__float2half(dist), __short_as_half(matID)));
        __stcs(&albedos[idx], packRGB10A2(albedo, false));
    }

    __device__ __forceinline__ void setMotionVec(
        uint32_t idx,
        float2 motionVec
    ) const {
        __stcs(&motionVector[idx], make_half2(__float2half(motionVec.x), __float2half(motionVec.y)));
    }

    __device__ __forceinline__ void setDualMotionVec(
        uint32_t idx,
        float2 motionVec
    ) const {
        __stcs(&dualMotionVector[idx], make_half2(__float2half(motionVec.x), __float2half(motionVec.y)));
    }

    __device__ __forceinline__ void setDualMotionVec(
        uint32_t idx,
        half2 motionVec
    ) const {
        __stcs(&dualMotionVector[idx], motionVec);
    }

    __device__ __forceinline__ void setInvalidMotionVec(
        uint32_t idx
    ) const {
        uint32_t sentinelBits = 0xFFFFFFFF;
        __stcs(&motionVector[idx], reinterpret_cast<const half2&>(sentinelBits));
    }

    // Marks that this pixel's reservoir should be excluded from display for this
    // frame (e.g. directly viewing an area light), while still participating
    // normally in reservoir math and reuse. Stored in the top bit of the RGB10A2
    // albedo alpha field so the motion vector stays a real reprojection vector.
    // Assumes setGeometry (which writes A2 = 0) has already run for this pixel.
    __device__ __forceinline__ void setSkipShadeFlag(uint32_t idx) const {
        uint32_t packed = albedos[idx];
        packed |= (1u << 31);
        __stcs(&albedos[idx], packed);
    }

    __device__ __forceinline__ bool getSkipShade(uint32_t idx) const {
        return (__ldcs(&albedos[idx]) >> 31) & 1u;
    }
};

__host__ inline void* allocateGBuffer(GBuffer& r, uint32_t numPixel) {
    numPixel = (numPixel + 31) & ~31;

    void* raw;
    cudaMalloc(&raw, numPixel * GBUFFER_SIZE);

    char* ptr = static_cast<char*>(raw);
    r.normals = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);
    r.distance_matID = reinterpret_cast<half2*>(ptr); ptr += numPixel * sizeof(half2);
    r.albedos = reinterpret_cast<uint32_t*>(ptr); ptr += numPixel * sizeof(uint32_t);
    r.motionVector = reinterpret_cast<half2*>(ptr); ptr += numPixel * sizeof(half2);
    r.dualMotionVector = reinterpret_cast<half2*>(ptr); ptr += numPixel * sizeof(half2);

    return raw;
}

// Guide buffers for the OptiX denoiser, written once per frame at the primary hit.
// Deliberately separate from the GBuffer: the denoiser wants flat, full-precision
// float4 guides, and specifically the GEOMETRIC normal (low frequency) rather than
// the normal-mapped shading normal the GBuffer stores for ReSTIR reuse. The noisy
// color input comes from CommonParams::accum_buffer; the temporal flow reuses
// GBuffer::motionVector -- neither is duplicated here.
struct DenoiserGuides {
    float4* __restrict__ albedo; // primary-surface base color (albedo-demodulation guide)
    float4* __restrict__ normal; // primary-surface GEOMETRIC normal (edge-stopping guide)
    float2* __restrict__ flow;   // temporal motion vector, in pixels, OptiX convention:
                                 // the vector FROM the current pixel TO its previous-frame
                                 // position, i.e. (prevPixel - currPixel) = -(gbuffer MV).

    __device__ __forceinline__ void setGuides(uint32_t idx, float3 alb, float3 geoNormal, float2 flowVec) const {
        __stcs(&albedo[idx], make_float4(alb.x, alb.y, alb.z, 0.0f));
        __stcs(&normal[idx], make_float4(geoNormal.x, geoNormal.y, geoNormal.z, 0.0f));
        __stcs(&flow[idx], flowVec);
    }

    __device__ __forceinline__ float3 getAlbedo(uint32_t idx) const {
        float4 a = __ldcs(&albedo[idx]);
        return make_float3(a.x, a.y, a.z);
    }

    __device__ __forceinline__ float3 getNormal(uint32_t idx) const {
        float4 n = __ldcs(&normal[idx]);
        return make_float3(n.x, n.y, n.z);
    }
};

// Same single-cudaMalloc arena pattern as allocateGBuffer: carve the field pointers
// out of one block; free by cudaFree-ing the returned base. Single-buffered (the
// temporal denoiser warps the previous OUTPUT, not the previous guides).
__host__ inline void* allocateDenoiserGuides(DenoiserGuides& r, uint32_t numPixel) {
    numPixel = (numPixel + 31) & ~31;

    void* raw;
    cudaMalloc(&raw, numPixel * DENOISER_GUIDES_SIZE);

    char* ptr = static_cast<char*>(raw);
    r.albedo = reinterpret_cast<float4*>(ptr); ptr += numPixel * sizeof(float4);
    r.normal = reinterpret_cast<float4*>(ptr); ptr += numPixel * sizeof(float4);
    r.flow   = reinterpret_cast<float2*>(ptr); ptr += numPixel * sizeof(float2);

    return raw;
}

// Roughly 30 format strings live in this one function. Compiled out entirely at
// DEBUG_MODE 0; call it through DEBUG_PRINT_PIXEL rather than directly.
#if DEBUG_MODE == 1
__device__ inline void printPixelData(const Reservoir& r, const GBuffer& g, uint32_t pixelIdx, uint32_t frame) {
    // ==========================================
    // 1. UNPACK RESERVOIR DATA
    // ==========================================
    // Routed through accessors so this works under either reservoir layout.
    float W; float3 unF;
    r.getWF(pixelIdx, W, unF);
    uint32_t initSeed = r.getInitRandomSeed(pixelIdx);

    // Unpack Path Flags
    uint32_t M, pathLength, rcVertexInd;
    TechniqueType technique;
    r.getPathFlags(pixelIdx, M, pathLength, rcVertexInd, technique);
    uint32_t tech = technique;

    // Unpack RC Vertex Geometry
    uint32_t rcPrimID, rcInstanceID;
    float2 rcBary;
    float3 rcWi, rcRadiance;
    r.getRcVertexGeometry_globalLoad(pixelIdx, rcPrimID, rcBary, rcWi, rcRadiance, rcInstanceID);

    // Unpack Cached Values
    float jacobian, neePDF;
    r.getCachedValues_globalLoad(pixelIdx, jacobian, neePDF);

    // ==========================================
    // 2. UNPACK GBUFFER DATA
    // ==========================================
    // Unpack Normal
    float3 normal = unpackOct(g.normals[pixelIdx]);

    float dist = g.getDepth(pixelIdx);
    int matID = g.getMatID(pixelIdx);

    // Unpack Albedo
    float3 albedo;
    uint32_t matTypeFlag;
    unpackRGB10A2(g.albedos[pixelIdx], albedo, matTypeFlag);

    // Unpack Motion Vector
    float2 mv = __half22float2(g.motionVector[pixelIdx]);

    // ==========================================
    // 3. FORMATTED PRINT
    // ==========================================
    printf("==================================================\n");
    printf(" DATA FOR PIXEL INDEX: %u at Frame %u\n", pixelIdx, frame);
    printf("==================================================\n");
    printf("[ RESERVOIR STATE ]\n");
    printf("  W (Weight)         : %f\n", W);
    printf("  F (Unpacked RGB)   : (%.3f, %.3f, %.3f)\n", unF.x, unF.y, unF.z);
    printf("  Init Random Seed   : %u\n", initSeed);
    printf("  Path Flags         : M = %u | Length = %u | RcIdx = %u | Tech = %u", M, pathLength, (rcVertexInd == 254 || rcVertexInd == 255) ? pathLength : rcVertexInd, tech);
    if (rcVertexInd == 254)
        printf(" with Direction Copy Sub Strategy\n");
    else if (rcVertexInd == 255)
        printf(" with Full Replay Sub Strategy\n");
    printf("\n");
    printf("[ RC VERTEX GEOMETRY ]\n");
    if (rcPrimID == 0xFFFFFFFF)
        printf("  Prim ID            : Environment\n");
    else
        printf("  Prim ID            : %u\n", rcPrimID);
    if (rcInstanceID == 0xFFFFFFFF)
        printf("  Instance ID        : none (env / untransformed)\n");
    else
        printf("  Instance ID        : %u\n", rcInstanceID);
    printf("  Barycentrics       : (%.3f, %.3f)\n", rcBary.x, rcBary.y);
    printf("  Wi (Oct Unpacked)  : (%.3f, %.3f, %.3f)\n", rcWi.x, rcWi.y, rcWi.z);
    printf("  Radiance           : (%.3f, %.3f, %.3f)\n", rcRadiance.x, rcRadiance.y, rcRadiance.z);
    printf("  Cached Jacobian    : %f\n", jacobian);
    printf("  NEE Light PDF      : %f\n", neePDF);
    printf("\n");
    printf("[ G-BUFFER STATE ]\n");
    printf("  Normal             : (%.3f, %.3f, %.3f)\n", normal.x, normal.y, normal.z);
    printf("  Hit Distance (t)   : %f\n", dist);
    printf("  Material           : %d\n", matID);
    printf("  Albedo             : (%.3f, %.3f, %.3f)\n", albedo.x, albedo.y, albedo.z);
    printf("  Material Flag      : %u\n", matTypeFlag);
    printf("  Motion Vector      : (%.3f, %.3f)\n", mv.x, mv.y);
    printf("==================================================\n\n");
}
#endif // DEBUG_MODE == 1

struct ShiftResult {
    bool isValid;               // True if visibility, footprint, and pdfs pass
    float3 contribution;        // The physical throughput * radiance
    float jacobian;             // The evaluated Jacobian for this shift
    float new_cached_jacobian;  // To save to the reservoir if this path wins
};

struct ShiftResultBuffer {
    uint4* __restrict__ buffer;

    __device__ __forceinline__ void setResult(uint32_t idx, bool isValid, float3 contribution, float jacobian, float new_cached_jacobian) const {
        uint4 data;
        data.x = isValid;
        data.y = toRGB9E5(contribution);
        data.z = __float_as_uint(jacobian);
        data.w = __float_as_uint(new_cached_jacobian);
        buffer[idx] = data;
    }

    __device__ __forceinline__ void getResult(uint32_t idx, bool& isValid, float3& contribution, float& jacobian, float& new_cached_jacobian) const {
        uint4 data = __ldg(&buffer[idx]);
        isValid = data.x;
        contribution = fromRGB9E5(data.y);
        jacobian = __uint_as_float(data.z);
        new_cached_jacobian = __uint_as_float(data.w);
    }
};

__host__ inline void* allocateShiftResultBuffer(ShiftResultBuffer& r, uint32_t numPixel) {
    numPixel = (numPixel + 31) & ~31;

    void* raw;
    cudaMalloc(&raw, numPixel * sizeof(uint4));

    char* ptr = static_cast<char*>(raw);
    r.buffer = reinterpret_cast<uint4*>(ptr); ptr += numPixel * sizeof(uint4);

    return raw;
}

__device__ __forceinline__ bool needNeePDF(TechniqueType type) {
    return (type & (SHIFT_K_IS_D | SHIFT_K_IS_D_MINUS_1));
}

__device__ __forceinline__ bool K_is_D(TechniqueType type) {
    return (type & SHIFT_K_IS_D);
}

__device__ __forceinline__ bool K_is_D_minus_1(TechniqueType type) {
    return (type & SHIFT_K_IS_D_MINUS_1);
}

__device__ __forceinline__ bool K_less_D_minus_1(TechniqueType type) {
    return (type & SHIFT_K_LESS_D_MINUS_1);
}

__device__ __forceinline__ bool K_is_D_minus_1_nee(TechniqueType type) {
    return (type & SHIFT_K_IS_D_MINUS_1) && (type & SHIFT_IS_NEE);
}

__device__ __forceinline__ bool is_nee(TechniqueType type) {
    return (type & (SHIFT_IS_NEE));
}

__device__ __forceinline__ bool is_bsdf(TechniqueType type) {
    return !(type & (SHIFT_IS_NEE));
}

__device__ __forceinline__ bool is_internal_rc_vertex(TechniqueType type) {
    return !(type & SHIFT_K_IS_D);
}

__device__ __forceinline__ bool is_env(TechniqueType type) {
    return (type & SHIFT_IS_ENV);
}

__device__ __forceinline__ bool is_area(TechniqueType type) {
    return !(type & SHIFT_IS_ENV);
}