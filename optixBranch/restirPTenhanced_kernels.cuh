#pragma once
#include "optixStructs.cuh"
#include "settings.cuh"

__global__ void computeDualMV(
    GBuffer gbuffer,
    GBuffer prevGbuffer,
    uint32_t w,
    uint32_t h
);

__global__ void computeDuplicationMapKernel(
    Reservoir lastFrameReservoir,
    uint8_t* __restrict__ duplication_map,
    uint32_t w,
    uint32_t h
);

__global__ void displayWinningReservoirs(PipelineParams params);

__global__ void initLinks(uint32_t* buffer, uint32_t dimension);

__global__ void shuffleLinks(uint32_t* bufferA, uint32_t* bufferB, uint32_t dimension, uint32_t iteration);

__global__ void resolvePairsPassA(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t dimension);

__global__ void resolvePairsPassB(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, uint32_t dimension);

__global__ void extractDeltasKernel(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, short2* outputTex, uint32_t dimension);

__global__ void resolveSpatialReuse(PipelineParams allParams);

// ---------------------------------------------------------------------------
// Reuse texture validation (debug only). See validateReuseTextures() in
// restirPTenhanced_spatialReuseTextures.cuh for the host side driver.
// ---------------------------------------------------------------------------
#if VALIDATE_REUSE_TEXTURES == 1

#define REUSE_VALIDATION_MAX_PRINTS 32
#define REUSE_VALIDATION_HIST_BINS  32

struct ReuseTextureStats {
    unsigned long long checked;   // pixels that produced a usable partner
    unsigned long long sumDx;     // signed, stored two's complement
    unsigned long long sumDy;
    unsigned long long sumD2;     // dx*dx + dy*dy
    unsigned int selfLinks;       // pixel paired with itself
    unsigned int brokenInvolution;// A -> B but B -/-> A
    unsigned int outOfRange;      // delta longer than half the texture, or partner off buffer
    unsigned int offScreen;       // partner fell outside the frame (legal)
    unsigned int maxAbsDelta;
    unsigned int printsUsed;
    unsigned int hist[REUSE_VALIDATION_HIST_BINS]; // histogram of round(|delta|)
};

__global__ void validateReuseTextureTexSpace(
    const short2* __restrict__ tex,
    uint32_t dimension,
    uint32_t textureId,
    ReuseTextureStats* stats
);

__global__ void validateReuseTextureScreenSpace(
    uint32_t w,
    uint32_t h,
    uint32_t frame_index,
    uint32_t textureId,
    uint32_t texSize,
    const short2* __restrict__ tex,
    ReuseTextureStats* stats
);

__global__ void countLinkIds(
    const uint32_t* __restrict__ linkBuf,
    uint32_t* __restrict__ counts,
    uint32_t dimension,
    ReuseTextureStats* stats
);

__global__ void checkLinkCounts(
    const uint32_t* __restrict__ counts,
    uint32_t numLinks,
    ReuseTextureStats* stats
);

#endif // VALIDATE_REUSE_TEXTURES == 1