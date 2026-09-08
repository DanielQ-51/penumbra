#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"
#include "restirPTenhanced_helpers.cuh"
#include "restirPTenhanced_kernels.cuh"
#include "settings.cuh"


#ifndef DUPE_MAP_SEARCH_RADIUS
#define DUPE_MAP_SEARCH_RADIUS 8
#endif

#ifndef DUPE_MAP_SEARCH_STRIDE
#define DUPE_MAP_SEARCH_STRIDE 1
#endif

// Dual motion vectors for occlusions, after Zeng et al. 2021, "Temporally Reliable
// Motion Vectors for Real-time Ray Tracing", Eqn. 6:
//
//     x^O_{i-1} = y + (x_i - z),   y = back-project(x_i),   z = forward-project(y)
//
// z - y is the occluder's screen displacement, so the expression collapses to
// x_i - mv_occluder: reproject by the OCCLUDER's motion rather than our own, where
// the occluder is whatever occupied x_i's traditional back-projection last frame.
// The y/z round trip exists only to identify which surface that is.
//
// The paper obtains z by applying the occluder's object transform T. We instead read
// the occluder's own stored motion vector out of the previous frame's gbuffer, which
// covers [i-2, i-1] rather than [i-1, i] -- a constant-velocity approximation. Exact
// T would require a per-pixel instance id in the gbuffer.
__global__ void computeDualMV(
    GBuffer gbuffer,
    GBuffer prevGbuffer,
    uint32_t w,
    uint32_t h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int center_idx = y * w + x;

    half2 self_mv = gbuffer.getMV_notstreaming(center_idx);

    // The dual slot must always decode to a finite half2: the temporal forward pass
    // feeds it straight into roundf(). Defaulting to this pixel's own motion vector
    // means an unresolvable occluder lookup can never plant a NaN there.
    half2 dual_mv = self_mv;

    // 0xFFFFFFFF = env miss, no reprojectable surface. Temporal never reads the dual
    // slot for those pixels, so propagating the sentinel is correct.
    if (reinterpret_cast<const uint32_t&>(self_mv) != 0xFFFFFFFFu) {
        // y in Eqn. 6.
        int back_x = x - (int)roundf(__half2float(self_mv.x));
        int back_y = y - (int)roundf(__half2float(self_mv.y));

        if (back_x >= 0 && back_x < (int)w && back_y >= 0 && back_y < (int)h) {
            half2 occ_mv = prevGbuffer.getMV_notstreaming(back_y * w + back_x);
            float ox = __half2float(occ_mv.x);
            float oy = __half2float(occ_mv.y);
            // isfinite also rejects the 0xFFFFFFFF sentinel, which decodes to NaN,
            // and any garbage half pattern in an unwritten history pixel.
            if (isfinite(ox) && isfinite(oy)) dual_mv = occ_mv;
        }
    }

    gbuffer.setDualMotionVec(center_idx, dual_mv);
}

__global__ void computeDuplicationMapKernel(
    Reservoir lastFrameReservoir,
    uint8_t* __restrict__ duplication_map,
    uint32_t w,
    uint32_t h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int center_idx = y * w + x;

    uint32_t currSeed = lastFrameReservoir.getSeed_notstreaming(center_idx);

    if (currSeed == 0u) {
        duplication_map[center_idx] = 0u;
        return;
    }

    int match_count = 0;
    int valid_pixels = 0;

    for (int dy = -DUPE_MAP_SEARCH_RADIUS; dy <= DUPE_MAP_SEARCH_RADIUS; dy += DUPE_MAP_SEARCH_STRIDE) {
        for (int dx = -DUPE_MAP_SEARCH_RADIUS; dx <= DUPE_MAP_SEARCH_RADIUS; dx += DUPE_MAP_SEARCH_STRIDE) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                continue;
            }

            uint32_t neighbor_seed = lastFrameReservoir.getSeed_notstreaming(ny * w + nx);

            if (neighbor_seed == currSeed) {
                match_count++;
            }

            valid_pixels++;
        }
    }

    __stcs(&duplication_map[center_idx],(uint8_t)(((float)match_count / (float)valid_pixels) * 255.0f));
}

__global__ void displayWinningReservoirs(PipelineParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.common.w || y >= params.common.h) return;

    int pixelIdx = y * params.common.w + x;

    if (IS_DEBUG_PIXEL(x, y)) {
        DEBUG_PRINTF("frame %u at the end seed: %u\n", params.common.frame_index, params.restir.reservoir.initRandomSeed[pixelIdx]);
        DEBUG_PRINT_PIXEL(params.restir.reservoir, params.restir.gbuffer, pixelIdx, params.common.frame_index);
    }

    half2 mv = params.restir.gbuffer.getMV(pixelIdx);
    if (reinterpret_cast<const uint32_t&>(mv) != 0xFFFFFFFF && !params.restir.gbuffer.getSkipShade(pixelIdx)) {
        float3 output = fireflyClamp(params.restir.reservoir.getShadingColor_streaming(pixelIdx));
#if DEBUG_MODE == 1
        if ((isnan(output.x) || isnan(output.y) || isnan(output.z))) {
            DEBUG_PRINTF("nan at pixel idx: %d\n", pixelIdx);
        }
#endif

#if DEBUG_VISUALIZE_TYPE == 1
        uint32_t pathFlags = params.restir.reservoir.getPathFlagsRaw(pixelIdx);
        output = debugVisualizeTechnique(extractType(pathFlags), extractRcInd(pathFlags));
#endif

#if ACCUMULATE_FRAMES == 1
        params.common.accum_buffer[pixelIdx] += f4(output);
#else
        params.common.accum_buffer[pixelIdx] = f4(output);
#endif
    }
}

__global__ void initLinks(uint32_t* buffer, uint32_t dimension) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    int pixelIdx = y * dimension + x;
    buffer[pixelIdx] = pixelIdx / 2;
}

__global__ void shuffleLinks(uint32_t* bufferA, uint32_t* bufferB, uint32_t dimension, uint32_t iteration) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dimension / 2|| y >= dimension / 2) return;

    int startX = 2 * x;
    int startY = 2 * y;

    int pixelIdx = y * dimension + x;

    if (iteration % 2 == 1) {
        startX += 1;
        startY += 1;
    }

    int2 coords[4] = {
        make_int2(startX % dimension, startY % dimension),
        make_int2((startX + 1) % dimension, startY % dimension),
        make_int2(startX % dimension, (startY + 1) % dimension),
        make_int2((startX + 1) % dimension, (startY + 1) % dimension)
    };

    uint32_t ids[4];
    for (int i = 0; i < 4; i++) {
        ids[i] = bufferA[coords[i].y * dimension + coords[i].x];
    }

    RNGState localState = load_rng(pixelIdx, iteration, hash_uint32(dimension), nullptr);

    for (int i = 3; i > 0; i--) {
        uint32_t j = (uint32_t)(rand(&localState) * (i + 1));
        j = min(j, (uint32_t)i);

        uint32_t temp = ids[i];
        ids[i] = ids[j];
        ids[j] = temp;
    }

    for (int i = 0; i < 4; i++) {
        bufferB[coords[i].y * dimension + coords[i].x] = ids[i];
    }
}

__global__ void resolvePairsPassA(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t packed_xy = (y << 16) | x;

    atomicCAS(&indexTableSlot0[link_id], 0xFFFFFFFF, packed_xy);
}

__global__ void resolvePairsPassB(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t packed_xy = (y << 16) | x;

    if (indexTableSlot0[link_id] != packed_xy) {
        indexTableSlot1[link_id] = packed_xy;
    }
}

__global__ void extractDeltasKernel(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, short2* outputTex, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t my_packed_xy = (y << 16) | x;

    uint32_t slot0 = indexTableSlot0[link_id];
    uint32_t slot1 = indexTableSlot1[link_id];

    uint32_t partner_packed = (slot0 == my_packed_xy) ? slot1 : slot0;

    int partner_x = partner_packed & 0xFFFF;
    int partner_y = partner_packed >> 16;

    int dx = partner_x - (int)x;
    int dy = partner_y - (int)y;

    // Break long links so the texture tiles. Signed math is mandatory here: with
    // dimension being uint32_t the comparisons promote dx/dy to unsigned, every
    // negative delta takes the first branch, and the second branch undoes it.
    const int S = (int)dimension;
    if (dx >  S / 2) dx -= S;
    if (dx < -S / 2) dx += S;
    if (dy >  S / 2) dy -= S;
    if (dy < -S / 2) dy += S;

    outputTex[y * dimension + x] = make_short2(dx, dy);
}

// ===========================================================================
// Reuse texture validation
//
// Three independent checks, cheapest / most localized first:
//
//   1. countLinkIds + checkLinkCounts  -- runs on the *link id* buffer, before
//      deltas are extracted. Every link id in [0, N/2) must appear exactly
//      twice. This isolates initLinks/shuffleLinks from everything after.
//
//   2. validateReuseTextureTexSpace    -- runs on the finished short2 texture.
//      Checks the texture is a perfect self-inverting matching in texture
//      space: no zero deltas, |delta| <= S/2, and tex[p + tex[p]] == -tex[p].
//
//   3. validateReuseTextureScreenSpace -- the screen wide test. Runs the full
//      production path (get_paired_neighbor, including the per frame
//      transpose / flip / offset) and asserts the property resolveSpatialReuse
//      actually depends on: if A pairs with B then B pairs with A. If this
//      fails, resolveSpatialReuse reads shiftResultBuffer[t][B] believing it
//      holds "B's path shifted to A", when it actually holds "B's path shifted
//      to somebody else" -> silently wrong MIS weights and wrong radiance.
// ===========================================================================
#if VALIDATE_REUSE_TEXTURES == 1

__device__ __forceinline__ void accumulateReuseDelta(ReuseTextureStats* stats, int dx, int dy) {
    const int d2 = dx * dx + dy * dy;

    atomicAdd(&stats->checked, 1ull);
    atomicAdd(&stats->sumDx, (unsigned long long)(long long)dx);
    atomicAdd(&stats->sumDy, (unsigned long long)(long long)dy);
    atomicAdd(&stats->sumD2, (unsigned long long)(long long)d2);
    atomicMax(&stats->maxAbsDelta, (unsigned int)max(abs(dx), abs(dy)));

    unsigned int bin = (unsigned int)(sqrtf((float)d2) + 0.5f);
    bin = min(bin, (unsigned int)(REUSE_VALIDATION_HIST_BINS - 1));
    atomicAdd(&stats->hist[bin], 1u);
}

__device__ __forceinline__ bool reuseValidationCanPrint(ReuseTextureStats* stats) {
    return atomicAdd(&stats->printsUsed, 1u) < REUSE_VALIDATION_MAX_PRINTS;
}

__global__ void countLinkIds(
    const uint32_t* __restrict__ linkBuf,
    uint32_t* __restrict__ counts,
    uint32_t dimension,
    ReuseTextureStats* stats
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    const uint32_t numLinks = dimension * dimension / 2;
    const uint32_t id = linkBuf[y * dimension + x];

    if (id >= numLinks) {
        atomicAdd(&stats->outOfRange, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u] LINK ID OUT OF RANGE at (%u,%u): id %u, valid range is [0,%u). "
                   "The link buffer was never initialized here.\n",
                   dimension, x, y, id, numLinks);
        }
        return;
    }

    atomicAdd(&counts[id], 1u);
}

__global__ void checkLinkCounts(
    const uint32_t* __restrict__ counts,
    uint32_t numLinks,
    ReuseTextureStats* stats
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLinks) return;

    const uint32_t c = counts[i];
    if (c != 2u) {
        atomicAdd(&stats->brokenInvolution, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex] LINK ID %u APPEARS %u TIME(S), expected exactly 2. "
                   "The shuffles are not a permutation of the initial link ids.\n", i, c);
        }
    }
}

__global__ void validateReuseTextureTexSpace(
    const short2* __restrict__ tex,
    uint32_t dimension,
    uint32_t textureId,
    ReuseTextureStats* stats
) {
    const int S = (int)dimension;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= S || y >= S) return;

    const short2 raw = tex[y * S + x];
    const int dx = raw.x;
    const int dy = raw.y;

    // (a) a pixel must never be paired with itself
    if (dx == 0 && dy == 0) {
        atomicAdd(&stats->selfLinks, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  tex-space] SELF LINK at (%d,%d): delta is (0,0)\n", textureId, x, y);
        }
        return;
    }

    // (b) after wrap breaking, every delta must land inside [-S/2, S/2]
    if (abs(dx) > S / 2 || abs(dy) > S / 2) {
        atomicAdd(&stats->outOfRange, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  tex-space] LONG LINK NOT BROKEN at (%d,%d): delta (%d,%d), "
                   "each component must be within +/-%d\n", textureId, x, y, dx, dy, S / 2);
        }
    }

    // (c) self inversion. The texture tiles, so the partner lookup wraps, but the
    //     stored delta must be the exact negation (screen space uses absolute coords).
    const int px = ((x + dx) % S + S) % S;
    const int py = ((y + dy) % S + S) % S;
    const short2 back = tex[py * S + px];

    if ((int)back.x != -dx || (int)back.y != -dy) {
        atomicAdd(&stats->brokenInvolution, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  tex-space] NOT SELF INVERTING: (%d,%d) delta (%d,%d) -> partner (%d,%d), "
                   "but partner stores (%d,%d), expected (%d,%d)\n",
                   textureId, x, y, dx, dy, px, py, (int)back.x, (int)back.y, -dx, -dy);
        }
        return;
    }

    accumulateReuseDelta(stats, dx, dy);
}

__global__ void validateReuseTextureScreenSpace(
    uint32_t w,
    uint32_t h,
    uint32_t frame_index,
    uint32_t textureId,
    uint32_t texSize,
    const short2* __restrict__ tex,
    ReuseTextureStats* stats
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)w || y >= (int)h) return;

    const int2 screenDim = make_int2((int)w, (int)h);
    const int2 self      = make_int2(x, y);

    const int2 partner = get_paired_neighbor(self, textureId, frame_index, texSize, screenDim, tex);

    // Partner pushed off the frame. Legal, and symmetric (the partner never runs).
    if (partner.x < 0 || partner.y < 0) {
        atomicAdd(&stats->offScreen, 1u);
        return;
    }

    if (partner.x >= (int)w || partner.y >= (int)h) {
        atomicAdd(&stats->outOfRange, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  frame %u  screen] PARTNER OUT OF BOUNDS: (%d,%d) -> (%d,%d), screen is %ux%u. "
                   "get_paired_neighbor's bounds test let a bad coord through.\n",
                   textureId, frame_index, x, y, partner.x, partner.y, w, h);
        }
        return;
    }

    if (partner.x == x && partner.y == y) {
        atomicAdd(&stats->selfLinks, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  frame %u  screen] SELF PAIR at (%d,%d): a pixel would reuse from itself\n",
                   textureId, frame_index, x, y);
        }
        return;
    }

    // The property the paper's amortization rests on, and the one resolveSpatialReuse
    // assumes when it reads shiftResultBuffer[t][partnerIdx] as "shifted to me".
    const int2 back = get_paired_neighbor(partner, textureId, frame_index, texSize, screenDim, tex);

    if (back.x != x || back.y != y) {
        atomicAdd(&stats->brokenInvolution, 1u);
        if (reuseValidationCanPrint(stats)) {
            printf("[reuseTex %u  frame %u  screen] PAIRING NOT SYMMETRIC: (%d,%d) -> (%d,%d) -> (%d,%d). "
                   "Pixel (%d,%d) shifts its path to (%d,%d), but (%d,%d) shifts to (%d,%d), "
                   "so the cached shift result is read by the wrong pixel.\n",
                   textureId, frame_index,
                   x, y, partner.x, partner.y, back.x, back.y,
                   x, y, partner.x, partner.y,
                   partner.x, partner.y, back.x, back.y);
        }
        return;
    }

    accumulateReuseDelta(stats, partner.x - x, partner.y - y);
}

#endif // VALIDATE_REUSE_TEXTURES == 1

__global__ void resolveSpatialReuse(
    PipelineParams allParams)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= allParams.common.w || y >= allParams.common.h) return;

    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers
    
    const Reservoir& reservoirIn = restir.reservoir;
    const Reservoir& reservoirOut = restir.lastFrameReservoir;

    const uint32_t selfIdx = y * params.w + x;

    uint32_t currentConfidence, self_pathLength, self_rcVertexIndex;
    TechniqueType self_type;
    reservoirIn.getPathFlags(selfIdx, currentConfidence, self_pathLength, self_rcVertexIndex, self_type);
    float3 self_F; float self_W;
    reservoirIn.getWF(selfIdx, self_W, self_F);
    const float currentTargetPdf = targetFunction(self_F);

    if (currentConfidence == 0u) {
        reservoirOut.setPathFlags(selfIdx, 0u);   // M stays 0 (not inflated to totalConfidence)
        reservoirOut.setW_noCS(selfIdx, 0.0f);
        return;
    }

    int2 neighborCoord[NUM_REUSE_TEXTURES];
    uint32_t neighborPixelIdx[NUM_REUSE_TEXTURES];
    uint32_t neighborConfidence[NUM_REUSE_TEXTURES];   // 0 if that partner doesn't exist
    // Prefetch each accepted neighbor's {W,F} here in the confidence loop and stash it,
    // so the resampling loop below consumes it from registers instead of making a second
    // scattered global trip to the same pixel (the old SOA two-pass read).
    float3 neighbor_F_cache[NUM_REUSE_TEXTURES];
    float  neighbor_W_cache[NUM_REUSE_TEXTURES];
    float totalNeighborConfidence = 0.0f;

    for (uint32_t t = 0; t < NUM_REUSE_TEXTURES; ++t) {
        neighborCoord[t] = get_paired_neighbor(
            make_int2(x, y), t, params.frame_index,
            restir.reuseTextureSizes[t], make_int2(params.w, params.h),
            restir.reuseTextures[t]);

        neighborConfidence[t] = 0u;
        
        const bool pairAccepted =
               (neighborCoord[t].x >= 0) && (neighborCoord[t].y >= 0)
            && isSpatialPairAccepted(allParams, make_int2(x, y), neighborCoord[t]);

        if (pairAccepted) {
            neighborPixelIdx[t] = neighborCoord[t].y * params.w + neighborCoord[t].x;
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("frame %u at spatial neighbor %u: %u\n", params.frame_index, t,
                             restir.reservoir.initRandomSeed[neighborPixelIdx[t]]);
                DEBUG_PRINT_PIXEL(restir.reservoir, restir.gbuffer, neighborPixelIdx[t], params.frame_index);
            }
            uint32_t m, pathLen, rcIdx; TechniqueType type;
            reservoirIn.getPathFlags(neighborPixelIdx[t], m, pathLen, rcIdx, type);
            reservoirIn.getWF(neighborPixelIdx[t], neighbor_W_cache[t], neighbor_F_cache[t]);

            neighborConfidence[t] = m;
            totalNeighborConfidence += (float)m;
        }
    }

    const float totalConfidence = (float)currentConfidence + totalNeighborConfidence;

    float canonicalMisWeight = (totalConfidence > 0.0f)
        ? ((float)currentConfidence / totalConfidence) : 1.0f;
    float weightSum = 0.0f;

    // P2 6.3: vector-valued resampling weights, accumulated in parallel with the scalar
    // ones. Scalar weights drive selection/resampling; this RGB sum drives shading.
    // Its luminance equals weightSum by construction (targetFunction == luminance), so it
    // is a drop-in for F_selected * W_final -- same brightness, chroma averaged over all
    // candidates instead of taken from the single winner. That is what kills color noise.
    float3 shadingWeightSum = f3(0.0f);

    int    winningTexture = -1;   // -1 means the canonical won
    float3 winning_shiftedF = f3(0.0f);
    float  winning_shiftedTargetPdf = 0.0f;
    float  winning_newCachedJacobian = 0.0f;
    uint32_t winning_pixelIdx = 0;

    RNGState localState = load_rng(
        hash_uint32(selfIdx),
        hash_uint32(params.frame_index),
        hash_uint32(0xA5A5A5A5u), nullptr
    );

    for (uint32_t t = 0; t < NUM_REUSE_TEXTURES; ++t) {
        if (neighborConfidence[t] == 0u) continue;

        bool   forwardValid,  backwardValid;
        float3 forwardContribution, backwardContribution;
        float  forwardJacobian, forwardNewCachedJacobian;
        float  backwardJacobian, backwardNewCachedJacobian;
        restir.shiftResultBuffer[t].getResult(neighborPixelIdx[t],
            forwardValid,  forwardContribution,  forwardJacobian,  forwardNewCachedJacobian);
        restir.shiftResultBuffer[t].getResult(selfIdx,
            backwardValid, backwardContribution, backwardJacobian, backwardNewCachedJacobian);

        // --- canonical's MIS contribution from this pair, evaluated at X_c ---
        //     pairDenominator = c_c*p̂(X_c) + (Σc_j)*p̂_←neighbor(X_c)
        const float backwardTargetPdf = backwardValid ? targetFunction(backwardContribution) : 0.0f;
        const float pairDenominator_atCurrent =
              (float)currentConfidence * currentTargetPdf
            + totalNeighborConfidence * backwardTargetPdf * backwardJacobian;
        if (pairDenominator_atCurrent > 0.0f) {
            canonicalMisWeight +=
                ((float)neighborConfidence[t] / totalConfidence)
              * ((float)currentConfidence * currentTargetPdf / pairDenominator_atCurrent);
        }

        // --- neighbor's resampling weight, evaluated at Y = T(neighbor -> current) ---
        if (!forwardValid) continue;
        const float3 neighbor_F         = neighbor_F_cache[t];  // prefetched in the confidence loop
        const float  neighbor_W         = neighbor_W_cache[t];
        const float  neighbor_targetPdf = targetFunction(neighbor_F);          // p̂(X_neighbor)
        const float  shiftedTargetPdf   = targetFunction(forwardContribution); // p̂(Y)

        // pairDenominator = c_c*p̂(Y) + (Σc_j)*p̂_←neighbor(Y);  p̂_←neighbor(Y)=p̂(X_nb)/J,
        // so multiply the whole ratio through by J -> forwardJacobian in the first term.
        const float pairDenominator_atNeighbor =
              (float)currentConfidence * shiftedTargetPdf * forwardJacobian
            + totalNeighborConfidence * neighbor_targetPdf;
        if (pairDenominator_atNeighbor <= 0.0f) continue;

        const float neighborMisWeight =
              ((float)neighborConfidence[t] / totalConfidence)
            * (totalNeighborConfidence * neighbor_targetPdf / pairDenominator_atNeighbor);

        // Eq 1:  w = m * p̂(Y) * W * |dT/dX|
        const float neighborResamplingWeight =
              neighborMisWeight * shiftedTargetPdf * neighbor_W * forwardJacobian;
        if (!(neighborResamplingWeight > 0.0f)) continue;

        weightSum += neighborResamplingWeight;
        // same weight, RGB instead of its luminance
        shadingWeightSum += neighborMisWeight * forwardContribution * neighbor_W * forwardJacobian;

        if (rand(&localState) < (neighborResamplingWeight / weightSum)) {
            winningTexture = (int)t;
            winning_shiftedF = forwardContribution;
            winning_shiftedTargetPdf = shiftedTargetPdf;
            winning_newCachedJacobian = forwardNewCachedJacobian;
            winning_pixelIdx = neighborPixelIdx[t];
        }
    }

    const float canonicalResamplingWeight = canonicalMisWeight * currentTargetPdf * self_W;
    weightSum += canonicalResamplingWeight;
    // canonical uses the identity shift, so no Jacobian factor
    shadingWeightSum += canonicalMisWeight * self_F * self_W;

    if (canonicalResamplingWeight > 0.0f && rand(&localState) < (canonicalResamplingWeight / weightSum))
        winningTexture = -1;

    const uint32_t new_M = min((uint32_t)(totalConfidence + 0.5f), 255u);

    // ---- 5. Publish (every field, unconditionally) ----
    if (winningTexture >= 0) {
        float W_final = (winning_shiftedTargetPdf > 0.0f) ? (weightSum / winning_shiftedTargetPdf) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        const uint32_t winnerIdx = winning_pixelIdx;
        // One grouped 32B load of the winner's descriptor (winner_M / winner_unusedJac
        // aren't used downstream but ride along for free in the same cache line).
        uint32_t winner_M, winner_pathLength, winner_rcVertexIndex; TechniqueType winner_type;
        uint32_t winner_rcPrimID, winner_rcInstanceID; float2 winner_rcBarycentrics;
        float3   winner_rcWi, winner_rcRadiance;
        float    winner_unusedJac, winner_cachedNeePdf;
        reservoirIn.getShiftDescriptor(winnerIdx,
            winner_M, winner_pathLength, winner_rcVertexIndex, winner_type,
            winner_rcPrimID, winner_rcBarycentrics, winner_rcWi, winner_rcRadiance, winner_rcInstanceID,
            winner_unusedJac, winner_cachedNeePdf);

        reservoirOut.saveReservoirAll(
            selfIdx, W_final, winning_shiftedF,
            reservoirIn.getSeed_notstreaming(winnerIdx), new_M,
            winner_pathLength, winner_rcVertexIndex, winner_type,
            winner_rcInstanceID, winner_rcPrimID, winner_rcBarycentrics, winner_rcWi, winner_rcRadiance,
            winning_newCachedJacobian,   // suffix pdf recomputed for OUR prefix
            winner_cachedNeePdf);        // shift-invariant: shared rc vertex / light
    } else {
        float W_final = (currentTargetPdf > 0.0f) ? (weightSum / currentTargetPdf) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        uint32_t current_rcPrimID, current_rcInstanceID; float2 current_rcBarycentrics;
        float3   current_rcWi, current_rcRadiance;
        reservoirIn.getRcVertexGeometry_globalLoad(selfIdx, current_rcPrimID,
            current_rcBarycentrics, current_rcWi, current_rcRadiance, current_rcInstanceID);
        float current_cachedJacobian, current_cachedNeePdf;
        reservoirIn.getCachedValues_globalLoad(selfIdx,
            current_cachedJacobian, current_cachedNeePdf);

        reservoirOut.saveReservoirAll(
            selfIdx, W_final, self_F,
            reservoirIn.getSeed_notstreaming(selfIdx), new_M,
            self_pathLength, self_rcVertexIndex, self_type,
            current_rcInstanceID, current_rcPrimID, current_rcBarycentrics, current_rcWi, current_rcRadiance,
            current_cachedJacobian, current_cachedNeePdf);  // unchanged: our path didn't move
    }

    // ==============================================================================
    // 6. SHADING  (replaces displayWinningReservoirs)
    //
    // Everything above resamples and republishes the reservoir. Nothing below feeds
    // back into it -- this section only writes the frame's color.
    //
    // We shade with shadingWeightSum (P2 6.3) rather than F_selected * W_final. Both
    // carry the same luminance, but the vector sum marginalizes over the index choice,
    // so chroma is the MIS-weighted average of all candidates instead of one sampled
    // hue. Without this, a converged reservoir gives correct brightness with a random
    // color per frame -- the colored speckle on flat walls.
    // ==============================================================================
    {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("frame %u at the end seed: %u\n", params.frame_index, restir.reservoir.initRandomSeed[selfIdx]);
            DEBUG_PRINT_PIXEL(restir.reservoir, restir.gbuffer, selfIdx, params.frame_index);
        }
        // 0xFFFFFFFF = environment miss: its color already went straight to accum_buffer
        // in candidate gen, and there is no reservoir here to shade.
        // skip-shade = directly-viewed emitter: emission already accumulated, and we do
        // not want the (noisy) indirect reservoir drawn on top of it this frame.
        half2 motionVec = restir.gbuffer.getMV(selfIdx);
        const bool suppressShading =
               (reinterpret_cast<const uint32_t&>(motionVec) == 0xFFFFFFFF)
            || restir.gbuffer.getSkipShade(selfIdx);

        if (!suppressShading) {
            float3 shaded = fireflyClamp(shadingWeightSum);

            if (isnan(shaded.x) || isnan(shaded.y) || isnan(shaded.z) ||
                isinf(shaded.x) || isinf(shaded.y) || isinf(shaded.z)) {
                shaded = f3(0.0f);
            }

#if DEBUG_VISUALIZE_TYPE == 1
            {
                const uint32_t shownFlags = (winningTexture >= 0)
                    ? reservoirIn.getPathFlagsRaw(neighborPixelIdx[winningTexture])
                    : reservoirIn.getPathFlagsRaw(selfIdx);
                shaded = debugVisualizeTechnique(extractType(shownFlags), extractRcInd(shownFlags));
            }
#endif

#if ACCUMULATE_FRAMES == 1
            params.accum_buffer[selfIdx] += f4(shaded);
#else
            params.accum_buffer[selfIdx] = f4(shaded);
#endif
        }
    }
}

