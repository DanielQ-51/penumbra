#include <optix.h>
#include <optix_device.h>
#include "settings.cuh"      // first: defines USE_RAY_CONES etc. before helpers.cuh (via optixUtils) is pulled in
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"
#include "restirPTenhanced_helpers.cuh"


extern "C" {
    __constant__ PipelineParams allParams;
}

extern "C" __global__ void __raygen__restirCandidateGeneration() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 0, nullptr);

    uint32_t seed = localState.getSeed();
    restir.reservoir.setInitRandomSeed(pixelIdx, seed);
    half2 jitter;
    Ray r = params.camera.generateCameraRayRecordOffset(localState, x, y, jitter);

    float3 throughput = f3(1.0f);
    float3 suffixThroughput = f3(1.0f);
    float lastPDF;      // lobe-specific: for the cached Jacobian and footprints
    float lastPDF_marg; // marginalized: for the bsdf-vs-light MIS weight at the next hit
    bool prevDelta;
    float lastCosine;

    #if DEBUG_MODE == 1
    float3 lastPOS_GETRIDOFME; // for displaying the debug paths
    #endif
    float w_sum = 0.0f;
    uint32_t F = 0;
    uint32_t pathFlags = 0;
    uint32_t pathRcVertexIndex = FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND; // mark unchosen
    uint4 pathRcVertexGeometry = make_uint4(0, 0, 0, 0);
    uint4 actualRcVertexGeometry = make_uint4(0, 0, 0, 0);
    uint32_t rcRadiance = 0u; // written into the reservoir unconditionally at finalize_pixel
    float pathCachedJacobian = 0.0f;
    float actualCachedJacobian = 0.0f;
    float neepdf = -1.0f;

    float primaryFootprint;

    // Ray-cone texture LOD state (2 registers). coneWidth = cone radius at the current
    // ray origin (world units); coneSpread = accumulated half-angle. Seeded to a
    // zero-width cone whose spread is one pixel's angular size (2*tan(fov/2)/height).
    float coneWidth  = 0.0f;
    float coneSpread = 2.0f * params.camera.fovScale / (float)params.h;

    /**
     * Trace Primary Hit, special case
     */

{
    // to allow gbuffer to write coalesced, we defer SER for first bounce to the end
    SurfaceHit hitData = traceClosestNoSER(params, r);
    if (!hitData.isHit)
    {
        float3 contribution = params.shadeContext.lightSampler.envMap.sampleDir(r.direction);

#if ACCUMULATE_FRAMES == 1
        params.accum_buffer[pixelIdx] += f4(contribution);
#else
        params.accum_buffer[pixelIdx] = f4(contribution);
#endif
        restir.reservoir.setPathFlags(pixelIdx, 0);
        restir.reservoir.setCachedJacobian(pixelIdx, -1.0f); // gate this out of shifts, like the empty-hit case
        restir.reservoir.setF(pixelIdx, 0u); // no reservoir here: zero contribution, not last-cycle stale radiance
        restir.gbuffer.setInvalidMotionVec(pixelIdx);
        restir.denoiserGuides.setGuides(pixelIdx, f3(0.0f), f3(0.0f), f2(0.0f)); // env miss: no surface albedo/normal/flow
        save_rng(pixelIdx, &localState, nullptr);
        return;
    }

    int materialID;
    float2 uv;
    float3 shadingPos;
    bool backface;
    float3 geoNormal;
    float3 normal;
    float3 ImplicitEmission;
    float lod;
    const Triangle& tri = params.shadeContext.scene[hitData.primId];

    // Grow the cone across the primary segment (eye -> hit), then bake this hit's LOD.
    coneWidth += coneSpread * hitData.t;

    getDataGeoLOD(
        tri,
        params.shadeContext,
        hitData.barycentrics,
        r.direction,
        coneWidth,

        materialID,
        uv,
        shadingPos,
        geoNormal,
        normal,
        backface,
        ImplicitEmission,
        lod,

        hitData.instanceId
    );

    if (IS_DEBUG_PIXEL(x, y)) {
        DEBUG_PRINTF("candidate gen shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
    }

    float3 albedo;

    getAlbedo(
        params.shadeContext.materials,
        materialID,
        params.shadeContext.textures,
        uv,
        albedo,
        lod
    );

    restir.gbuffer.setGeometry(pixelIdx, normal, hitData.t, materialID, albedo);

    float2 currPixelPos = make_float2((float)x + __half2float(jitter.x), (float)y + __half2float(jitter.y));
    float2 lastPixelPos;

    // Temporal-denoiser flow, in pixels. Empirically OptiX wants (curr - prev) here
    // (== our gbuffer MV); the (prev - curr) the docs suggest smears flat surfaces
    // under motion. If ghosting reappears, flip this back to lastPixelPos-currPixelPos.
    // Zero on frame 0 (no history; gated by temporalModeUsePreviousLayers).
    float2 denoiserFlow = f2(0.0f);
    if (params.frame_index != 0) {
        restir.lastFrameCamera.worldToRaster(shadingPos, lastPixelPos);
        restir.gbuffer.setMotionVec(pixelIdx, currPixelPos - lastPixelPos); // validity is double checked in temporal phase
        denoiserFlow = currPixelPos - lastPixelPos;
    }

    // Still in the pre-SER (coalesced) write region, next to setGeometry.
    restir.denoiserGuides.setGuides(pixelIdx, albedo, geoNormal, denoiserFlow); // geometric normal, full-precision albedo, flow

    primaryFootprint =
        (RECON_FOOTPRINT_C_CONSTANT / 100.0f) *
        (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));

    float3 contribution = backface ? f3(0.0f) : ImplicitEmission;

    if (luminance(contribution) > EPSILON) {
#if ACCUMULATE_FRAMES == 1
        params.accum_buffer[pixelIdx] += f4(contribution);
#else
        params.accum_buffer[pixelIdx] = f4(contribution);
#endif
        restir.gbuffer.setSkipShadeFlag(pixelIdx); // still run a reservoir and reuse normally, but dont display the reservoir this frame
    }

    float3 incomingDirLocal;
    toLocal(r.direction, normal, incomingDirLocal);

    bool currDelta = params.shadeContext.materials[materialID].isSpecular;

    // Handle DI, special case
    if (!currDelta) {
        // This block ALWAYS consumes 5 + 1 = 6 rng calls
        float3 lightNormal;
        float3 emission;
        float3 shadingPosToLightNormalized;
        float t_max;
        float pdf_nee;

        uint32_t neePrimID; // 0xFFFFFFFF for env, otherwise the triangle primID
        uint32_t lightInstanceID;
        float2 neeBarycentrics;

        bool sampledEnv = sample_ReSTIR_rc_data(params.shadeContext.lightSampler,
            rand(&localState), rand4(&localState),
            shadingPos,
            params.shadeContext.vertices,
            emission,
            shadingPosToLightNormalized,
            lightNormal,
            t_max,
            pdf_nee,
            neePrimID,
            neeBarycentrics,
            lightInstanceID,
            params.shadeContext.transformationMatrices
        );

        float3 shadingPosToLightLocal;
        toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

        bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
        bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
        if (!surfaceBackface && !lightBackface) {
            float bsdfPDF;

            pdf_eval(
                params.shadeContext.materials,
                materialID,
                params.shadeContext.textures,
                incomingDirLocal,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                bsdfPDF,
                uv,
                lod
            );

            float3 f_val_nee;
            f_eval(
                params.shadeContext.materials,
                materialID,
                params.shadeContext.textures,
                incomingDirLocal,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                f_val_nee,
                uv,
                TRANSPORTMODE_RADIANCE, lod
            );

            float3 contribution;
            float misWeight;

            float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
            float cosSurface = dot(normal, shadingPosToLightNormalized);
            if (sampledEnv) {
                contribution = f_val_nee * emission * cosSurface / pdf_nee;
                misWeight = powerHeuristicTwoStrategy(
                    pdf_nee,
                    bsdfPDF
                );
            } else {
                contribution =
                    f_val_nee * emission * cosLight * // NEE contribution
                    cosSurface / (pdf_nee * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                misWeight = powerHeuristicTwoStrategy(
                    (t_max * t_max) * pdf_nee / cosLight, // convert area pdf to SA
                    bsdfPDF // alt strat
                );
            }

            bool occluded = traceVisibility(
                params,
                Ray((shadingPos + (dot(shadingPosToLightNormalized, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON), shadingPosToLightNormalized),
                t_max * (1.0f - EPSILON2)
            );

            if (!occluded) {

                float w_i = targetFunction(contribution * misWeight);

                w_sum += w_i;

                float roll = rand(&localState);
                if (w_sum > 0.0f && roll < w_i / w_sum && t_max > EPSILON3) {

                    F = toRGB9E5(contribution * misWeight);
                    pathFlags = packPathFlags(
                        1,          // M = 1
                        2,          // Path Length
                        2,          // Rc vertex index (forces the light to be the rc vertex, k=d)
                        sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D : PATH_TYPE_NEE_AREA_K_EQ_D
                    );

                    //neepdf = (t_max * t_max * pdf_nee) / cosLight;
                    neepdf = pdf_nee; // must store origin measure pdf for k=d

                    actualRcVertexGeometry = packRcGeometry(
                        neePrimID,  // Also flags whether or not it is an environment or area light via sentinel value
                        neeBarycentrics,
                        shadingPosToLightNormalized,   // undefined for k=d, but we store the direction of the sampled dir
                        lightInstanceID
                    );
                    
                    rcRadiance = toRGB9E5(emission / neepdf); // RGB9E5-range encode; decoded by *neepdf in evaluateHybridShift

                    actualCachedJacobian = 1.0f; // di case.
                }
            } else {
                rand(&localState);
            }

        } else {
            rand(&localState);
        }
    }

    float3 outgoing;
    float3 f_val_bsdf;
    float pdf_bsdf;
    float pdf_bsdf_marg;

    sample_f_eval_lobe(
        localState,
        params.shadeContext.materials,
        materialID,
        params.shadeContext.textures,
        incomingDirLocal,
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        backface,
        outgoing,
        f_val_bsdf,
        pdf_bsdf,
        pdf_bsdf_marg,
        uv,
        TRANSPORTMODE_RADIANCE,
        lod
    );

    if (pdf_bsdf < EPSILON)
    {
        goto finalize_pixel;
    }

    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    float rr_roll = rand(&localState);
    if (rr_roll > p) {
        goto finalize_pixel;
    }
    throughput /= p;

    throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
    lastCosine = fabsf(outgoing.z);

    // This surface's contribution to the cone spread for the NEXT segment (rougher
    // surfaces defocus the cone faster; ~0 for mirrors, so specular stays sharp).
#if USE_RAY_CONES
    coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif

    toWorld(outgoing, normal, outgoing);

    r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
    r.direction = outgoing;

    prevDelta = currDelta;
    lastPDF = pdf_bsdf;
    lastPDF_marg = pdf_bsdf_marg;
    #if DEBUG_MODE == 1
    lastPOS_GETRIDOFME = shadingPos;
    #endif
}
    // reorder with empty flag: carries all alive threads to new homes while the returned threads
    // are left behind
    optixReorder(0u, 0u);

    for (int depth = 1; depth < params.max_depth; depth++)
    {
        SurfaceHit hitData = traceClosest(params, r);
        if (!hitData.isHit) // ENVIRONMENT
        {
            float pdf_sampleLight = params.shadeContext.lightSampler.evaluateEnvPdf(r.direction);
            float3 envEmission = params.shadeContext.lightSampler.envMap.sampleDir(r.direction);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF_marg, // primary strategy (marginal bsdf pdf: MIS cares about the physical ray)
                pdf_sampleLight // alternate strategy
            );

            float w_i = targetFunction(throughput * envEmission * misWeight);
            w_sum += w_i;

            float roll = rand(&localState);
            if (w_sum > 0.0f && roll < w_i / w_sum) {
                F = toRGB9E5(throughput * envEmission * misWeight);

                uint32_t pathType;
                uint32_t rcInd;
                if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                    // k = d
                    neepdf = pdf_sampleLight;
                    actualRcVertexGeometry = packRcGeometry(
                        0xFFFFFFFF, // flags a env hit
                        f2(0.0f),   // undefined for env hit
                        r.direction,   // undefined for k=d, but we store the direction of the sampled dir
                        0xFFFFFFFF // there is no instance ID
                    );
                    rcRadiance = toRGB9E5(envEmission / neepdf); // RGB9E5-range encode; decoded by *neepdf in evaluateHybridShift
                    actualCachedJacobian = prevDelta ? 1.0f : lastPDF; // direction copy for that case (if not direction copy, this variable isnt neccesary)
                    pathType = PATH_TYPE_BSDF_ENV_K_EQ_D;
                    rcInd = prevDelta ?
                        FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY : // direction copy is impossible if the prev vertex was full specular
                        FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_DIRECTION_COPY; // direction copy for environment map lowers variance
                    rcInd = FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY;
                    } else if (pathRcVertexIndex == depth) {
                    // k = d - 1
                    // This means the previous iteration, the previous vertex was marked as the rc vertex, thus, the rcvertexgeometry
                    // holds a rcWi that points towards this current vertex, which is correct.

                    neepdf = pdf_sampleLight;
                    // raw emission, RGB9E5-range encoded; decoded by *neepdf in evaluateHybridShift
                    actualRcVertexGeometry = pathRcVertexGeometry;
                    rcRadiance = toRGB9E5(envEmission / neepdf);
                    actualCachedJacobian = pathCachedJacobian;
                    pathType = PATH_TYPE_BSDF_ENV_K_EQ_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                } else {
                    // k < d - 1
                    actualRcVertexGeometry = pathRcVertexGeometry;
                    rcRadiance = toRGB9E5(suffixThroughput * envEmission * misWeight);
                    actualCachedJacobian = pathCachedJacobian;
                    neepdf = -1.0f;
                    pathType = PATH_TYPE_BSDF_ENV_K_LESS_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                }

                pathFlags = packPathFlags(
                    1,          // M = 1
                    depth + 1,          // Path Length
                    rcInd,
                    pathType
                );
            }

            goto finalize_pixel;
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 geoNormal;
        float3 normal;
        float3 emission;
        float lod;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        // Grow the cone across this segment (prev vertex -> this hit), then bake LOD.
        coneWidth += coneSpread * hitData.t;

        getDataGeoLOD(
            tri,
            params.shadeContext,
            hitData.barycentrics,
            r.direction,
            coneWidth,

            materialID,
            uv,
            shadingPos,
            geoNormal,
            normal,
            backface,
            emission,
            lod,

            hitData.instanceId
        );


        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("candidate gen shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
        }



        if (IS_DEBUG_PIXEL(x, y)) {
            #if DEBUG_MODE == 1
            DEBUG_DRAWLINE(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos,
                (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) ?
                f3(1.0f, 0.0f, 0.0f) :
                f3(0.0f, 1.0f, 0.0f), 3
            );
            #endif
        }


        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        float3 outgoing;
        float3 f_val_bsdf;
        float pdf_bsdf;
        float pdf_bsdf_marg;

        sample_f_eval_lobe(
            localState,
            params.shadeContext.materials,
            materialID,
            params.shadeContext.textures,
            incomingDirLocal,
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            backface,
            outgoing,
            f_val_bsdf,
            pdf_bsdf,
            pdf_bsdf_marg,
            uv,
            TRANSPORTMODE_RADIANCE,
            lod
        );

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Check dual footprint for rc connectability
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;

        if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) { // to catch undefined pdfs
            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper

            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                pathRcVertexIndex = depth + 1;

                float3 out_world;
                toWorld(outgoing, normal, out_world);

                pathRcVertexGeometry = packRcGeometry(
                    hitData.primId,
                    hitData.barycentrics,
                    out_world,
                    hitData.instanceId
                );

                // p_(x_k-1 -> x_k) * G(x_k-1 -> x_k) * P(x_k -> x_k+1)
                pathCachedJacobian = lastPDF * pdf_bsdf * fabsf(incomingDirLocal.z) / (hitData.t * hitData.t);
            }
        }

        float3 lightEmission = backface ? f3(0.0f) : emission;

        if (luminance(lightEmission) > EPSILON) {
            float sampleLightPDF = params.shadeContext.lightSampler.evaluateMeshPdf(tri);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF_marg, // primary strategy (marginal bsdf pdf for MIS)
                (hitData.t * hitData.t * sampleLightPDF / (fabsf(incomingDirLocal.z))) // alternate strategy
            );

            float w_i = targetFunction(throughput * lightEmission * misWeight);
            w_sum += w_i;

            float roll = rand(&localState);
            if (w_sum > 0.0f && roll < w_i / w_sum && hitData.t > EPSILON3) {
                F = toRGB9E5(throughput * lightEmission * misWeight);

                uint32_t pathType;
                uint32_t rcInd;
                if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                    // k = d
                    neepdf = sampleLightPDF; // must store area pdf for k=d
                    actualRcVertexGeometry = packRcGeometry(
                        hitData.primId,
                        hitData.barycentrics,
                        f3(0.0f),
                        hitData.instanceId
                    );
                    rcRadiance = toRGB9E5(suffixThroughput * lightEmission / neepdf); // RGB9E5-range encode; decoded by *neepdf in evaluateHybridShift
                    //actualCachedJacobian = lastPDF * (fabsf(incomingDirLocal.z) / (hitData.t * hitData.t));
                    actualCachedJacobian = 1.0f; // we rely on a full replay for this
                    pathType = PATH_TYPE_BSDF_AREA_K_EQ_D;
                    rcInd = FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY;
                } else if (pathRcVertexIndex == depth) {
                    // k = d - 1
                    neepdf = (hitData.t * hitData.t * sampleLightPDF) / fabsf(incomingDirLocal.z);
                    // raw emission, RGB9E5-range encoded; decoded by *neepdf in evaluateHybridShift
                    actualRcVertexGeometry = pathRcVertexGeometry;
                    rcRadiance = toRGB9E5(lightEmission / neepdf);
                    actualCachedJacobian = pathCachedJacobian;
                    pathType = PATH_TYPE_BSDF_AREA_K_EQ_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                } else {
                    // k < d - 1
                    actualRcVertexGeometry = pathRcVertexGeometry;
                    rcRadiance = toRGB9E5(suffixThroughput * lightEmission * misWeight);
                    actualCachedJacobian = pathCachedJacobian;
                    pathType = PATH_TYPE_BSDF_AREA_K_LESS_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                    neepdf = -1.0f;
                }

                pathFlags = packPathFlags(
                    1,          // M = 1
                    depth + 1,          // Path Length
                    rcInd,
                    pathType    // This was chosen via NEE
                );
            }
        }


        if (!currDelta) {
            float3 lightNormal;
            float3 emission;
            float3 shadingPosToLightNormalized;
            float t_max;
            float pdf_nee;

            uint32_t neePrimID; // 0xFFFFFFFF for env, otherwise the triangle primID
            uint32_t lightInstanceID;
            float2 neeBarycentrics;

            bool sampledEnv = sample_ReSTIR_rc_data(params.shadeContext.lightSampler,
                rand(&localState), rand4(&localState),
                shadingPos,
                params.shadeContext.vertices,
                emission,
                shadingPosToLightNormalized,
                lightNormal,
                t_max,
                pdf_nee,
                neePrimID,
                neeBarycentrics,
                lightInstanceID,

                params.shadeContext.transformationMatrices
            );

            float3 shadingPosToLightLocal;
            toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

            bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
            bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
            if (!surfaceBackface && !lightBackface) {
                float bsdfPDF;

                pdf_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
                    incomingDirLocal,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    bsdfPDF,
                    uv,
                    lod
                );

                float3 f_val_nee;
                f_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
                    incomingDirLocal,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    f_val_nee,
                    uv,
                    TRANSPORTMODE_RADIANCE, lod
                );

                float3 contributionSansThroughput;
                float misWeight;

                float cosLight;
                float cosSurface;
                if (sampledEnv) {
                    cosSurface = dot(normal, shadingPosToLightNormalized);
                    contributionSansThroughput = f_val_nee * emission * cosSurface / pdf_nee;
                    misWeight = powerHeuristicTwoStrategy(
                        pdf_nee,
                        bsdfPDF
                    );
                } else {
                    cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                    cosSurface = dot(normal, shadingPosToLightNormalized);

                    contributionSansThroughput =
                        f_val_nee * emission * cosLight * // NEE contribution
                        cosSurface / (pdf_nee * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                    misWeight = powerHeuristicTwoStrategy(
                        (t_max * t_max) * pdf_nee / cosLight, // convert area pdf to SA
                        bsdfPDF // alt strat
                    );
                }

                bool occluded = traceVisibility(
                    params,
                    Ray((shadingPos + (dot(shadingPosToLightNormalized, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON), shadingPosToLightNormalized),
                    t_max * (1.0f - EPSILON2)
                );

                if (!occluded) {
                    float w_i = targetFunction(throughput * contributionSansThroughput * misWeight);
                    w_sum += w_i;

                    float roll = rand(&localState);
                    if (w_sum > 0.0f && roll < w_i / w_sum && t_max > EPSILON3) {
                        F = toRGB9E5(throughput * contributionSansThroughput * misWeight);

                        uint32_t pathType;
                        if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                            // k = d
                            neepdf = pdf_nee; // must store original measure for k=d
                            actualRcVertexGeometry = packRcGeometry(
                                neePrimID,
                                neeBarycentrics,
                                shadingPosToLightNormalized,
                                lightInstanceID
                            );
                            rcRadiance = toRGB9E5(emission / neepdf); // emission ONLY, RGB9E5-range encoded; decoded by *neepdf in evaluateHybridShift
                            actualCachedJacobian = 1.0f;
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D : PATH_TYPE_NEE_AREA_K_EQ_D;
                        } else if (pathRcVertexIndex == depth + 1) {
                            // k = d - 1
                            neepdf = sampledEnv ? (pdf_nee) : ((t_max * t_max * pdf_nee) / cosLight);
                            // raw emission, RGB9E5-range encoded; decoded by *neepdf in evaluateHybridShift
                            rcRadiance = toRGB9E5(suffixThroughput * emission / neepdf);
                            actualRcVertexGeometry = updateRcVertexWi(pathRcVertexGeometry, shadingPosToLightNormalized);
                            actualCachedJacobian = lastPDF * (fabsf(incomingDirLocal.z) / (hitData.t * hitData.t));
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D_MINUS_1 : PATH_TYPE_NEE_AREA_K_EQ_D_MINUS_1;
                        } else {
                            // k < d - 1
                            actualRcVertexGeometry = pathRcVertexGeometry;
                            rcRadiance = toRGB9E5(suffixThroughput * contributionSansThroughput * misWeight);
                            actualCachedJacobian = pathCachedJacobian;
                            neepdf = -1.0f;
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_LESS_D_MINUS_1 : PATH_TYPE_NEE_AREA_K_LESS_D_MINUS_1;
                        }

                        pathFlags = packPathFlags(
                            1,          // M = 1
                            depth + 2,          // Path Length
                            (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) ? // Rc vertex index (sentinel means it hasnt been found yet)
                                (depth + 2) : // not yet found, so set this sampled one as rc vertex; k=d special case
                                pathRcVertexIndex, // rc vertex found; save it
                            pathType    // This was chosen via NEE
                        );
                    }
                } else {
                    rand(&localState);
                }
            } else {
                rand(&localState);
            }
        }

        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);

        if (rand(&localState) > p)   // survive with probability p
        {
            goto finalize_pixel;
        }
        throughput /= p;
        if (pathRcVertexIndex != FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND && depth + 1> pathRcVertexIndex)
            suffixThroughput /= p;
        if (pdf_bsdf < EPSILON)
        {
            goto finalize_pixel;
        }

        throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
        if (pathRcVertexIndex != FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND && depth + 1 > pathRcVertexIndex)
            suffixThroughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;

        toWorld(outgoing, normal, outgoing);

        r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf_bsdf;
        lastPDF_marg = pdf_bsdf_marg;
        lastCosine = fabsf(dot(outgoing, normal));

        // This surface's contribution to the cone spread for the next segment.
#if USE_RAY_CONES
        coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif
        #if DEBUG_MODE == 1
        lastPOS_GETRIDOFME = shadingPos;
        #endif
    }

finalize_pixel:
    if (w_sum <= 0.0f) {
        restir.reservoir.setW(pixelIdx, 1.0f);
        restir.reservoir.setCachedJacobian(pixelIdx, -1.0f);
        restir.reservoir.setPathFlags(pixelIdx, packPathFlags(1, 0, 0, 0));
        restir.reservoir.setF(pixelIdx, 0u); // empty sample: zero contribution, not last-cycle stale radiance
        return;
    }

    float p_hat = targetFunction(fromRGB9E5(F));
    float W = (p_hat > EPSILON) ? (w_sum / p_hat) : 0.0f;

    restir.reservoir.saveReservoirFinal(
        pixelIdx,
        W,
        F,
        pathFlags,
        actualRcVertexGeometry,
        rcRadiance,
        actualCachedJacobian,
        neepdf
    );
}

extern "C" __global__ void __raygen__restirTemporalReuse() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers



    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;
    
    half2 mv = restir.gbuffer.getMV(pixelIdx);
    int2 historyCoord = make_int2(-1, -1);
    uint32_t reorderHint = 0u;

    uint32_t mvBits = reinterpret_cast<const uint32_t&>(mv);
    if (mvBits != 0xFFFFFFFF) { // 0xFFFFFFFF = no reprojectable surface (env miss). Skip-shade pixels keep a real MV and reuse normally.
        if (isHistoryValid(allParams, make_int2(x, y), mv, historyCoord)) { // check primary movtion vec
            reorderHint = 0xFFFFFFFF;
        } 
        #if TEMPORAL_USE_DUAL_MV == 1
        else {
            mv = restir.gbuffer.getDualMV(pixelIdx);
            if (isHistoryValid(allParams, make_int2(x, y), mv, historyCoord)) { // check dual motion vec
                reorderHint = 0xFFFFFFFF;
            }
        }
        #endif
    }

    // Optionally go one step beyond stream compaction, and sort by morton code
#if TEMPORAL_SER_SORT_MORTON_CODE == 1
    uint32_t cx = x >> 5;
    uint32_t cy = y >> 5;

    uint32_t spatial_hint = (expandBits(cx) | (expandBits(cy) << 1)) & 0x7Fu;
    if (reorderHint != 0u) {
        reorderHint = spatial_hint | 0x80u;
    }

    optixReorder(reorderHint, 8);
#else
    optixReorder(reorderHint, 1);
#endif

    if (reorderHint == 0u)
        return;

    uint32_t historyIdx = historyCoord.x + historyCoord.y * params.w;

    if (IS_DEBUG_PIXEL(x, y)) {
        DEBUG_PRINTF("Fresh candidate gen reservoir: \n");
        DEBUG_PRINT_PIXEL(restir.reservoir, restir.gbuffer, pixelIdx, params.frame_index);
        DEBUG_PRINTF("History reservoir: \n");
        DEBUG_PRINT_PIXEL(restir.lastFrameReservoir, restir.prevGbuffer, historyIdx, params.frame_index - 1);
    }

#if USE_DUPLICATION_MAP
    uint8_t dupe_val = __ldg(&restir.duplication_map[historyIdx]);
    float D = (float)dupe_val / 255.0f;

    float cCap = lerp(LERP_MCAP, 1.0f, powf(D, 0.1f));
#else
    // Duplication map disabled: original hard M-cap (Lin 2022), cCap = LERP_MCAP.
    float cCap = LERP_MCAP;
#endif

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Proceed to perform shift
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    // One grouped 32B load of the whole history shift descriptor (was 4+ separate
    // dependent __ldg's when the reservoir was SOA).
    uint32_t hist_M_int;
    uint32_t hist_pathLength;
    uint32_t hist_rcVertexIndex;
    TechniqueType hist_type;
    uint32_t hist_rcPrimID;
    float2 hist_rcBarycentrics;
    float3 hist_rcWi;
    float3 hist_rcRadiance;
    uint32_t hist_rcInstanceID;
    float hist_cachedJacobianDenom;
    float hist_cachedNeePdf;
    restir.lastFrameReservoir.getShiftDescriptor(historyIdx,
        hist_M_int, hist_pathLength, hist_rcVertexIndex, hist_type,
        hist_rcPrimID, hist_rcBarycentrics, hist_rcWi, hist_rcRadiance, hist_rcInstanceID,
        hist_cachedJacobianDenom, hist_cachedNeePdf);

    float hist_M = fminf(cCap, hist_M_int);

    uint32_t hist_seed = restir.lastFrameReservoir.getSeed_notstreaming(historyIdx);

    if (IS_DEBUG_PIXEL(x, y)) {
        DEBUG_PRINTF("frame %u at the start of temporal seed: %u\n", params.frame_index, hist_seed);
    }

    // ==============================================================================
    // 1. UNPACK CURRENT PATH DATA (Needed for the Backward Shift)
    // ==============================================================================
    // One grouped 32B load of the whole current shift descriptor.
    uint32_t curr_M, curr_pathLength, curr_rcVertexIndex;
    TechniqueType curr_type;
    uint32_t curr_rcPrimID;
    float2 curr_rcBarycentrics;
    float3 curr_rcWi;
    float3 curr_rcRadiance;
    uint32_t curr_rcInstanceID;
    float curr_cachedJacobianDenom;
    float curr_cachedNeePdf;
    restir.reservoir.getShiftDescriptor(pixelIdx,
        curr_M, curr_pathLength, curr_rcVertexIndex, curr_type,
        curr_rcPrimID, curr_rcBarycentrics, curr_rcWi, curr_rcRadiance, curr_rcInstanceID,
        curr_cachedJacobianDenom, curr_cachedNeePdf);

    float curr_W; float3 curr_F;
    restir.reservoir.getWF(pixelIdx, curr_W, curr_F);
    float curr_p_hat = targetFunction(curr_F);

    uint32_t curr_seed = restir.reservoir.getSeed_notstreaming(pixelIdx);

    uint32_t new_M = curr_M + (uint32_t)hist_M;
    float hist_W; float3 hist_F;
    restir.lastFrameReservoir.getWF(historyIdx, hist_W, hist_F);
    float hist_p_hat = targetFunction(hist_F);

    // ==============================================================================
    // 2. THE FORWARD SHIFT (History Path -> Current Pixel)
    // ==============================================================================
    ShiftResult fwdResult;
    if (hist_M_int > 0 && hist_cachedJacobianDenom != -1.0f) {
        fwdResult = evaluateHybridShift<false>(
            allParams,
            x, y,
            hist_seed, hist_pathLength, hist_rcVertexIndex, hist_type,
            hist_rcPrimID, hist_rcInstanceID, hist_rcBarycentrics, hist_rcWi, hist_rcRadiance,
            hist_cachedNeePdf, hist_cachedJacobianDenom
        );

        if (fwdResult.isValid && fwdResult.new_cached_jacobian < EPSILON3) {
            fwdResult = {false, f3(0), 0.0f, 0.0f};
        }
    } else {
        fwdResult = {false, f3(0), 0.0f, 0.0f};
    }


    // ==============================================================================
    // 3. THE BACKWARD SHIFT (Current Path -> History Pixel)
    // ==============================================================================
    ShiftResult bwdResult;
    bool needs_bwd_shift = (curr_cachedJacobianDenom != -1.0f);

    if (IS_DEBUG_PIXEL(x, y) && !needs_bwd_shift) {
        DEBUG_PRINTF("Backwards shift judged to not be needed.\n");
    }
    //optixReorder(needs_bwd_shift, 1);

    if (needs_bwd_shift) {
        // just for now, we want to print out everything.
        bwdResult = evaluateHybridShift<true>(
            allParams,
            historyCoord.x, historyCoord.y, // Backward shift originates from the history pixel
            curr_seed, curr_pathLength, curr_rcVertexIndex, curr_type,
            curr_rcPrimID, curr_rcInstanceID, curr_rcBarycentrics, curr_rcWi, curr_rcRadiance,
            curr_cachedNeePdf, curr_cachedJacobianDenom
        );
    } else {
        bwdResult = {false, f3(0), 0.0f, 0.0f};
    }


    // ==============================================================================
    // 4. UNBIASED MIS WEIGHT CALCULATIONS (Factored to avoid NaN)
    // ==============================================================================
    float w_tentative = 0.0f;
    float mis_weight_curr = 1.0f; // Safely defaults to 1.0 if backward shift fails

    float fwd_phat = targetFunction(fwdResult.contribution);
    // A. Evaluate History Path MIS (Evaluated at Y_h = fwdResult)
    if (fwdResult.isValid) {
        // denom = M_c * p_c(Y_h) * J_{h->c} + M_h * p_h(X_h)
        float denom_hist = (curr_M * fwd_phat * fwdResult.jacobian) + (hist_M * hist_p_hat);
        if (denom_hist > 0.0f) {
            float mis_weight_hist = (hist_M * hist_p_hat) / denom_hist;
            w_tentative = mis_weight_hist * fwd_phat * hist_W * fwdResult.jacobian;
        }
    }
    if (IS_DEBUG_PIXEL(x, y))
        DEBUG_PRINTF("forward shift resulted in a jacobian of %f\n forward shfit produced an F of: <%f, %f, %f>, new Jacobian Denom: %f\n", fwdResult.jacobian, fwdResult.contribution.x, fwdResult.contribution.y, fwdResult.contribution.z, fwdResult.new_cached_jacobian);

    float bwd_phat = targetFunction(bwdResult.contribution);
    // B. Evaluate Current Path MIS (Evaluated at X_c)
    if (bwdResult.isValid) {
        // denom = M_c * p_c(X_c) + M_h * p_h(X_{c->h}) * J_{c->h}
        float denom_curr = (curr_M * curr_p_hat) + (hist_M * bwd_phat * bwdResult.jacobian);
        if (denom_curr > 0.0f) {
            mis_weight_curr = (curr_M * curr_p_hat) / denom_curr;
        }
    }

    if (IS_DEBUG_PIXEL(x, y) && needs_bwd_shift) {
        DEBUG_PRINTF("backwards shift resulted in a jacobian of %f\n backwards shfit produced an F of: <%f, %f, %f>, new Jacobian Denom: %f\n", bwdResult.jacobian, bwdResult.contribution.x, bwdResult.contribution.y, bwdResult.contribution.z, bwdResult.new_cached_jacobian);

    }


    float w_curr_weighted = mis_weight_curr * curr_W * curr_p_hat;


    // ==============================================================================
    // 5. UNIFIED RESERVOIR UPDATE
    // ==============================================================================
    float w_sum = w_curr_weighted + w_tentative;
    RNGState refreshedLocalState = load_rng(hash_uint32(pixelIdx), hash_uint32(params.frame_index), hash_uint32(0), nullptr);

    bool history_won = false;
    if (w_sum > 0.0f && rand(&refreshedLocalState) < (w_tentative / w_sum)) {
        history_won = true;
    }

    if (history_won) {
        float W_final = (fwd_phat > 0.0f) ? (w_sum / fwd_phat) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        restir.reservoir.saveReservoirAll(
            pixelIdx,
            W_final,
            fwdResult.contribution,
            hist_seed,
            new_M,
            hist_pathLength,
            hist_rcVertexIndex,
            hist_type,
            hist_rcInstanceID,
            hist_rcPrimID,
            hist_rcBarycentrics,
            hist_rcWi,
            hist_rcRadiance,
            fwdResult.new_cached_jacobian,
            hist_cachedNeePdf
        );
    } else {
        float W_final = (curr_p_hat > 0.0f) ? (w_sum / curr_p_hat) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        if (curr_M == 0 || W_final == 0.0f) {
            restir.reservoir.setPathFlags(pixelIdx, packPathFlags(1, 0, 0, 0));
            restir.reservoir.setW_noCS(pixelIdx, 0.0f);
            restir.reservoir.setCachedJacobian(pixelIdx, -1.0f); // gate out of subsequent shifts, matching the candidate-gen empty case
            restir.reservoir.setF(pixelIdx, 0u);                 // avoid stale radiance leaking into p_hat
        } else {
            // Only M changes; repack from the fields we already unpacked (equivalent to updateM).
            restir.reservoir.setPathFlags(pixelIdx, packPathFlags(new_M, curr_pathLength, curr_rcVertexIndex, curr_type));
            restir.reservoir.setW_noCS(pixelIdx, W_final);
        }
    }
}

/**
 *
 */
extern "C" __global__ void __raygen__restirSpatialReuse() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    uint32_t texIdx = launch_index.z;
    int pixelIdx = y*params.w + x;

    uint32_t self_M, self_pathLength, self_rcVertexIndex;
    TechniqueType self_type;
    restir.reservoir.getPathFlags(pixelIdx, self_M, self_pathLength, self_rcVertexIndex, self_type);
    const float self_cachedJacobian = restir.reservoir.getCachedJacobian_globalLoad(pixelIdx);

    half2 mv = restir.gbuffer.getMV(pixelIdx);

    const int2 partnerCoord = get_paired_neighbor(
        make_int2(x, y), texIdx, params.frame_index,
        restir.reuseTextureSizes[texIdx],
        make_int2(params.w, params.h),
        restir.reuseTextures[texIdx]
    );

    // Pair acceptance: a property of (P, N), decided before touching any path.
    const bool pairAccepted =
           (partnerCoord.x >= 0) && (partnerCoord.y >= 0)
        && isSpatialPairAccepted(allParams, make_int2(x, y), partnerCoord);

    // Self-shiftability: a property of THIS pixel's path. An accepted pair whose
    // shift fails is still accepted -- it stays in c_tot and in the MIS denominators.
    const bool selfShiftable = pairAccepted
        && (self_M > 0u)
        && (self_cachedJacobian != -1.0f)
        && (reinterpret_cast<const uint32_t&>(mv) != 0xFFFFFFFF);
        
    uint32_t reorderHint = selfShiftable ? 0xFFFFFFFFu : 0u;
#if TEMPORAL_SER_SORT_MORTON_CODE == 1
    uint32_t spatial_hint = (expandBits(partnerCoord.x >> 5) |
                            (expandBits(partnerCoord.y >> 5) << 1)) & 0x7Fu;
    if (reorderHint != 0u) reorderHint = spatial_hint | 0x80u;
    optixReorder(reorderHint, 8);
#else
    optixReorder(reorderHint, 1);
#endif

    ShiftResult result;

    if (selfShiftable) {
        uint32_t self_rcPrimID, self_rcInstanceID; float2 self_rcBary; float3 self_rcWi, self_rcRadiance;
        restir.reservoir.getRcVertexGeometry_globalLoad(pixelIdx, self_rcPrimID, self_rcBary,
                                                        self_rcWi, self_rcRadiance, self_rcInstanceID);
        const uint32_t self_seed = restir.reservoir.getSeed_notstreaming(pixelIdx);
        float self_cachedNee = -1.0f;
        if (needNeePDF(self_type)) self_cachedNee = restir.reservoir.getCachedNEE_globalLoad(pixelIdx);

        // Replay OUR path at the PARTNER's pixel: T(self -> partner).
        result = evaluateHybridShift<false>(
            allParams,
            partnerCoord.x, partnerCoord.y,
            self_seed, self_pathLength, self_rcVertexIndex, self_type,
            self_rcPrimID, self_rcInstanceID, self_rcBary, self_rcWi, self_rcRadiance,
            self_cachedNee, self_cachedJacobian
        );
    }

    restir.shiftResultBuffer[texIdx].setResult(
        pixelIdx, 
        result.isValid, 
        result.contribution, 
        result.jacobian, 
        result.new_cached_jacobian
    );
}