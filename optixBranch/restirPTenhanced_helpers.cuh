#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "optixStructs.cuh"
#include "settings.cuh"
#include "restirPTenhanced_shiftHelpers.cuh"

template<bool isReverseShift>
__device__ __forceinline__ ShiftResult evaluateHybridShift(
    const PipelineParams& allParams,
    uint32_t x, uint32_t y,           // The pixel generating the offset ray
    uint32_t seed,                        // The RNG seed to replay
    uint32_t pathLength,                  // Base path data
    uint32_t rcVertexIndex,               // Base path data
    TechniqueType type,                       // Base path data
    uint32_t rcPrimID,                    // Base path data
    uint32_t rcInstanceID,
    float2 rcBarycentrics,                    // Base path data
    float3 rcWi,                              // Base path data
    float3 rcRadiance,                        // Base path data
    float cached_nee,                         // Base path data
    float jacobianDenom                       // Base path data
) {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    // k=d and k=d-1 store raw emission divided by the light pdf, so it fits in RGB9E5's range
    // (a bright sun overflows the ~65408 ceiling and loses its magnitude entirely). Decode here.
    // rcRadiance is by-value, so the caller's copy stays encoded for re-storage.
    if (needNeePDF(type) && cached_nee > 0.0f) {
        rcRadiance *= cached_nee;
    }

    uint32_t reorderHint = (rcVertexIndex == FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY) ? 0u : 0xFFFFFFFF;
    optixReorder(reorderHint, 1); // ser so good

    RNGState localState = load_rng(seed); // seed path using other pixel's start seed

    Ray r;
    if constexpr (!isReverseShift) {
        r = params.camera.generateCameraRay(localState, x, y);
    } else {
        r = restir.lastFrameCamera.generateCameraRay(localState, x, y);
    }

    if constexpr (!isReverseShift) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("frame %u using rng with state: %u for temporal reuse, replaying from %u, %u, with initial camera ray o(%f, %f, %f), d(%f, %f, %f)\n", params.frame_index, seed, x, y, r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);
        }
    }

    // Ray-cone texture LOD state, reconstructed on the shifted domain exactly like
    // candidate gen (deterministic in the replayed geometry, so the shifted path gets
    // its own correct LOD). coneWidth = cone radius at the current origin; coneSpread =
    // accumulated half-angle, seeded to the replaying camera's per-pixel angle.
    float coneWidth = 0.0f;
    float coneSpread;
    if constexpr (!isReverseShift) coneSpread = 2.0f * params.camera.fovScale / (float)params.h;
    else                           coneSpread = 2.0f * restir.lastFrameCamera.fovScale / (float)params.h;

    // handles all k=d bsdf cases except for environment hit with a non specular previous vertex
    // Note: a "full replay" is also done in NEE k=d paths, but its still handled in the other block since it shares the rc shadow ray logic
    if (rcVertexIndex == FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY) {
        float3 throughput = f3(1.0f);

        // these three may be unnecesary
        float lastPDF;
        float lastPDF_marg; // marginal bsdf pdf, for the bsdf-vs-light MIS at the ending vertex
        bool prevDelta;
        float lastCosine;

        float3 lastPos;
        float3 lastNormal;
        int lastMaterialID;
        float2 lastUV;
        bool lastBackface;
        float3 lastInDirLocal;

        float primaryFootprint;
    {
        SurfaceHit hitData = traceClosest(params, r);

        if (!hitData.isHit) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("SHIFT ABORT [%s]: primary ray miss for full replay\n", isReverseShift ? "REVERSE" : "FORWARD");
            }
            return {false, f3(0), 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
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

        coneWidth += coneSpread * hitData.t; // grow cone across this segment, then bake LOD

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
        if constexpr (!isReverseShift) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("replaying shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        primaryFootprint =
            (RECON_FOOTPRINT_C_CONSTANT / 100.0f) *
            (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));

        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        lastPos = shadingPos;
        lastMaterialID = materialID;
        lastUV = uv;
        lastBackface = backface;
        lastInDirLocal = incomingDirLocal;
        lastNormal = normal;

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;
        if (!currDelta) {
            // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
            rand(&localState);
            rand4(&localState);

            rand(&localState); // to do the reservoir roll
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

        if (pdf_bsdf < EPSILON && !K_is_D(type) && (pathLength != 2))
        {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("SHIFT ABORT [%s]: full replay scattering pdf zero for primary hit\n", isReverseShift ? "REVERSE" : "FORWARD");
            }
            return {false, f3(0), 0.0f, 0.0f}; // something went wrong, cant finish temporal shift
        }

        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        float rr_roll = rand(&localState);
        if (rr_roll > p) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("SHIFT ABORT [%s]: FULL REPLAY RR failed", isReverseShift ? "REVERSE" : "FORWARD");
            }
            return {false, f3(0), 0.0f, 0.0f};
        }
        throughput /= p;

        throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
        lastCosine = fabsf(outgoing.z);
#if USE_RAY_CONES
        coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif
        toWorld(outgoing, normal, outgoing);

        r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf_bsdf;
        lastPDF_marg = pdf_bsdf_marg;
    }
        #if DEBUG_MODE == 1
        float3 lastPOS_GETRIDOFME = r.origin;
        #endif
        // depth + 1 is the "index" of the curr vertex, so this stops at y_k-1
        for (int depth = 1; depth + 1 < pathLength; depth++) {
            SurfaceHit hitData = traceClosest(params, r);

            if (!hitData.isHit) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: full replay secondary ray missed scene\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
            }

            int materialID;
            float2 uv;
            float3 shadingPos;
            bool backface;
            float3 geoNormal;
            float3 normal;
            float3 lightEmission;
            float lod;
            const Triangle& tri = params.shadeContext.scene[hitData.primId];

            coneWidth += coneSpread * hitData.t; // grow cone across this segment, then bake LOD

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
                lightEmission,
                lod,

                hitData.instanceId
            );

            if constexpr (!isReverseShift) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_DRAWLINE(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos,
                        f3(0.0f, 1.0f, 1.0f), 3
                    );
                    DEBUG_PRINTF("replaying shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
                }
            }

            float3 incomingDirLocal;
            toLocal(r.direction, normal, incomingDirLocal);

            // needed for recon
            if (depth + 1 == rcVertexIndex - 1) {
                lastPos = shadingPos;
                lastMaterialID = materialID;
                lastUV = uv;
                lastBackface = backface;
                lastInDirLocal = incomingDirLocal;
                lastNormal = normal;
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

            lightEmission = backface ? f3(0.0f) : lightEmission;
            if (luminance(lightEmission) > 0.0f) {
                rand(&localState); // for the reservoir roll
            }

            bool currDelta = params.shadeContext.materials[materialID].isSpecular;

            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper

            bool isValid = true;
            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                isValid = false;
            }

            if (!isValid) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: full replay failed reciprocality on dual footprint\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }

            if (!currDelta) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // for the reservoir roll
            }

            float lum = luminance(throughput);
            float p = clamp(lum, 0.05f, 1.0f);
            float rr_roll = rand(&localState);
            if (rr_roll > p) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: FULL REPLAY RR failed", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }
            throughput /= p;

            if (pdf_bsdf < EPSILON && !K_is_D(type) && (pathLength != depth + 2))
            {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: full replay scattering pdf zero for secondary bounce\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }

            throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
            toWorld(outgoing, normal, outgoing);
            r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
            r.direction = outgoing;

            prevDelta = currDelta;
            lastPDF = pdf_bsdf;
            lastPDF_marg = pdf_bsdf_marg;
            lastCosine = fabsf(dot(outgoing, normal));
#if USE_RAY_CONES
            coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif
            #if DEBUG_MODE == 1
            lastPOS_GETRIDOFME = shadingPos;
            #endif
        }

        // now we are on the last bounce
        SurfaceHit hitData = traceClosest(params, r);

        if (is_env(type)) {
            if (hitData.isHit) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: last full replay hit scene when it should hit env\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }
            ShiftResult result;

            float pdf_sampleLight = params.shadeContext.lightSampler.evaluateEnvPdf(r.direction);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF_marg, // primary strategy (marginal bsdf pdf for MIS)
                pdf_sampleLight // alternate strategy
            );

            result.contribution =
                throughput *
                params.shadeContext.lightSampler.envMap.sampleDir(r.direction) *
                misWeight;

            result.isValid = true;
            result.jacobian = 1.0f;
            result.new_cached_jacobian = 1.0f;

            return result;
        } else {
            if (!hitData.isHit) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: last full replay hit env when it should hit scene\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }
        }

        if (!is_bsdf(type)) {
            DEBUG_PRINTF("Error: full replay for bsdf has wrongly packed type or wrongly chosen rcvertexindex flag\n");
            return {false, f3(0), 0.0f, 0.0f};
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 geoNormal;
        float3 normal;
        float3 lightEmission;
        float lod;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        coneWidth += coneSpread * hitData.t; // final segment to the light vertex

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
            lightEmission,
            lod,

            hitData.instanceId
        );

        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        if constexpr (!isReverseShift) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_DRAWLINE(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos,
                    f3(0.0f, 1.0f, 1.0f), 3
                );
                DEBUG_PRINTF("replaying shading pos depth %u: %f, %f, %f\n", pathLength-1, shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        lightEmission = backface ? f3(0.0f) : lightEmission;
        if (luminance(lightEmission) > 0.0f) {
            float sampleLightPDF = params.shadeContext.lightSampler.evaluateMeshPdf(tri);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF_marg, // primary strategy (marginal bsdf pdf for MIS)
                (hitData.t * hitData.t * sampleLightPDF / (fabsf(incomingDirLocal.z))) // alternate strategy
            );

            ShiftResult result;

            result.contribution = throughput * lightEmission * misWeight;

            result.isValid = true;
            result.jacobian = 1.0f;
            result.new_cached_jacobian = 1.0f;

            return result;

        } else {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("SHIFT ABORT [%s]: full replay ended on non emissive surface\n", isReverseShift ? "REVERSE" : "FORWARD");
            }
            return {false, f3(0), 0.0f, 0.0f};
        }

    } else { // handles all nee k=d cases, and the k=d environment case when the previous vertex is not specular
        float3 throughput = f3(1.0f);

        // these three may be unnecesary
        float lastPDF;
        bool prevDelta;
        float lastCosine;

        float3 lastPos;
        float3 lastNormal;

        // for x_k-1 emissive flag, when reconstructing rng at x_k-1
        uint32_t lastMaterialID_packedWithEmissiveFlag;
        float2 lastUV;
        bool lastBackface;
        float3 lastInDirLocal;

        // Cone state captured AS OF x_{k-1}: coneWidth at x_{k-1}, the spread that will
        // cross the x_{k-1}->x_k reconnection segment (includes x_{k-1}'s surface term),
        // and x_{k-1}'s own texture LOD. Fed to the reconnection helpers.
        float lastConeWidth  = 0.0f;
        float lastConeSpread = 0.0f;
        float lastLod        = 0.0f;

        float primaryFootprint;
    {
        SurfaceHit hitData = traceClosest(params, r);

        if (!hitData.isHit) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("SHIFT ABORT [%s]: recon missed scene on primary hit\n", isReverseShift ? "REVERSE" : "FORWARD");
            }
            return {false, f3(0), 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
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

        coneWidth += coneSpread * hitData.t; // grow cone across this segment, then bake LOD

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
        if constexpr (!isReverseShift) {
            if (IS_DEBUG_PIXEL(x, y)) {
                DEBUG_PRINTF("replaying shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        primaryFootprint =
            (RECON_FOOTPRINT_C_CONSTANT / 100.0f) *
            (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));

        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        lastPos = shadingPos;
        lastMaterialID_packedWithEmissiveFlag = materialID;
        lastUV = uv;
        lastBackface = backface;
        lastInDirLocal = incomingDirLocal;
        lastNormal = normal;
#if USE_RAY_CONES
        lastConeWidth  = coneWidth;
        lastConeSpread = coneSpread + RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
        lastLod        = lod;
#endif

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;

        uint32_t loopBound = (K_is_D(type)) ? pathLength : rcVertexIndex;

        if (loopBound > 2) {
            // emissive primary hit does not roll a reservoir. since x_k-1 emissive flag is for reconstructing the reservoir roll there,
            // we thus dont need to store the emissive flag for primary hit

            if (!currDelta) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // to do the reservoir roll
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
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon primary hit scattering pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f}; // something went wrong, cant finish temporal shift
            }

            float lum = luminance(throughput);
            float p = clamp(lum, 0.05f, 1.0f);
            float rr_roll = rand(&localState);
            if (rr_roll > p) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon RR failed", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }
            throughput /= p;

            if (!(K_is_D(type) && pathLength == 2 && is_nee(type))) { // We dont want to update scattering throughput for this case
                throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
            }
            lastCosine = fabsf(outgoing.z);
#if USE_RAY_CONES
            coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif
            toWorld(outgoing, normal, outgoing);

            r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
            r.direction = outgoing;

            prevDelta = currDelta;
            lastPDF = pdf_bsdf;
        }
    }
        #if DEBUG_MODE == 1
        float3 lastPOS_GETRIDOFME = r.origin;
        #endif

        uint32_t loopBound = (K_is_D(type)) ?
            pathLength:
            rcVertexIndex;
        // depth + 1 is the "index" of the curr vertex, so this stops at y_k-1
        for (int depth = 1; depth + 1 < loopBound; depth++) {
            SurfaceHit hitData = traceClosest(params, r);

            if (!hitData.isHit) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon secondary hit missed scene\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
            }

            int materialID;
            float2 uv;
            float3 shadingPos;
            bool backface;
            float3 geoNormal;
            float3 normal;
            float3 lightEmission;
            float lod;
            const Triangle& tri = params.shadeContext.scene[hitData.primId];

            coneWidth += coneSpread * hitData.t; // grow cone across this segment, then bake LOD

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
                lightEmission,
                lod,

                hitData.instanceId
            );
            if constexpr (!isReverseShift) {

                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_DRAWLINE(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos,
                        f3(0.0f, 1.0f, 1.0f), 3
                    );
                    DEBUG_PRINTF("replaying shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
                }
            }

            float3 incomingDirLocal;
            toLocal(r.direction, normal, incomingDirLocal);

            // needed for recon
            if (depth + 1 == loopBound - 1) { // if this is the last iteration
                lastPos = shadingPos;
                lastMaterialID_packedWithEmissiveFlag = materialID;
                if (luminance(lightEmission) > 0.0f) {
                    // sets the msb to 1 to flag x_k-1 is emissive
                    lastMaterialID_packedWithEmissiveFlag = lastMaterialID_packedWithEmissiveFlag | 0x80000000;
                }

                lastUV = uv;
                lastBackface = backface;
                lastInDirLocal = incomingDirLocal;
                lastNormal = normal;
#if USE_RAY_CONES
                lastConeWidth  = coneWidth;
                lastConeSpread = coneSpread + RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
                lastLod        = lod;
#endif

                // change: moved break to this block so that the rng state handed to the shift
                // helpers is as of arriving to x_k-1
                //break;
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

            lightEmission = backface ? f3(0.0f) : lightEmission;
            if (luminance(lightEmission) > 0.0f) {
                rand(&localState); // for the reservoir roll
            }

            bool currDelta = params.shadeContext.materials[materialID].isSpecular;

            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper

            bool isValid = true;
            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                isValid = false;
            }
            
            if (!isValid) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon failed dual footprint\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }

            if (depth + 1 == loopBound - 1) {
                break;
            }

            if (!currDelta) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // for the reservoir roll
            }

            float lum = luminance(throughput);
            float p = clamp(lum, 0.05f, 1.0f);
            float rr_roll = rand(&localState);
            if (rr_roll > p) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon RR failed", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }
            throughput /= p;

            if (pdf_bsdf < EPSILON)
            {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon secondary hit scattering pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }

            throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;

            toWorld(outgoing, normal, outgoing);
            r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
            r.direction = outgoing;

            prevDelta = currDelta;
            lastPDF = pdf_bsdf;
            lastCosine = fabsf(dot(outgoing, normal));
#if USE_RAY_CONES
            coneSpread += RAYCONE_ROUGHNESS_SPREAD * params.shadeContext.materials[materialID].roughness;
#endif
            #if DEBUG_MODE == 1
            lastPOS_GETRIDOFME = shadingPos;
            #endif
        }

        int rc_xk_materialID;
        float2 rc_xk_uv;
        float3 rc_xk_pos;
        bool rc_xk_backface;
        float3 rc_xk_normal;

        // Ray-cone LOD at the reconnection vertex x_k, propagating the prefix cone
        // (captured at x_{k-1}) across the reconnection segment. Computed from the
        // GEOMETRIC rc position/normal (pre-normal-map) so it can also feed x_k's own
        // normal-map fetch below -- keeping x_k's shading consistent with the forward
        // pass. Only meaningful for a real surface rc vertex; env rc vertices
        // (rcPrimID == 0xFFFFFFFF) are emission-only, so rc_lod stays 0 there.
        float rc_lod = 0.0f;

        if (rcPrimID != 0xFFFFFFFF) {
            if (rcPrimID >= params.shadeContext.triNum) {
                if (IS_DEBUG_PIXEL(x, y)) {
                    DEBUG_PRINTF("SHIFT ABORT [%s]: recon wrongly initialized rcPrimID\n", isReverseShift ? "REVERSE" : "FORWARD");
                }
                return {false, f3(0), 0.0f, 0.0f};
            }

            const Triangle& tri = params.shadeContext.scene[rcPrimID];

#if USE_RAY_CONES
            {
                float ru = rcBarycentrics.x, rv = rcBarycentrics.y;
                float3 ap = f3(__ldg(&params.shadeContext.vertices->positions[tri.aInd]));
                float3 bp = f3(__ldg(&params.shadeContext.vertices->positions[tri.bInd]));
                float3 cp = f3(__ldg(&params.shadeContext.vertices->positions[tri.cInd]));
                float3 rcpos = (1.0f - ru - rv) * ap + ru * bp + rv * cp;
                float3 gn    = cross(bp - ap, cp - ap); // geometric normal
                if (rcInstanceID != 0xFFFFFFFF) {
                    rcpos = transformPosition(params.shadeContext.transformationMatrices, rcInstanceID, rcpos);
                    gn    = transformNormalRigid(params.shadeContext.transformationMatrices, rcInstanceID, gn);
                }
                float3 seg = rcpos - lastPos;
                float dist = length(seg);
                float coneWidth_xk = lastConeWidth + lastConeSpread * dist;
                float ndd = fmaxf(fabsf(dot(normalize(gn), seg / fmaxf(dist, 1e-8f))), 1e-3f);
                rc_lod = fmaxf(0.0f, tri.lodDelta + log2f(fmaxf(coneWidth_xk, 1e-8f) / ndd) + RAYCONE_LOD_BIAS);
            }
#endif

            getDataWithoutInDirectionAndEmission(
                tri,
                params.shadeContext,
                rcBarycentrics,
                lastPos,

                rc_xk_materialID,
                rc_xk_uv,
                rc_xk_pos,
                rc_xk_normal,
                rc_xk_backface,

                rcInstanceID,
                rc_lod
            );
        }

        if (K_less_D_minus_1(type)) {
            return perform_K_less_than_D_minus_1_reconnection(
                params,
                localState,
                type,
                x, y,
                isReverseShift,
                (loopBound == 2), // xkminus1IsPrimary (x_k-1 is the primary hit)

                // x_k parameters (from the rcPrimID getData block)
                rc_xk_materialID,           // rc_xk_materialID
                rc_xk_uv,                   // rc_xk_uv
                rc_xk_pos,                  // rc_xk_pos
                rc_xk_backface,             // rc_xk_backface
                rc_xk_normal,               // rc_xk_normal

                (lastMaterialID_packedWithEmissiveFlag & 0x80000000), // reinterpet the msb emissive flag as prevEmission

                // x_{k-1} / y_{k-1} parameters (cached from the prefix loop)
                (lastMaterialID_packedWithEmissiveFlag & 0x7FFFFFFF),           // xkminus1_materialID (remove msb)
                lastUV,                   // xkminus1_uv
                lastPos,                  // xkminus1_pos
                lastBackface,             // xkminus1_backface
                lastNormal,               // xkminus1_normal
                lastInDirLocal, // xkminus1_inDirLocal

                // Ray/Path state
                throughput,               // throughput entering x_k-1
                rcWi,                     // rcWi
                rcRadiance,               // rcRadiance (contains suffix throughput)
                jacobianDenom,            // jacobian_denominator

                lastLod,                  // xkminus1_lod (prefix cone LOD at x_{k-1})
                rc_lod                    // rc_lod (reconnection-segment LOD at x_k)
            );
        }
        else if (K_is_D_minus_1(type)) {
            return perform_K_is_D_minus_1_reconnection(
                params,
                localState,
                type,
                x, y,
                isReverseShift,
                (loopBound == 2), // xkminus1IsPrimary (x_k-1 is the primary hit)

                // x_k parameters
                rc_xk_materialID,           // rc_xk_materialID
                rc_xk_uv,                   // rc_xk_uv
                rc_xk_pos,                  // rc_xk_pos
                rc_xk_backface,             // rc_xk_backface
                rc_xk_normal,               // rc_xk_normal

                (lastMaterialID_packedWithEmissiveFlag & 0x80000000), // reinterpet the msb emissive flag as prevEmission
                // x_{k-1} / y_{k-1} parameters
                (lastMaterialID_packedWithEmissiveFlag & 0x7FFFFFFF),           // xkminus1_materialID (remove msb)
                lastUV,                   // xkminus1_uv
                lastPos,                  // xkminus1_pos
                lastBackface,             // xkminus1_backface
                lastNormal,               // xkminus1_normal
                lastInDirLocal, // xkminus1_inDirLocal

                // Ray/Path state
                throughput,
                rcWi,
                cached_nee,               // pdf_sampledLight_nee_sa
                rcRadiance,               // lightEmissionRaw
                jacobianDenom,

                lastLod,                  // xkminus1_lod (prefix cone LOD at x_{k-1})
                rc_lod                    // rc_lod (reconnection-segment LOD at x_k)
            );
        }
        else if (K_is_D(type)){ // K = D case
            return perform_K_is_D_reconnection(
                params,
                localState,
                type,
                x, y,
                isReverseShift,
                (loopBound == 2), // xkminus1IsPrimary (x_k-1 is the primary hit)

                // x_k parameters
                rc_xk_materialID,           // rc_xk_materialID
                rc_xk_uv,                   // rc_xk_uv
                rc_xk_pos,                  // rc_xk_pos
                rc_xk_backface,             // rc_xk_backface
                rc_xk_normal,               // rc_xk_normal

                (lastMaterialID_packedWithEmissiveFlag & 0x80000000), // reinterpet the msb emissive flag as prevEmission
                // x_{k-1} / y_{k-1} parameters
                (lastMaterialID_packedWithEmissiveFlag & 0x7FFFFFFF),           // xkminus1_materialID (remove msb)
                lastUV,                   // xkminus1_uv
                lastPos,                  // xkminus1_pos
                lastBackface,             // xkminus1_backface
                lastNormal,               // xkminus1_normal
                lastInDirLocal, // xkminus1_inDirLocal

                // Ray/Path state
                throughput,
                rcWi,                     // xkminus1_to_xk_direction_normalized (for env hits)
                cached_nee,               // pdf_sampledLight_nee
                rcRadiance,               // lightEmissionRaw
                jacobianDenom,

                lastLod                   // xkminus1_lod (x_k is the light: emission only)
            );
        } else {
            DEBUG_PRINTF("Alert: invalid path type with respect to k vs d, type is: %u, with seed %u and path length %u\n", type, seed, pathLength);
            return {false, f3(0), 0.0f, 0.0f};
        }
    }
}

__device__ __forceinline__ bool isHistoryValid(const PipelineParams& params, int2 currentCoord, half2 motionVec, int2& out_coords) {
    out_coords = make_int2(-1, -1);
    int2 history_coord = make_int2(currentCoord.x - (int)roundf(__half2float(motionVec.x)),
                            currentCoord.y - (int)roundf(__half2float(motionVec.y)));
    uint32_t current_idx = currentCoord.x + currentCoord.y * params.common.w;
    uint32_t history_idx = history_coord.x + history_coord.y * params.common.w;

    if (history_coord.x < 0 || history_coord.x >= params.common.w ||
        history_coord.y < 0 || history_coord.y >= params.common.h) {
        return false;
    }

    uint32_t current_id = params.restir.gbuffer.getMatID(current_idx);
    uint32_t history_id = params.restir.prevGbuffer.getMatID(history_idx);

    if (current_id != history_id) {
        return false;
    }

    float3 currNorm = params.restir.gbuffer.getNormal(current_idx);
    float3 pastNorm = params.restir.prevGbuffer.getNormal(history_idx);

    if (dot(currNorm, pastNorm) < NORMAL_REJECTION_THRESHOLD) {
        return false;
    }

    // the pixel jitter is easily recreated since it is always spawned from the first two random numbers after the seed
    RNGState localState = load_rng(current_idx, params.common.frame_index, 0, nullptr);
    Ray r = params.common.camera.generateCameraRay(localState, currentCoord.x, currentCoord.y);

    float3 current_pos = r.at(params.restir.gbuffer.getDepth(current_idx));

    localState = load_rng(history_idx, params.common.frame_index - 1, 0, nullptr);
    r = params.restir.lastFrameCamera.generateCameraRay(localState, history_coord.x, history_coord.y);

    float3 history_pos = r.at(params.restir.prevGbuffer.getDepth(history_idx));

    float3 pos_diff = history_pos - current_pos;

    float plane_distance = abs(dot(pos_diff, currNorm));

    float depth_tolerance = PLANAR_DIST_REJECTION_THRESHOLD + 
        (length(current_pos - params.common.camera.cameraOrigin) * PLANAR_DIST_REJECTION_THRESHOLD);

    if (plane_distance > depth_tolerance) {
        return false;
    }

    float true_distance = length(pos_diff);
    if (true_distance > depth_tolerance * TRUE_DIST_REJECTION_THRESHOLD) {
        return false;
    }

    out_coords = history_coord;
    return true;
}

__device__ __forceinline__ bool isSpatialNeighborValid(const PipelineParams& params, int2 currentCoord, int2 neighborCoord) {

    uint32_t current_idx = currentCoord.x + currentCoord.y * params.common.w;
    uint32_t neighbor_idx = neighborCoord.x + neighborCoord.y * params.common.w;

    uint32_t current_id = params.restir.gbuffer.getMatID(current_idx);
    uint32_t neighbor_id = params.restir.gbuffer.getMatID(neighbor_idx);

    if (current_id != neighbor_id) {
        return false;
    }

    // Normal Threshold Check
    float3 currNorm = params.restir.gbuffer.getNormal(current_idx);
    float3 neighNorm = params.restir.gbuffer.getNormal(neighbor_idx);

    if (dot(currNorm, neighNorm) < NORMAL_REJECTION_THRESHOLD) {
        return false;
    }

    // Current Pixel
    RNGState localStateCurr = load_rng(current_idx, params.common.frame_index, 0, nullptr);
    Ray rCurr = params.common.camera.generateCameraRay(localStateCurr, currentCoord.x, currentCoord.y);
    float3 current_pos = rCurr.at(params.restir.gbuffer.getDepth(current_idx));

    // Neighbor Pixel
    RNGState localStateNeigh = load_rng(neighbor_idx, params.common.frame_index, 0, nullptr);
    Ray rNeigh = params.common.camera.generateCameraRay(localStateNeigh, neighborCoord.x, neighborCoord.y);
    float3 neighbor_pos = rNeigh.at(params.restir.gbuffer.getDepth(neighbor_idx));

    // Planarity and Distance Checks
    float3 pos_diff = neighbor_pos - current_pos;

    // Check if the neighbor is roughly on the same plane as the current pixel
    float plane_distance = fabsf(dot(pos_diff, currNorm));
    float depth_tolerance = PLANAR_DIST_REJECTION_THRESHOLD + 
        (length(current_pos - params.common.camera.cameraOrigin) * PLANAR_DIST_REJECTION_THRESHOLD);

    if (plane_distance > depth_tolerance) {
        return false;
    }

    // Check if the neighbor is physically too far away in 3D space
    float true_distance = length(pos_diff);
    if (true_distance > depth_tolerance * TRUE_DIST_REJECTION_THRESHOLD) {
        return false;
    }

    return true;
}

__device__ __forceinline__ bool isSpatialPairAccepted(
    const PipelineParams& allParams, int2 a, int2 b)
{
    return isSpatialNeighborValid(allParams, a, b)
        && isSpatialNeighborValid(allParams, b, a);
}

__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ int2 get_frame_offset(uint32_t frame_idx, uint32_t texture_id, int2 texture_size) {
    uint32_t hx = hash_uint32(frame_idx ^ (texture_id * 1973u));
    uint32_t hy = hash_uint32(hx);
    return make_int2(hx % texture_size.x, hy % texture_size.y);
}

__device__ bool get_frame_flip_x(uint32_t frame_idx, uint32_t texture_id) {
    uint32_t h = hash_uint32(frame_idx ^ (texture_id * 31337u));
    return (h & 1) != 0;
}

__device__ bool get_frame_transpose(uint32_t frame_idx, uint32_t texture_id) {
    uint32_t h = hash_uint32(frame_idx ^ (texture_id * 8128u));
    return (h & 1) != 0;
}

__device__ int2 get_paired_neighbor(
    int2 screen_pixel,
    uint32_t texture_id,
    uint32_t frame_idx,
    uint32_t texture_size_1d,
    int2 screen_dimension,
    const short2* __restrict__ pairing_buffer)
{
    int2 texture_size = make_int2(texture_size_1d, texture_size_1d);
    int2 random_offset = get_frame_offset(frame_idx, texture_id, texture_size);
    bool flip_x        = get_frame_flip_x(frame_idx, texture_id);
    bool transpose     = get_frame_transpose(frame_idx, texture_id);

    int2 tex_coord = screen_pixel;

    if (transpose) {
        tex_coord = make_int2(tex_coord.y, tex_coord.x);
    }

    // Apply flip across X
    if (flip_x) {
        tex_coord.x = texture_size.x - 1 - tex_coord.x;
    }

    // Apply offset and wrap around the boundaries
    tex_coord.x = (tex_coord.x + random_offset.x) % texture_size.x;
    tex_coord.y = (tex_coord.y + random_offset.y) % texture_size.y;

    // Ensure positive modulo wrapping
    if (tex_coord.x < 0) tex_coord.x += texture_size.x;
    if (tex_coord.y < 0) tex_coord.y += texture_size.y;

    // Flatten 2D coord to 1D index for the raw short2* buffer
    uint32_t flat_index = tex_coord.y * texture_size.x + tex_coord.x;

    // Read the raw delta and cast up to int2 for math
    short2 raw_short_delta = pairing_buffer[flat_index];
    int2 final_delta = make_int2(raw_short_delta.x, raw_short_delta.y);

    // Apply inverse transformations to the delta. The forward map above is
    // transpose then flip, whose linear part is a 90 degree rotation, so the
    // inverse has to undo them in the opposite order: flip first, then transpose.
    // Doing it the other way round applies the rotation a second time instead of
    // cancelling it, and the pairing stops being self inverting.
    if (flip_x) {
        final_delta.x = -final_delta.x;
    }
    if (transpose) {
        final_delta = make_int2(final_delta.y, final_delta.x);
    }

    // Apply the delta to get the actual neighbor pixel coordinate
    int2 paired_pixel = make_int2(screen_pixel.x + final_delta.x, screen_pixel.y + final_delta.y);

    // Boundary check: If the delta pushes the pixel off-screen, return an invalid coordinate
    if (paired_pixel.x < 0 || paired_pixel.x >= screen_dimension.x ||
        paired_pixel.y < 0 || paired_pixel.y >= screen_dimension.y)
    {
        return make_int2(-1, -1);
    }

    return paired_pixel;
}

__device__ __forceinline__ float3 debugVisualizeTechnique(uint32_t type, uint32_t rcInd) {
    // 0. Special Reconnection Indices -> Overriding Colors
    if (rcInd == 0xFF) {
        // Full Replay Technique: White
        return make_float3(0.0f, 1.0f, 0.0f); 
    } 
    else if (rcInd == 0xFE) {
        // Direction Copy: black/greu
        return make_float3(0.1f, 0.1f, 0.1f); 
    }

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // 1. Reconnection Depth -> Primary Color Axis
    if (type & SHIFT_K_IS_D) {
        color = make_float3(1.0f, 0.0f, 0.0f); // Base: Red
    }
    else if (type & SHIFT_K_IS_D_MINUS_1) {
        color = make_float3(0.0f, 1.0f, 0.0f); // Base: Green
    }
    else if (type & SHIFT_K_LESS_D_MINUS_1) {
        color = make_float3(0.0f, 0.0f, 1.0f); // Base: Blue
    }

    // 2. Light Type -> Shift to Secondary Color
    if (type & SHIFT_IS_ENV) {
        if (type & SHIFT_K_IS_D)                 color.y = 1.0f; // Red -> Yellow
        else if (type & SHIFT_K_IS_D_MINUS_1)    color.z = 1.0f; // Green -> Cyan
        else if (type & SHIFT_K_LESS_D_MINUS_1)  color.x = 1.0f; // Blue -> Magenta
    }

    // 3. Sampling Method -> Intensity
    // NEE = Vivid (1.0), BSDF = Dim/Muted (0.35)
    float intensity = (type & SHIFT_IS_NEE) ? 1.0f : 0.35f;

    return color * intensity;
}

__device__ __forceinline__ float3 debugVisualizeTechniqueAndLength(uint32_t type, uint32_t pathLength) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // 1. Reconnection Depth (K) -> Base Hue
    if (type & SHIFT_K_IS_D)                 color = make_float3(1.0f, 0.0f, 0.0f); // Base: Red
    else if (type & SHIFT_K_IS_D_MINUS_1)    color = make_float3(0.0f, 1.0f, 0.0f); // Base: Green
    else if (type & SHIFT_K_LESS_D_MINUS_1)  color = make_float3(0.0f, 0.0f, 1.0f); // Base: Blue

    // 2. Light Type -> Shift to Secondary Hue
    if (type & SHIFT_IS_ENV) {
        if (type & SHIFT_K_IS_D)                 color.y = 1.0f; // Red -> Yellow
        else if (type & SHIFT_K_IS_D_MINUS_1)    color.z = 1.0f; // Green -> Cyan
        else if (type & SHIFT_K_LESS_D_MINUS_1)  color.x = 1.0f; // Blue -> Magenta
    }

    // 3. Sampling Method -> Saturation
    // NEE = Pure/Vivid. BSDF = Mix with 65% white (Pastel/Chalky)
    if (!(type & SHIFT_IS_NEE)) {
        float pastel_blend = 0.65f;
        color.x = color.x + (1.0f - color.x) * pastel_blend;
        color.y = color.y + (1.0f - color.y) * pastel_blend;
        color.z = color.z + (1.0f - color.z) * pastel_blend;
    }

    // 4. Path Length -> Brightness / Value
    // Assuming minimum path length is 2.
    // Length 2 = 100% brightness. Each extra bounce darkens it by 15%.
    float brightness = fmaxf(1.0f - (float)(pathLength - 2) * 0.15f, 0.15f);

    return make_float3(color.x * brightness, color.y * brightness, color.z * brightness);
}