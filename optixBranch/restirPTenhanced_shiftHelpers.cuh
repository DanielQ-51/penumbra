#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "optixStructs.cuh"
#include "settings.cuh"
#include "optixUtils.cuh"
#include "reflectors.cuh"

// Overall convention: RNG state passed to the function is as of arriving at x_k-1


/** Overall RR and bsdf scaling logic is split between two cases:
 *
 * If NEE, then it must be that x_k-1 rolled a RR and performed throughput scaling,
 * then hit x_k, spawned the NEE sample for this reservoir without rolling RR.
 *
 * If BSDF, then it must be that x_k-1 rolled a RR and performed throughput scaling,
 * then hit x_k, which rolled a RR and performed throughput scaling, then bounced and hit the light.
 *
 * We assume that no RR has been done for x_k-1 or x_k for the throughput passed to this function
 *
 */
__device__ __forceinline__ inline ShiftResult perform_K_is_D_minus_1_reconnection(
    const CommonParams& params,
    RNGState localState,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,
    bool xkminus1IsPrimary,
    
    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    bool xkminus1_emissive,
    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 rcWi,
    float pdf_sampledLight_nee_sa, // Candidate generation ensures that for these paths, this pdf is always in solid angle already
    float3 lightEmissionRaw,
    float jacobian_denominator,

    float xkminus1_lod, // ray-cone texture LOD at x_{k-1} (from the prefix replay)
    float rc_lod        // ray-cone texture LOD at x_k (prefix cone across the reconnection segment)
) {
    ShiftResult result;

    float3 xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
    float xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);

    // Shading systems expect the normal to be facing opposite direction from the incoming direction
    if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
        rc_xk_normal = -rc_xk_normal;
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (dot(xkminus1_to_xk_direction_normalized, xkminus1_normal) > 0.0f ? xkminus1_normal : -xkminus1_normal) * VIS_EPSILON, xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon failed reconnection visibility test for k=d-1\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // First, scale RR for x_k-1. This can be done right away since we do RR before the throughput scaling at each vertex.
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}

    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    // If x_k-1 is the primary hit, candidate gen used the depth-0 rng order (NEE
    // BEFORE the bsdf sample), not the secondary order (sample first). Roll x_k-1's
    // NEE draws here, before the lobe replay, so the lobe-selection draw lands on the
    // right rng. Reconnection guarantees x_k-1 is non-delta, so its NEE always happened.
    if (xkminus1IsPrimary && !params.shadeContext.materials[xkminus1_materialID].isSpecular) {
        rand(&localState);
        rand4(&localState);
        rand(&localState); // reservoir roll
    }

    float3 xkminus1_f_val;
    float xkminus1_pdf;
    float xkminus1_pdf_marg; // unused here (this helper's path MIS uses the x_k pdf); API requires it
    f_pdf_eval_replayLobe(
        localState,
        params.shadeContext.materials,
        xkminus1_materialID,
        params.shadeContext.textures,
        xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
        xkminus1_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        xkminus1_backface, // not sure why needed?
        xkminus1_f_val,
        xkminus1_pdf,
        xkminus1_pdf_marg,
        xkminus1_uv,
        TRANSPORTMODE_RADIANCE, xkminus1_lod
    );

    if (xkminus1IsPrimary) {
        // primary order: after the bsdf sample, only the RR roll remains
        rand(&localState); // RR for x_k-1
    } else {
        // secondary order: sample, then emissive roll, then NEE, then RR
        if (xkminus1_emissive) {
            rand(&localState); // if emissive, consume one rand for reservoir roll
        }

        if (is_bsdf(pathType)) { // means that x_k-1 performed nee, but then bounced onto x_k
            // x_k-1 MUST be non-delta, guaranteed by the dual footprint check, so we do the nee padding
            // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
            rand(&localState);
            rand4(&localState);

            rand(&localState); // for the reservoir roll
        }

        // to roll RR for x_k-1
        rand(&localState);
    }

    // now, since we do the bsdf sample before nee and before implicit light eval, the pre-bsdf sample rng
    // is the same as the entering x_k rng, so we good now

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / xkminus1_pdf;

    // Next, scale RR for x_k, but only if it was a bsdf sample This can be done now since we do RR before the throughput scaling at each vertex.
    // since RR is done after bsdf sampling, we dont need to pad an rng here
    if (is_bsdf(pathType)) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // Now, perform throughput scaling for exiting x_k

    float3 rc_xk_f_val;
    float rc_xk_bsdf_pdf;
    float rc_xk_bsdf_pdf_marg;

    // This engine has the incoming direciton pointing into the surface
    float3 rc_xk_inDirLocal = toLocal(xkminus1_to_xk_direction_normalized, rc_xk_normal);
    float3 rc_xk_outDirLocal = toLocal(rcWi, rc_xk_normal);
    float rc_xk_outgoing_cosine = fabsf(rc_xk_outDirLocal.z);

    if (is_bsdf(pathType)) { 
        // if it was a bsdf path, then the last segment x_k outwards was
        // sampled via the lobe specific bsdf scatter
        f_pdf_eval_replayLobe(
            localState,
            params.shadeContext.materials,
            rc_xk_materialID,
            params.shadeContext.textures,
            rc_xk_inDirLocal, // direction to the y_k-1 from y_k-2
            rc_xk_outDirLocal, // direction to the rc vertex from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            rc_xk_backface,
            rc_xk_f_val,
            rc_xk_bsdf_pdf,
            rc_xk_bsdf_pdf_marg,
            rc_xk_uv,
            TRANSPORTMODE_RADIANCE, rc_lod
        );
    } else {
        // if it was a nee path, then the last segment x_k outwards was
        // sampled via the marginalized bsdf
        f_pdf_eval(
            params.shadeContext.materials,
            rc_xk_materialID,
            params.shadeContext.textures,
            rc_xk_inDirLocal, // direction to the y_k-1 from y_k-2
            rc_xk_outDirLocal, // direction to the rc vertex from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            rc_xk_f_val,
            rc_xk_bsdf_pdf,
            rc_xk_uv,
            TRANSPORTMODE_RADIANCE, rc_lod
        );
    }
    

    // nee case evaluated x_k's bsdf marginally (f_pdf_eval), so its marginal == pdf.
    if (!is_bsdf(pathType)) rc_xk_bsdf_pdf_marg = rc_xk_bsdf_pdf;

    float rc_xk_p_sampled_light = is_nee(pathType) ? (pdf_sampledLight_nee_sa) : (rc_xk_bsdf_pdf);

    throughput *= rc_xk_f_val * rc_xk_outgoing_cosine / rc_xk_p_sampled_light;

    // MIS uses the MARGINAL bsdf pdf (throughput/jacobian keep the lobe-specific one).
    float path_misWeight = powerHeuristicTwoStrategy(
        (is_nee(pathType)) ? pdf_sampledLight_nee_sa : rc_xk_bsdf_pdf_marg,
        (is_nee(pathType)) ? rc_xk_bsdf_pdf_marg : pdf_sampledLight_nee_sa
    );

    result.contribution = throughput * lightEmissionRaw * path_misWeight;

    // Now, calculate jacobian.

    float rc_xk_incoming_cosine = fabsf(rc_xk_inDirLocal.z);
    float geometryTerm = rc_xk_incoming_cosine / (xkminus1_to_xk_distance * xkminus1_to_xk_distance);
    float jacobian_numerator = is_bsdf(pathType) ?
        (xkminus1_pdf * geometryTerm * rc_xk_bsdf_pdf) : // if it was the bsdf case, it should be the normal jacobian
        (xkminus1_pdf * geometryTerm) // if it was a NEE case, the last pdf cancels out since the light selection pdf is equal for both domains paths
    ;

    result.jacobian = jacobian_numerator / jacobian_denominator;
    result.new_cached_jacobian = jacobian_numerator;
    result.isValid = true;

    // Safety Checks
    if (jacobian_numerator <= 0.0f || jacobian_denominator <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k=d-1 zero p_new_suffix or jacobianDenom\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    if (xkminus1_pdf <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k=d-1 xkminus1_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (rc_xk_bsdf_pdf <= 0.0f && !(is_nee(pathType))) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k=d-1 rc_xk_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (targetFunction(result.contribution) <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k=d-1 phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}


/**
 * The most straightforward reconneciton, since the reconnection is agnostic to the suffix geometry, and can easily just
 * process a clean incoming radiance at the rc vertex.
 *
 * We assume that no RR has been done for x_k-1 or x_k for the throughput passed to this function
 */
__device__ __forceinline__ inline ShiftResult perform_K_less_than_D_minus_1_reconnection(
    const CommonParams& params,
    RNGState localState,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,
    bool xkminus1IsPrimary,
    
    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    bool xkminus1_emissive,
    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 rcWi,

    // As this is meant to be stitched onto the end of the prefix, this must only include RR and bsdf throughput scaling terms starting at x_k+1, since all that
    // for x_k is handled here in the shift. Contains the suffix throughput (including RR scaling), the emission, and the path mis term
    float3 rcRadiance,

    float jacobian_denominator,

    float xkminus1_lod, // ray-cone texture LOD at x_{k-1} (from the prefix replay)
    float rc_lod        // ray-cone texture LOD at x_k (prefix cone across the reconnection segment)
) {
    ShiftResult result;

    float3 xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
    float xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);

    // Shading systems expect the normal to be facing opposite direction from the incoming direction
    if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
        rc_xk_normal = -rc_xk_normal;
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (dot(xkminus1_to_xk_direction_normalized, xkminus1_normal) > 0.0f ? xkminus1_normal : -xkminus1_normal) * VIS_EPSILON, xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon failed reconnection visibility test for k<d-1\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // First, scale RR for x_k-1. This can be done right away since we do RR before the throughput scaling at each vertex.
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}
    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    // Primary x_k-1 used the depth-0 rng order (NEE before the bsdf sample); roll its
    // NEE draws before the lobe replay so the lobe-selection draw is aligned.
    if (xkminus1IsPrimary && !params.shadeContext.materials[xkminus1_materialID].isSpecular) {
        rand(&localState);
        rand4(&localState);
        rand(&localState); // reservoir roll
    }

    float3 xkminus1_f_val;
    float xkminus1_pdf;
    float xkminus1_pdf_marg; // unused here (this helper's path MIS uses the x_k pdf); API requires it
    f_pdf_eval_replayLobe(
        localState,
        params.shadeContext.materials,
        xkminus1_materialID,
        params.shadeContext.textures,
        xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
        xkminus1_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        xkminus1_backface,
        xkminus1_f_val,
        xkminus1_pdf,
        xkminus1_pdf_marg,
        xkminus1_uv,
        TRANSPORTMODE_RADIANCE, xkminus1_lod
    );
    
    if (xkminus1IsPrimary) {
        // primary order: after the bsdf sample, only the RR roll remains
        rand(&localState); // RR for x_k-1
    } else {
        if (xkminus1_emissive) {
            rand(&localState); // if emissive, consume one rand for reservoir roll
        }

        if (!params.shadeContext.materials[xkminus1_materialID].isSpecular) {
            // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
            rand(&localState);
            rand4(&localState);

            rand(&localState); // for the reservoir roll
        }

        // to roll RR for x_k-1
        rand(&localState);
    }

    // now, rng is the correct rng entering x_k, and because the first rng consumption comes from the
    // sampling given no env hit, we can evaluate x_k sampling now

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / xkminus1_pdf;

    // Next, scale RR for x_k
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}

    // Now, perform throughput scaling for exiting x_k

    float3 rc_xk_f_val;
    float rc_xk_bsdf_pdf;
    float rc_xk_bsdf_pdf_marg;

    // This engine has the incoming direciton pointing into the surface
    float3 rc_xk_inDirLocal = toLocal(xkminus1_to_xk_direction_normalized, rc_xk_normal);
    float3 rc_xk_outDirLocal = toLocal(rcWi, rc_xk_normal);
    float rc_xk_outgoing_cosine = fabsf(rc_xk_outDirLocal.z);

    f_pdf_eval_replayLobe(
        localState,
        params.shadeContext.materials,
        rc_xk_materialID,
        params.shadeContext.textures,
        rc_xk_inDirLocal, // direction to the y_k-1 from y_k-2
        rc_xk_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        rc_xk_backface,
        rc_xk_f_val,
        rc_xk_bsdf_pdf,
        rc_xk_bsdf_pdf_marg, // unused here (k<d-1 has no path MIS); API requires it
        rc_xk_uv,
        TRANSPORTMODE_RADIANCE, rc_lod
    );

    throughput *= rc_xk_f_val * rc_xk_outgoing_cosine / rc_xk_bsdf_pdf;

    result.contribution = throughput * rcRadiance;

    // Now, calculate jacobian.

    float rc_xk_incoming_cosine = fabsf(rc_xk_inDirLocal.z);
    float geometryTerm = rc_xk_incoming_cosine / (xkminus1_to_xk_distance * xkminus1_to_xk_distance);
    float jacobian_numerator = (xkminus1_pdf * geometryTerm * rc_xk_bsdf_pdf); // if it was the bsdf case, it should be the normal jacobian

    result.jacobian = jacobian_numerator / jacobian_denominator;
    result.new_cached_jacobian = jacobian_numerator;

    result.isValid = true;

    // Safety Checks
    if (jacobian_numerator <= 0.0f || jacobian_denominator <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k<d-1 zero p_new_suffix or jacobianDenom\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    if (xkminus1_pdf <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k<d-1 xkminus1_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (rc_xk_bsdf_pdf <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k<d-1 rc_xk_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (targetFunction(result.contribution) <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: internal recon k<d-1 phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}

__device__ __forceinline__ inline ShiftResult perform_K_is_D_reconnection(
    const CommonParams& params,
    RNGState localState,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,
    bool xkminus1IsPrimary,
    
    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    bool xkminus1_emissive,
    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 xkminus1_to_xk_direction_normalized, // only used for environment cases
    float pdf_sampledLight_nee, // Either in solid angle or area depending on whether its env or area light
    float3 lightEmissionRaw,

    float jacobian_denominator,

    float xkminus1_lod // ray-cone texture LOD at x_{k-1} (x_k is the light: emission only, no LOD)
) {
    ShiftResult result;

    float xkminus1_to_xk_distance;

    if (is_area(pathType)) { // if its an area light (triangle)
        xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
        xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);
    } else {
        // xkminus1_to_xk_direction_normalized is already defined, and passed in
        xkminus1_to_xk_distance = 1e30;
    }

    if (is_area(pathType)) { // Normal isnt defined for environment ending vertex
        // Shading systems expect the normal to be facing opposite direction from the incoming direction
        if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
            rc_xk_normal = -rc_xk_normal;
        }
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (dot(xkminus1_to_xk_direction_normalized, xkminus1_normal) > 0.0f ? xkminus1_normal : -xkminus1_normal) * VIS_EPSILON, xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon failed reconnection visibility test for k=d\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // Now, this should make sure that pdf_sampled_Light_nee is in solid angle
    if (is_area(pathType)) {
        float xk_lightCosine = fabsf(dot(rc_xk_normal, xkminus1_to_xk_direction_normalized));
        pdf_sampledLight_nee *= (xkminus1_to_xk_distance * xkminus1_to_xk_distance) / xk_lightCosine;
    }

    // If the path was generated by bsdf, then it must be that RR was rolled at x_k-1, before then bouncing to x_k.
    // On the contrary, if it was nee, then nee was cast from x_k-1, before any RR was rolled for x_k-1, so it isnt applied
    if (is_bsdf(pathType)) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    float3 xkminus1_f_val;
    float xkminus1_bsdf_pdf;
    float xkminus1_bsdf_pdf_marg;

    if (is_bsdf(pathType)) {
        // Primary x_k-1 rolled NEE before its bsdf sample (depth-0 order); align first.
        // (k=d bsdf paths are normally routed to full replay, so this branch is rarely hit.)
        if (xkminus1IsPrimary && !params.shadeContext.materials[xkminus1_materialID].isSpecular) {
            rand(&localState);
            rand4(&localState);
            rand(&localState);
        }
        // if it was a bsdf path, then the last segment x_k outwards was
        // sampled via the lobe specific bsdf scatter
        f_pdf_eval_replayLobe(
            localState,
            params.shadeContext.materials,
            xkminus1_materialID,
            params.shadeContext.textures,
            xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
            xkminus1_outDirLocal, // direction to the rc vertex aka light from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            xkminus1_backface,
            xkminus1_f_val,
            xkminus1_bsdf_pdf,
            xkminus1_bsdf_pdf_marg,
            xkminus1_uv,
            TRANSPORTMODE_RADIANCE, xkminus1_lod
        );
    } else {
        // if it was a nee path, then the last segment x_k outwards was
        // sampled via the marginalized bsdf
        f_pdf_eval(
            params.shadeContext.materials,
            xkminus1_materialID,
            params.shadeContext.textures,
            xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
            xkminus1_outDirLocal, // direction to the rc vertex aka light from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            xkminus1_f_val,
            xkminus1_bsdf_pdf,
            xkminus1_uv,
            TRANSPORTMODE_RADIANCE, xkminus1_lod
        );
    }
    

    // nee case evaluated x_k-1's bsdf marginally (f_pdf_eval), so its marginal == pdf.
    if (!is_bsdf(pathType)) xkminus1_bsdf_pdf_marg = xkminus1_bsdf_pdf;

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    float p_sampled_light = is_bsdf(pathType) ? xkminus1_bsdf_pdf : pdf_sampledLight_nee;

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / p_sampled_light;

    // MIS uses the MARGINAL bsdf pdf (throughput/jacobian keep the lobe-specific one).
    float path_misWeight = powerHeuristicTwoStrategy(
        (is_nee(pathType)) ? pdf_sampledLight_nee : xkminus1_bsdf_pdf_marg,
        (is_nee(pathType)) ? xkminus1_bsdf_pdf_marg : pdf_sampledLight_nee
    );

    result.contribution = throughput * lightEmissionRaw * path_misWeight;

    // For only this case, we execute a bsdf direction copy, which reduces variance but has a
    // non-identity jacobian
    if (is_bsdf(pathType) && is_env(pathType)) {
        float jacobian_numerator = xkminus1_bsdf_pdf;
        result.jacobian = jacobian_numerator / jacobian_denominator;
        result.new_cached_jacobian = jacobian_numerator;
    } else {
        result.jacobian = 1.0f; // a full random replay path
        result.new_cached_jacobian = 1.0f;
    }

    result.isValid = true;

    // Add safety checks at the end like the other funcions

    if (p_sampled_light <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon k=d p_sampled_light zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 2. Ensure the BSDF at y_{k-1} can actually scatter towards the light.
    // Even if this is an NEE path, if the BSDF PDF is zero, the connection is physically impossible.
    if (xkminus1_bsdf_pdf <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon k=d xkminus1_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 3. Ensure we aren't dividing by zero when calculating the Jacobian.
    // This is only applicable for the BSDF Environment hit, since the other paths hardcode jacobian = 1.0f.
    if (is_bsdf(pathType) && is_env(pathType) && jacobian_denominator <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon k=d env bsdf zero jacobian_denominator\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 4. Ensure the final target function evaluates to a valid, non-zero weight.
    if (targetFunction(result.contribution) <= 0.0f) {
        if (IS_DEBUG_PIXEL(x, y)) {
            DEBUG_PRINTF("SHIFT ABORT [%s]: recon k=d phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}

/**
 * ============================================================================
 * UNIFIED RECONNECTION HELPER
 * ============================================================================
 *
 * Branch-merged equivalent of the three helpers above:
 *   perform_K_less_than_D_minus_1_reconnection  (k < d-1)
 *   perform_K_is_D_minus_1_reconnection         (k = d-1)
 *   perform_K_is_D_reconnection                 (k = d)
 *
 * Every step below is the exact arithmetic of the corresponding step in those
 * three functions; the only thing that changed is that the case selection moved
 * from a 3-way code branch into scalar predicates/selects around ONE pass. Every
 * `if` that survives is either (a) an rng-consuming step, which cannot be
 * predicated away, or (b) a guard against touching x_k data that is genuinely
 * undefined for that case (k=d env vertices carry no material/uv/pos).
 *
 * Parameter aliasing (all three call sites already pass the same locals):
 *   rcWi                   -> x_k outgoing direction for k<d, and the
 *                             x_{k-1}->env direction for the k=d env case.
 *   rcRadiance             -> the suffix radiance (k<d-1, MIS already baked in)
 *                             or the raw light emission (k=d-1, k=d).
 *   pdf_sampledLight_nee   -> solid-angle nee pdf for k=d-1; solid-angle OR area
 *                             pdf for k=d (converted here); unused for k<d-1.
 *   rc_lod                 -> unused for k=d (the light is emission only).
 *
 * Debug printouts are intentionally omitted here.
 *
 * We assume that no RR has been done for x_k-1 or x_k for the throughput passed in.
 */
__device__ __forceinline__ inline ShiftResult perform_reconnection_unified(
    const CommonParams& params,
    RNGState localState,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,
    bool xkminus1IsPrimary,

    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,
    float3 rc_xk_geoNormal,   // geometric (pre-normal-map) normal at x_k; undefined for k=d env

    bool xkminus1_emissive,
    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_geoNormal, // geometric normal at x_{k-1}; the visibility ray offsets along THIS
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 rcWi,                   // x_k outgoing dir (k<d) / x_k-1 -> env dir (k=d env)
    float pdf_sampledLight_nee,    // k=d-1: already solid angle. k=d: solid angle (env) or area (tri)
    float3 rcRadiance,             // suffix radiance (k<d-1) or raw light emission (k=d-1, k=d)
    float jacobian_denominator,

    float xkminus1_lod, // ray-cone texture LOD at x_{k-1} (from the prefix replay)
    float rc_lod        // ray-cone texture LOD at x_k (prefix cone across the reconnection segment)
) {
    ShiftResult result;

    // ---- case predicates -----------------------------------------------------
    const bool kIsD     = K_is_D(pathType);
    const bool kIsDm1   = K_is_D_minus_1(pathType);
    const bool kLessDm1 = K_less_D_minus_1(pathType);
    const bool bsdfPath = is_bsdf(pathType);
    const bool neePath  = is_nee(pathType);
    // only the k=d case can terminate on the environment; for k<d the rc vertex is a surface
    const bool envEnd   = kIsD && is_env(pathType);

    // ---- 1. reconnection segment --------------------------------------------
    // (k=d env has no rc position/normal at all: the direction is handed in directly)
    float3 xkminus1_to_xk_direction_normalized;
    float xkminus1_to_xk_distance;

    if (envEnd) {
        xkminus1_to_xk_direction_normalized = rcWi;
        xkminus1_to_xk_distance = 1e30;
    } else {
        xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
        xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);

        // Shading systems expect the normal to be facing opposite direction from the incoming
        // direction. Flip the geometric normal with it so the two stay on the same side.
        if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
            rc_xk_normal = -rc_xk_normal;
            rc_xk_geoNormal = -rc_xk_geoNormal;
        }
    }

    // ---- 2. reconnection visibility -----------------------------------------
    // Offset along the GEOMETRIC normal, matching every other ray spawn in the engine
    // (getDataGeo, renderer.cu, candidate gen's NEE). Offsetting along the normal-mapped
    // shading normal can push the origin below the actual surface wherever the two
    // disagree, which self-intersects and reports a false occlusion -- silently killing
    // the shift on exactly the normal-mapped geometry where reconnection matters most.
    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (dot(xkminus1_to_xk_direction_normalized, xkminus1_geoNormal) > 0.0f ? xkminus1_geoNormal : -xkminus1_geoNormal) * VIS_EPSILON, xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // ---- 3. k=d area light: convert the stored area pdf to solid angle -------
    // (k=d-1 stores a solid-angle pdf already; k<d-1 has no light pdf)
    if (kIsD && !envEnd) {
        float xk_lightCosine = fabsf(dot(rc_xk_normal, xkminus1_to_xk_direction_normalized));
        pdf_sampledLight_nee *= (xkminus1_to_xk_distance * xkminus1_to_xk_distance) / xk_lightCosine;
    }

    // ---- 4. RR at x_k-1 ------------------------------------------------------
    // k<d: x_k-1 always rolled RR before bouncing onto x_k.
    // k=d: only the bsdf path bounced out of x_k-1; a nee cast happens before x_k-1 RR.
    if (!kIsD || bsdfPath) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // ---- 5. outgoing direction at x_k-1 -------------------------------------
    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    const bool xkminus1NonDelta = !params.shadeContext.materials[xkminus1_materialID].isSpecular;

    // k=d + nee is the only case where the outgoing direction at x_k-1 was NOT bsdf-sampled
    // (it is the nee connection), so it is evaluated marginally and consumes no rng.
    const bool xkm1UsesLobeReplay = !kIsD || bsdfPath;

    // ---- 6. primary-hit rng alignment ---------------------------------------
    // If x_k-1 is the primary hit, candidate gen used the depth-0 rng order (NEE
    // BEFORE the bsdf sample), not the secondary order (sample first). Roll the x_k-1
    // NEE draws here, before the lobe replay, so the lobe-selection draw lands on the
    // right rng. Reconnection guarantees x_k-1 is non-delta, so its NEE always happened.
    if (xkm1UsesLobeReplay && xkminus1IsPrimary && xkminus1NonDelta) {
        rand(&localState);
        rand4(&localState);
        rand(&localState); // reservoir roll
    }

    // ---- 7. bsdf at x_k-1 ----------------------------------------------------
    float3 xkminus1_f_val;
    float xkminus1_pdf;
    float xkminus1_pdf_marg;

    if (xkm1UsesLobeReplay) {
        // the outgoing segment was bsdf-sampled -> replay the lobe it picked
        f_pdf_eval_replayLobe(
            localState,
            params.shadeContext.materials,
            xkminus1_materialID,
            params.shadeContext.textures,
            xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
            xkminus1_outDirLocal, // direction to the rc vertex from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            xkminus1_backface,
            xkminus1_f_val,
            xkminus1_pdf,
            xkminus1_pdf_marg,
            xkminus1_uv,
            TRANSPORTMODE_RADIANCE, xkminus1_lod
        );
    } else {
        // k=d nee: the last segment was sampled via the marginalized bsdf
        f_pdf_eval(
            params.shadeContext.materials,
            xkminus1_materialID,
            params.shadeContext.textures,
            xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
            xkminus1_outDirLocal, // direction to the rc vertex aka light from y_k-1
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            xkminus1_f_val,
            xkminus1_pdf,
            xkminus1_uv,
            TRANSPORTMODE_RADIANCE, xkminus1_lod
        );
        // evaluated marginally, so its marginal == pdf
        xkminus1_pdf_marg = xkminus1_pdf;
    }

    // ---- 8. rng padding from x_k-1 up to entering x_k ------------------------
    // k=d never reads localState again (there is no x_k scattering), so it is skipped.
    if (!kIsD) {
        if (xkminus1IsPrimary) {
            // primary order: after the bsdf sample, only the RR roll remains
            rand(&localState); // RR for x_k-1
        } else {
            // secondary order: sample, then emissive roll, then NEE, then RR
            if (xkminus1_emissive) {
                rand(&localState); // if emissive, consume one rand for reservoir roll
            }

            // k<d-1: pad whenever x_k-1 is non-delta (it cast its own NEE).
            // k=d-1: only the bsdf path reads the stream past this point, and there
            //        x_k-1 MUST be non-delta (guaranteed by the dual footprint check).
            const bool padNee = kLessDm1 ? xkminus1NonDelta : bsdfPath;
            if (padNee) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // for the reservoir roll
            }

            // to roll RR for x_k-1
            rand(&localState);
        }
    }

    // now the rng is the correct rng entering x_k, and because the first rng consumption
    // comes from the sampling given no env hit, we can evaluate x_k sampling now

    // ---- 9. throughput scaling leaving x_k-1 --------------------------------
    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    // for k=d nee the last segment generation pdf is the light pdf, not the bsdf pdf
    float xkminus1_p_sampled = (kIsD && neePath) ? pdf_sampledLight_nee : xkminus1_pdf;

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / xkminus1_p_sampled;

    // ---- 10. RR at x_k -------------------------------------------------------
    // k<d-1: x_k always continued, so it rolled RR.
    // k=d-1: only the bsdf path bounced out of x_k; nee is cast before the x_k RR.
    // k=d:   x_k is the light, no RR.
    // Since RR is done after bsdf sampling, we dont need to pad an rng here.
    if (kLessDm1 || (kIsDm1 && bsdfPath)) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // ---- 11. bsdf at x_k + throughput scaling leaving x_k --------------------
    // Guarded: for k=d, x_k is a light/env vertex and rc_xk_materialID / uv / pos are
    // not even initialized by the caller, so this block must not be entered.
    float rc_xk_bsdf_pdf = 0.0f;
    float rc_xk_bsdf_pdf_marg = 0.0f;
    float rc_xk_incoming_cosine = 0.0f;

    if (!kIsD) {
        // This engine has the incoming direciton pointing into the surface
        float3 rc_xk_inDirLocal = toLocal(xkminus1_to_xk_direction_normalized, rc_xk_normal);
        float3 rc_xk_outDirLocal = toLocal(rcWi, rc_xk_normal);
        float rc_xk_outgoing_cosine = fabsf(rc_xk_outDirLocal.z);
        rc_xk_incoming_cosine = fabsf(rc_xk_inDirLocal.z);

        float3 rc_xk_f_val;

        // k=d-1 nee is the only sub-case where the outgoing segment at x_k is the nee
        // connection rather than a bsdf sample, so it is evaluated marginally.
        const bool xkUsesLobeReplay = kLessDm1 || bsdfPath;

        if (xkUsesLobeReplay) {
            f_pdf_eval_replayLobe(
                localState,
                params.shadeContext.materials,
                rc_xk_materialID,
                params.shadeContext.textures,
                rc_xk_inDirLocal, // direction to the rc vertex from y_k-1
                rc_xk_outDirLocal, // direction leaving the rc vertex
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                rc_xk_backface,
                rc_xk_f_val,
                rc_xk_bsdf_pdf,
                rc_xk_bsdf_pdf_marg,
                rc_xk_uv,
                TRANSPORTMODE_RADIANCE, rc_lod
            );
        } else {
            f_pdf_eval(
                params.shadeContext.materials,
                rc_xk_materialID,
                params.shadeContext.textures,
                rc_xk_inDirLocal, // direction to the rc vertex from y_k-1
                rc_xk_outDirLocal, // direction leaving the rc vertex
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                rc_xk_f_val,
                rc_xk_bsdf_pdf,
                rc_xk_uv,
                TRANSPORTMODE_RADIANCE, rc_lod
            );
            // nee case evaluated the x_k bsdf marginally (f_pdf_eval), so its marginal == pdf.
            rc_xk_bsdf_pdf_marg = rc_xk_bsdf_pdf;
        }

        float rc_xk_p_sampled_light = (kIsDm1 && neePath) ? pdf_sampledLight_nee : rc_xk_bsdf_pdf;

        throughput *= rc_xk_f_val * rc_xk_outgoing_cosine / rc_xk_p_sampled_light;
    }

    // ---- 12. path MIS + contribution ----------------------------------------
    // k<d-1 has no path MIS here (rcRadiance already carries the suffix mis term).
    // k=d-1 MISes at x_k, k=d MISes at x_k-1; both use the MARGINAL bsdf pdf
    // (throughput/jacobian keep the lobe-specific one).
    float mis_bsdf_pdf_marg = kIsD ? xkminus1_pdf_marg : rc_xk_bsdf_pdf_marg;
    float path_misWeight = kLessDm1 ? 1.0f : powerHeuristicTwoStrategy(
        neePath ? pdf_sampledLight_nee : mis_bsdf_pdf_marg,
        neePath ? mis_bsdf_pdf_marg : pdf_sampledLight_nee
    );

    result.contribution = throughput * rcRadiance * path_misWeight;

    // ---- 13. jacobian --------------------------------------------------------
    float jacobian_numerator;
    if (kIsD) {
        // only meaningful for the env+bsdf direction-copy case below
        jacobian_numerator = xkminus1_pdf;
    } else {
        float geometryTerm = rc_xk_incoming_cosine / (xkminus1_to_xk_distance * xkminus1_to_xk_distance);
        jacobian_numerator = (kIsDm1 && neePath) ?
            (xkminus1_pdf * geometryTerm) : // NEE case: the last pdf cancels out since the light selection pdf is equal for both domains paths
            (xkminus1_pdf * geometryTerm * rc_xk_bsdf_pdf) // bsdf case: the normal jacobian
        ;
    }

    // k=d is a full random replay path (jacobian 1), EXCEPT the bsdf direction copy
    // into the environment, which reduces variance but has a non-identity jacobian.
    if (kIsD && !(bsdfPath && envEnd)) {
        result.jacobian = 1.0f;
        result.new_cached_jacobian = 1.0f;
    } else {
        result.jacobian = jacobian_numerator / jacobian_denominator;
        result.new_cached_jacobian = jacobian_numerator;
    }

    result.isValid = true;

    // ---- 14. safety checks ---------------------------------------------------
    // (all aborts return the same value, so the ordering between them is irrelevant)

    // k<d: the jacobian is always live, so both ends must be positive.
    if (!kIsD && (jacobian_numerator <= 0.0f || jacobian_denominator <= 0.0f)) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // k=d: only the env bsdf direction copy divides by the denominator.
    if (kIsD && bsdfPath && envEnd && jacobian_denominator <= 0.0f) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // k=d: the generation pdf of the last segment (light pdf for nee, bsdf pdf otherwise).
    if (kIsD && xkminus1_p_sampled <= 0.0f) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // Ensure the BSDF at y_{k-1} can actually scatter towards x_k. Even on an NEE
    // path, if the BSDF PDF is zero the connection is physically impossible.
    if (xkminus1_pdf <= 0.0f) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // k<d: same for x_k, except k=d-1 nee where the last pdf is the light pdf, not the x_k one.
    if (!kIsD && !(kIsDm1 && neePath) && rc_xk_bsdf_pdf <= 0.0f) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    // Ensure the final target function evaluates to a valid, non-zero weight.
    if (targetFunction(result.contribution) <= 0.0f) {
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}
