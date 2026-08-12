#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include "textureView.cuh"
#include <curand_kernel.h>

// Compile-time switch for ReSTIR PT BSDF evaluation.
//   1 = lobe-specific: sample_f_eval_lobe / f_pdf_eval_replayLobe do per-lobe work
//       (what the shift's lobe replay needs).
//   0 = marginalized only: both forward to the marginal sample_f_eval / f_pdf_eval,
//       reproducing the pre-lobe behavior. The shift helpers' extra rng padding is
//       left in place but is harmless (marginal eval ignores the rng state).
// Lives here (not settings.cuh) because reflectors.cuh is shared with the software
// renderer, which doesn't have optixBranch/settings.cuh on its include path.
#ifndef LOBE_SPECIFIC_BSDF
#define LOBE_SPECIFIC_BSDF 1
#endif

__device__ inline void cosine_f(const float3& baseColor, float3& newColor)
{
    newColor = baseColor/PI;
}

__device__ inline void cosine_pdf(const float3& wo_local, float& pdf)
{
    pdf = fmaxf(wo_local.z, EPSILON)/PI;
}


__device__ inline void cosine_sample_f(RNGState& localState, const float3& baseColor, float3& wo, float3& f_val, float& pdf)
{
    float u1 = rand(&localState);

    u1 = fminf(u1, 1.0f-EPSILON);
    float u2 = rand(&localState);

    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(1.0f - u1);

    wo = f3(x,y,z);

    cosine_f(baseColor, f_val);
    cosine_pdf(wo, pdf);
}

__device__ inline void cosine_emit(RNGState& localState, float3& wo, float& pdf)
{
    float u1 = rand(&localState);

    u1 = fminf(u1, 1.0f-EPSILON);
    float u2 = rand(&localState);

    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(1.0f - u1);

    wo = f3(x,y,z);
    cosine_pdf(wo, pdf);
}

__device__ inline void mirror_f(float3& f_val, float3 wo)
{
    float cos_theta = fmaxf(wo.z, EPSILON);
    f_val = f3(1.0f / cos_theta);
}

__device__ inline void mirror_pdf(float& pdf)
{
    pdf = 1.0f;
}

__device__ inline void mirror_sample_f(float3 wi, float3& wo, float3& f_val, float& pdf)
{
    wo = f3(-wi.x, -wi.y, wi.z);
    float cos_theta = fmaxf(wo.z, EPSILON);
    f_val = f3(1.0f / cos_theta);
    pdf = 1.0f;
}

__device__ inline float D_GGX(const float3& h, float alpha)
{
    float cosThetaH = h.z;
    float alpha2 = alpha * alpha;
    float denom = cosThetaH*cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom);
}

__device__ inline float G1_Smith(float nDotV, float alpha)
{
    float a = alpha * sqrtf(1.0f - nDotV*nDotV) / nDotV;
    return 2.0f / (1.0f + sqrtf(1.0f + a*a));
}

__device__ inline float G1_GGX(const float3& v, const float3& h, float alpha)
{
    float cosTheta = v.z;
    float tanTheta = sqrtf(1.0f - cosTheta * cosTheta) / cosTheta;
    float a = 1.0f / (alpha * tanTheta);
    if (a < 1.6f)
        return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
    else
        return 1.0f;
}

__device__ inline float G_Smith(const float3& wi, const float3& wo, const float3& h, float alpha)
{
    return G1_GGX(wi, h, alpha) * G1_GGX(wo, h, alpha);
}

__device__ inline float3 Fresnel_Conductor(float cosTheta, const float3& eta, const float3& k) {
    float3 cosTheta2 = f3(cosTheta*cosTheta);
    float3 sinTheta2 = f3(1.0f) - cosTheta2;

    float3 eta2 = eta * eta;
    float3 k2 = k * k;

    float3 t0 = eta2 - k2 - sinTheta2;
    float3 a2plusb2 = sqrtf3(t0*t0 + 4.0f * eta2 * k2);
    float3 t1 = a2plusb2 + cosTheta2;
    float3 a = sqrtf3(0.5f * (a2plusb2 + t0));
    float3 t2 = 2.0f * cosTheta * a;

    float3 Rs = (t1 - t2) / (t1 + t2);
    float3 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    float3 t4 = t2 * sinTheta2;
    float3 Rp = (t3 - t4) / (t3 + t4);
    //return (Rs + Rp) * 0.5f;
    return (t1 - t2) / (t1 + t2);
}

__device__ inline void microfacet_metal_f(const float3& eta, const float3& k, float roughness, const float3& wi, const float3& wo, float3& f_val)
{
    if (wi.z <= 0.0f || wo.z <= 0.0f) {
        f_val = f3(0.0f);
        return;
    }
    float nDotWi = wi.z;
    float nDotWo = wo.z;

    float3 h = normalize(wi + wo);

    if (h.z <= 0.0f) h = f3(-h.x, -h.y, -h.z); // flip so h.z > 0

    float alpha = roughness * roughness;

    float D = D_GGX(h, alpha);
    float G = G_Smith(wi, wo, h, alpha);

    float3 f = Fresnel_Conductor(dot(wi, h), eta, k);

    f_val = (D * G * f) / fmaxf(4.0f * nDotWi * nDotWo, EPSILON);
}

__device__ inline void microfacet_pdf(const float& roughness, const float3& wi, const float3& wo, float& pdf)
{
    float3 h = normalize(wi + wo);
    float D = D_GGX(h, roughness *roughness);
    float denom = 4.0f * dot(wo, h);
    pdf = (D * h.z) / (denom);
}

__device__ inline void microfacet_metal_sample_f(RNGState& localState, const float3& eta, const float3& k, float roughness, const float3& wi,
    float3& wo, float3& f_val, float& pdf)
{
    float u1 = rand(&localState);

    float alpha = roughness * roughness;
    float phi = 2.0f * PI * rand(&localState);
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha*alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));

    float3 h_local;
    h_local.x = sinTheta * cosf(phi);
    h_local.y = sinTheta * sinf(phi);
    h_local.z = cosTheta; // points along local normal

    wo = 2.0f * dot(wi, h_local) * h_local - wi;
    if (wo.z <= 0.0f) wo.z = -wo.z;

    microfacet_metal_f(eta, k, roughness, wi, wo, f_val); // f_val set
    microfacet_pdf(roughness, wi, wo, pdf); // pdf set
}

// fabs is used here for costheta
__device__ inline float schlick_fresnel(float cosTheta, float etaI, float etaT)
{
    float R0 = (etaI - etaT) / (etaI + etaT);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf(1.0f - fabsf(cosTheta), 5.0f);
}

// -----------------------------------------------------------------------------
// EVAL (given wi, wo, etaI, etaT, and reflect boolean)
// -----------------------------------------------------------------------------
__device__ inline void smooth_dielectric_f(
    const float3& wi, const float3& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float3& f_val)
{
    printf("never called");
    float cosThetaI = wi.z;
    float cosThetaO = wo.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        // Perfect specular reflection: wo.z = wi.z
        // BSDF value for delta reflection (handled as Dirac delta)
        if (TIR)
            f_val = f3(1.0f / fabsf(cosThetaO));
        else
            f_val = f3(F / fabsf(cosThetaO));
    } else {
        // Perfect specular refraction
        float eta = etaI / etaT;
        f_val = f3((1.0f - F) * eta * eta / fabsf(cosThetaO));
    }
}

// -----------------------------------------------------------------------------
// SAMPLE_F (importance sample the reflection/refract direction)
// -----------------------------------------------------------------------------
__device__ inline void smooth_dielectric_sample_f(RNGState& localState,
    const float3& wi, float etaI, float etaT, float3& wo, float3& f_val, float& pdf)
{
    float cosThetaI = wi.z; // always pos

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;
    if (cosThetaT2 < 0.0f)
    {
        wo = f3(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;

        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f3(0.0f);
            return;
        }

        f_val = f3(1.0f / cosThetaO);
        pdf = 1.0f;
        return;
    }

    F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (rand(&localState) < F) {
        wo = f3(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f3(0.0f);
            return;
        }
        pdf = F;
        f_val = f3(F / fabsf(cosThetaO));
    }
    else {
        float eta = etaI / etaT;

        wo = f3(
            -eta * wi.x,
            -eta * wi.y,
            - (sqrtf(cosThetaT2))
        );
        wo = normalize(wo);

        float cosThetaO = wo.z;

        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f3(0.0f);
            return;
        }
        // BSDF term
        f_val = f3((1.0f - F) * eta * eta / fabsf(cosThetaO));
        pdf = 1.0f - F;
    }

}

// -----------------------------------------------------------------------------
// PDF (for MIS weighting)
// -----------------------------------------------------------------------------
__device__ inline void smooth_dielectric_pdf(
    const float3& wi, const float3& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float& pdf)
{
    printf("never called");
    float cosThetaI = wi.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        if (TIR)
            pdf = 1.0f;
        else
            pdf = F;
    } else {
        pdf = 1.0f - F;
    }
}
// wi is always point away from the surface on the positive z hemisphere. since we flipped the intersect normal if it was a backface before converting to local
__device__ inline void dumb_smooth_dielectric_sample_f(RNGState& localState,
    const float3& wi, float etaSurface, bool backface, int transportMode, float3& wo, float3& f_val, float& pdf)
{
    float etaI, etaT;
    if (backface)
    {
        etaI = etaSurface;
        etaT = 1.0f;
    }
    else
    {
        etaI = 1.0f;
        etaT = etaSurface;
    }

    float cosThetaI = fminf(fmaxf(wi.z, EPSILON), 1.0f);

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;

    F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (cosThetaT2 < 0.0f || F >= 0.99999f)
    {
        wo = f3(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        f_val = f3(1.0f / fmaxf(cosThetaO, EPSILON));
        pdf = 1.0f;
        return;
    }

    if (rand(&localState) < F) {
        wo = f3(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        pdf = F;
        f_val = f3(F / fmaxf(cosThetaO, EPSILON));
    }
    else {
        float eta = etaI / etaT;

        wo = f3(
            -eta * wi.x,
            -eta * wi.y,
            - (sqrtf(cosThetaT2))
        );

        float cosThetaO = wo.z;

        // BSDF term
        float denom = fmaxf(fabsf(cosThetaO), EPSILON);
        f_val = f3((1.0f - F)/ denom);
        pdf = 1.0f - F;

        // adjoint bsdf handling
        if (transportMode == TRANSPORTMODE_IMPORTANCE)
        {
            // do nothing, throughput conserved
        }
        else if (transportMode == TRANSPORTMODE_RADIANCE)
        {
            f_val *= eta * eta;
        }
    }
}
__device__ inline void thin_dielectric_sample_f(RNGState& localState,
    const float3& wi, float etaSurface, bool backface, int transportMode, float3& wo, float3& f_val, float& pdf)
{
    // 1. Setup IOR
    // A thin wall is effectively a sheet suspended in air (1.0).
    // The interaction "Air -> Material -> Air" is symmetric, so we don't swap
    // IORs based on backface. We always calculate F for Air(1.0) hitting Surface(eta).
    float etaI = 1.0f;
    float etaT = etaSurface;

    float cosThetaI = fminf(fmaxf(wi.z, EPSILON), 1.0f);

    // 2. Calculate Single-Interface Fresnel
    float F_single = schlick_fresnel(cosThetaI, etaI, etaT);

    // 3. Adjust for Thin-Walled "Double Reflection"
    // F_thin accounts for the infinite series of internal bounces
    // (front reflection + transmitted-then-back-reflected light).
    // Formula: R = 2F / (1 + F)
    float F = (2.0f * F_single) / (1.0f + F_single);

    // 4. Sample
    if (rand(&localState) < F)
    {
        // --- REFLECTION ---
        // Standard reflection vector
        wo = f3(-wi.x, -wi.y, wi.z);

        float cosThetaO = wo.z;

        pdf = F;

        // BSDF = F * (1 / |cosTheta|)
        f_val = f3(F / fmaxf(cosThetaO, EPSILON));
    }
    else
    {
        // --- TRANSMISSION ---
        // Pass straight through (wo = -wi).
        // No refraction bending.
        wo = f3(-wi.x, -wi.y, -wi.z);

        float cosThetaO = wo.z;

        pdf = 1.0f - F;

        // BSDF = (1-F) * (1 / |cosTheta|)
        // Note: No 'eta * eta' scaling because the ray starts and ends
        // in the same medium (Air), so radiance is conserved naturally.
        float denom = fmaxf(fabsf(cosThetaO), EPSILON);
        f_val = f3((1.0f - F) / denom);
    }
}

// convention: wi always faces away from the surface (same dir as surface normal)
__device__ inline void leaf_f(const float3& albedo, float ior, float currIOR, float roughness, float transmission, const float3& wi, const float3& wo, float3& f_val)
{
    bool is_reflection = wo.z * wi.z > 0.0f;

    float3 h;
    float F;

    F = schlick_fresnel(wi.z, currIOR, ior);

    if (is_reflection) // if it reflected with respect to the surface normal (add f_val from both events, reflection and diffuse)
    {
        h = normalize(wi + wo);
        float microfacet_F = schlick_fresnel(dot(wi, h), currIOR, ior);
        float nDotWi = wi.z;
        float nDotWo = wo.z;

        if (h.z <= 0.0f) h = -h; // flip so h.z > 0

        float alpha = roughness * roughness;

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);

        float3 f_cuticle = f3(D * G * microfacet_F / fmaxf(4.0f * nDotWi * nDotWo, EPSILON));

        if (microfacet_F < 0.0f) printf("negative microfacetF");

        if (microfacet_F > 1.0f) printf("greater than 1 microfacetF %f: dot = %f, currIOR = %f, ior = %f\n", microfacet_F, dot(wi, h), currIOR, ior);

        float3 f_diffuse_val;
        cosine_f(albedo, f_diffuse_val);

        f_val = (1.0f-microfacet_F) * (1.0f - transmission) * f_diffuse_val + f_cuticle;
    }
    else
    {
        cosine_f(albedo, f_val);
        f_val *= transmission * (1.0f - F);
    }
}

__device__ inline void leaf_pdf(float ior, float currIOR, float roughness, float transmission, const float3& wi, const float3& wo, float& pdf)
{
    bool is_reflection = wo.z * wi.z > 0.0f;

    float3 h;
    float F;

    F = schlick_fresnel(abs(wi.z), currIOR, ior);

    F = fminf(F, 1.0f - 0.1f * roughness);

    float p_specular = F;
    float p_diffuse_refl = (1.0f - F) * (1.0f - transmission);
    float p_diffuse_trans = (1.0f - F) * transmission;

    if (is_reflection) // if it reflected with respect to the surface normal
    {
        h = normalize(wi + wo);
        if (h.z < 0.0f) h = -h;

        //float nDotWi = wi.z;
        //float nDotWo = wo.z;
        float alpha = roughness * roughness;

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);

        float denom = 4.0f * dot(wo, h);
        float pdf_cuticle_bounce = (D * h.z) / (denom);

        float pdf_diffuse;

        cosine_pdf(wo, pdf_diffuse);

        pdf = (p_specular * pdf_cuticle_bounce) + (p_diffuse_refl * pdf_diffuse);
    }
    else
    {
        float pdf_trans;
        cosine_pdf(-wo, pdf_trans);

        pdf = pdf_trans * p_diffuse_trans;
    }
}

__device__ inline void leaf_sample_f(RNGState& localState, const float3& wi, float ior, float currIOR, float roughness, const float3& albedo, float transmission, float3& wo, float3& f_val, float& pdf)
{
    float F = schlick_fresnel(wi.z, currIOR, ior);

    if (rand(&localState) < F) // reflection on cuticle layer
    {
        float u1 = rand(&localState);

        float alpha = roughness * roughness;
        float phi = 2.0f * PI * rand(&localState);
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha*alpha - 1.0f) * u1));
        float sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));

        float3 h_local;
        h_local.x = sinTheta * cosf(phi);
        h_local.y = sinTheta * sinf(phi);
        h_local.z = cosTheta; // points along local normal

        wo = 2.0f * dot(wi, h_local) * h_local - wi;
    }
    else // transmit through cuticle
    {
        if (rand(&localState) < transmission) // go through leaf
        {
            cosine_sample_f(localState, albedo, wo, f_val, pdf);
            wo.z = -wo.z;
        }
        else // diffuse bounce off leaf
        {
            cosine_sample_f(localState, albedo, wo, f_val, pdf);
        }
    }

    leaf_f(albedo, ior, currIOR, roughness, transmission, wi, wo, f_val);
    leaf_pdf(ior, currIOR, roughness, transmission, wi, wo, pdf);
}

__device__ inline void microfacet_dielectric_f(
    float etaI, float etaT, float roughness,
    const float3& wi, const float3& wo, float3& f_val)
{
    // Determine if this is reflection or transmission based on hemispheres
    bool is_reflection = wo.z * wi.z > 0.0f;

    float cosThetaWi = wi.z;
    float cosThetaWo = wo.z;
    float alpha = roughness * roughness;

    if (is_reflection)
    {
        // Reflection: Half vector is standard half-angle
        float3 h = normalize(wi + wo);
        if (h.z <= 0.0f) h = -h; // Ensure h points into upper hemisphere

        // Calculate D, G, F
        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);

        // Fresnel term (using dot(wi, h))
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);

        // Microfacet Reflection Formula
        // f_r = (D * G * F) / (4 * cosThetaWi * cosThetaWo)
        float denom = 4.0f * fabsf(cosThetaWi * cosThetaWo);
        f_val = f3((D * G * F) / fmaxf(denom, EPSILON));
    }
    else
    {
        // Transmission: Half vector depends on IOR
        // Walter et al. 2007 Eq. 16
        // h = - normalize(etaI * wi + etaT * wo)
        float3 h = normalize(etaI * wi + etaT * wo);
        if (h.z < 0.0f) h = -h; // Flip to align with normal

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        // Microfacet Transmission Formula
        // f_t = ( |wi.h| * |wo.h| * etaT^2 * D * G * (1-F) ) / ( |wi.n| * |wo.n| * (etaI * wi.h + etaT * wo.h)^2 )

        float sqrtDenom = etaI * wiDotH + etaT * woDotH;
        float denom = fmaxf(fabsf(cosThetaWi * cosThetaWo), EPSILON) * sqrtDenom * sqrtDenom;

        // Note: We include etaT^2 here as per standard derivation,
        // but see sample_f for final adjoint/transport handling.
        float num = fabsf(wiDotH * woDotH) * (etaT * etaT) * D * G * (1.0f - F);

        f_val = f3(num / fmaxf(denom, EPSILON));
    }
}

__device__ inline void microfacet_dielectric_pdf(
    float etaI, float etaT, float roughness,
    const float3& wi, const float3& wo, float& pdf)
{
    bool is_reflection = wo.z * wi.z > 0.0f;
    float alpha = roughness * roughness;

    if (is_reflection)
    {
        float3 h = normalize(wi + wo);
        if (h.z < 0.0f) h = -h;

        // Jacobian d_wh / d_wo = 1 / (4 * wo.h)
        float D = D_GGX(h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);

        // We weigh the PDF by F because we chose reflection with probability F
        float pdf_h = (D * h.z);
        float denom = 4.0f * fabsf(dot(wo, h));

        pdf = F * (pdf_h / fmaxf(denom, EPSILON));
    }
    else
    {
        float3 h = normalize(etaI * wi + etaT * wo);
        if (h.z < 0.0f) h = -h;

        float D = D_GGX(h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);
        float pdf_h = (D * h.z);

        // Jacobian for refraction
        // d_wh / d_wo = (etaT^2 * |wo.h|) / (etaI * wi.h + etaT * wo.h)^2
        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);
        float sqrtDenom = etaI * wiDotH + etaT * woDotH;

        float jacobian = (etaT * etaT * fabsf(woDotH)) / fmaxf(sqrtDenom * sqrtDenom, EPSILON);

        // We weigh PDF by (1-F) because we chose transmission with probability (1-F)
        pdf = (1.0f - F) * pdf_h * jacobian;
    }
}

__device__ inline void microfacet_dielectric_sample_f(RNGState& localState,
    const float3& wi, float etaSurface, float roughness, bool backface, int transportMode, float3& wo, float3& f_val, float& pdf)
{
    float etaI, etaT;
    if (backface)
    {
        etaI = etaSurface;
        etaT = 1.0f;
    }
    else
    {
        etaI = 1.0f;
        etaT = etaSurface;
    }

    // 1. Sample Microfacet Normal (h) using GGX
    float u1 = rand(&localState);
    float alpha = roughness * roughness;
    float phi = 2.0f * PI * rand(&localState);

    // Standard GGX sampling (matching leaf_sample_f implementation)
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));

    float3 h;
    h.x = sinTheta * cosf(phi);
    h.y = sinTheta * sinf(phi);
    h.z = cosTheta;

    // Ensure h aligns with wi to avoid artifacts
    if (dot(wi, h) < 0.0f) h = -h;

    // 2. Fresnel and Selection
    float wiDotH = dot(wi, h);
    float F = schlick_fresnel(fabsf(wiDotH), etaI, etaT);

    if (rand(&localState) < F)
    {
        // --- Reflection ---
        wo = 2.0f * wiDotH * h - wi;
    }
    else
    {
        // --- Transmission (Refraction) ---
        float eta = etaI / etaT;
        float c = dot(wi, h);
        float term = 1.0f - eta * eta * (1.0f - c * c);

        // Check for Total Internal Reflection (TIR)
        if (term < 0.0f)
        {
            // TIR: Default to reflection (probability of TIR is handled by F going to 1.0 usually,
            // but for safety in microfacet sampling, we reflect)
            wo = 2.0f * c * h - wi;
        }
        else
        {
            // Refract vector
            wo = (eta * c - sqrtf(term)) * h - eta * wi;
        }
    }

    // 3. Evaluate f and pdf
    microfacet_dielectric_f(etaI, etaT, roughness, wi, wo, f_val);
    microfacet_dielectric_pdf(etaI, etaT, roughness, wi, wo, pdf);

    // 4. Adjoint / Transport Mode Handling
    // Matching 'dumb_smooth_dielectric' convention:
    // If we transmitted (signs differ) and we are in Radiance mode, scale by eta^2.
    // (eta = etaI / etaT)
    if (wo.z * wi.z < 0.0f && transportMode == TRANSPORTMODE_RADIANCE)
    {
        float eta = etaI / etaT;
        f_val *= eta * eta;
    }
}

// =============================================================================
// glTF metallic-roughness principled BSDF (MAT_GLTF_PRINCIPLED_BSDF)
//
// Two-lobe, opaque, energy-conserving per the glTF 2.0 spec (Appendix B):
//   f = (1 - F) * baseColor*(1-metallic)/PI   +   F * D * G / (4 nv nl)
// with F0 = lerp(0.04, baseColor, metallic), Schlick Fresnel.
//
// MARGINALIZED (evaluate-all-lobes) formulation: sample_f picks a lobe only to
// generate a direction, then returns the FULL mixed f and the FULL marginal pdf
// (identical to what f_eval / pdf_eval return for the same directions). The lobe
// choice is a replayable RNG draw, so this stays compatible with ReSTIR shifts.
//
// Convention: v and l are both local, z-up, pointing AWAY from the surface
// (callers pass v = -wi, l = wo), matching the microfacet_* helpers.
// =============================================================================

// Colored Schlick Fresnel from reflectance-at-normal-incidence F0.
__device__ __forceinline__ float3 fresnelSchlickF0(float cosTheta, const float3& F0)
{
    float m  = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m5 = (m * m) * (m * m) * m;
    return F0 + (f3(1.0f) - F0) * m5;
}

// Lobe-selection probability. MUST be identical in sample_f and pdf, so it lives
// in one place. Naturally -> ~1 for pure metal (cDiff -> 0), floored so a
// dielectric's specular lobe is always samplable.
__device__ __forceinline__ float principled_specProb(const float3& F0, const float3& cDiff)
{
    float ls = luminance(F0);
    float ld = luminance(cDiff);
    return fmaxf(ls / (ls + ld + 1e-4f), 0.1f);
}

__device__ inline void principled_f(const float3& baseColor, float metallic, float roughness,
    const float3& v, const float3& l, float3& f_val)
{
    if (v.z <= 0.0f || l.z <= 0.0f) { f_val = f3(0.0f); return; }

    roughness = fmaxf(roughness, 0.025f); // avoid delta-limit NaNs at roughness 0
    float alpha = roughness * roughness;

    float3 h  = normalize(v + l);
    float3 F0 = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
    float3 F  = fresnelSchlickF0(fmaxf(dot(v, h), 0.0f), F0);

    float D = D_GGX(h, alpha);
    float G = G_Smith(v, l, h, alpha);

    float3 spec  = (D * G) * F / fmaxf(4.0f * v.z * l.z, EPSILON);
    float3 cDiff = baseColor * (1.0f - metallic);
    float3 diff  = (f3(1.0f) - F) * cDiff * (1.0f / PI);

    f_val = diff + spec;
}

__device__ inline void principled_pdf(const float3& baseColor, float metallic, float roughness,
    const float3& v, const float3& l, float& pdf)
{
    if (v.z <= 0.0f || l.z <= 0.0f) { pdf = 0.0f; return; }

    roughness = fmaxf(roughness, 0.025f);
    float alpha = roughness * roughness;

    float3 h = normalize(v + l);
    float D  = D_GGX(h, alpha);

    float specPdf = D * h.z / fmaxf(4.0f * dot(v, h), EPSILON);
    float diffPdf = l.z * (1.0f / PI);

    float3 F0    = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
    float3 cDiff = baseColor * (1.0f - metallic);
    float pSpec  = principled_specProb(F0, cDiff);

    pdf = pSpec * specPdf + (1.0f - pSpec) * diffPdf;
}

__device__ inline void principled_sample_f(RNGState& localState, const float3& baseColor, float metallic, float roughness,
    const float3& v, float3& l, float3& f_val, float& pdf)
{
    roughness = fmaxf(roughness, 0.025f);
    float alpha = roughness * roughness;

    float3 F0    = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
    float3 cDiff = baseColor * (1.0f - metallic);
    float pSpec  = principled_specProb(F0, cDiff);

    if (rand(&localState) < pSpec)
    {
        // GGX NDF half-vector sampling (matches microfacet_metal_sample_f)
        float u1  = rand(&localState);
        float phi = 2.0f * PI * rand(&localState);
        float cosT = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
        float sinT = sqrtf(fmaxf(1.0f - cosT * cosT, 0.0f));
        float3 h = f3(sinT * cosf(phi), sinT * sinf(phi), cosT);
        l = 2.0f * dot(v, h) * h - v; // reflect v about h
    }
    else
    {
        // cosine-weighted diffuse
        float pdfTmp;
        cosine_emit(localState, l, pdfTmp);
    }

    if (l.z <= 0.0f) { f_val = f3(0.0f); pdf = 0.0f; return; }

    // Return the FULL mixed value and FULL marginal pdf (not the chosen lobe's).
    principled_f(baseColor, metallic, roughness, v, l, f_val);
    principled_pdf(baseColor, metallic, roughness, v, l, pdf);
}

// =============================================================================
// Lobe-specific principled BSDF (for ReSTIR PT shift mappings).
//
// The marginalized principled_* above blend BOTH lobes -- ideal for NEE and MIS,
// but a blended pdf contaminates a specular reconnection's Jacobian. These split
// the same diffuse/specular terms into a SINGLE lobe so the shift can evaluate
// exactly the lobe the base path used -- reconstructed via RNG replay, so no lobe
// index is ever stored.
//
//   p_total (generation pdf) = P_select * p_dir       // what the shift / throughput use
//   marginal pdf (above)     = sum over lobes of p_total
// =============================================================================

enum PrincipledLobe { PRINCIPLED_LOBE_DIFFUSE = 0, PRINCIPLED_LOBE_SPECULAR = 1 };

// Pick a lobe from one uniform sample. Mirrors the branch in principled_sample_f
// (u < pSpec -> specular). Returns the lobe and its discrete selection probability.
__device__ __forceinline__ int principled_lobeFromU(float u, const float3& F0, const float3& cDiff, float& P_select)
{
    float pSpec = principled_specProb(F0, cDiff);
    if (u < pSpec) { P_select = pSpec;        return PRINCIPLED_LOBE_SPECULAR; }
    else           { P_select = 1.0f - pSpec; return PRINCIPLED_LOBE_DIFFUSE;  }
}

// Evaluate ONLY the given lobe's contribution (no blend). Same terms as principled_f.
__device__ inline void principled_f_lobe(const float3& baseColor, float metallic, float roughness,
    const float3& v, const float3& l, int lobe, float3& f_val)
{
    if (v.z <= 0.0f || l.z <= 0.0f) { f_val = f3(0.0f); return; }

    roughness = fmaxf(roughness, 0.025f);
    float alpha = roughness * roughness;

    float3 h  = normalize(v + l);
    float3 F0 = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
    float3 F  = fresnelSchlickF0(fmaxf(dot(v, h), 0.0f), F0);

    if (lobe == PRINCIPLED_LOBE_SPECULAR) {
        float D = D_GGX(h, alpha);
        float G = G_Smith(v, l, h, alpha);
        f_val = (D * G) * F / fmaxf(4.0f * v.z * l.z, EPSILON);
    } else {
        float3 cDiff = baseColor * (1.0f - metallic);
        f_val = (f3(1.0f) - F) * cDiff * (1.0f / PI);
    }
}

// Directional pdf of ONLY the given lobe (NOT multiplied by P_select). Same terms
// as the two halves of principled_pdf.
__device__ inline void principled_pdf_lobe(const float3& baseColor, float metallic, float roughness,
    const float3& v, const float3& l, int lobe, float& pdf)
{
    if (v.z <= 0.0f || l.z <= 0.0f) { pdf = 0.0f; return; }

    roughness = fmaxf(roughness, 0.025f);
    float alpha = roughness * roughness;

    if (lobe == PRINCIPLED_LOBE_SPECULAR) {
        float3 h = normalize(v + l);
        float D  = D_GGX(h, alpha);
        pdf = D * h.z / fmaxf(4.0f * dot(v, h), EPSILON);
    } else {
        pdf = l.z * (1.0f / PI);
    }
}

// Lobe-specific counterpart of principled_sample_f: picks a lobe (1 rng draw) then a
// direction (2 draws) -- SAME rng footprint -- but returns the chosen lobe's f and its
// generation pdf p_total = P_select * p_dir (not the blended f / marginal pdf).
__device__ inline void principled_sample_f_lobe(RNGState& localState, const float3& baseColor, float metallic, float roughness,
    const float3& v, float3& l, float3& f_val, float& pdf)
{
    roughness = fmaxf(roughness, 0.025f);
    float alpha = roughness * roughness;

    float3 F0    = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
    float3 cDiff = baseColor * (1.0f - metallic);

    float P_select;
    int lobe = principled_lobeFromU(rand(&localState), F0, cDiff, P_select);

    if (lobe == PRINCIPLED_LOBE_SPECULAR)
    {
        float u1  = rand(&localState);
        float phi = 2.0f * PI * rand(&localState);
        float cosT = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
        float sinT = sqrtf(fmaxf(1.0f - cosT * cosT, 0.0f));
        float3 h = f3(sinT * cosf(phi), sinT * sinf(phi), cosT);
        l = 2.0f * dot(v, h) * h - v; // reflect v about h
    }
    else
    {
        float pdfTmp;
        cosine_emit(localState, l, pdfTmp); // consumes 2 draws
    }

    if (l.z <= 0.0f) { f_val = f3(0.0f); pdf = 0.0f; return; }

    principled_f_lobe(baseColor, metallic, roughness, v, l, lobe, f_val);
    float p_dir;
    principled_pdf_lobe(baseColor, metallic, roughness, v, l, lobe, p_dir);
    pdf = P_select * p_dir;
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ inline void f_eval(const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, const float3& wo, float etaI, float etaT, float3& f_val, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_f(albedo, f_val);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_f(f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        //smooth_dielectric_f(-wi, wo, etaI, etaT, reflect_dielectric, TIR, f_val);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_f(albedo, mat.ior, etaI, mat.roughness, trans, -wi, wo, f_val);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_f(f_val, wo);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_f(etaI, etaT, mat.roughness, -wi, wo, f_val);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            // smooth-dielectric glass: delta BSDF, f not evaluated directly.
        }
        else
        {
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y; // glTF: roughness in G
                metallic  *= mr.z; // glTF: metallic in B
            }
            principled_f(albedo, metallic, roughness, -wi, wo, f_val);
        }
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ inline void sample_f_eval(RNGState& localState, const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, float etaI, float etaT, bool backface, float3& wo, float3& f_val, float& pdf, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_sample_f(localState, f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
        //smooth_dielectric_sample_f(localState, -wi, etaI, etaT, wo, f_val, pdf);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_sample_f(localState, -wi, mat.ior, etaI, mat.roughness, albedo, trans, wo, f_val, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_sample_f(-wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_THINDIELECTRIC)
    {
        thin_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_sample_f(localState, -wi, mat.ior, mat.roughness, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            // smooth-dielectric glass: full handoff to the tested dielectric
            // sampler; no principled lobes. Refraction comes from mat.ior +
            // backface (entry/exit through the closed mesh geometry).
            dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
        }
        else
        {
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y;
                metallic  *= mr.z;
            }
            principled_sample_f(localState, albedo, metallic, roughness, -wi, wo, f_val, pdf);
        }
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ inline void pdf_eval(const Material* __restrict__ materials, int materialID, const TextureView& textures, const float3& wi, const float3& wo,
    float etaI, float etaT, float& pdf, const float2 uv, float lod = 0.0f)
{
    const Material& mat = materials[materialID];
    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_pdf(mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        pdf = 999999999.0f;
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_pdf(mat.ior, etaI, mat.roughness, trans, -wi, wo, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_pdf(pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_pdf(etaI, etaT, mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            pdf = 999999999.0f; // delta dielectric
        }
        else
        {
            float3 albedo = f3(mat.albedo);
            if (mat.baseColorTex >= 0)
                albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y;
                metallic  *= mr.z;
            }
            principled_pdf(albedo, metallic, roughness, -wi, wo, pdf);
        }
    }
}
__device__ inline void f_pdf_eval(const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, const float3& wo, float etaI, float etaT, float3& f_val, float& pdf, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_f(albedo, f_val);
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_f(f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val);
        microfacet_pdf(mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        //smooth_dielectric_f(-wi, wo, etaI, etaT, reflect_dielectric, TIR, f_val);
        pdf = 999999999.0f;
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_f(albedo, mat.ior, etaI, mat.roughness, trans, -wi, wo, f_val);
        leaf_pdf(mat.ior, etaI, mat.roughness, trans, -wi, wo, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_f(f_val, wo);
        mirror_pdf(pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_f(etaI, etaT, mat.roughness, -wi, wo, f_val);
        microfacet_dielectric_pdf(etaI, etaT, mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            pdf = 999999999.0f; // delta dielectric; f not evaluated directly
        }
        else
        {
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y;
                metallic  *= mr.z;
            }
            principled_f(albedo, metallic, roughness, -wi, wo, f_val);
            principled_pdf(albedo, metallic, roughness, -wi, wo, pdf);
        }
    }
}

// =============================================================================
// ReSTIR PT lobe-specific dispatchers.
//
// sample_f_eval_lobe  -- forward BSDF-sampled continuation bounces.
// f_pdf_eval_replayLobe -- reconnection endpoints, where the outgoing direction is
//                          fixed and the lobe is reconstructed by replaying the rng.
//
// Both keep the SAME rng footprint as sample_f_eval, so a replayed stream stays
// aligned. Only the principled opaque BSDF is genuinely multi-lobe; every other
// material is single-lobe (p_total == marginal) and behaves identically to the
// marginal path. NEE and all MIS weights must keep using the marginal f_eval /
// pdf_eval -- do NOT route those through these.
// =============================================================================

// Lobe-specific continuation sampling. Identical to sample_f_eval except the
// principled opaque branch returns the sampled lobe's f and generation pdf (p_total)
// instead of the blended f / marginal pdf. No lobe index escapes.
__device__ inline void sample_f_eval_lobe(RNGState& localState, const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, float etaI, float etaT, bool backface, float3& wo, float3& f_val, float& pdf, float& pdf_marg, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    // pdf      = lobe-specific generation pdf (p_total) -> use for throughput and Jacobians.
    // pdf_marg = fully marginalized bsdf pdf of `wo`    -> use STRICTLY for MIS weights.
    // They are equal for every single-lobe material; they differ only for the
    // multi-lobe principled BSDF, where the marginal is filled below.
#if LOBE_SPECIFIC_BSDF
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    pdf_marg = -1.0f; // <0 sentinel: overwritten only by the multi-lobe principled path

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_sample_f(localState, f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_sample_f(localState, -wi, mat.ior, etaI, mat.roughness, albedo, trans, wo, f_val, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_sample_f(-wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_THINDIELECTRIC)
    {
        thin_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_sample_f(localState, -wi, mat.ior, mat.roughness, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
        }
        else
        {
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y;
                metallic  *= mr.z;
            }
            principled_sample_f_lobe(localState, albedo, metallic, roughness, -wi, wo, f_val, pdf);
            principled_pdf(albedo, metallic, roughness, -wi, wo, pdf_marg); // marginal, for MIS only
        }
    }

    if (pdf_marg < 0.0f) pdf_marg = pdf; // single-lobe materials: marginal == sampling pdf
#else
    sample_f_eval(localState, materials, materialID, textures, wi, etaI, etaT, backface, wo, f_val, pdf, uv, transportMode, lod);
    pdf_marg = pdf;
#endif
}

__device__ inline void sample_f_eval_lobe_returnRoughness(RNGState& localState, const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, float etaI, float etaT, bool backface, float3& wo, float3& f_val, float& pdf, float& pdf_marg, const float2 uv,
    float& rough, int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    // pdf      = lobe-specific generation pdf (p_total) -> use for throughput and Jacobians.
    // pdf_marg = fully marginalized bsdf pdf of `wo`    -> use STRICTLY for MIS weights.
    // They are equal for every single-lobe material; they differ only for the
    // multi-lobe principled BSDF, where the marginal is filled below.
#if LOBE_SPECIFIC_BSDF
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    float trans = mat.transmission;
    if (mat.transTex >= 0)
        trans = sampleTex(textures, mat.transTex, uv, lod).x;

    pdf_marg = -1.0f; // <0 sentinel: overwritten only by the multi-lobe principled path

    rough = mat.roughness;

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_sample_f(localState, f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_sample_f(localState, -wi, mat.ior, etaI, mat.roughness, albedo, trans, wo, f_val, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_sample_f(-wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_THINDIELECTRIC)
    {
        thin_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_sample_f(localState, -wi, mat.ior, mat.roughness, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF)
    {
        if (mat.isSpecular)
        {
            dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
        }
        else
        {
            float metallic = mat.metallic;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                rough *= mr.y;
                metallic  *= mr.z;
            }
            principled_sample_f_lobe(localState, albedo, metallic, rough, -wi, wo, f_val, pdf);
            principled_pdf(albedo, metallic, rough, -wi, wo, pdf_marg); // marginal, for MIS only
        }
    }

    if (pdf_marg < 0.0f) pdf_marg = pdf; // single-lobe materials: marginal == sampling pdf
#else
    sample_f_eval(localState, materials, materialID, textures, wi, etaI, etaT, backface, wo, f_val, pdf, uv, transportMode, lod);
    pdf_marg = pdf;
#endif
}

// Replay-and-evaluate for a reconnection endpoint. `wo` is fixed by the reconnection
// geometry. This rolls the SAME rng sequence sample_f_eval would have consumed for
// this material -- the lobe-selection draw reconstructs the lobe the base path used,
// the remaining direction draws are rolled purely to carry the stream forward -- then
// returns that lobe's f and generation pdf (p_total).
//
// Only the reconnectable rough materials are rng-aligned here: diffuse, metal, and the
// principled opaque BSDF. Delta materials are never reconnection endpoints. Leaf and
// microfacet-dielectric are intentionally NOT handled (their sample paths have
// branch-dependent draw counts that need padding first); they fall back to marginal
// evaluation WITHOUT advancing the stream, so they must not be used as endpoints yet.
__device__ inline void f_pdf_eval_replayLobe(RNGState& localState, const Material* __restrict__ materials, int materialID, const TextureView& textures,
    const float3& wi, const float3& wo, float etaI, float etaT, bool backface, float3& f_val, float& pdf, float& pdf_marg, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE, float lod = 0.0f)
{
    // pdf = lobe-specific p_total (throughput / Jacobian); pdf_marg = marginal (MIS only).
#if LOBE_SPECIFIC_BSDF
    const Material& mat = materials[materialID];
    float3 albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    pdf_marg = -1.0f; // <0 sentinel: overwritten only by the multi-lobe principled path

    if (mat.type == MAT_DIFFUSE)
    {
        rand(&localState); rand(&localState); // cosine_sample_f consumes 2 draws
        cosine_f(albedo, f_val);
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        rand(&localState); rand(&localState); // microfacet_metal_sample_f consumes 2 draws
        microfacet_metal_f(f3(mat.eta), f3(mat.k), mat.roughness, -wi, wo, f_val);
        microfacet_pdf(mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_GLTF_PRINCIPLED_BSDF && !mat.isSpecular)
    {
        float metallic = mat.metallic;
        float roughness = mat.roughness;
        if (mat.mrTex >= 0) {
            float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
            roughness *= mr.y;
            metallic  *= mr.z;
        }

        float3 v = -wi;
        float3 F0    = f3(0.04f) * (1.0f - metallic) + albedo * metallic;
        float3 cDiff = albedo * (1.0f - metallic);

        // Draw 1: reconstruct the lobe. Draws 2-3: advance past the discarded direction.
        float P_select;
        int lobe = principled_lobeFromU(rand(&localState), F0, cDiff, P_select);
        rand(&localState); rand(&localState);

        principled_f_lobe(albedo, metallic, roughness, v, wo, lobe, f_val);
        float p_dir;
        principled_pdf_lobe(albedo, metallic, roughness, v, wo, lobe, p_dir);
        pdf = P_select * p_dir;
        principled_pdf(albedo, metallic, roughness, v, wo, pdf_marg); // marginal, for MIS only
    }
    else
    {
        // Unsupported reconnection endpoint (delta / leaf / microfacet dielectric):
        // marginal fallback, no rng consumed. Not a valid endpoint yet -- see note above.
        f_pdf_eval(materials, materialID, textures, wi, wo, etaI, etaT, f_val, pdf, uv, transportMode, lod);
    }

    if (pdf_marg < 0.0f) pdf_marg = pdf; // single-lobe / fallback: marginal == pdf
#else
    f_pdf_eval(materials, materialID, textures, wi, wo, etaI, etaT, f_val, pdf, uv, transportMode, lod);
    pdf_marg = pdf;
#endif
}


__device__ inline void getAlbedo(
    const Material* __restrict__ materials,
    int materialID,
    const TextureView& textures,
    const float2 uv,

    float3& albedo,
    float lod = 0.0f
) {
    const Material& mat = materials[materialID];
    albedo = f3(mat.albedo);
    if (mat.baseColorTex >= 0)
        albedo = f3(sampleTex(textures, mat.baseColorTex, uv, lod));

    if (mat.type == MAT_METAL) {
        float3 eta = f3(mat.eta);
        float3 k = f3(mat.k);
        albedo = ((eta - f3(1.0f)) * (eta - f3(1.0f)) + k * k)/
                 ((eta + f3(1.0f)) * (eta + f3(1.0f)) + k * k);
    } else if (mat.type == MAT_DELTAMIRROR || mat.type == MAT_SMOOTHDIELECTRIC || mat.type == MAT_THINDIELECTRIC ||
               (mat.type == MAT_GLTF_PRINCIPLED_BSDF && mat.isSpecular)) {
        albedo = f3(1.0f);
    }
}
