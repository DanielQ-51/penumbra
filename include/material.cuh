#pragma once

#include "util.cuh"

enum TextureType {
    TEX_ALBEDO = 0,
    TEX_METAL = 1,
    TEX_ROUGHNESS = 2,
    TEX_EMISSION = 3,
    TEX_OCCLUSION = 4,
    TEX_TRANSMISSION = 5,
    TEX_NORMAL = 6
};

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_SMOOTHDIELECTRIC = 2,
    MAT_MICROFACETDIELECTRIC = 3,
    MAT_LEAF = 4,
    MAT_FLOWER = 5,
    MAT_DELTAMIRROR = 6,
    MAT_THINDIELECTRIC = 7,
    MAT_GLTF_PRINCIPLED_BSDF = 8
};

struct Material
{
    // Texture slots — indices into TextureView.handles; -1 means "none".
    int baseColorTex = -1;
    int mrTex        = -1; // glTF metallic-roughness (B=metallic, G=roughness)
    int normalTex    = -1; 
    int emissiveTex  = -1;
    int occlusionTex = -1; // likely wont be used
    int transTex     = -1; // transmission / opacity map (legacy leaf)

    float normalScale = 1.0f; // glTF normalTexture.scale; multiplies the map's tangent-plane (xy)

    int type;

    float4 albedo;
    float roughness;

    float4 eta;
    float4 k;
    float ior;

    float metallic;
    float specular;
    float transmission;

    bool isSpecular;
    bool boundary; // for mediums tack calculations

    bool thinWalled;

    float4 absorption;

    int priority; // dielectric priority, for nested dielectrics/medium stack

    __host__ Material()
        : type(MAT_DIFFUSE), albedo(f4(0.8f)),
          roughness(0.5f), eta(f4(0)), k(f4(0)),
          ior(1.5f), metallic(0.0f), specular(1.0f), transmission(0.0f) {}

    __host__ static Material Diffuse(const float4& color) {
        Material m;
        m.type = MAT_DIFFUSE;

        m.albedo = color;
        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;
        m.isSpecular = false;
        return m;
    }

    __host__ static Material DiffuseTextured(int baseColorTexIndex) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.baseColorTex = baseColorTexIndex;

        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material Metal(const float4& n, const float4& k, float roughness = 0.1f) {
        Material m;
        m.type = MAT_METAL;
        m.eta = n;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);  // metals usually reflect via Fresnel, not albedo tint
        m.metallic = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material SmoothDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_SMOOTHDIELECTRIC;
        m.ior = ior;
        m.albedo = f4(1.0f);
        m.roughness = 0.0f;

        m.priority = pri;
        m.isSpecular = true;
        m.boundary = true;

        m.absorption = k;
        m.thinWalled = false;
        return m;
    }

    __host__ static Material ThinDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_THINDIELECTRIC;
        m.ior = ior;
        m.albedo = f4(1.0f);
        m.roughness = 0.0f;

        m.priority = pri;
        m.isSpecular = true;
        m.boundary = true;

        m.absorption = k;
        m.thinWalled = false;
        return m;
    }

    __host__  static Material MicrofacetDielectric(float ior = 1.5f, float roughness = 0.0f, const float4& k = f4()) {
        Material m;
        m.type = MAT_MICROFACETDIELECTRIC;
        m.ior = ior;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);

        m.thinWalled = false;
        return m;
    }

    __host__ static Material Leaf(int baseColorTexIndex, float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;
        m.baseColorTex = baseColorTexIndex;

        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }

    __host__ static Material Leaf(int baseColorTexIndex, int transTexIndex, float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;
        m.baseColorTex = baseColorTexIndex;
        m.transTex = transTexIndex;

        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }

    // glTF metallic-roughness principled material. baseColor is linear RGBA;
    // metallic/roughness are the scalar factors (multiplied by mrTex.b / mrTex.g
    // at shading time if mrTexIndex >= 0). Opaque; no transmission in v1.
    __host__ static Material Principled(const float4& baseColor, float metallic, float roughness,
        int baseColorTexIndex = -1, int mrTexIndex = -1)
    {
        Material m;
        m.type = MAT_GLTF_PRINCIPLED_BSDF;
        m.albedo = baseColor;
        m.metallic = metallic;
        m.roughness = roughness;
        m.baseColorTex = baseColorTexIndex;
        m.mrTex = mrTexIndex;
        m.boundary = false;
        m.thinWalled = false;
        m.isSpecular = false;
        return m;
    }

    // A principled material that is a smooth (refractive) dielectric — glass.
    // Stays MAT_GLTF_PRINCIPLED_BSDF; the isSpecular flag makes the dispatchers
    // hand off wholesale to the smooth-dielectric functions (no other lobes).
    // ior + backface drive refraction through the closed mesh; absorption/priority
    // are for the medium stack (colored/nested glass) once that's wired.
    __host__ static Material PrincipledGlass(float ior = 1.5f, const float4& absorption = f4(), int priority = 0)
    {
        Material m;
        m.type = MAT_GLTF_PRINCIPLED_BSDF;
        m.ior = ior;
        m.albedo = f4(1.0f);
        m.roughness = 0.0f;

        m.isSpecular = true;   // -> dispatchers delegate to dumb_smooth_dielectric_*
        m.boundary = true;
        m.priority = priority;
        m.absorption = absorption;
        m.thinWalled = false;
        return m;
    }

    __host__ static Material Mirror()
    {
        Material m;
        m.type = MAT_DELTAMIRROR;

        m.isSpecular = true;
        m.roughness = 0.0f;

        m.albedo = f4(1.0f);

        return m;
    }
};