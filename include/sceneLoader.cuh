#pragma once

#include "textureManager.cuh"
#include "sceneContexts.cuh"
#include "material.cuh"
#include "util.cuh"
#include "helpers.cuh"
#include "configParser.cuh"

#include <tinygltf/tiny_gltf_v3.h>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <memory>
#include <algorithm>
#include <utility>
#include <thread>
#include <atomic>

#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

// -----------------------------------------------------------------------------
// Scene loader — an alternative to the manual scene-building path.
//
// Canonical model: object-space meshes + an instance list. Nothing here bakes
// transforms into stored vertices; that only happens in a backend finalizer
// (software = flatten to world space, OptiX = keep object-space + per-instance
// matrices). This file provides the canonical structs, the glTF accessor
// readers, and loadGLTF. The finalizers (buildFlattened / buildInstanced) come
// next.
// -----------------------------------------------------------------------------

// ---- Transform ------------------------------------------------------------
// Row-major 4x4 that transforms a column vector: p' = M * p, m[row*4 + col].
struct Mat4 {
    float m[16];

    __host__ static Mat4 identity() {
        Mat4 t{};
        t.m[0] = t.m[5] = t.m[10] = t.m[15] = 1.0f;
        return t;
    }

    // glTF matrices are column-major: (row,col) lives at src[col*4 + row].
    __host__ static Mat4 fromColumnMajor(const double* src) {
        Mat4 t;
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
                t.m[row * 4 + col] = (float)src[col * 4 + row];
        return t;
    }

    // T * R * S from glTF node TRS. q is (x,y,z,w).
    __host__ static Mat4 fromTRS(const double* tr, const double* q, const double* s) {
        float x = (float)q[0], y = (float)q[1], z = (float)q[2], w = (float)q[3];
        float sx = (float)s[0], sy = (float)s[1], sz = (float)s[2];

        // Rotation matrix from the quaternion.
        float r00 = 1.0f - 2.0f * (y*y + z*z), r01 = 2.0f * (x*y - w*z),     r02 = 2.0f * (x*z + w*y);
        float r10 = 2.0f * (x*y + w*z),        r11 = 1.0f - 2.0f * (x*x + z*z), r12 = 2.0f * (y*z - w*x);
        float r20 = 2.0f * (x*z - w*y),        r21 = 2.0f * (y*z + w*x),     r22 = 1.0f - 2.0f * (x*x + y*y);

        Mat4 t = identity();
        // Columns of (R*S) are R's columns scaled by s.
        t.m[0] = r00 * sx; t.m[1] = r01 * sy; t.m[2]  = r02 * sz; t.m[3]  = (float)tr[0];
        t.m[4] = r10 * sx; t.m[5] = r11 * sy; t.m[6]  = r12 * sz; t.m[7]  = (float)tr[1];
        t.m[8] = r20 * sx; t.m[9] = r21 * sy; t.m[10] = r22 * sz; t.m[11] = (float)tr[2];
        return t;
    }

    // a * b (matrix product).
    __host__ static Mat4 multiply(const Mat4& a, const Mat4& b) {
        Mat4 t{};
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += a.m[r * 4 + k] * b.m[k * 4 + c];
                t.m[r * 4 + c] = sum;
            }
        return t;
    }

    // Transform a position (w = 1).
    __host__ float3 transformPoint(float3 p) const {
        return make_float3(
            m[0]*p.x + m[1]*p.y + m[2] *p.z + m[3],
            m[4]*p.x + m[5]*p.y + m[6] *p.z + m[7],
            m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]);
    }

    // Transform a direction (w = 0, no translation). Uses the upper 3x3, which
    // is correct for rigid + uniform scale; non-uniform scale would want the
    // inverse-transpose for normals (v1 accepts the small error).
    __host__ float3 transformDir(float3 d) const {
        return make_float3(
            m[0]*d.x + m[1]*d.y + m[2] *d.z,
            m[4]*d.x + m[5]*d.y + m[6] *d.z,
            m[8]*d.x + m[9]*d.y + m[10]*d.z);
    }
};

// ---- Canonical mesh -------------------------------------------------------
// A single-material renderable unit (one glTF primitive). Vertices are unified
// (position[i]/normal[i]/uv[i] share index i), matching glTF.
struct SceneMesh {
    std::vector<float4> positions;   // object space, w unused
    std::vector<float4> normals;     // w unused
    std::vector<float2> uvs;         // empty if the primitive had no TEXCOORD_0
    std::vector<float4> tangents;    // xyz + handedness sign in w; empty if absent
    std::vector<uint3>  indices;     // triangle corner indices into the arrays above

    int    materialID  = -1;         // index into SceneLoader::materials
    bool   alphaTested = false;      // material alphaMode == MASK/BLEND -> needs anyhit
    float3 emissive    = make_float3(0.0f, 0.0f, 0.0f); // material emissiveFactor -> light path
};

// ---- Canonical instance ---------------------------------------------------
struct SceneInstance {
    int  meshIndex = -1;             // index into SceneLoader::meshes
    Mat4 transform = Mat4::identity();
};

// =============================================================================
// glTF accessor reading
//
// The accessor -> bufferView -> buffer dereference, resolved once into a plain
// {base, count, stride, types} view, then small typed readers on top of it.
// =============================================================================

struct AccessorView {
    const uint8_t* base = nullptr;   // first element
    size_t count        = 0;
    size_t stride       = 0;         // bytes between consecutive elements
    int    componentType = 0;        // TG3_COMPONENT_TYPE_*
    int    type          = 0;        // TG3_TYPE_*

    bool valid() const { return base != nullptr; }
};

// Resolve an accessor index into a byte view, with bounds checking. Returns an
// invalid view (base == nullptr) for out-of-range indices, sparse accessors
// (not supported in v1), or any read that would fall outside the buffer.
inline AccessorView resolveAccessor(const tg3_model& model, int accessorIndex)
{
    AccessorView v;
    if (accessorIndex < 0 || (uint32_t)accessorIndex >= model.accessors_count) return v;

    const tg3_accessor& acc = model.accessors[accessorIndex];
    if (acc.sparse.is_sparse) return v; // TODO: sparse accessors
    if (acc.buffer_view < 0 || (uint32_t)acc.buffer_view >= model.buffer_views_count) return v;

    const tg3_buffer_view& bv = model.buffer_views[acc.buffer_view];
    if (bv.buffer < 0 || (uint32_t)bv.buffer >= model.buffers_count) return v;

    const tg3_buffer& buf = model.buffers[bv.buffer];

    int stride = tg3_accessor_byte_stride(&acc, &bv);
    int elemSize = tg3_component_size(acc.component_type) * tg3_num_components(acc.type);
    if (stride <= 0 || elemSize <= 0 || acc.count == 0) return v;

    size_t start = (size_t)bv.byte_offset + (size_t)acc.byte_offset;
    size_t lastEnd = start + (size_t)(acc.count - 1) * (size_t)stride + (size_t)elemSize;
    if (lastEnd > buf.data.count) return v;

    v.base          = buf.data.data + start;
    v.count         = (size_t)acc.count;
    v.stride        = (size_t)stride;
    v.componentType = acc.component_type;
    v.type          = acc.type;
    return v;
}

inline int findAttribute(const tg3_primitive& prim, const char* name)
{
    for (uint32_t i = 0; i < prim.attributes_count; i++) {
        if (tg3_str_equals_cstr(prim.attributes[i].key, name))
            return prim.attributes[i].value;
    }
    return -1;
}

inline void readVec3(const AccessorView& v, std::vector<float4>& out)
{
    if (!v.valid() || v.componentType != TG3_COMPONENT_TYPE_FLOAT || v.type != TG3_TYPE_VEC3) return;
    out.reserve(out.size() + v.count);
    for (size_t i = 0; i < v.count; i++) {
        const float* p = reinterpret_cast<const float*>(v.base + i * v.stride);
        out.push_back(make_float4(p[0], p[1], p[2], 0.0f));
    }
}

inline void readVec2(const AccessorView& v, std::vector<float2>& out)
{
    if (!v.valid() || v.componentType != TG3_COMPONENT_TYPE_FLOAT || v.type != TG3_TYPE_VEC2) return;
    out.reserve(out.size() + v.count);
    for (size_t i = 0; i < v.count; i++) {
        const float* p = reinterpret_cast<const float*>(v.base + i * v.stride);
        out.push_back(make_float2(p[0], p[1]));
    }
}

inline void readVec4(const AccessorView& v, std::vector<float4>& out)
{
    if (!v.valid() || v.componentType != TG3_COMPONENT_TYPE_FLOAT || v.type != TG3_TYPE_VEC4) return;
    out.reserve(out.size() + v.count);
    for (size_t i = 0; i < v.count; i++) {
        const float* p = reinterpret_cast<const float*>(v.base + i * v.stride);
        out.push_back(make_float4(p[0], p[1], p[2], p[3]));
    }
}

inline void readIndices(const AccessorView& v, std::vector<uint32_t>& out)
{
    if (!v.valid() || v.type != TG3_TYPE_SCALAR) return;
    out.reserve(out.size() + v.count);
    for (size_t i = 0; i < v.count; i++) {
        const uint8_t* p = v.base + i * v.stride;
        uint32_t idx = 0;
        switch (v.componentType) {
            case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:  idx = *reinterpret_cast<const uint8_t*>(p);  break;
            case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: idx = *reinterpret_cast<const uint16_t*>(p); break;
            case TG3_COMPONENT_TYPE_UNSIGNED_INT:   idx = *reinterpret_cast<const uint32_t*>(p); break;
            default: return; // unsupported index type
        }
        out.push_back(idx);
    }
}

// =============================================================================
// glTF material extensions (KHR_materials_*) — tiny helpers over tg3_value.
// =============================================================================

// Find a named extension on a material's ext block; returns its value or null.
inline const tg3_value* findExtension(const tg3_extras_ext& ext, const char* name)
{
    for (uint32_t i = 0; i < ext.extensions_count; i++)
        if (tg3_str_equals_cstr(ext.extensions[i].name, name))
            return &ext.extensions[i].value;
    return nullptr;
}

// Look up a field by key inside an OBJECT-typed value.
inline const tg3_value* objField(const tg3_value* obj, const char* key)
{
    if (!obj || obj->type != TG3_VALUE_OBJECT) return nullptr;
    for (uint32_t i = 0; i < obj->object_count; i++)
        if (tg3_str_equals_cstr(obj->object_data[i].key, key))
            return &obj->object_data[i].value;
    return nullptr;
}

// Read a numeric value (INT or REAL) with a default.
inline double valNumber(const tg3_value* v, double def)
{
    if (!v) return def;
    if (v->type == TG3_VALUE_REAL) return v->real_val;
    if (v->type == TG3_VALUE_INT)  return (double)v->int_val;
    return def;
}

// =============================================================================
// GPUScene — device-resident result of a finalizer. Owns its device memory
// (freed in the destructor); the ShadeContext points into it. Textures are
// borrowed from the SceneLoader's TextureManager, so the loader must outlive
// the GPUScene. Returned by unique_ptr so the raw device pointers are never
// moved/double-freed.
// =============================================================================
struct GPUScene {
    Vertices*  d_vertices  = nullptr;   // device Vertices struct
    float4*    d_positions = nullptr;   // arrays the Vertices points at
    float4*    d_normals   = nullptr;
    float2*    d_uvs       = nullptr;
    Triangle*  d_scene     = nullptr;
    Material*  d_materials = nullptr;
    std::unique_ptr<LightSamplerManager> lightManager;

    ShadeContext shadeContext = {};

    // Host geometry for the engine's acceleration-structure build (world space
    // for the flattened path).
    std::vector<float4>   hostPositions;
    std::vector<Triangle> hostTriangles;

    GPUScene() = default;
    GPUScene(const GPUScene&) = delete;
    GPUScene& operator=(const GPUScene&) = delete;

    ~GPUScene() {
        cudaFree(d_vertices);
        cudaFree(d_positions);
        cudaFree(d_normals);
        cudaFree(d_uvs);
        cudaFree(d_scene);
        cudaFree(d_materials);
        // lightManager frees its own CDFs/triLights; textures owned by SceneLoader.
    }
};

// Config-authored texture "PBR role" -> upload colorspace. Albedo/emission are
// stored sRGB (hardware linearizes on sample); everything else (metal/roughness/
// occlusion/transmission/normal maps) is data, not color, and stays linear.
inline TexColorSpace colorSpaceForTextureType(TextureType t) {
    switch (t) {
        case TEX_ALBEDO:
        case TEX_EMISSION:
            return TEX_SRGB;
        default: // TEX_METAL, TEX_ROUGHNESS, TEX_OCCLUSION, TEX_TRANSMISSION, TEX_NORMAL
            return TEX_LINEAR;
    }
}

// =============================================================================
// SceneLoader — accumulate from mixed sources, then finalize.
// =============================================================================
class SceneLoader {
public:
    std::vector<SceneMesh>     meshes;
    std::vector<SceneInstance> instances;
    std::vector<Material>      materials;
    TextureManager             textures;

    int addMaterial(const Material& m) {
        materials.push_back(m);
        return (int)materials.size() - 1;
    }

    // Global multiplier applied to every material's emission (on top of the glTF
    // emissiveFactor x emissiveStrength). Tune it so emissive surfaces sit on the
    // same brightness scale as your env map. Set before loadGLTF.
    void setEmissiveScale(float s) { emissiveScale_ = s; }

    std::string resolveAssetPath(const std::string& relPath) const {
        return std::string(ROOT_DIR) + "/" + relPath;
    }

    // ---- Config-driven textures --------------------------------------------
    void loadTexturesFromConfig(const std::vector<TextureConfig>& cfgs) {
        for (const TextureConfig& tc : cfgs) {
            int idx = textures.addFromFile(resolveAssetPath(tc.path), colorSpaceForTextureType(tc.type));
            if (idx < 0) {
                std::cerr << "SceneLoader::loadTexturesFromConfig: failed to load '" << tc.name
                          << "' (" << tc.path << ")\n";
                continue;
            }
            if (namedTextures_.count(tc.name))
                std::cerr << "SceneLoader::loadTexturesFromConfig: duplicate texture name '"
                          << tc.name << "', overwriting\n";
            namedTextures_[tc.name] = idx;
        }
    }

    // ---- Config-driven materials --------------------------------------------
    void loadMaterialsFromConfig(const std::vector<MaterialConfig>& cfgs) {
        for (const MaterialConfig& mc : cfgs) addMaterial(buildMaterialFromConfig(mc));
    }

    // ---- Config-driven OBJ meshes -------------------------------------------
    // OBJ meshes are never instanced (no SceneInstance/Mat4) — any placement is
    // the static `offset` readObjSimple bakes directly into vertex positions at
    // parse time, matching current engine behavior. Kept on their own flat,
    // independently-indexed accumulator (objScene_) rather than forced into
    // SceneMesh's unified-index model — buildFlattened() merges the two below.
    bool loadOBJ(const MeshConfig& cfg, float3 offset = make_float3(0.0f, 0.0f, 0.0f)) {
        readObjSimple(resolveAssetPath(cfg.path),
                      objScene_.points, objScene_.normals, objScene_.colors, objScene_.uvs,
                      objScene_.triangles, objScene_.lightTriangles, objScene_.lightDescriptors,
                      make_float3(0.0f, 0.0f, 0.0f),          // c: legacy per-file tint, unused downstream
                      cfg.emissionMultiplier * cfg.emissionColor,
                      cfg.materialID, offset,
                      0xFFFFFFFF);                            // instanceID: "no per-instance transform" sentinel
        return true;
    }

    void loadOBJs(const std::vector<MeshConfig>& cfgs) {
        for (const MeshConfig& c : cfgs) loadOBJ(c);
    }

    // ---- Orchestrator: drive a full hybrid (OBJ + glTF + config materials/
    // textures) scene load from one RenderConfig. Order matters: materials/
    // textures must exist before anything references them by index/name; OBJ
    // meshes reference materials by raw index (MeshConfig::materialID); glTF
    // assets build their own materials internally (loadMaterials(), unaffected
    // by config-authored ones) and are loaded last since maxTextureDim/
    // emissionScale are applied as loader-global state immediately before each
    // entry's loadGLTF() call.
    bool loadFromConfig(const RenderConfig& config) {
        loadTexturesFromConfig(config.textures);
        loadMaterialsFromConfig(config.materials);

        loadOBJs(config.meshes);

        for (const GLTFConfig& g : config.gltfs) {
            if (g.maxTextureDim > 0) textures.setMaxDimension((int)g.maxTextureDim);
            setEmissiveScale(g.emissionScale);
            loadGLTF(resolveAssetPath(g.path));
        }

        // Defensive bounds check: a MeshConfig.materialID that doesn't index into
        // `materials` would otherwise become a silent device-side OOB read.
        for (Triangle& t : objScene_.triangles) {
            if (t.materialID < 0 || t.materialID >= (int)materials.size()) {
                std::cerr << "SceneLoader::loadFromConfig: OBJ triangle references out-of-range "
                             "materialID " << t.materialID << " (materials.size()=" << materials.size()
                          << "), clamping to 0\n";
                t.materialID = 0;
            }
        }
        return true;
    }

    // Load a glTF/GLB. Appends object-space meshes, instances (world transforms
    // from the node hierarchy, pre-multiplied by `root`), materials, and
    // textures. Returns false on parse failure. Env map is handled separately.
    bool loadGLTF(const std::string& path, const Mat4& root = Mat4::identity())
    {
        tg3_parse_options opts; tg3_parse_options_init(&opts);
        tg3_error_stack errors; tg3_error_stack_init(&errors);
        tg3_model model;

        // tg3_parse_file (with TINYGLTF3_ENABLE_FS defined for tiny_gltf_v3.c)
        // reads the file AND resolves external .bin buffers / images relative to
        // its directory — required for multi-file .gltf (e.g. Sponza). It
        // auto-detects GLB vs JSON, so this path serves both.
        tg3_error_code err = tg3_parse_file(&model, &errors,
                                            path.c_str(), (uint32_t)path.size(), &opts);
        if (err != TG3_OK || tg3_errors_has_error(&errors)) {
            std::cerr << "loadGLTF: parse failed for " << path << "\n";
            for (uint32_t i = 0; i < errors.count; i++)
                std::cerr << "  " << (errors.entries[i].message ? errors.entries[i].message : "(null)") << "\n";
            tg3_model_free(&model);
            tg3_error_stack_free(&errors);
            return false;
        }

        // base directory for external image URIs (empty/none for self-contained GLB)
        size_t slash = path.find_last_of("/\\");
        baseDir_ = (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);

        // per-file lookup tables (indices into THIS model)
        gltfTextureMap_.clear();
        gltfMaterialMap_.assign(model.materials_count, -1);

        // Decode all images up front, in parallel — PNG/JPG decode is the load-time
        // bottleneck and is embarrassingly parallel across images.
        decodeAllImages(model);
        loadMaterials(model);
        imageCache_.clear();
        imageCache_.shrink_to_fit();

        int sceneIdx = (model.default_scene >= 0) ? model.default_scene : 0;
        if (model.scenes_count > 0) {
            const tg3_scene& scene = model.scenes[(uint32_t)sceneIdx];
            std::unordered_map<int, std::vector<int>> meshCache; // glTF mesh -> our SceneMesh indices
            for (uint32_t i = 0; i < scene.nodes_count; i++)
                traverseNode(model, scene.nodes[i], root, meshCache);
        }

        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        return true;
    }

    // Software finalizer: bake every instance transform into a single world-space
    // triangle soup, upload it, build the light sampler, and return a GPUScene
    // whose ShadeContext is ready to render. No instancing / no transform matrices
    // (that's buildInstanced, the OptiX path). The SceneLoader must outlive the
    // returned GPUScene (it owns the textures the ShadeContext references).
    std::unique_ptr<GPUScene> buildFlattened(EnvMapView env, float envWeight = 0.5f)
    {
        auto gpu = std::make_unique<GPUScene>();

        std::vector<float4>   points;    // world-space
        std::vector<float4>   normals;   // world-space
        std::vector<float2>   uvs;
        std::vector<Triangle> tris;
        std::vector<Triangle> lightTris;
        std::vector<LightDescriptor> lightDesc;

        for (const SceneInstance& inst : instances) {
            const SceneMesh& sm = meshes[inst.meshIndex];
            const Mat4& M = inst.transform;

            int baseVert = (int)points.size();
            for (size_t v = 0; v < sm.positions.size(); v++) {
                float3 p = M.transformPoint(f3(sm.positions[v]));
                points.push_back(make_float4(p.x, p.y, p.z, 0.0f));
                float3 n = M.transformDir(f3(sm.normals[v]));
                float len = length(n);
                if (len > 1e-12f) { float inv = 1.0f / len; n = make_float3(n.x*inv, n.y*inv, n.z*inv); }
                normals.push_back(make_float4(n.x, n.y, n.z, 0.0f));
            }

            bool hasUV = !sm.uvs.empty();
            int baseUV = (int)uvs.size();
            if (hasUV) for (const float2& uv : sm.uvs) uvs.push_back(uv);

            bool emissive = luminance(sm.emissive) > 0.0f;
            float4 emission = emissive ? make_float4(sm.emissive.x, sm.emissive.y, sm.emissive.z, 0.0f) : f4();
            int lightStart = (int)lightTris.size();

            // Ray-cone LOD base term (Δ_tri = 0.5*log2(texelArea/worldArea)) is per
            // triangle, but its texture dimensions are constant across the mesh. Baked
            // from the base-color texture; skipped (Δ=0 -> mip 0) without UVs or a texture.
            int  baseTex = (sm.materialID >= 0) ? materials[sm.materialID].baseColorTex : -1;
            int2 texWH   = (baseTex >= 0) ? textures.dims(baseTex) : make_int2(0, 0);
            bool bakeLOD = hasUV && baseTex >= 0 && texWH.x > 0 && texWH.y > 0;

            for (const uint3& t : sm.indices) {
                int a = baseVert + (int)t.x, b = baseVert + (int)t.y, c = baseVert + (int)t.z;
                int uva = hasUV ? baseUV + (int)t.x : -1;
                int uvb = hasUV ? baseUV + (int)t.y : -1;
                int uvc = hasUV ? baseUV + (int)t.z : -1;

                int myTriInd   = (int)tris.size();
                int myLightInd = emissive ? (int)lightTris.size() : -51;

                int matID = sm.materialID;

                Triangle tri(a, b, c, a, b, c, matID, uva, uvb, uvc, emission, myLightInd, myTriInd);

                if (bakeLOD) {
                    float3 p0 = f3(points[a]), p1 = f3(points[b]), p2 = f3(points[c]);
                    float  pa = length(cross(p1 - p0, p2 - p0)); // world parallelogram area
                    float2 t0 = sm.uvs[t.x], t1 = sm.uvs[t.y], t2 = sm.uvs[t.z];
                    float  ta = fabsf((t1.x - t0.x) * (t2.y - t0.y) -
                                      (t2.x - t0.x) * (t1.y - t0.y))   // UV parallelogram area
                                * (float)texWH.x * (float)texWH.y;     // -> texel area
                    if (pa > 1e-20f && ta > 1e-20f)
                        tri.lodDelta = 0.5f * log2f(ta / pa);
                }

                tris.push_back(tri);
                if (emissive) lightTris.push_back(tri);
            }

            if (emissive) {
                LightDescriptor ld;
                ld.type = 0;
                ld.startInd = lightStart;
                ld.numPrim = (int)lightTris.size() - lightStart;
                ld.instanceID = 0xFFFFFFFF; // flattened -> world space, no per-instance transform
                float power = 0.0f;
                for (int li = lightStart; li < (int)lightTris.size(); li++) {
                    const Triangle& lt = lightTris[li];
                    float3 ap = f3(points[lt.aInd]), bp = f3(points[lt.bInd]), cp = f3(points[lt.cInd]);
                    float area = 0.5f * length(cross(bp - ap, cp - ap));
                    power += area * luminance(lt.emission) * h_PI;
                }
                ld.totalPower = power;
                lightDesc.push_back(ld);
            }
        }

        // --- Merge OBJ-sourced flat data ---------------------------------------
        // OBJ triangles use independent per-corner position/normal/uv indices
        // (with -1 sentinels for missing normal/uv), unlike glTF's unified
        // indexing above — offset each index stream independently, skipping the
        // -1 sentinel. readObjSimple already tracks its own running offsets
        // within objScene_'s accumulation across multiple loadOBJ() calls, so
        // objScene_ arrives here already internally consistent; this just
        // re-bases that soup onto the final merged arrays once.
        {
            const int baseVert     = (int)points.size();
            const int baseNorm     = (int)normals.size();
            const int baseUV       = (int)uvs.size();
            const int baseTri      = (int)tris.size();
            const int baseLightTri = (int)lightTris.size();

            points.insert(points.end(),   objScene_.points.begin(),  objScene_.points.end());
            normals.insert(normals.end(), objScene_.normals.begin(), objScene_.normals.end());
            uvs.insert(uvs.end(),         objScene_.uvs.begin(),     objScene_.uvs.end());

            auto off = [](int idx, int base) { return (idx < 0) ? idx : idx + base; };

            for (Triangle t : objScene_.triangles) {
                t.aInd = off(t.aInd, baseVert);   t.bInd = off(t.bInd, baseVert);   t.cInd = off(t.cInd, baseVert);
                t.naInd = off(t.naInd, baseNorm); t.nbInd = off(t.nbInd, baseNorm); t.ncInd = off(t.ncInd, baseNorm);
                t.uvaInd = off(t.uvaInd, baseUV); t.uvbInd = off(t.uvbInd, baseUV); t.uvcInd = off(t.uvcInd, baseUV);
                t.triInd = baseTri + t.triInd;                                // re-sequence into merged tris[]
                if (t.lightInd != -51) t.lightInd = baseLightTri + t.lightInd; // -51 = "not a light" sentinel
                tris.push_back(t);
            }
            for (Triangle lt : objScene_.lightTriangles) {
                lt.aInd = off(lt.aInd, baseVert);   lt.bInd = off(lt.bInd, baseVert);   lt.cInd = off(lt.cInd, baseVert);
                lt.naInd = off(lt.naInd, baseNorm); lt.nbInd = off(lt.nbInd, baseNorm); lt.ncInd = off(lt.ncInd, baseNorm);
                lt.uvaInd = off(lt.uvaInd, baseUV); lt.uvbInd = off(lt.uvbInd, baseUV); lt.uvcInd = off(lt.uvcInd, baseUV);
                lt.triInd = baseTri + lt.triInd;
                lt.lightInd = baseLightTri + lt.lightInd;
                lightTris.push_back(lt);
            }
            for (LightDescriptor ld : objScene_.lightDescriptors) {
                ld.startInd += baseLightTri;
                ld.instanceID = 0xFFFFFFFF; // normalize to the same "no transform" sentinel glTF instances use
                lightDesc.push_back(ld);
            }
        }

        if (tris.empty()) {
            std::cerr << "buildFlattened: no triangles.\n";
            return gpu;
        }

        // --- Upload the SoA vertices ---
        Vertices temp;
        cudaMalloc(&temp.positions, points.size()  * sizeof(float4));
        cudaMalloc(&temp.normals,   normals.size() * sizeof(float4));
        cudaMalloc(&temp.uvs,       std::max<size_t>(uvs.size(), 1) * sizeof(float2));
        cudaMemcpy(temp.positions, points.data(),  points.size()  * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(temp.normals,   normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
        if (!uvs.empty())
            cudaMemcpy(temp.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);

        cudaMalloc(&gpu->d_vertices, sizeof(Vertices));
        cudaMemcpy(gpu->d_vertices, &temp, sizeof(Vertices), cudaMemcpyHostToDevice);
        gpu->d_positions = temp.positions;
        gpu->d_normals   = temp.normals;
        gpu->d_uvs       = temp.uvs;

        // --- Scene triangles ---
        cudaMalloc(&gpu->d_scene, tris.size() * sizeof(Triangle));
        cudaMemcpy(gpu->d_scene, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

        // --- Materials ---
        cudaMalloc(&gpu->d_materials, materials.size() * sizeof(Material));
        cudaMemcpy(gpu->d_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

        // --- Light sampler (allocates d_lights; owned by lightManager) ---
        Triangle* d_lights = nullptr;
        gpu->lightManager = std::make_unique<LightSamplerManager>(
            lightDesc, lightTris, points, d_lights, env, envWeight);

        // --- ShadeContext ---
        ShadeContext sc = {};
        sc.materials              = gpu->d_materials;
        sc.textures               = textures.getView();   // borrowed from this loader
        sc.lights                 = d_lights;
        sc.scene                  = gpu->d_scene;
        sc.vertices               = gpu->d_vertices;
        sc.transformationMatrices = nullptr;
        sc.lightSampler           = gpu->lightManager->getSampler();
        sc.lightNum               = (int)lightTris.size();
        sc.triNum                 = (uint32_t)tris.size();
        gpu->shadeContext = sc;

        // --- Host geometry for the engine's BVH build ---
        gpu->hostPositions = std::move(points);
        gpu->hostTriangles = std::move(tris);

        std::cout << "buildFlattened: " << gpu->hostTriangles.size() << " tris, "
                  << gpu->hostPositions.size() << " verts, "
                  << lightTris.size() << " light tris, "
                  << materials.size() << " materials, "
                  << textures.count() << " textures\n";
        return gpu;
    }

    // Next increment: std::unique_ptr<GPUScene> buildInstanced(...) for OptiX
    // (object-space upload + per-instance transformationMatrices + GAS/IAS).

private:
    // A decoded RGBA8 image, produced in parallel by decodeAllImages().
    struct DecodedImage {
        std::vector<unsigned char> rgba;
        int  w = 0, h = 0;
        bool ok = false;
    };

    // Flat, independently-indexed accumulation for config-driven OBJ meshes —
    // mirrors exactly what readObjSimple already produces. Kept separate from
    // SceneMesh's unified-index model (see loadOBJ's comment); buildFlattened()
    // merges this into its final output alongside the glTF-instance flattening.
    struct ObjSceneData {
        std::vector<float4>   points;
        std::vector<float4>   normals;
        std::vector<float4>   colors;   // legacy/unused tint param, kept for readObjSimple's signature
        std::vector<float2>   uvs;
        std::vector<Triangle> triangles;
        std::vector<Triangle> lightTriangles;
        std::vector<LightDescriptor> lightDescriptors;
    };
    ObjSceneData objScene_;

    std::unordered_map<std::string, int> namedTextures_; // config-authored texture name -> texture index

    std::string                     baseDir_;
    std::unordered_map<int, int>    gltfTextureMap_;  // glTF texture idx -> our texture idx
    std::vector<int>                gltfMaterialMap_; // glTF material idx -> our material idx
    int                             defaultMaterialID_ = -1;
    float                           emissiveScale_ = 1.0f; // global emission multiplier
    std::vector<DecodedImage>       imageCache_;      // glTF image idx -> decoded pixels

    int resolveTextureName(const std::string& name) const {
        auto it = namedTextures_.find(name);
        if (it == namedTextures_.end()) {
            std::cerr << "SceneLoader: unknown texture name '" << name << "', using none (-1)\n";
            return -1;
        }
        return it->second;
    }

    // Maps one Materials: config line to a Material via the matching Material::
    // factory. Texture-slot fields hold a texture NAME, resolved against
    // namedTextures_ (populated by loadTexturesFromConfig, which must run first).
    // Missing params fall back to the same defaults as the corresponding factory.
    Material buildMaterialFromConfig(const MaterialConfig& mc) {
        auto getF = [&](const char* k, float def) -> float {
            auto it = mc.params.find(k); return it != mc.params.end() ? std::stof(it->second) : def;
        };
        auto getI = [&](const char* k, int def) -> int {
            auto it = mc.params.find(k); return it != mc.params.end() ? std::stoi(it->second) : def;
        };
        auto getV4 = [&](const char* k, float4 def) -> float4 {
            auto it = mc.params.find(k); return it != mc.params.end() ? parseVec4(it->second, def.w) : def;
        };
        auto getTex = [&](const char* k) -> int {
            auto it = mc.params.find(k); return it != mc.params.end() ? resolveTextureName(it->second) : -1;
        };

        if (mc.type == "DIFFUSE") {
            int baseTex = getTex("baseColorTex");
            return baseTex >= 0 ? Material::DiffuseTextured(baseTex)
                                 : Material::Diffuse(getV4("albedo", f4(0.8f, 0.8f, 0.8f, 1.0f)));
        }
        if (mc.type == "METAL")
            return Material::Metal(getV4("eta", f4(1, 1, 1, 1)), getV4("k", f4(1, 1, 1, 1)), getF("roughness", 0.1f));
        if (mc.type == "SMOOTH_DIELECTRIC")
            return Material::SmoothDielectric(getF("ior", 1.5f), getV4("absorption", f4()), getI("priority", 0));
        if (mc.type == "THIN_DIELECTRIC")
            return Material::ThinDielectric(getF("ior", 1.5f), getV4("absorption", f4()), getI("priority", 0));
        if (mc.type == "MICROFACET_DIELECTRIC")
            return Material::MicrofacetDielectric(getF("ior", 1.5f), getF("roughness", 0.0f), getV4("k", f4()));
        if (mc.type == "LEAF") {
            int baseTex = getTex("baseColorTex"), transTex = getTex("transTex");
            float ior = getF("ior", 1.5f), rough = getF("roughness", 0.7f), trans = getF("transmission", 0.05f);
            float4 albedo = getV4("albedo", f4());
            return transTex >= 0 ? Material::Leaf(baseTex, transTex, ior, rough, albedo, trans)
                                  : Material::Leaf(baseTex, ior, rough, albedo, trans);
        }
        if (mc.type == "PRINCIPLED") {
            Material m = Material::Principled(getV4("baseColor", f4(0.8f, 0.8f, 0.8f, 1.0f)),
                                               getF("metallic", 0.0f), getF("roughness", 0.5f),
                                               getTex("baseColorTex"), getTex("mrTex"));
            m.normalTex    = getTex("normalTex");
            m.emissiveTex  = getTex("emissiveTex");
            m.occlusionTex = getTex("occlusionTex");
            return m;
        }
        if (mc.type == "PRINCIPLED_GLASS")
            return Material::PrincipledGlass(getF("ior", 1.5f), getV4("absorption", f4()), getI("priority", 0));
        if (mc.type == "MIRROR" || mc.type == "DELTAMIRROR")
            return Material::Mirror();

        std::cerr << "SceneLoader::buildMaterialFromConfig: unknown material type '" << mc.type
                  << "', using default Diffuse\n";
        return Material::Diffuse(f4(0.8f));
    }

    // Decode one glTF image (embedded bufferView or external URI) with stb into
    // imageCache_[i]. Each thread writes a distinct slot, so this is race-free.
    void decodeOneImage(const tg3_model& model, uint32_t i)
    {
        const tg3_image& img = model.images[i];
        int w = 0, h = 0, n = 0;
        unsigned char* data = nullptr;

        if (img.buffer_view >= 0 && (uint32_t)img.buffer_view < model.buffer_views_count) {
            const tg3_buffer_view& bv = model.buffer_views[(uint32_t)img.buffer_view];
            if (bv.buffer >= 0 && (uint32_t)bv.buffer < model.buffers_count) {
                const tg3_buffer& buf = model.buffers[(uint32_t)bv.buffer];
                if (bv.byte_offset + bv.byte_length <= buf.data.count)
                    data = stbi_load_from_memory(buf.data.data + bv.byte_offset,
                                                 (int)bv.byte_length, &w, &h, &n, 4);
            }
        } else if (img.uri.data && img.uri.len > 0 &&
                   !tg3_is_data_uri(img.uri.data, img.uri.len)) {
            std::string uri(img.uri.data, img.uri.len);
            data = stbi_load((baseDir_ + uri).c_str(), &w, &h, &n, 4);
        }

        if (data) {
            DecodedImage& di = imageCache_[i];
            di.w = w; di.h = h; di.ok = true;
            di.rgba.assign(data, data + (size_t)w * h * 4);
            stbi_image_free(data);
        }
    }

    // Decode every image across a thread pool (stb decode is thread-safe).
    void decodeAllImages(const tg3_model& model)
    {
        imageCache_.assign(model.images_count, DecodedImage{});
        if (model.images_count == 0) return;

        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned nthreads = std::min<unsigned>(hw, model.images_count);
        std::atomic<uint32_t> next{0};

        auto worker = [&]() {
            uint32_t i;
            while ((i = next.fetch_add(1u)) < model.images_count)
                decodeOneImage(model, i);
        };

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (unsigned t = 0; t < nthreads; t++) pool.emplace_back(worker);
        for (std::thread& th : pool) th.join();
    }

    // Register a glTF texture (dedup by glTF index) from the pre-decoded cache.
    // Returns our texture index, or -1.
    int registerTexture(const tg3_model& model, int gltfTexIndex, TexColorSpace cs)
    {
        if (gltfTexIndex < 0 || (uint32_t)gltfTexIndex >= model.textures_count) return -1;
        auto it = gltfTextureMap_.find(gltfTexIndex);
        if (it != gltfTextureMap_.end()) return it->second;

        int our = -1;
        const tg3_texture& tex = model.textures[(uint32_t)gltfTexIndex];
        if (tex.source >= 0 && (uint32_t)tex.source < imageCache_.size()) {
            // Bytes were decoded in parallel by decodeAllImages(); here we only
            // do the (serial) mip-gen + GPU upload. `cs` (sRGB vs linear) is a
            // per-texture property, so the same decoded image can be uploaded
            // under different colorspaces.
            const DecodedImage& di = imageCache_[(uint32_t)tex.source];
            if (di.ok)
                our = textures.addFromMemory(di.rgba.data(), di.w, di.h, cs);
        }
        gltfTextureMap_[gltfTexIndex] = our;
        return our;
    }

    void loadMaterials(const tg3_model& model)
    {
        for (uint32_t i = 0; i < model.materials_count; i++) {
            const tg3_material& gm = model.materials[i];

            // --- Glass: KHR_materials_transmission -> smooth dielectric (v1) ---
            const tg3_value* trExt = findExtension(gm.ext, "KHR_materials_transmission");
            float transmission = (float)valNumber(objField(trExt, "transmissionFactor"), 0.0);
            if (transmission > 0.0f) {
                const tg3_value* iorExt = findExtension(gm.ext, "KHR_materials_ior");
                float ior = (float)valNumber(objField(iorExt, "ior"), 1.5);
                // Absorption (KHR_materials_volume attenuation) left clear until the
                // medium stack is wired; thicknessFactor is ignored on purpose — the
                // path tracer refracts through the real closed geometry, not a scalar.
                //gltfMaterialMap_[i] = addMaterial(Material::PrincipledGlass(ior));
                gltfMaterialMap_[i] = addMaterial(Material::ThinDielectric());
                continue;
            }

            // --- Opaque metallic-roughness principled ---
            const tg3_pbr_metallic_roughness& pbr = gm.pbr_metallic_roughness;

            float4 baseColor = make_float4((float)pbr.base_color_factor[0], (float)pbr.base_color_factor[1],
                                           (float)pbr.base_color_factor[2], (float)pbr.base_color_factor[3]);
            float metallic  = (float)pbr.metallic_factor;
            float roughness = (float)pbr.roughness_factor;

            int baseTex = registerTexture(model, pbr.base_color_texture.index,          TEX_SRGB);
            int mrTex   = registerTexture(model, pbr.metallic_roughness_texture.index,  TEX_LINEAR);

            Material m = Material::Principled(baseColor, metallic, roughness, baseTex, mrTex);
            m.normalTex   = registerTexture(model, gm.normal_texture.index,   TEX_LINEAR);
            m.normalScale = (float)gm.normal_texture.scale; // glTF normalTexture.scale (default 1.0)
            m.emissiveTex = registerTexture(model, gm.emissive_texture.index, TEX_SRGB);

            gltfMaterialMap_[i] = addMaterial(m);
        }
    }

    int ensureDefaultMaterial()
    {
        if (defaultMaterialID_ < 0)
            defaultMaterialID_ = addMaterial(Material::Principled(f4(0.8f), 0.0f, 0.5f));
        return defaultMaterialID_;
    }

    // Build all SceneMeshes for a glTF mesh (one per primitive); returns their indices.
    std::vector<int> buildMeshesForGltfMesh(const tg3_model& model, int gltfMeshIndex)
    {
        std::vector<int> result;
        const tg3_mesh& gm = model.meshes[(uint32_t)gltfMeshIndex];

        for (uint32_t p = 0; p < gm.primitives_count; p++) {
            const tg3_primitive& prim = gm.primitives[p];
            if (prim.mode != -1 && prim.mode != 4) continue; // 4 = TRIANGLES only for v1

            if (prim.material >= 0 && (uint32_t)prim.material < model.materials_count) {
                const tg3_material& gmat = model.materials[(uint32_t)prim.material];
                
                // 1. Ignore Dirt/Decals: These rely heavily on "BLEND" alpha. 
                // Skipping these removes the floating dirt squares.
                if (tg3_str_equals_cstr(gmat.alpha_mode, "BLEND")) {
                    continue; 
                }

                // 2. Ignore Subsurface/Volume meshes: If it relies on volume, 
                // skip it so it doesn't render as a flat, ugly solid.
                if (findExtension(gmat.ext, "KHR_materials_volume") != nullptr) {
                    continue; 
                }
                
                // NOTE: We leave "MASK" alone here, because MASK is usually 
                // chainlink fences and leaves. Even if they render a bit solidly 
                // right now, they usually look better than missing geometry.
            }

            SceneMesh sm;
            readVec3(resolveAccessor(model, findAttribute(prim, "POSITION")),   sm.positions);
            if (sm.positions.empty()) continue;
            readVec3(resolveAccessor(model, findAttribute(prim, "NORMAL")),     sm.normals);
            readVec2(resolveAccessor(model, findAttribute(prim, "TEXCOORD_0")), sm.uvs);
            readVec4(resolveAccessor(model, findAttribute(prim, "TANGENT")),    sm.tangents);

            std::vector<uint32_t> idx;
            if (prim.indices >= 0) {
                readIndices(resolveAccessor(model, prim.indices), idx);
            } else {
                idx.resize(sm.positions.size());
                for (uint32_t i = 0; i < idx.size(); i++) idx[i] = i; // non-indexed
            }
            for (size_t i = 0; i + 3 <= idx.size(); i += 3)
                sm.indices.push_back(make_uint3(idx[i], idx[i + 1], idx[i + 2]));

            if (sm.normals.size() != sm.positions.size())
                computeSmoothNormals(sm);

            if (prim.material >= 0 && (uint32_t)prim.material < gltfMaterialMap_.size()) {
                sm.materialID = gltfMaterialMap_[prim.material];
                const tg3_material& gmat = model.materials[(uint32_t)prim.material];
                sm.alphaTested = tg3_str_equals_cstr(gmat.alpha_mode, "MASK") ||
                                 tg3_str_equals_cstr(gmat.alpha_mode, "BLEND");
                // emission = emissiveFactor x emissiveStrength x globalScale.
                // KHR_materials_emissive_strength (default 1) is what lets emitters
                // exceed the [0,1] factor; emissiveScale_ places glTF's uncalibrated
                // scale onto this engine's radiance scale (tune vs the env map).
                float strength = (float)valNumber(
                    objField(findExtension(gmat.ext, "KHR_materials_emissive_strength"), "emissiveStrength"), 1.0);
                float es = strength * emissiveScale_;
                sm.emissive = make_float3((float)gmat.emissive_factor[0] * es,
                                          (float)gmat.emissive_factor[1] * es,
                                          (float)gmat.emissive_factor[2] * es);
            } else {
                sm.materialID = ensureDefaultMaterial();
            }

            meshes.push_back(std::move(sm));
            result.push_back((int)meshes.size() - 1);
        }
        return result;
    }

    void traverseNode(const tg3_model& model, int nodeIndex, const Mat4& parentWorld,
                      std::unordered_map<int, std::vector<int>>& meshCache)
    {
        if (nodeIndex < 0 || (uint32_t)nodeIndex >= model.nodes_count) return;
        const tg3_node& node = model.nodes[(uint32_t)nodeIndex];

        Mat4 local = node.has_matrix ? Mat4::fromColumnMajor(node.matrix)
                                     : Mat4::fromTRS(node.translation, node.rotation, node.scale);
        Mat4 world = Mat4::multiply(parentWorld, local);

        if (node.mesh >= 0) {
            auto it = meshCache.find(node.mesh);
            const std::vector<int>* meshIndices;
            if (it == meshCache.end()) {
                meshCache[node.mesh] = buildMeshesForGltfMesh(model, node.mesh);
                meshIndices = &meshCache[node.mesh];
            } else {
                meshIndices = &it->second;
            }
            for (int mi : *meshIndices) {
                SceneInstance inst;
                inst.meshIndex = mi;
                inst.transform = world;
                instances.push_back(inst);
            }
        }

        for (uint32_t c = 0; c < node.children_count; c++)
            traverseNode(model, node.children[c], world, meshCache);
    }

    // Area-weighted smooth normals (fallback when a primitive has no NORMAL).
    static void computeSmoothNormals(SceneMesh& sm)
    {
        sm.normals.assign(sm.positions.size(), make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        for (const uint3& tri : sm.indices) {
            float3 p0 = f3(sm.positions[tri.x]);
            float3 p1 = f3(sm.positions[tri.y]);
            float3 p2 = f3(sm.positions[tri.z]);
            float3 n = cross(p1 - p0, p2 - p0); // area-weighted (unnormalized)
            sm.normals[tri.x].x += n.x; sm.normals[tri.x].y += n.y; sm.normals[tri.x].z += n.z;
            sm.normals[tri.y].x += n.x; sm.normals[tri.y].y += n.y; sm.normals[tri.y].z += n.z;
            sm.normals[tri.z].x += n.x; sm.normals[tri.z].y += n.y; sm.normals[tri.z].z += n.z;
        }
        for (float4& nf : sm.normals) {
            float3 n = f3(nf);
            float len = length(n);
            if (len > 1e-12f) { float inv = 1.0f / len; nf.x = n.x * inv; nf.y = n.y * inv; nf.z = n.z * inv; }
            else { nf.x = 0.0f; nf.y = 1.0f; nf.z = 0.0f; }
        }
    }
};


void printPrincipledMaterials(const SceneLoader& loader) {
    const std::vector<Material>& materials = loader.materials;
    
    std::cout << "====================================================\n";
    std::cout << "Loaded Scene Materials: " << materials.size() << "\n";
    std::cout << "====================================================\n";

    for (size_t i = 0; i < materials.size(); ++i) {
        const Material& m = materials[i];
        
        std::cout << "Material [" << i << "]: ";
        
        if (m.type != MAT_GLTF_PRINCIPLED_BSDF) {
            std::cout << "Non-Principled Material (Type ID: " << m.type << ")\n";
            std::cout << "----------------------------------------------------\n";
            continue;
        }

        // Differentiate between Principled (Opaque) and Principled Glass based on your struct
        if (m.isSpecular) {
            std::cout << "Principled Glass\n";
            std::cout << "  IOR:                " << m.ior << "\n";
            std::cout << "  Absorption (k):     (" << m.absorption.x << ", " 
                                                << m.absorption.y << ", " 
                                                << m.absorption.z << ")\n";
            std::cout << "  Medium Priority:    " << m.priority << "\n";
            std::cout << "  Is Boundary:        " << (m.boundary ? "True" : "False") << "\n";
        } else {
            std::cout << "Principled BSDF (Opaque)\n";
            std::cout << "  Base Color (RGBA):  (" << m.albedo.x << ", " 
                                                << m.albedo.y << ", " 
                                                << m.albedo.z << ", " 
                                                << m.albedo.w << ")\n";
            std::cout << "  Metallic:           " << m.metallic << "\n";
            std::cout << "  Roughness:          " << m.roughness << "\n";
            
            // Texture Maps
            std::cout << "  Textures:\n";
            std::cout << "    - Base Color:     " << (m.baseColorTex >= 0 ? std::to_string(m.baseColorTex) : "None") << "\n";
            std::cout << "    - Metallic/Rough: " << (m.mrTex >= 0 ? std::to_string(m.mrTex) : "None") << "\n";
            std::cout << "    - Normal:         " << (m.normalTex >= 0 ? std::to_string(m.normalTex) : "None") << "\n";
            std::cout << "    - Emissive:       " << (m.emissiveTex >= 0 ? std::to_string(m.emissiveTex) : "None") << "\n";
        }
        std::cout << "----------------------------------------------------\n";
    }
}