#pragma once

#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "material.cuh"

#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

__host__ inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

__host__ inline float3 parseVec3(const std::string& val) {
    float3 v;
    std::stringstream ss(val);
    ss >> v.x >> v.y >> v.z;
    return v;
}

// Parses "(x,y,z)" or "(x,y,z,w)" — used by Materials: lines, where the vector
// syntax needs commas (unlike parseVec3's bare-whitespace "x y z" convention,
// which the Meshes:/Volumes: line grammar already relies on elsewhere).
__host__ inline float4 parseVec4(const std::string& val, float defaultW = 0.0f) {
    std::string s = trim(val);
    if (!s.empty() && s.front() == '(') s.erase(s.begin());
    if (!s.empty() && s.back() == ')') s.pop_back();
    std::replace(s.begin(), s.end(), ',', ' ');
    std::stringstream ss(s);
    float x = 0.0f, y = 0.0f, z = 0.0f, w = defaultW;
    ss >> x >> y >> z;
    if (!(ss >> w)) w = defaultW;
    return make_float4(x, y, z, w);
}

__host__ inline bool parseBool(const std::string& val) {
    std::string v = val;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "true");
}

__host__ inline TextureType parseTextureType(const std::string& val) {
    if (val == "ALBEDO") return TEX_ALBEDO;
    if (val == "METAL") return TEX_METAL;
    if (val == "ROUGHNESS") return TEX_ROUGHNESS;
    if (val == "EMISSION") return TEX_EMISSION;
    if (val == "OCCLUSION") return TEX_OCCLUSION;
    if (val == "TRANSMISSION") return TEX_TRANSMISSION;
    if (val == "NORMAL") return TEX_NORMAL;
    std::cerr << "parseTextureType: unknown texture type '" << val << "', defaulting to ALBEDO\n";
    return TEX_ALBEDO;
}

// Splits a "key=value, key=value, ..." string on top-level commas only — commas
// inside "(...)" (vector params like albedo=(0.4,0.4,0.8)) must not be split on.
__host__ inline void parseKeyValueList(const std::string& s, std::unordered_map<std::string, std::string>& out) {
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        char c = (i < s.size()) ? s[i] : ',';
        if (c == '(') depth++;
        else if (c == ')') depth--;
        else if (c == ',' && depth == 0) {
            std::string piece = trim(s.substr(start, i - start));
            start = i + 1;
            if (piece.empty()) continue;
            size_t eq = piece.find('=');
            if (eq == std::string::npos) continue;
            out[trim(piece.substr(0, eq))] = trim(piece.substr(eq + 1));
        }
    }
}

struct MeshConfig {
    std::string path;
    float emissionMultiplier;
    float3 emissionColor;
    int materialID;
};

struct GLTFConfig {
    std::string path;
    uint32_t maxTextureDim = 2048;
    float emissionScale = 1.0f;
};

// for non gltf textures
struct TextureConfig {
    std::string name; // referenced by Materials: lines' texture-slot fields
    std::string path;
    TextureType type;
};

// One Materials: line. Params are kept as raw text and typed/resolved later by
// SceneLoader::buildMaterialFromConfig, which knows the Material:: factory methods
// and can resolve texture-name references against the loaded texture table —
// configParser.cuh itself stays free of Material-construction knowledge.
struct MaterialConfig {
    std::string type; // "DIFFUSE","METAL","SMOOTH_DIELECTRIC","THIN_DIELECTRIC",
                       // "MICROFACET_DIELECTRIC","LEAF","PRINCIPLED","PRINCIPLED_GLASS","MIRROR"
    std::unordered_map<std::string, std::string> params;
};

struct VolConfig {
    std::string path;
    float emissionMultiplier;
    float tempScale;
    float3 albedo;
    float densityScale;
    float anisotropy;
};

struct RenderConfig {
    // Window / System
    int width = 0;
    int height = 0;

    std::string name;

    // Integrator Settings
    std::string integratorType;
    int sampleCount = 0;
    int maxDepth = 0;
    int bvhLeafSize = 0;
    bool sampleEnvironment = false;
    bool postProcess = false;

    // BDPT Settings
    int bdptEyeDepth = 0;
    int bdptLightDepth = 0;
    bool bdptLightTrace = false;
    bool bdptNee = false;
    bool bdptNaive = false;
    bool bdptConnection = false;
    bool bdptDrawPath = false;
    bool bdptDoMis = false;
    bool bdptPaintWeight = false;
    bool vcmDoMerge = false;
    bool doSPPM = false;

    float vcmMergeConst = 0.0f;
    float vcmInitialMergeRadiusMultiplier = 0.0f;

    // Camera
    bool pinholeCamera = false;
    float3 camPos;
    float3 camRot;
    float camFov = 0.0f;
    float camApeture = 0.0f;
    float camFocalDist = 0.0f;

    // Environment map
    std::string envMapPath;        // EnvMap: <path>, relative to ROOT_DIR
    float envMapRotation = 0.0f;   // EnvMap Rotation: <degrees>, rotates around the Y-axis

    // Assets
    std::vector<MeshConfig> meshes;
    std::vector<GLTFConfig> gltfs;
    std::vector<VolConfig> volumes;
    std::vector<TextureConfig> textures;
    std::vector<MaterialConfig> materials;
    std::string materialsFile; // MaterialsFile: <path> — optional shared Materials:/Textures: library
};

// Decodes one Textures: line — "name; type; path" — into a TextureConfig, shared
// between loadConfig's own Textures: section and loadMaterialsLibrary below.
__host__ inline void parseTextureLine(const std::string& line, std::vector<TextureConfig>& out) {
    TextureConfig tex;
    std::stringstream ss(line);
    std::string seg;
    if (std::getline(ss, seg, ';')) tex.name = trim(seg);
    if (std::getline(ss, seg, ';')) tex.type = parseTextureType(trim(seg));
    if (std::getline(ss, seg, ';')) tex.path = trim(seg);
    out.push_back(tex);
}

// Decodes one Materials: line — "type; key=value, key=value, ..." — into a
// MaterialConfig, shared between loadConfig's own Materials: section and
// loadMaterialsLibrary below.
__host__ inline void parseMaterialLine(const std::string& line, std::vector<MaterialConfig>& out) {
    MaterialConfig mat;
    std::stringstream ss(line);
    std::string seg;
    if (std::getline(ss, seg, ';')) mat.type = trim(seg);
    if (std::getline(ss, seg, ';')) parseKeyValueList(trim(seg), mat.params);
    out.push_back(mat);
}

// Loads a shared Materials:/Textures: library file, referenced via a scene
// config's "MaterialsFile: <path>" directive. Deliberately recognizes ONLY
// Materials:/Textures: sections (plus its own MaterialsFile: for one level of
// chaining) — it must not honor scalar keys like "width:", or a shared library
// could clobber the including scene's settings. Entries append to
// config.materials/config.textures in file-scan order, so MeshConfig::materialID
// keeps meaning "index into this list" regardless of whether entries came from a
// library file or the scene file's own local sections.
//
// Path is resolved against ROOT_DIR directly (not the ASSET_PATH macro, which
// isn't visible here — it's #define'd in main.cu/hostSetup.cuh AFTER their
// #include of this file) so this doesn't depend on process CWD.
__host__ inline bool loadMaterialsLibrary(const std::string& filepath, RenderConfig& config, int depth = 0) {
    if (depth > 4) {
        std::cerr << "loadMaterialsLibrary: MaterialsFile chain too deep at " << filepath << "\n";
        return false;
    }
    std::string resolvedPath = std::string(ROOT_DIR) + "/" + filepath;
    std::ifstream file(resolvedPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open materials library: " << resolvedPath << std::endl;
        return false;
    }

    std::string line;
    enum class LibSection { None, Materials, Textures };
    LibSection section = LibSection::None;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Same "MaterialsFile starts with Materials" ambiguity as loadConfig()
        // above — must be checked before the section-header prefix match.
        if (line.rfind("MaterialsFile", 0) == 0) {
            size_t delimiterPos = line.find(':');
            if (delimiterPos != std::string::npos) {
                std::string value = trim(line.substr(delimiterPos + 1));
                if (!value.empty()) loadMaterialsLibrary(value, config, depth + 1);
            }
            continue;
        }

        if (line.rfind("Materials", 0) == 0) { section = LibSection::Materials; continue; }
        if (line.rfind("Textures", 0) == 0) { section = LibSection::Textures; continue; }

        if (section == LibSection::Textures) {
            parseTextureLine(line, config.textures);
        } else if (section == LibSection::Materials) {
            parseMaterialLine(line, config.materials);
        }
    }
    return true;
}

__host__ inline bool loadConfig(const std::string& filepath, RenderConfig& config) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file: " << filepath << std::endl;
        return false;
    }

    std::string line;

    enum class ParseSection { None, Meshes, GLTFs, Volumes, Materials, Textures };
    ParseSection section = ParseSection::None;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Detect section headers
        if (line.rfind("Meshes", 0) == 0) {
            section = ParseSection::Meshes;
            continue;
        }

        if (line.rfind("GLTF", 0) == 0) {
            // Matches both "GLTF" and "GLTFs" headers
            section = ParseSection::GLTFs;
            continue;
        }

        if (line.rfind("Volumes", 0) == 0) {
            section = ParseSection::Volumes;
            continue;
        }

        // MaterialsFile: <path> is a scalar directive, not a section header — must
        // be checked before the "Materials" section-header prefix match below,
        // since "MaterialsFile" itself starts with "Materials" and would otherwise
        // be swallowed as a (bogus) section header.
        if (line.rfind("MaterialsFile", 0) == 0) {
            size_t delimiterPos = line.find(':');
            if (delimiterPos != std::string::npos) {
                std::string value = trim(line.substr(delimiterPos + 1));
                if (!value.empty()) {
                    config.materialsFile = value;
                    loadMaterialsLibrary(value, config);
                }
            }
            continue;
        }

        if (line.rfind("Materials", 0) == 0) {
            section = ParseSection::Materials;
            continue;
        }

        if (line.rfind("Textures", 0) == 0) {
            section = ParseSection::Textures;
            continue;
        }

        if (section == ParseSection::Meshes) {
            // Mesh Line Format: path; multiplier * emission; materialID
            // Example: scenedata/smallbox.obj; 1.0 * (0.0, 0.0, 0.0); 2

            MeshConfig mesh;
            std::stringstream ss(line);
            std::string segment;

            // 1. Path
            if(std::getline(ss, segment, ';')) mesh.path = trim(segment);

            // 2. Emission Complex Logic
            if(std::getline(ss, segment, ';')) {
                std::string complexEm = trim(segment);
                size_t starPos = complexEm.find('*');
                size_t openParen = complexEm.find('(');
                size_t closeParen = complexEm.find(')');

                if (starPos != std::string::npos && openParen != std::string::npos) {
                    // Parse Multiplier
                    mesh.emissionMultiplier = std::stof(complexEm.substr(0, starPos));

                    // Parse Vector (0.0, 0.0, 0.0) -> replace commas with space for easier parsing
                    std::string vecStr = complexEm.substr(openParen + 1, closeParen - openParen - 1);
                    std::replace(vecStr.begin(), vecStr.end(), ',', ' ');
                    mesh.emissionColor = parseVec3(vecStr);
                }
            }

            // 3. Material ID
            if(std::getline(ss, segment, ';')) mesh.materialID = std::stoi(trim(segment));

            config.meshes.push_back(mesh);
        } else if (section == ParseSection::GLTFs) {
            // GLTF Line Format: path; maxTextureDim; emissionScale
            // Example: scenedata/sponza.gltf; 2048; 1.0

            GLTFConfig gltf;
            std::stringstream ss(line);
            std::string segment;

            // 1. Path
            if(std::getline(ss, segment, ';')) gltf.path = trim(segment);

            // 2. Max Texture Dimension
            if(std::getline(ss, segment, ';')) {
                std::string dimStr = trim(segment);
                if (!dimStr.empty()) {
                    gltf.maxTextureDim = static_cast<uint32_t>(std::stoul(dimStr));
                }
            }

            // 3. Emission Scale
            if(std::getline(ss, segment, ';')) {
                std::string emStr = trim(segment);
                if (!emStr.empty()) {
                    gltf.emissionScale = std::stof(emStr);
                }
            }

            config.gltfs.push_back(gltf);
        } else if (section == ParseSection::Volumes) {
            VolConfig vol;
            std::stringstream ss(line);
            std::string segment;

            // 1. Path
            if(std::getline(ss, segment, ';')) vol.path = trim(segment);

            // 2. Albedo Vector
            if(std::getline(ss, segment, ';')) {
                std::string vecStr = trim(segment);
                size_t openParen = vecStr.find('(');
                size_t closeParen = vecStr.find(')');

                if (openParen != std::string::npos && closeParen != std::string::npos) {
                    vecStr = vecStr.substr(openParen + 1, closeParen - openParen - 1);
                    std::replace(vecStr.begin(), vecStr.end(), ',', ' ');
                    vol.albedo = parseVec3(vecStr);
                }
            }

            // 3. Density Scale
            if(std::getline(ss, segment, ';')) vol.densityScale = std::stof(trim(segment));

            // 4. Temperature Scale
            if(std::getline(ss, segment, ';')) vol.tempScale = std::stof(trim(segment));

            // 5. Emission Multiplier
            if(std::getline(ss, segment, ';')) vol.emissionMultiplier = std::stof(trim(segment));

            if(std::getline(ss, segment, ';')) vol.anisotropy = std::stof(trim(segment));

            config.volumes.push_back(vol);
        } else if (section == ParseSection::Materials) {
            parseMaterialLine(line, config.materials);
        } else if (section == ParseSection::Textures) {
            parseTextureLine(line, config.textures);
        }
        else {
            // Standard Key-Value Parsing
            size_t delimiterPos = line.find(':');
            if (delimiterPos == std::string::npos) continue; // Headers without values

            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));

            if (value.empty()) continue; // Skip headers like "BDPT Specific Settings:"

            // Mapping
            if (key == "width") config.width = std::stoi(value);
            else if (key == "height") config.height = std::stoi(value);
            else if (key == "Integrator") config.integratorType = value;
            else if (key == "Name") config.name = value;
            else if (key == "Sample Count") config.sampleCount = std::stoi(value);
            else if (key == "Unidirectional Max Depth") config.maxDepth = std::stoi(value);
            else if (key == "BVH recommended leaf size") config.bvhLeafSize = std::stoi(value);
            else if (key == "Bidirectional Eye Depth") config.bdptEyeDepth = std::stoi(value);
            else if (key == "Bidirectional Light Depth") config.bdptLightDepth = std::stoi(value);

            // Booleans
            else if (key == "BDPT_LIGHTTRACE") config.bdptLightTrace = parseBool(value);
            else if (key == "BDPT_NEE") config.bdptNee = parseBool(value);
            else if (key == "BDPT_NAIVE") config.bdptNaive = parseBool(value);
            else if (key == "BDPT_CONNECTION") config.bdptConnection = parseBool(value);
            else if (key == "BDPT_DRAWPATH") config.bdptDrawPath = parseBool(value);
            else if (key == "BDPT_DOMIS") config.bdptDoMis = parseBool(value);
            else if (key == "BDPT_PAINTWEIGHT") config.bdptPaintWeight = parseBool(value);
            else if (key == "Pinhole Camera") config.pinholeCamera = parseBool(value);
            else if (key == "SAMPLE_ENVIRONMENT") config.sampleEnvironment = parseBool(value);
            else if (key == "Post Process") config.postProcess = parseBool(value);
            else if (key == "VCM_DOMERGE") config.vcmDoMerge = parseBool(value);

            // Vectors & Floats
            else if (key == "Camera Position") config.camPos = parseVec3(value);
            else if (key == "Camera Rotation") config.camRot = parseVec3(value);
            else if (key == "Camera FOV") config.camFov = std::stof(value);
            else if (key == "Camera Apeture") config.camApeture = std::stof(value);
            else if (key == "Camera FocalDist") config.camFocalDist = std::stof(value);
            else if (key == "EnvMap") config.envMapPath = value;
            else if (key == "EnvMap Rotation") config.envMapRotation = std::stof(value);
            else if (key == "VCM Merge Radius Power Factor") config.vcmMergeConst = std::stof(value);
            else if (key == "VCM Initial Merge Radius Multiplier") config.vcmInitialMergeRadiusMultiplier = std::stof(value);
        }
    }
    return true;
}
