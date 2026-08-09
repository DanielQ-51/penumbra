#pragma once

#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

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

__host__ inline bool parseBool(const std::string& val) {
    std::string v = val;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "true");
}

struct MeshConfig {
    std::string path;
    float emissionMultiplier;
    float3 emissionColor;
    int materialID;
};

struct GLTFConfig {
    std::string path;
    uint32_t maxTextureDim;
    float emissionScale;
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

    // Assets
    std::vector<MeshConfig> meshes;
    std::vector<GLTFConfig> gltfs;
    std::vector<VolConfig> volumes;
};

__host__ inline bool loadConfig(const std::string& filepath, RenderConfig& config) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file: " << filepath << std::endl;
        return false;
    }

    std::string line;

    enum class ParseSection { None, Meshes, GLTFs, Volumes };
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
            else if (key == "VCM Merge Radius Power Factor") config.vcmMergeConst = std::stof(value);
            else if (key == "VCM Initial Merge Radius Multiplier") config.vcmInitialMergeRadiusMultiplier = std::stof(value);
        }
    }
    return true;
}