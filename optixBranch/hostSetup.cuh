#pragma once

#include "objects.cuh"
#include "util.cuh"
#include "helpers.cuh"
#include "volumeRendering.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
#include "restirPTenhanced_host.cuh"
#include "unidirectional_host.cuh"
#include <chrono>
#include <iostream>
#include <exception>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include "textureManager.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <iomanip>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>   // [STACK_FIX] needed for optixUtilComputeStackSizes / optixPipelineSetStackSize

#include "sceneLoader.cuh"
#include "configParser.cuh"

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

#ifndef PTX_DIR
#define PTX_DIR "" 
#endif

__host__ std::string read_file_to_string(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline void gltfSmokeTest(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "gltf: cannot open " << path << "\n"; return; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());

    tg3_parse_options opts;  tg3_parse_options_init(&opts);
    tg3_error_stack errors;  tg3_error_stack_init(&errors);
    tg3_model model;

    tg3_error_code err = tg3_parse_auto(
        &model, &errors,
        bytes.data(), bytes.size(),
        nullptr, 0,              // base_dir: unused for self-contained GLB
        &opts);

    if (err != TG3_OK || tg3_errors_has_error(&errors)) {
        std::cerr << "gltf parse failed:\n";
        for (uint32_t i = 0; i < errors.count; i++)
            std::cerr << "  " << (errors.entries[i].message ? errors.entries[i].message : "(null)") << "\n";
    } else {
        std::cout << "glTF ok: "
                  << model.meshes_count    << " meshes, "
                  << model.materials_count << " materials, "
                  << model.nodes_count     << " nodes, "
                  << model.textures_count  << " textures, "
                  << model.images_count    << " images, "
                  << model.accessors_count << " accessors, "
                  << model.buffers_count   << " buffers\n";
    }

    const tg3_mesh& mesh = model.meshes[0];
    const tg3_primitive& prim = mesh.primitives[0];

    AccessorView pv = resolveAccessor(model, findAttribute(prim, "POSITION"));
    std::vector<float4> pos; readVec3(pv, pos);
    std::cout << "POSITION: " << pos.size() << " verts\n";
    for (int i = 0; i < 3 && i < (int)pos.size(); i++)
        std::cout << "  (" << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << ")\n";

    AccessorView iv = resolveAccessor(model, prim.indices);
    std::vector<uint32_t> idx; readIndices(iv, idx);
    std::cout << "indices: " << idx.size() << "\n";

    tg3_model_free(&model);      // one call frees the whole arena
    tg3_error_stack_free(&errors);
}

__host__ int initOptixSystem(OptixEngineState& engineState) {
    cudaFree(0); 
    CUcontext cuCtx = 0;

    if (optixInit() != OPTIX_SUCCESS) {
        std::cerr << "Failed to initialize OptiX!" << std::endl;
        return -1;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4;
    options.logCallbackFunction = [](uint32_t level, const char* tag, const char* message, void*) {
        std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
    };

    OptixDeviceContext context = nullptr;
    if (optixDeviceContextCreate(cuCtx, &options, &context) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create OptiX context!" << std::endl;
        return -1;
    }

    std::cout << "OptiX Context Created Successfully. Ready to build." << std::endl;

    std::string ptxCode;
    try {
        std::string ptxPath = std::string(PTX_DIR) + "/renderer.ptx";
        
        std::cout << "Loading PTX from: " << ptxPath << std::endl;
        ptxCode = read_file_to_string(ptxPath);
        
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    std::string restirPTX;
    try {
        std::string ptxPath = std::string(PTX_DIR) + "/restirPTenhancedShaders.ptx";
        
        std::cout << "Loading PTX from: " << ptxPath << std::endl;
        restirPTX = read_file_to_string(ptxPath);
        
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOptions = {}; 
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.numPayloadValues = 2; 
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.usesPrimitiveTypeFlags = (OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipelineOptions.pipelineLaunchParamsVariableName = "allParams"; 

    OptixModule module = nullptr;
    optixModuleCreate(
        context, 
        &moduleOptions, 
        &pipelineOptions, 
        ptxCode.c_str(), 
        ptxCode.size(), 
        nullptr, nullptr, 
        &module
    );

    OptixModule restirModule = nullptr;
    optixModuleCreate(
        context, 
        &moduleOptions, 
        &pipelineOptions, 
        restirPTX.c_str(), 
        restirPTX.size(), 
        nullptr, nullptr, 
        &restirModule
    );

    OptixProgramGroupOptions pgOptions = {};
    
    OptixProgramGroupDesc raygenUnidirectionalDesc = {};
    raygenUnidirectionalDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenUnidirectionalDesc.raygen.module            = module;
    raygenUnidirectionalDesc.raygen.entryFunctionName = "__raygen__unidirectional"; 

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = nullptr;             
    missDesc.miss.entryFunctionName = nullptr;

    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = module;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__gather";

    OptixProgramGroupDesc restirCandidateGenDesc = {};
    restirCandidateGenDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    restirCandidateGenDesc.raygen.module            = restirModule;
    restirCandidateGenDesc.raygen.entryFunctionName = "__raygen__restirCandidateGeneration"; 

    OptixProgramGroupDesc restirSpatialDesc = {};
    restirSpatialDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    restirSpatialDesc.raygen.module            = restirModule;
    restirSpatialDesc.raygen.entryFunctionName = "__raygen__restirSpatialReuse"; 

    OptixProgramGroupDesc restirTemporalDesc = {};
    restirTemporalDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    restirTemporalDesc.raygen.module            = restirModule;
    restirTemporalDesc.raygen.entryFunctionName = "__raygen__restirTemporalReuse"; 

    OptixProgramGroupDesc programGroupDescs[] = { 
        raygenUnidirectionalDesc, 
        missDesc,
        hitgroupDesc, 
        restirCandidateGenDesc,
        restirSpatialDesc,
        restirTemporalDesc
    };
    OptixProgramGroup programGroups[6];

    optixProgramGroupCreate(context, programGroupDescs, 6, &pgOptions, nullptr, nullptr, programGroups);

    OptixProgramGroup raygenUnidirectionalProgramGroup = programGroups[0];
    OptixProgramGroup missProgramGroup                 = programGroups[1];
    OptixProgramGroup hitgroupProgramGroup             = programGroups[2];
    OptixProgramGroup restirCandidateGenGroup          = programGroups[3];
    OptixProgramGroup restirSpatialGroup               = programGroups[4];
    OptixProgramGroup restirTemporalGroup              = programGroups[5];

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;   // [STACK_FIX] shaders trace via optixTraverse; 0 is UB (was 0)

    optixPipelineCreate(
        context,
        &pipelineOptions,
        &linkOptions,
        programGroups, 6,
        nullptr, nullptr,
        &pipeline
    );

    // [STACK_FIX] BEGIN - set an explicit pipeline stack size.
    // Previously no stack size was set, so OptiX used a default derived from
    // maxTraceDepth=0 and assumed a single-GAS scene. Adding the IAS -> GAS
    // (2-level) traversal pushed the deepest raygen (temporal reuse) past that
    // default, overflowing the continuation stack on frame 1. Size it explicitly
    // and declare maxTraversableGraphDepth = 2 for the instance layer.
    {
        OptixStackSizes stackSizes = {};
        for (auto pg : programGroups) {
            optixUtilAccumulateStackSizes(pg, &stackSizes, pipeline);
        }

        uint32_t dcStackTraversal = 0, dcStackState = 0, continuationStack = 0;
        optixUtilComputeStackSizes(
            &stackSizes,
            /* maxTraceDepth */ 1,   // [STACK_FIX] must match linkOptions.maxTraceDepth
            /* maxCCDepth    */ 0,
            /* maxDCDepth    */ 0,
            &dcStackTraversal, &dcStackState, &continuationStack
        );

        optixPipelineSetStackSize(
            pipeline,
            dcStackTraversal,
            dcStackState,
            continuationStack,
            /* maxTraversableGraphDepth */ 2   // IAS -> GAS
        );
    }
    // [STACK_FIX] END

    struct RaygenRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    RaygenRecord rgRecords[4];
    optixSbtRecordPackHeader(programGroups[0], &rgRecords[0]); 
    optixSbtRecordPackHeader(programGroups[3], &rgRecords[1]); 
    optixSbtRecordPackHeader(programGroups[4], &rgRecords[2]); 
    optixSbtRecordPackHeader(programGroups[5], &rgRecords[3]); 

    CUdeviceptr d_rgRecordArray;
    cudaMalloc(reinterpret_cast<void**>(&d_rgRecordArray), sizeof(RaygenRecord) * 4);
    cudaMemcpy(reinterpret_cast<void*>(d_rgRecordArray), rgRecords, sizeof(RaygenRecord) * 4, cudaMemcpyHostToDevice);

    RaygenRecord hgRecord;
    optixSbtRecordPackHeader(hitgroupProgramGroup, &hgRecord); 
    CUdeviceptr d_hgRecord;
    cudaMalloc(reinterpret_cast<void**>(&d_hgRecord), sizeof(RaygenRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_hgRecord), &hgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    RaygenRecord msRecord;
    optixSbtRecordPackHeader(missProgramGroup, &msRecord); 

    CUdeviceptr d_msRecord;
    cudaMalloc(reinterpret_cast<void**>(&d_msRecord), sizeof(RaygenRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_msRecord), &msRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    auto buildMenu = [&](int arrayIndex) -> OptixShaderBindingTable {
        OptixShaderBindingTable sbt = {};
        
        sbt.raygenRecord = d_rgRecordArray + (arrayIndex * sizeof(RaygenRecord));

        sbt.missRecordBase          = d_msRecord;
        sbt.missRecordStrideInBytes = sizeof(RaygenRecord);
        sbt.missRecordCount         = 1;
        
        sbt.hitgroupRecordBase          = d_hgRecord;
        sbt.hitgroupRecordStrideInBytes = sizeof(RaygenRecord);
        sbt.hitgroupRecordCount         = 1;
        return sbt;
    };

    engineState.sbt_unidirectional  = buildMenu(0);
    engineState.sbt_restirCandidate = buildMenu(1);
    engineState.sbt_restirSpatial   = buildMenu(2);
    engineState.sbt_restirTemporal  = buildMenu(3);

    engineState.context = context;
    engineState.pipeline = pipeline;
    engineState.raygenUnidirectionalProgramGroup = raygenUnidirectionalProgramGroup;
    engineState.raygenRestirCandidateProgramGroup = restirCandidateGenGroup;
    engineState.raygenRestirSpatialProgramGroup = restirSpatialGroup;
    engineState.raygenRestirTemporalProgramGroup = restirTemporalGroup;
    engineState.hitgroupProgramGroup = hitgroupProgramGroup;
    engineState.module = module;
    engineState.restirModule = restirModule;
    engineState.d_rgRecord = d_rgRecordArray;
    engineState.d_hgRecord = d_hgRecord;
    engineState.d_msRecord = d_msRecord;

    std::cout << "OptiX engine setup complete." << std::endl;
    return 0;
}

__host__ int optixEngineCleanup(OptixEngineState& engineState) {
    cudaFree(reinterpret_cast<void*>(engineState.d_rgRecord));
    cudaFree(reinterpret_cast<void*>(engineState.d_msRecord));
    cudaFree(reinterpret_cast<void*>(engineState.d_hgRecord));
    optixPipelineDestroy(engineState.pipeline);
    optixProgramGroupDestroy(engineState.raygenUnidirectionalProgramGroup);
    optixProgramGroupDestroy(engineState.raygenRestirCandidateProgramGroup);
    optixProgramGroupDestroy(engineState.raygenRestirSpatialProgramGroup);
    optixProgramGroupDestroy(engineState.raygenRestirTemporalProgramGroup);
    optixProgramGroupDestroy(engineState.hitgroupProgramGroup);
    optixProgramGroupDestroy(engineState.missProgramGroup);
    optixModuleDestroy(engineState.module);
    optixModuleDestroy(engineState.restirModule);
    optixDeviceContextDestroy(engineState.context);

    std::cout << "OptiX engine cleanup complete." << std::endl;
    return 0;
}

using namespace std;

__host__ OptixTraversableHandle buildOptixGAS(
    OptixDeviceContext context,
    const std::vector<float3>& vertices,
    const std::vector<uint3>& indices,
    CUdeviceptr& out_d_gas_output_buffer // We keep this to free it later
) {
    std::cout << "Begin OptiX GAS build." << std::endl;
    // 1. Upload Vertices to GPU
    CUdeviceptr d_vertices;
    size_t vertices_size = vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice);

    // 2. Upload Indices to GPU
    CUdeviceptr d_indices;
    size_t indices_size = indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size);
    cudaMemcpy(reinterpret_cast<void*>(d_indices), indices.data(), indices_size, cudaMemcpyHostToDevice);

    // 3. Describe the geometry to OptiX
    uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    // Vertex data
    triangle_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices         = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers       = &d_vertices;

    // Index data
    triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes  = sizeof(uint3);
    triangle_input.triangleArray.numIndexTriplets    = static_cast<uint32_t>(indices.size());
    triangle_input.triangleArray.indexBuffer         = d_indices;

    triangle_input.triangleArray.flags               = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords       = 1;

    // 4. Set up build options (Fast trace speed, allow compaction if desired)
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // 5. Ask OptiX how much memory it needs
    OptixAccelBufferSizes gas_buffer_sizes;
    optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes);

    // 6. Allocate memory for the build process
    CUdeviceptr d_temp_buffer_gas;
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&out_d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes);

    // 7. Execute the build
    OptixTraversableHandle gas_handle = 0;
    optixAccelBuild(
        context,
        0,                  // CUDA stream
        &accel_options,
        &triangle_input,
        1,                  // Number of build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        out_d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr,            // Emitted properties (used for compaction)
        0                   // Num emitted properties
    );

    cudaDeviceSynchronize();

    cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas));
    cudaFree(reinterpret_cast<void*>(d_vertices));
    cudaFree(reinterpret_cast<void*>(d_indices));

    std::cout << "OptiX GAS build complete." << std::endl;
    return gas_handle;
}

__host__ OptixTraversableHandle buildOptixIAS(
    OptixDeviceContext context,
    CUdeviceptr d_instances,
    size_t numInstance,
    CUdeviceptr& out_d_ias_output_buffer // Keep this to free during cleanup
) {
    if (numInstance < 1) {
        throw std::runtime_error("Error: Attempting to build an IAS with 0 instances.");
    }

    std::cout << "Begin OptiX IAS build with " << numInstance << " instances." << std::endl;

    OptixBuildInput ias_input = {};
    ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_input.instanceArray.instances = d_instances;
    ias_input.instanceArray.numInstances = static_cast<uint32_t>(numInstance);

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    optixAccelComputeMemoryUsage(context, &ias_accel_options, &ias_input, 1, &ias_buffer_sizes);

    CUdeviceptr d_temp_buffer_ias;
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_ias), ias_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&out_d_ias_output_buffer), ias_buffer_sizes.outputSizeInBytes);

    OptixTraversableHandle iasHandle = 0;
    optixAccelBuild(
        context,
        0,                  // CUDA stream
        &ias_accel_options,
        &ias_input,
        1,                  // Number of build inputs
        d_temp_buffer_ias,
        ias_buffer_sizes.tempSizeInBytes,
        out_d_ias_output_buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &iasHandle,
        nullptr,            
        0                   
    );

    cudaDeviceSynchronize();

    cudaFree(reinterpret_cast<void*>(d_temp_buffer_ias));

    std::cout << "OptiX IAS build complete." << std::endl;
    return iasHandle;
}

int initRender(OptixEngineState& engineState, string configPath, int renderNumber)
{
    RenderConfig config;
    loadConfig(configPath, config);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::cout << "------------------------------------------------------------------------------------------------------ \n" << 
        "Began render number " << renderNumber << ": \"" << config.name << "\"\n\n";

    std::cout << "Current time: " 
              << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
              << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    int w = config.width;
    int h = config.height;

    int integratorChoice = matchIntegrator(config.integratorType); 

    int sampleCount = config.sampleCount;
    int maxDepth = config.maxDepth;

    Camera camera;
    if (config.pinholeCamera)
        camera = Camera::Pinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov);
    else
        camera = Camera::NotPinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov, 
            config.camApeture, config.camFocalDist);

    camera.preCompute();

    Image image = Image(w, h);
    image.postProcess = config.postProcess;

    if (integratorChoice == OPTIX_NORMAL)
    {
        std::cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount <<" with a max depth of " << 
            maxDepth << ".\nIntegrating with Optix Naive + NEE Unidirectional MIS." << 
            endl << endl;
    } else if (integratorChoice == OPTIX_RESTIR_PT) {
        std::cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount <<" with a max depth of " << 
            maxDepth << ".\nIntegrating with Optix ReSTIR PT." << 
            endl << endl;
    }

    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors; // unused now
    vector<float2> uvs;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<Material> mats;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading environment map
    //---------------------------------------------------------------------------------------------------------------------------------------------------

#if USE_ENV_MAP == 1
    //EnvironmentMapManager envManager(ASSET_PATH("assets/environment/lakeside_sunrise_1k.exr"));
    EnvironmentMapManager envManager(ASSET_PATH("assets/environment/sunflowers_puresky_1k.exr"));
#else
    EnvironmentMapManager envManager(ASSET_PATH("assets/environment/black.exr"));
#endif
    //envManager.setRotation(70.0f + (float)renderNumber);
    envManager.setRotation(130.0f);
    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Textures
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    TextureManager texManager;

    // All of these are 8-bit sRGB base-color maps; the hardware linearizes them.
    int tex_enkidu      = texManager.addFromFile(ASSET_PATH("assets/textures/enkidutexture.bmp"), TEX_SRGB);
    int tex_enkiduChibi = texManager.addFromFile(ASSET_PATH("assets/textures/enkiduchibitexture.bmp"), TEX_SRGB);
    int tex_leaf        = texManager.addFromFile(ASSET_PATH("assets/textures/leaftex2.bmp"), TEX_SRGB);
    int tex_leafAutumn  = texManager.addFromFile(ASSET_PATH("assets/textures/leafautumn.bmp"), TEX_SRGB);
    int tex_wood        = texManager.addFromFile(ASSET_PATH("assets/textures/wood.bmp"), TEX_SRGB);
    int tex_wall        = texManager.addFromFile(ASSET_PATH("assets/textures/wall.bmp"), TEX_SRGB);
    int tex_glove       = texManager.addFromFile(ASSET_PATH("assets/textures/Material.006_baseColor.bmp"), TEX_SRGB);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Creating Materials
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    Material principledWood = Material::Principled(f4(1.0f), 0.0f, 0.15f, tex_wood, -1);

    Material wood = Material::Leaf(tex_wood, 1.5f, 0.3f, f4(), 0.00f);
    Material wall = Material::DiffuseTextured(tex_wall);
    Material lambertTextured = Material::DiffuseTextured(tex_enkidu);
    Material lambert2Textured = Material::DiffuseTextured(tex_enkiduChibi);

    Material lambertBlue = Material::Diffuse(f4(0.4f,0.4f,0.8f));
    Material lambertGrey = Material::Diffuse(f4(0.8f,0.8f,0.8f));
    Material lambertGreyBlue = Material::Diffuse(f4(0.6f,0.6f,0.7f));
    Material lambertWhite = Material::Diffuse(f4(0.9f,0.9f,0.9f));
    Material lambertGreen = Material::Diffuse(f4(0.2f,0.6f,0.6f));
    Material lambertRed = Material::Diffuse(f4(0.90f,0.1f,0.1f));
    Material lambertVeryGreen = Material::Diffuse(f4(0.1f,0.9f,0.1f));
    Material lambertBLACK = Material::Diffuse(f4(0.0f,0.0f,0.0f));
    Material lambert95 = Material::Diffuse(f4(0.95f,0.95f,0.95f));
    Material lambert1 = Material::Diffuse(f4(1.0f));
    Material lambert50 = Material::Diffuse(f4(0.5f,0.5f,0.5f));

    float4 eta_steel = f4(0.14f, 0.16f, 0.13f);   // real part (R,G,B,alpha)
    float4 k_steel   = f4(4.1f, 2.3f, 3.1f);     // imaginary part (absorption)


    float4 eta_gold = f4(0.17f, 0.35f, 1.5f);  // real part of refractive index
    float4 k_gold   = f4(3.1f, 2.7f, 1.9f);   // imaginary part, absorption
    float roughness_polished = 0.05f;  
    float roughness_rough = 0.15f;  
    float roughness_rougher = 0.35f;  

    Material gold = Material::Metal(eta_gold, eta_gold, roughness_polished);
    Material gold15 = Material::Metal(eta_gold, eta_gold, roughness_rough);
    Material steel = Material::Metal(eta_steel, eta_steel, roughness_rough);
    Material steelSmooth = Material::Metal(eta_steel, eta_steel, roughness_polished);
    Material steel25 = Material::Metal(eta_steel, eta_steel, 0.25f);
    Material roughSteel = Material::Metal(eta_steel, eta_steel, roughness_rougher);

    float ior = 1.5f;

    Material glass = Material::SmoothDielectric(ior, f4(0.0f), 1);
    Material diamond = Material::SmoothDielectric(2.42f, f4(0.0f), 1);

    Material water = Material::SmoothDielectric(1.333f, f4(), 2);
    Material tea = Material::SmoothDielectric(1.333f, 2.5f * f4(0.180f, 1.5f, 2.996f), 2);

    Material ice = Material::SmoothDielectric(1.31f, f4(0.2f), 0);

    Material air = Material::SmoothDielectric(1.0f, f4(0.0f), 99);

    //Material leaf = Material::Leaf(1.5f, 0.6f, f4(0.8f, 0.25f, 0.28f), 0.2f);
    Material leaf = Material::Leaf(tex_leaf, 1.5f, 0.10f, f4(0.22f, 0.75f, 0.28f), 0.15f);
    Material leafAutumn = Material::Leaf(tex_leafAutumn, 1.5f, 0.8f, f4(0.22f, 0.75f, 0.28f), 0.6f);
    Material canopy = Material::Leaf(tex_leaf, 1.5f, 0.9f, f4(0.22f, 0.75f, 0.28f), 0.7f);
    Material leafStem = Material::Diffuse(f4(0.90f, 0.9f, 0.83f));
    Material sky = Material::Diffuse(f4(0.4f, 0.4f, 1.00f));

    Material mirror = Material::Mirror();
    Material thinGlass = Material::ThinDielectric(1.5f);


    Material blade = Material::Metal(f4(2.88f, 2.49f, 2.12f), f4(3.05f, 2.97f, 2.76f), 0.15f);

    Material liners = Material::Metal(f4(1.80f, 1.40f, 0.40f), f4(2.10f, 2.80f, 4.20f), 0.35f);

    Material hardware = Material::Metal(eta_steel, k_steel, 0.45f);

    float4 cf_albedo = f4(0.03f, 0.03f, 0.03f);
    Material handles = Material::Diffuse(cf_albedo);

    Material glove = Material::Leaf(tex_glove, 1.5f, 0.4f, f4(), 0.00f);

    mats.push_back(air); // index 0

    mats.push_back(lambertBlue); // index 1
    mats.push_back(lambertWhite); // index 2
    mats.push_back(lambertGreen); // index 3
    mats.push_back(gold); // index 4
    mats.push_back(glass); // index 5
    mats.push_back(lambertRed); // index 6
    mats.push_back(steel); // index 7
    mats.push_back(tea); // index 8
    mats.push_back(ice); // index 9
    mats.push_back(water); // index 10
    mats.push_back(lambertTextured); // index 11
    mats.push_back(lambert2Textured); // index 12
    mats.push_back(leaf); // index 13
    mats.push_back(leafStem); // index 14
    mats.push_back(sky); // index 15
    mats.push_back(leafAutumn); // index 16
    mats.push_back(lambertGrey); // index 17
    mats.push_back(diamond); // index 18
    mats.push_back(mirror); // index 19
    mats.push_back(lambertBLACK); // index 20
    mats.push_back(lambert95); // index 21
    mats.push_back(lambert50); // index 22
    mats.push_back(lambertVeryGreen); // index 23
    mats.push_back(wood); // index 24
    mats.push_back(lambertGreyBlue); // index 25
    mats.push_back(wall); // index 26
    mats.push_back(roughSteel); // index 27
    mats.push_back(thinGlass); // index 28
    mats.push_back(steelSmooth); // index 29
    mats.push_back(steel25); // index 30
    mats.push_back(gold15); // index 31
    mats.push_back(lambert1); // index 32
    mats.push_back(blade); // index 33
    mats.push_back(liners); // index 34
    mats.push_back(hardware); // index 35
    mats.push_back(handles); // index 36
    mats.push_back(glove); // index 37
    mats.push_back(principledWood); // index 38

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    vector<LightDescriptor> lightDesc;

    for (MeshConfig c : config.meshes)
    {
        readObjSimple(ASSET_PATH(c.path), points, normals, colors, uvs, mesh, lightsvec, lightDesc, f3(),
                c.emissionMultiplier * c.emissionColor, c.materialID);
    }

    Vertices* verts;
    Triangle* scene;

    cudaMalloc(&verts,  sizeof(Vertices));
    Vertices temp;

    cudaMalloc(&temp.positions, sizeof(float4) * points.size());
    cudaMalloc(&temp.normals, sizeof(float4) * normals.size());
    cudaMalloc(&temp.uvs,  sizeof(float2) * uvs.size());

    cudaMemcpy(temp.positions, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.normals, normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(verts, &temp, sizeof(Vertices), cudaMemcpyHostToDevice);

    if (mesh.size() == 0) {
        cout << "Error: No triangles loaded." << endl;
        return 1;
    }
    cout << "scene data read. There are " << mesh.size() << " Triangles and " << lightsvec.size() << " +1 lights" << endl;

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene read took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;
    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    SceneLoader loader = {};
    loader.textures.setMaxDimension(2048);
    loader.setEmissiveScale(1.0f);
    loader.loadGLTF(ASSET_PATH("assets/gltf/main_sponza/NewSponza_Main_glTF_003.gltf"));
    loader.loadGLTF(ASSET_PATH("assets/gltf/pkg_a_curtains/NewSponza_Curtains_glTF.gltf"));
    //loader.loadGLTF(ASSET_PATH("assets/gltf/pkg_d_10k_candles/NewSponza_4_Combined_glTF.gltf"));
    //loader.loadGLTF(ASSET_PATH("assets/gltf/main_sponza/blendersponza/updatedsponza.gltf"));

    printPrincipledMaterials(loader);
    std::unique_ptr<GPUScene> gpuScene = loader.buildFlattened(envManager.getView(), 0.9f);

    points = gpuScene->hostPositions;
    mesh = gpuScene->hostTriangles;

    vector<float3> positions;
    vector<uint3> indices;

    positions.reserve(points.size());

    std::transform(points.begin(), points.end(), std::back_inserter(positions), 
        [](const float4& v) {
            return make_float3(v.x, v.y, v.z);
        }
    );

    for (Triangle& t : mesh) {
        indices.push_back(make_uint3(t.aInd, t.bInd, t.cInd));
    }

    CUdeviceptr out_d_gas_output_buffer;
    OptixTraversableHandle BLAShandle = buildOptixGAS(
        engineState.context,
        positions,
        indices,
        out_d_gas_output_buffer
    );

    std::vector<OptixInstance> instances;
    std::vector<float4> transformMatrices; // Switched to float4!

    OptixInstance inst = {};
    float transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f, 
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // 1. Push 3 float4 rows into your vector for the device shaders
    transformMatrices.push_back(make_float4(transform[0], transform[1], transform[2],  transform[3]));
    transformMatrices.push_back(make_float4(transform[4], transform[5], transform[6],  transform[7]));
    transformMatrices.push_back(make_float4(transform[8], transform[9], transform[10], transform[11]));

    // 2. OptixInstance still wants raw floats, so we keep the memcpy exactly as you had it
    memcpy(inst.transform, transform, sizeof(float) * 12);
    inst.instanceId = 0;
    inst.sbtOffset = 0;
    inst.visibilityMask = 255;
    inst.flags = OPTIX_INSTANCE_FLAG_NONE;
    inst.traversableHandle = BLAShandle;

    instances.push_back(inst);

    // --- Upload Instances ---
    CUdeviceptr d_instances;
    cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(OptixInstance) * instances.size());
    cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), sizeof(OptixInstance) * instances.size(), cudaMemcpyHostToDevice);

    // --- Upload Matrices ---
    float4* d_matrices;

    // transformMatrices.size() is now the number of float4s (3 per instance), 
    // so the memory allocation math becomes super clean!
    cudaMalloc(&d_matrices, transformMatrices.size() * sizeof(float4));
    cudaMemcpy(d_matrices, transformMatrices.data(), transformMatrices.size() * sizeof(float4), cudaMemcpyHostToDevice);
    
    CUdeviceptr out_d_ias_output_buffer;
    OptixTraversableHandle TLAShandle = buildOptixIAS(
        engineState.context,
        d_instances,
        instances.size(),
        out_d_ias_output_buffer
    );

    auto afterBVH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterBVH = afterBVH - afterRead;
    std::cout << "BVH construction took: " << elapsed_seconds_afterBVH.count() << " seconds" << std::endl << endl;

    Triangle* lights;

    // allocates and deallocates device pointer for lights (device light array)
    LightSamplerManager lightManager(lightDesc, lightsvec, points, lights, envManager.getView());
    
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));
    cudaMemset(out_colors, 0, w * h * sizeof(float4));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));
    cudaMemset(d_finalOutput, 0, w * h * sizeof(float4));

    float4* host_colors = new float4[w * h];

    ShadeContext sc = {};

    sc.lightNum = lightsvec.size();
    sc.lights = lights;
    sc.scene = scene;
    sc.vertices = verts;
    sc.materials = mats_d;
    sc.textures = texManager.getView();
    sc.lightSampler = lightManager.getSampler();
    sc.triNum = mesh.size();
    sc.transformationMatrices = d_matrices;

    CommonParams params = {};
    params.w = w;
    params.h = h;
    params.frame_index = 0;
    params.bvh_handle = TLAShandle; 
    params.accum_buffer = out_colors; 
    params.camera = camera;
    params.shadeContext = sc; // beep beep remove later
    gpuScene->shadeContext.transformationMatrices = d_matrices;
    params.shadeContext = gpuScene->shadeContext;
    
    params.shadeContext.lightSampler.printDebugState();
    params.max_depth = maxDepth;

    PipelineParams allParams = {};
    allParams.common = params;

    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);

    CUstream stream;
    cudaStreamCreate(&stream);

    if (integratorChoice == OPTIX_NORMAL) {
        launch_unidirectional(engineState, params, sampleCount);
    } else if (integratorChoice == OPTIX_RESTIR_PT) {
        launch_restir(engineState, params, sampleCount);
    } else {
        printf("Error: Integrator Unavaible in Optix Branch");
    }
    


    
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            image.setColor(i, j, host_colors[image.toIndex(i, j)]/sampleCount);
        }
    }

    // memory freeing
    cudaFree(reinterpret_cast<void*>(out_d_gas_output_buffer));
    cudaFree(reinterpret_cast<void*>(out_d_ias_output_buffer));
    cudaFree(reinterpret_cast<void*>(d_instances));
    cudaFree(reinterpret_cast<void*>(d_matrices));
    cudaFree(out_colors);
    cudaFree(d_finalOutput);
    cudaFree(verts);
    cudaFree(scene);
    cudaFree(mats_d);
    // textures are owned by texManager (RAII); no manual free needed

    cudaFree(temp.positions);
    cudaFree(temp.normals);
    cudaFree(temp.uvs);
    delete[] host_colors;

    std::string filename = std::string(ROOT_DIR) + "/renders/optix/" + config.name + "" + std::to_string(renderNumber) + ".bmp";
    image.saveImageBMP(filename);
    filename = "render.bmp";
    image.saveImageBMP(filename);
    image.saveImageCSV_MONO(0);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds_render = end - afterBVH;
    std::cout << "Render took: " << elapsed_seconds_render.count() << " seconds" << std::endl << endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Total Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total Elapsed time (ms): " << elapsed_ms.count() << " milliseconds" << std::endl;

    return 0;
}