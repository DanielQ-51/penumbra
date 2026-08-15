#pragma once

#include "objects.cuh"
#include "util.cuh"
#include "helpers.cuh"
#include "volumeRendering.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
#include "restirPTenhanced_host.cuh"
#include "unidirectional_host.cuh"
#include "profiling.cuh"
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
    // Post-processing (exposure/tonemap/gamma) now happens on the GPU via
    // cleanAndFormatImageNoOverlay + postProcessOnly below rather than on
    // the host. Capture the decision once and disable Image::postProcess so
    // saveImageBMP() doesn't re-apply it on top of the already-processed data.
    const bool gpuPostProcess = config.postProcess;
    image.postProcess = false;

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

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading environment map
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    std::string envMapPath = config.envMapPath.empty() ? "assets/environment/black.exr" : config.envMapPath;
    EnvironmentMapManager envManager(ASSET_PATH(envMapPath));
    envManager.setRotation(config.envMapRotation);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Scene (materials, textures, OBJ meshes, glTF assets) via the shared SceneLoader
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    SceneLoader loader;   // declared BEFORE gpuScene: locals destroy in reverse order, and
                           // gpuScene->shadeContext borrows device handles loader.textures owns.
    loader.loadFromConfig(config);

    std::unique_ptr<GPUScene> gpuScene = loader.buildFlattened(envManager.getView(), 0.5f);

    if (gpuScene->hostTriangles.empty()) {
        cout << "Error: No triangles loaded." << endl;
        return 1;
    }
    cout << "scene data read. There are " << gpuScene->hostTriangles.size() << " Triangles and "
         << gpuScene->shadeContext.lightNum << " +1 lights" << endl;

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene read took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    vector<float3> positions;
    vector<uint3> indices;

    positions.reserve(gpuScene->hostPositions.size());

    std::transform(gpuScene->hostPositions.begin(), gpuScene->hostPositions.end(), std::back_inserter(positions),
        [](const float4& v) {
            return make_float3(v.x, v.y, v.z);
        }
    );

    for (Triangle& t : gpuScene->hostTriangles) {
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

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));
    cudaMemset(out_colors, 0, w * h * sizeof(float4));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));
    cudaMemset(d_finalOutput, 0, w * h * sizeof(float4));

    float4* host_colors = new float4[w * h];

    CommonParams params = {};
    params.w = w;
    params.h = h;
    params.frame_index = 0;
    params.bvh_handle = TLAShandle;
    params.accum_buffer = out_colors;
    params.camera = camera;
    params.shadeContext = gpuScene->shadeContext;

    // buildFlattened() leaves transformationMatrices null (its output is already
    // world-space, so no per-instance transform is needed at shading time). But
    // traceClosest() (optixUtils.cuh) always reads a REAL OptiX instance ID via
    // optixHitObjectGetInstanceId() -- never the 0xFFFFFFFF "no transform" sentinel,
    // since that's only meaningful for bare-GAS/software tracing -- and passes it
    // into getBarycentrics(), which unconditionally dereferences transformationMatrices
    // for any non-sentinel id. With a real IAS (even our single identity instance),
    // that id is 0, so the pointer must be valid. d_matrices holds exactly that
    // identity transform, uploaded alongside the GAS/IAS build above.
    params.shadeContext.transformationMatrices = d_matrices;

    params.shadeContext.lightSampler.printDebugState();
    params.max_depth = maxDepth;

    PipelineParams allParams = {};
    allParams.common = params;

    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);

    CUstream stream;
    cudaStreamCreate(&stream);

#if PROFILE_TECHNIQUES
    // Separate profiling option: sweep CommonParams::debugVersion across the
    // PROFILE_ARMS table (profiling.cuh) for whichever integrator is compiled
    // in, and report paired per-variant timing stats. `frames` reuses the
    // config's sampleCount (= frameCount for ReSTIR, samples/arm for
    // unidirectional). `warmup` frames render but are excluded from the stats.
    launchProfile(engineState, params, integratorChoice,
                  /*warmup*/ 8, /*frames*/ (uint32_t)sampleCount);
#else
    if (integratorChoice == OPTIX_NORMAL) {
        launch_unidirectional(engineState, params, sampleCount);
    } else if (integratorChoice == OPTIX_RESTIR_PT) {
        launch_restir(engineState, params, sampleCount, config);
    } else {
        printf("Error: Integrator Unavaible in Optix Branch");
    }
#endif

    // Normalize (divide by sampleCount) on the GPU. No overlay buffer in
    // this pipeline, so the NoOverlay variant. Divisor is
    // currentSampleCount + 1, so pass sampleCount - 1 to divide by sampleCount.
    cleanAndFormatImageNoOverlay<<<gridSize, blockSize, 0, stream>>>(
        out_colors, d_finalOutput, w, h, sampleCount - 1
    );

    if (gpuPostProcess) {
        postProcessOnly<<<gridSize, blockSize, 0, stream>>>(
            d_finalOutput, d_finalOutput, w, h, config.exposure, image.use_fitted_aces
        );
    }

    cudaMemcpyAsync(host_colors, d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            image.setColor(i, j, host_colors[image.toIndex(i, j)]);
        }
    }

    // memory freeing
    cudaFree(reinterpret_cast<void*>(out_d_gas_output_buffer));
    cudaFree(reinterpret_cast<void*>(out_d_ias_output_buffer));
    cudaFree(reinterpret_cast<void*>(d_instances));
    cudaFree(reinterpret_cast<void*>(d_matrices));
    cudaFree(out_colors);
    cudaFree(d_finalOutput);
    // scene vertices/triangles/materials/textures are owned by gpuScene/loader (RAII)
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