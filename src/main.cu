
#include "deviceCode.cuh"
#include "fastIntegrators.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "volumeRendering.cuh"
#include "sceneContexts.cuh"
#include "softwareBVH.cuh"
#include "helpers.cuh"
#include <chrono>
#include <iostream>
#include <exception>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include "textureManager.cuh"
#include "configParser.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <iomanip>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/cuda/DeviceBuffer.h>

using namespace std;

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

int initRender(string configPath, int renderNumber, string animatedObjPath = "invalid")
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

    int eyePathDepth = config.bdptEyeDepth;
    int lightPathDepth = config.bdptLightDepth;
    
    int maxLeafSize = config.bvhLeafSize;
    float VCMMergeConstant = config.vcmMergeConst;
    float VCMInitialMergeRadiusMultiplier = config.vcmInitialMergeRadiusMultiplier;

    std::cout << VCMMergeConstant << " and " << VCMInitialMergeRadiusMultiplier << std::endl;

    Camera camera;
    if (config.pinholeCamera)
        camera = Camera::Pinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov);
    else
        camera = Camera::NotPinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov, 
            config.camApeture, config.camFocalDist);
        
    // temporary code for turntable
    
    //float angleRad = (-92.0f) * (h_PI / 180.0f);
    //float angleRad = ((renderNumber * -0.0f) + -122.0f) * (h_PI / 180.0f);
    //camera.cameraOrigin.x = 2.668f + 500.0f * std::sin(angleRad);
    //camera.cameraOrigin.z = -177.961f + 500.0f * std::cos(angleRad);
    //camera.yRot = config.camRot.y + angleRad;

    camera.preCompute();

    Image image = Image(w, h);
    image.postProcess = config.postProcess;

    if (integratorChoice == UNIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            maxDepth << ".\nIntegrating with Naive + NEE Unidirectional MIS." << 
            endl << endl;
    }
    if (integratorChoice == WAVEFRONT_UNIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            maxDepth << ".\nIntegrating with Naive + NEE Unidirectional MIS (Wavefront)." << 
            endl << endl;
    }
    else if (integratorChoice == BIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Bidirectional." << 
            endl << endl;
    }
    if (integratorChoice == NAIVE_UNIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            maxDepth << ".\nIntergating with Naive Unidirectional" << 
            endl << endl;
    }
    else if (integratorChoice == VCM)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Vertex Connection and Merging, with alpha parameter " <<
            VCMMergeConstant << " and initial merge radius multiplier " <<
            VCMInitialMergeRadiusMultiplier << 
            endl << endl;
    }
    else if (integratorChoice == SPPM)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Stochastic Progressive Photon Mapping, with alpha parameter " <<
            VCMMergeConstant << " and initial merge radius multiplier " <<
            VCMInitialMergeRadiusMultiplier << 
            endl << endl;

        // to turn the vcm integrator into an only merging integrator
        config.bdptConnection = false;
        config.bdptNaive = false;
        config.bdptNee = false;
        config.bdptLightTrace = false;
        config.bdptDoMis = false;
        config.vcmDoMerge = true;
        config.doSPPM = true;
    }
    else if (integratorChoice == VOLUME_SIMPLE)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            eyePathDepth << ".\nIntegrating with Volumetric Rendering." <<
            endl << endl;
    }

    updateConstants(config);

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));
    cudaMemset(out_colors, 0, w * h * sizeof(float4));

    float4* out_overlay;
    cudaMalloc(&out_overlay, w * h * sizeof(float4));
    cudaMemset(out_overlay, 0, w * h * sizeof(float4));

    Vertices vertices;
    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors; // unused now
    vector<float2> uvs;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<BVHnode> bvhvec;

    vector<float4> centroids;
    vector<float4> minboxes;
    vector<float4> maxboxes;

    vector<Material> mats;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading environment map
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    EnvironmentMapManager envManager(ASSET_PATH("assets/environment/lakeside_sunrise_2k.exr"));
    //EnvironmentMapManager envManager(ASSET_PATH("assets/environment/black.exr"));
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

    TextureView texView = texManager.getView();

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Creating Materials
    //---------------------------------------------------------------------------------------------------------------------------------------------------

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

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    vector<LightDescriptor> lightDesc;

    for (MeshConfig c : config.meshes)
    {
        readObjSimple(ASSET_PATH(c.path), points, normals, colors, uvs, mesh, lightsvec, lightDesc, f3(),
                c.emissionMultiplier * c.emissionColor, c.materialID);
    }

    if (animatedObjPath != "invalid" && config.name == "watersim")
    {
        readObjSimple(animatedObjPath, points, normals, colors, uvs, mesh, lightsvec, lightDesc, f3(),
                f3(), 10, f3(0.0f, -0.9f, 0.0f));
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

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Volumes
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vector<Volume> volumes;

    for (const VolConfig& c : config.volumes) {
        std::string volumePath = ASSET_PATH(c.path);

        float4 aabbMIN = f4(0.0f);
        float4 aabbMAX = f4(0.0f);
        void* d_density_gridBuffer = nullptr;
        void* d_temp_gridBuffer = nullptr;

        try {
            auto density_handle = nanovdb::io::readGrid(volumePath, "density");
            const nanovdb::NanoGrid<float>* density_hostGrid = density_handle.grid<float>();
            size_t gridSizeInBytes = density_handle.size();

            cudaMalloc(&d_density_gridBuffer, gridSizeInBytes);
            cudaMemcpy(d_density_gridBuffer, density_hostGrid, gridSizeInBytes, cudaMemcpyHostToDevice);

            const nanovdb::GridMetaData* currentGrid = density_handle.gridMetaData(0);
            auto bbox = currentGrid->worldBBox();
            auto min = bbox.min();
            auto max = bbox.max();
            
            aabbMIN = f4(min[0], min[1], min[2]);
            aabbMAX = f4(max[0], max[1], max[2]);
            
        } catch (const std::exception& e) {
            std::cout << "Could not read density grid from file: " << volumePath << "\nReason: " << e.what() << std::endl;
            continue;
        }
        
        try {
            auto temp_handle = nanovdb::io::readGrid(volumePath, "temperature");
            const nanovdb::NanoGrid<float>* temp_hostGrid = temp_handle.grid<float>();
            size_t gridSizeInBytes = temp_handle.size();

            cudaMalloc(&d_temp_gridBuffer, gridSizeInBytes);
            cudaMemcpy(d_temp_gridBuffer, temp_hostGrid, gridSizeInBytes, cudaMemcpyHostToDevice);
            
        } catch (const std::exception& e) {
            std::cout << "Could not read temperature grid from file: " << volumePath << "\nReason: " << e.what() << std::endl;
        }

        volumes.push_back(
            Volume(
                aabbMIN,
                aabbMAX,
                reinterpret_cast<nanovdb::NanoGrid<float>*>(d_density_gridBuffer),
                reinterpret_cast<nanovdb::NanoGrid<float>*>(d_temp_gridBuffer),
                c.densityScale,
                c.anisotropy,
                f4(c.albedo)
            )
        );

        cout << "Added volume at" << c.path <<" with " << ((d_density_gridBuffer) ? ("density") : ("")) 
             << ((d_temp_gridBuffer) ? (" and temperature") : "") << " grid(s), with" 
             << "\nanisotropy " << c.anisotropy
             << "\nalbedo (" << c.albedo.x << ", " << c.albedo.y << ", " << c.albedo.z << ")"
             << "\ndensity scale " << c.densityScale
             << "\n\n";
    }

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene construction took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;

    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH (on CPU)
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vertices.positions = points.data();
    vertices.normals   = normals.data();
    vertices.uvs    = uvs.data();
    
    vector<int> primType;
    vector<int> originalIndices;

    computeInfoForBVH(vertices, mesh, volumes, centroids, minboxes, maxboxes, primType, originalIndices);

    int volCount = 0;
    for(int type : primType) {
        if(type == TYPE_VOLUME) volCount++;
    }

    std::cout << "Volumes found in primType: " << volCount << std::endl;

    cout << "BVH data computed" << endl;

    vector<int> indvec(mesh.size() + volumes.size());
    for (int i = 0; i < indvec.size(); i++) indvec[i] = i;

    int failcount = 0;
    int backupCt = 0;
    int startNode = buildBVH(bvhvec, indvec, centroids, minboxes, maxboxes, 0, indvec.size(), maxLeafSize, failcount, backupCt);

    vector<int2> gpu_indirection(indvec.size());
    for (int i = 0; i < indvec.size(); i++) {
        int global_idx = indvec[i];
        gpu_indirection[i].x = primType[global_idx];      // type tag
        gpu_indirection[i].y = originalIndices[global_idx]; // the actual index in the triangle/volume array
    }

    BVHnode root = bvhvec[0];
    float3 sceneCenter = f3((root.aabbMAX + root.aabbMIN) * 0.5f);
    float sceneRadius = length(f3(root.aabbMAX) - sceneCenter) + 0.01f;
    float3 sceneMin = f3(root.aabbMIN);

    cout << "BVH built. Scene radius is " << sceneRadius << "." << endl;
    cout << "Largest leaf is size: " << failcount << "." << " Backup was called "<< backupCt << " times." << endl;
    //printBVH(bvhvec, indvec);

    printBVHSummary(bvhvec);

    for (int2 index : gpu_indirection) {
        if (index.x == TYPE_VOLUME) printf("volume found!!\n");
    }

    auto afterBVH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterBVH = afterBVH - afterRead;
    std::cout << "BVH construction took: " << elapsed_seconds_afterBVH.count() << " seconds" << std::endl << endl;
    
    BVHnode* BVH;
    int2* BVHindices;
    Volume* d_volumes;

    Triangle* lights;

    // allocates and deallocates device pointer for lights (device light array)
    LightSamplerManager lightManager(lightDesc, lightsvec, points, lights, envManager.getView());
    
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    //cudaMalloc(&lights, lightsvec.size() * sizeof(Triangle));
    cudaMalloc(&BVH, bvhvec.size() * sizeof(BVHnode));
    cudaMalloc(&BVHindices, indvec.size() * sizeof(int2));

    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    //cudaMemcpy(lights, lightsvec.data(), lightsvec.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(BVH, bvhvec.data(), bvhvec.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);
    cudaMemcpy(BVHindices, gpu_indirection.data(), gpu_indirection.size() * sizeof(int2), cudaMemcpyHostToDevice);

    if (volumes.size() > 0) {
        cudaMalloc(&d_volumes, volumes.size() * sizeof(Volume));
        cudaMemcpy(d_volumes, volumes.data(), volumes.size() * sizeof(Volume), cudaMemcpyHostToDevice);
    } else {
        d_volumes = nullptr;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Additional setup according to which integrator is used
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    
    if (integratorChoice == UNIDIRECTIONAL)
        launch_unidirectional(maxDepth, camera, mats_d, texView, BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);
    else if (integratorChoice == BIDIRECTIONAL)
    {
        int totalEyePathVertices = w * h * eyePathDepth;
        int totalLightPathVertices = w * h * lightPathDepth;

        PathVertices* eyePath_d;
        cudaMalloc(&eyePath_d, sizeof(PathVertices));

        PathVertices tempPaths;

        cudaMalloc(&tempPaths.materialID, sizeof(int) * totalEyePathVertices);
        cudaMalloc(&tempPaths.pt,         sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.n,          sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.wo,         sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.beta,       sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.d_vcm,     sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.d_vc,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.pdfFwd,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.misWeight,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.isDelta,    sizeof(bool) * totalEyePathVertices);
        cudaMalloc(&tempPaths.lightInd,   sizeof(int) * totalEyePathVertices);
        cudaMalloc(&tempPaths.uv,   sizeof(float2) * totalEyePathVertices);
        cudaMalloc(&tempPaths.backface,   sizeof(bool) * totalEyePathVertices);

        cudaMemset(tempPaths.materialID, 0, sizeof(int) * totalEyePathVertices);
        cudaMemset(tempPaths.pt,         0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.n,          0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.wo,         0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.beta,       0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.d_vcm,     0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.d_vc,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.pdfFwd,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.misWeight,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.isDelta,    0, sizeof(bool) * totalEyePathVertices);
        cudaMemset(tempPaths.lightInd,   0, sizeof(int) * totalEyePathVertices);
        cudaMemset(tempPaths.uv,   0, sizeof(float2) * totalEyePathVertices);
        cudaMemset(tempPaths.backface,   0, sizeof(bool) * totalEyePathVertices);

        cudaMemcpy(eyePath_d, &tempPaths, sizeof(PathVertices), cudaMemcpyHostToDevice);

        PathVertices* lightPath_d;
        cudaMalloc(&lightPath_d, sizeof(PathVertices));

        PathVertices tempPaths1;

        cudaMalloc(&tempPaths1.materialID, sizeof(int) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.pt,         sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.n,          sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.wo,         sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.beta,       sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.d_vcm,     sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.d_vc,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.pdfFwd,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.misWeight,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.isDelta,    sizeof(bool) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.lightInd,   sizeof(int) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.uv,   sizeof(float2) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.backface,   sizeof(bool) * totalLightPathVertices);

        cudaMemset(tempPaths1.materialID, 0, sizeof(int) * totalLightPathVertices);
        cudaMemset(tempPaths1.pt,         0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.n,          0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.wo,         0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.beta,       0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.d_vcm,     0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.d_vc,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.pdfFwd,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.misWeight,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.isDelta,    0, sizeof(bool) * totalLightPathVertices);
        cudaMemset(tempPaths1.lightInd,   0, sizeof(int) * totalLightPathVertices);
        cudaMemset(tempPaths1.uv,   0, sizeof(float2) * totalLightPathVertices);
        cudaMemset(tempPaths1.backface,   0, sizeof(bool) * totalLightPathVertices);

        cudaMemcpy(lightPath_d, &tempPaths1, sizeof(PathVertices), cudaMemcpyHostToDevice);

        launch_bidirectional(eyePathDepth, lightPathDepth, camera, eyePath_d, lightPath_d, mats_d, texView, BVH, 
            BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, w, h, 
            sceneCenter, sceneRadius, out_colors, out_overlay, config.postProcess);
        cudaFree(eyePath_d);
        cudaFree(lightPath_d);

        cudaFree(tempPaths.materialID);
        cudaFree(tempPaths.pt);
        cudaFree(tempPaths.n);
        cudaFree(tempPaths.wo);
        cudaFree(tempPaths.beta);
        cudaFree(tempPaths.d_vc);
        cudaFree(tempPaths.isDelta);
        cudaFree(tempPaths.lightInd);
        cudaFree(tempPaths.uv);
        cudaFree(tempPaths.d_vcm);
        cudaFree(tempPaths.backface);
        cudaFree(tempPaths.misWeight);
        cudaFree(tempPaths.pdfFwd);

        cudaFree(tempPaths1.materialID);
        cudaFree(tempPaths1.pt);
        cudaFree(tempPaths1.n);
        cudaFree(tempPaths1.wo);
        cudaFree(tempPaths1.beta);
        cudaFree(tempPaths1.d_vc);
        cudaFree(tempPaths1.isDelta);
        cudaFree(tempPaths1.lightInd);
        cudaFree(tempPaths1.uv);
        cudaFree(tempPaths1.d_vcm);
        cudaFree(tempPaths1.backface);
        cudaFree(tempPaths1.misWeight);
        cudaFree(tempPaths1.pdfFwd);
    }
    else if (integratorChoice == NAIVE_UNIDIRECTIONAL)
    {
        launch_naive_unidirectional(maxDepth, camera, mats_d, texView, BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);
    }
    else if (integratorChoice == SPPM)
    {
        int totalPhotons = w * h * lightPathDepth;

        Photons tempPhotons;
        cudaMalloc(&tempPhotons.pos_plus_vm, sizeof(float4) * totalPhotons);
        cudaMalloc(&tempPhotons.packedWi, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.packedNormal, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons.d_vcm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons.pos_plus_vm, 0, sizeof(float4) * totalPhotons);
        cudaMemset(tempPhotons.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.packedWi, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons.packedNormal, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons.d_vcm, 0, sizeof(float) * totalPhotons);

        Photons tempPhotons1;
        cudaMalloc(&tempPhotons1.pos_plus_vm, sizeof(float4) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedWi, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedNormal, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons1.d_vcm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons1.pos_plus_vm, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.packedWi, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons1.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.packedNormal, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons1.d_vcm, 0, sizeof(float) * totalPhotons);

        launch_SPPM(
            eyePathDepth, lightPathDepth, 
            camera, 
            &tempPhotons, &tempPhotons1, 
            mats_d, texView, 
            BVH, BVHindices, 
            verts, points.size(), 
            scene, mesh.size(), 
            lights, lightsvec.size(), sampleCount, 
            w, h, 
            sceneCenter, sceneRadius, sceneMin,
            out_colors, out_overlay,
            config.postProcess, VCMMergeConstant, VCMInitialMergeRadiusMultiplier
        );


        cudaFree(tempPhotons.pos_plus_vm);
        cudaFree(tempPhotons.beta_x);
        cudaFree(tempPhotons.beta_y);
        cudaFree(tempPhotons.beta_z);
        cudaFree(tempPhotons.packedWi);
        cudaFree(tempPhotons.packedNormal);
        cudaFree(tempPhotons.d_vcm);
        
        cudaFree(tempPhotons1.pos_plus_vm);
        cudaFree(tempPhotons1.beta_x);
        cudaFree(tempPhotons1.beta_y);
        cudaFree(tempPhotons1.beta_z);
        cudaFree(tempPhotons1.packedWi);
        cudaFree(tempPhotons1.packedNormal);
        cudaFree(tempPhotons1.d_vcm);
    }
    else if (integratorChoice == VCM)
    {
        int totalLightPathVertices = w * h * lightPathDepth;

        VCMPathVertices tempPaths;

        cudaMalloc(&tempPaths.pos_x, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.pos_y, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.pos_z, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_x, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_y, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_z, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedNormal, sizeof(uint32_t) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedWo, sizeof(uint32_t) * totalLightPathVertices);
        //cudaMalloc(&tempPaths.packedBeta, sizeof(uint32_t) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedInfo, sizeof(uint32_t) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedUV, sizeof(half2) * totalLightPathVertices);
        cudaMalloc(&tempPaths.d_vc, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.d_vcm, sizeof(float) * totalLightPathVertices);
        //cudaMalloc(&tempPaths.d_vm, sizeof(float) * totalLightPathVertices);

        cudaMemset(tempPaths.pos_x, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.pos_y, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.pos_z, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_x, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_y, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_z, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.packedNormal, 0, sizeof(uint32_t) * totalLightPathVertices);
        cudaMemset(tempPaths.packedWo, 0, sizeof(uint32_t) * totalLightPathVertices);
        //cudaMemset(tempPaths.packedBeta, 0, sizeof(uint32_t) * totalLightPathVertices);
        cudaMemset(tempPaths.packedInfo, 0, sizeof(uint32_t) * totalLightPathVertices);
        cudaMemset(tempPaths.packedUV, 0, sizeof(half2) * totalLightPathVertices);
        cudaMemset(tempPaths.d_vc, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.d_vcm, 0, sizeof(float) * totalLightPathVertices);
        //cudaMemset(tempPaths.d_vm, 0, sizeof(float) * totalLightPathVertices);

        //cudaMemcpy(lightPath_d, &tempPaths, sizeof(VCMPathVertices), cudaMemcpyHostToDevice);
        
        int totalPhotons = w * h * lightPathDepth;

        //Photons* photons_d;
        //cudaMalloc(&photons_d, sizeof(Photons));

        Photons tempPhotons;
        cudaMalloc(&tempPhotons.pos_plus_vm, sizeof(float4) * totalPhotons);
        //cudaMalloc(&tempPhotons.pos_y, sizeof(float) * totalPhotons);
        //cudaMalloc(&tempPhotons.pos_z, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.packedWi, sizeof(uint32_t) * totalPhotons);
        //cudaMalloc(&tempPhotons.packedPower, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.packedNormal, sizeof(uint32_t) * totalPhotons);
        //cudaMalloc(&tempPhotons.d_vc, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.d_vcm, sizeof(float) * totalPhotons);
        //cudaMalloc(&tempPhotons.d_vm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons.pos_plus_vm, 0, sizeof(float4) * totalPhotons);
        //cudaMemset(tempPhotons.pos_y, 0, sizeof(float) * totalPhotons);
        //cudaMemset(tempPhotons.pos_z, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.packedWi, 0, sizeof(uint32_t) * totalPhotons);
        //cudaMemset(tempPhotons.packedPower, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons.packedNormal, 0, sizeof(uint32_t) * totalPhotons);
        //cudaMemset(tempPhotons.d_vc, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.d_vcm, 0, sizeof(float) * totalPhotons);
        //cudaMemset(tempPhotons.d_vm, 0, sizeof(float) * totalPhotons);

        //cudaMemcpy(photons_d, &tempPhotons, sizeof(Photons), cudaMemcpyHostToDevice);

        //Photons* photons_sorted_d;
        //cudaMalloc(&photons_sorted_d, sizeof(Photons));

        Photons tempPhotons1;
        cudaMalloc(&tempPhotons1.pos_plus_vm, sizeof(float4) * totalPhotons);
        //cudaMalloc(&tempPhotons1.pos_y, sizeof(float) * totalPhotons);
        //cudaMalloc(&tempPhotons1.pos_z, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedWi, sizeof(uint32_t) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedNormal, sizeof(uint32_t) * totalPhotons);
        //cudaMalloc(&tempPhotons1.d_vc, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.d_vcm, sizeof(float) * totalPhotons);
        //cudaMalloc(&tempPhotons1.d_vm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons1.pos_plus_vm, 0, sizeof(float) * totalPhotons);
        //cudaMemset(tempPhotons1.pos_y, 0, sizeof(float) * totalPhotons);
        //cudaMemset(tempPhotons1.pos_z, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.packedWi, 0, sizeof(uint32_t) * totalPhotons);
        cudaMemset(tempPhotons1.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.packedNormal, 0, sizeof(uint32_t) * totalPhotons);
        //cudaMemset(tempPhotons1.d_vc, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.d_vcm, 0, sizeof(float) * totalPhotons);
        //cudaMemset(tempPhotons1.d_vm, 0, sizeof(float) * totalPhotons);

        //cudaMemcpy(photons_sorted_d, &tempPhotons1, sizeof(Photons), cudaMemcpyHostToDevice);
        

        // launch kernel
        launch_VCM(
            eyePathDepth, lightPathDepth, 
            camera, 
            &tempPaths, 
            &tempPhotons, &tempPhotons1, 
            mats_d, texView, 
            BVH, BVHindices, 
            verts, points.size(), 
            scene, mesh.size(), 
            lights, lightsvec.size(), sampleCount, 
            w, h, 
            sceneCenter, sceneRadius, sceneMin,
            out_colors, out_overlay,
            config.postProcess, VCMMergeConstant, VCMInitialMergeRadiusMultiplier
        );

        //cudaFree(lightPath_d);

        cudaFree(tempPaths.pos_x);
        cudaFree(tempPaths.pos_y);
        cudaFree(tempPaths.pos_z);
        cudaFree(tempPaths.packedNormal);
        cudaFree(tempPaths.packedWo);
        cudaFree(tempPaths.beta_x);
        cudaFree(tempPaths.beta_y);
        cudaFree(tempPaths.beta_z);
        cudaFree(tempPaths.packedInfo);
        cudaFree(tempPaths.packedUV);
        cudaFree(tempPaths.d_vc);
        cudaFree(tempPaths.d_vcm);
        //cudaFree(tempPaths.d_vm);

        //cudaFree(photons_d);
        
        cudaFree(tempPhotons.pos_plus_vm);
        //cudaFree(tempPhotons.pos_y);
        //cudaFree(tempPhotons.pos_z);
        cudaFree(tempPhotons.beta_x);
        cudaFree(tempPhotons.beta_y);
        cudaFree(tempPhotons.beta_z);
        cudaFree(tempPhotons.packedWi);
        cudaFree(tempPhotons.packedNormal);
        cudaFree(tempPhotons.d_vcm);
        //cudaFree(tempPhotons.d_vc);
        //cudaFree(tempPhotons.d_vm);

        //cudaFree(photons_sorted_d);
        
        cudaFree(tempPhotons1.pos_plus_vm);
        //cudaFree(tempPhotons1.pos_y);
        //cudaFree(tempPhotons1.pos_z);
        cudaFree(tempPhotons1.beta_x);
        cudaFree(tempPhotons1.beta_y);
        cudaFree(tempPhotons1.beta_z);
        cudaFree(tempPhotons1.packedWi);
        cudaFree(tempPhotons1.packedNormal);
        cudaFree(tempPhotons1.d_vcm);
        //cudaFree(tempPhotons1.d_vc);
        //cudaFree(tempPhotons1.d_vm);
    } else if (integratorChoice == WAVEFRONT_UNIDIRECTIONAL) {
        SceneContext sc;
        sc.BVH = BVH;
        sc.BVHindices = BVHindices;
        sc.lightNum = lightsvec.size();
        sc.lights = lights;
        sc.scene = scene;
        sc.triNum = mesh.size();
        sc.vertices = verts;
        sc.vertNum = points.size();
        sc.materials = mats_d;
        sc.textures = texView;
        sc.lightSampler = lightManager.getSampler();

        lightManager.getSampler().printDebugState();

        launch_wavefrontUnidirectional(
            camera,
            sc,
            sampleCount,
            maxDepth,
            w, h,
            sceneCenter, sceneRadius, sceneMin,
            out_colors, out_overlay,
            config.postProcess
        );
    } else if (integratorChoice == VOLUME_SIMPLE) {
        SceneContext sc;
        sc.BVH = BVH;
        sc.BVHindices = BVHindices;
        sc.lightNum = lightsvec.size();
        sc.lights = lights;
        sc.scene = scene;
        sc.triNum = mesh.size();
        sc.vertices = verts;
        sc.vertNum = points.size();
        sc.materials = mats_d;
        sc.textures = texView;
        sc.volumes = d_volumes;
        sc.lightSampler = lightManager.getSampler();
        
        lightManager.getSampler().printDebugState();
        launch_simple_volume(
            camera,
            sc,
            sampleCount,
            maxDepth,
            w, h,
            sceneCenter, sceneRadius, sceneMin,
            out_colors, out_overlay,
            config.postProcess
        );
    }
    

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Launch GPU Code - goes to functions in deviceCode.cu
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    float4* host_colors = new float4[w * h];
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    float4* host_overlay = new float4[w * h];
    cudaMemcpy(host_overlay, out_overlay, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w * h; i++)
    {
        host_colors[i] /= (float)sampleCount;

        if (isnan(host_colors[i].x) || isnan(host_colors[i].y) || isnan(host_colors[i].z)) {
            host_colors[i] = f4(1.0f, 0.0f, 1.0f); // Bright Pink for NaN
        }
        if (isinf(host_colors[i].x) || isinf(host_colors[i].y) || isinf(host_colors[i].z)) {
            host_colors[i] = f4(0.0f, 1.0f, 0.0f); // Bright Green for Inf
        }
    }

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            
            if (host_colors[image.toIndex(i, j)].x < 0 || host_colors[image.toIndex(i, j)].y < 0 || host_colors[image.toIndex(i, j)].z < 0)
                cout << i << ", " << j << " Negative color written: <" << host_colors[image.toIndex(i, j)].x << ", " << host_colors[image.toIndex(i, j)].y << ", " 
                    << host_colors[image.toIndex(i, j)].z << ">"<< endl;

            if (host_overlay[image.toIndex(i, j)].x == 0 && host_overlay[image.toIndex(i, j)].y == 0 && host_overlay[image.toIndex(i, j)].z == 0)
                image.setColor(i, j, host_colors[image.toIndex(i, j)]);
            else
                image.setColor(i, j, host_overlay[image.toIndex(i, j)]);
        }
    }
    
    // memory freeing
    cudaFree(out_colors);
    cudaFree(out_overlay);
    cudaFree(verts);
    cudaFree(scene);
    //cudaFree(lights); freed in the light manager
    cudaFree(BVH);
    cudaFree(BVHindices);
    cudaFree(mats_d);
    // textures are owned by texManager (RAII); no manual free needed

    for (const Volume& vol : volumes) {
        if (vol.density_pointer) cudaFree(vol.density_pointer);
        if (vol.temperature_pointer) cudaFree(vol.temperature_pointer);
    }
    
    if (d_volumes) cudaFree(d_volumes);

    cudaFree(temp.positions);
    cudaFree(temp.normals);
    cudaFree(temp.uvs);
    delete[] host_colors;
    delete[] host_overlay;

    std::string filename = std::string(ROOT_DIR) + "/renders/glove/" + config.name + "" + std::to_string(renderNumber) + ".bmp";
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

void printNvdbMetadata(const std::string& filePath) {
    try {
        // According to IO.h: "A negative value of n means read all grids in the file."
        auto handle = nanovdb::io::readGrid(filePath, -1);

        if (!handle || handle.gridCount() == 0) {
            std::cout << "Could not read grids from file: " << filePath << std::endl;
            return;
        }

        std::cout << "\nThe file \"" << filePath << "\" contains " << handle.gridCount() << " grids:\n";
        
        // Expanded header to include World Size
        std::cout << std::left 
                  << std::setw(4)  << "#"
                  << std::setw(15) << "Name"
                  << std::setw(12) << "Type"
                  << std::setw(10) << "Class"
                  << std::setw(12) << "Size (MB)"
                  << std::setw(12) << "# Voxels"
                  << std::setw(20) << "Index Res"
                  << "World Size (W x H x D)" << std::endl;

        std::cout << std::string(115, '-') << std::endl;

        for (uint32_t i = 0; i < handle.gridCount(); ++i) {
            const nanovdb::GridMetaData* currentGrid = handle.gridMetaData(i);
            if (!currentGrid) continue;

            char typeBuf[32];
            char classBuf[32];
            nanovdb::toStr(typeBuf, currentGrid->gridType());
            nanovdb::toStr(classBuf, currentGrid->gridClass());

            // --- Index Space Data ---
            auto indexBBox = currentGrid->indexBBox();
            auto indexDim = indexBBox.dim();
            double gridMB = static_cast<double>(currentGrid->gridSize()) / (1024.0 * 1024.0);
            
            // --- World Space Data ---
            auto worldBBox = currentGrid->worldBBox();
            auto worldMin = worldBBox.min();
            auto worldMax = worldBBox.max();
            auto worldDim = worldMax - worldMin; // Physical size in scene units
            
            // Calculate the exact center for camera look_at
            auto worldCenter = worldMin + (worldDim * 0.5);

            // Print Main Table Row
            std::cout << std::left
                      << std::setw(4)  << (i + 1)
                      << std::setw(15) << currentGrid->shortGridName()
                      << std::setw(12) << typeBuf
                      << std::setw(10) << (currentGrid->isUnknown() ? "?" : classBuf)
                      << std::fixed << std::setprecision(2) << std::setw(12) << gridMB
                      << std::setw(12) << currentGrid->activeVoxelCount()
                      << indexDim[0] << "x" << indexDim[1] << "x" << std::setw(9) << indexDim[2]
                      << std::fixed << std::setprecision(3) 
                      << worldDim[0] << " x " << worldDim[1] << " x " << worldDim[2]
                      << std::endl;

            // Print Scene Setup Helpers underneath each grid
            std::cout << "    -> [Scene Setup] Center: (" 
                      << worldCenter[0] << ", " << worldCenter[1] << ", " << worldCenter[2] 
                      << ") | Max Dimension: " << std::max({worldDim[0], worldDim[1], worldDim[2]}) 
                      << "\n" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main ()
{
    printNvdbMetadata(ASSET_PATH("assets/vdb/nvdb/industrial/smoke_0000.nvdb"));
    string configName = ASSET_PATH("configs/config.rendertron");

    initRender(configName, 0); 
    //return;
    for (int i = 0; i < 250; ++i) {
        char buf[128];
        snprintf(buf, sizeof(buf), "assets/scenedata/watersim/tenbillionobj/wateranim%04d.obj", i);
        //initRender(configName, i, buf); 
        initRender(configName, i);
    }

    cout << "All Renders Finished" << endl;

    RNGManager::cleanup();
    return 0;
}