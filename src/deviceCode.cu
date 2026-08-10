#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "deviceCode.cuh"
#include "configParser.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include <cub/cub.cuh>

__device__ __constant__ bool SAMPLE_ENVIRONMENT = false;

__device__ __constant__ bool BDPT_LIGHTTRACE;
__device__ __constant__ bool BDPT_NEE;
__device__ __constant__ bool BDPT_NAIVE;
__device__ __constant__ bool BDPT_CONNECTION;

__device__ __constant__ bool VCM_DOMERGE;

__device__ __constant__ bool DO_SPPM;

__device__ __constant__ bool BDPT_DRAWPATH;
__device__ __constant__ bool BDPT_DOMIS;
__device__ __constant__ bool BDPT_PAINTWEIGHT;

__device__ __constant__ float eta_vcm;
__device__ __constant__ float sceneRadius;
__device__ __constant__ float3 sceneCenter;
__device__ __constant__ float3 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

__host__ void updateConstants(const RenderConfig& config)
{
    cudaMemcpyToSymbol(BDPT_LIGHTTRACE, &config.bdptLightTrace, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_NAIVE, &config.bdptNaive, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_NEE, &config.bdptNee, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_CONNECTION, &config.bdptConnection, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_DRAWPATH, &config.bdptDrawPath, sizeof(bool));
    cudaMemcpyToSymbol(VCM_DOMERGE, &config.vcmDoMerge, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_DOMIS, &config.bdptDoMis, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_PAINTWEIGHT, &config.bdptPaintWeight, sizeof(bool));
    cudaMemcpyToSymbol(SAMPLE_ENVIRONMENT, &config.sampleEnvironment, sizeof(bool));
    cudaMemcpyToSymbol(DO_SPPM, &config.doSPPM, sizeof(bool));
    return;
}

__device__ void neePDF(const Vertices* __restrict__ vertices, const Triangle* __restrict__ scene, int lightNum, int lightTriInd, const Intersection& intersect,
    float& light_pdf, float etaI, float etaT, const Intersection* newIntersect)
{
    Triangle l = scene[lightTriInd];
    float3 apos = f3(vertices->positions[l.aInd]);
    float3 bpos = f3(vertices->positions[l.bInd]);
    float3 cpos = f3(vertices->positions[l.cInd]);
    float3 p = newIntersect->point;
    float3 n = newIntersect->normal;

    float3 surfaceToLight = p-intersect.point;
    float3 wi = normalize(surfaceToLight);

    float distanceSQR = lengthSquared(surfaceToLight);
    float3 lightNormal = f3(vertices->normals[l.naInd]);

    float cosThetaLight = dot(lightNormal, -wi);
    float cosThetaSurface = fabsf(dot(n, wi));

    float area = 0.5f * length(cross(bpos - apos, cpos - apos));

    light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
}

__device__ void nextEventEstimation(RNGState& localState, const Material* __restrict__ materials, TextureView textures, const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices, const Vertices* __restrict__ vertices,
    const Triangle* __restrict__ scene, const Triangle* __restrict__ lights, int lightNum, const Intersection& intersect, const float3& wo,
    float& light_pdf, float3& contribution, float3& surfaceToLight_local, float etaI, float etaT)
{
    contribution = f3(0.0f,0.0f,0.0f);
    Triangle l;
    float3 apos;
    float3 bpos;
    float3 cpos;
    float u;
    float v;
    float3 p;
    float3 n;

    if (lightNum == 0)
    {
        light_pdf = -1.0f;
        return;
    }
    int index = min(static_cast<int>(rand(&localState) * lightNum), lightNum - 1);
    l = lights[index];
    apos = f3(vertices->positions[l.aInd]);
    bpos = f3(vertices->positions[l.bInd]);
    cpos = f3(vertices->positions[l.cInd]);
    u = sqrtf(rand(&localState));
    v = rand(&localState);
    p = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos; // point on light

    float3 a_n = f3(__ldg(&vertices->normals[l.naInd]));
    float3 b_n = f3(__ldg(&vertices->normals[l.nbInd]));
    float3 c_n = f3(__ldg(&vertices->normals[l.ncInd]));

    float3 lightNormal = normalize((1.0f - u) * a_n + u * (1.0f - v) * b_n + u * v * c_n);
    n = intersect.normal;

    float3 surfaceToLight = p-intersect.point;
    float3 wi = normalize(surfaceToLight);

    float cosThetaLight = dot(lightNormal, -wi);
    float cosThetaSurface = dot(n, wi);

    if (cosThetaLight < 0.0f || cosThetaSurface < 0.0f)
    {
        contribution = f3();
        return;
    }

    Ray r = Ray(intersect.point + wi * EPSILON, wi);

    float t; 
    float3 dummy;
    triangleIntersect(vertices, &l, r, dummy, t);

    Intersection sceneIntersect = Intersection();
    float3 throughputScale = f3(1.0f);
    BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, length(surfaceToLight)*(1.0f - EPSILON), l.triInd);
    // following if statement tests for scene intersection (direct light) AND
    // whether the original light intersect was valid
    //if (!sceneIntersect.valid && t != -1.0f) // direct LOS from intersection to light
    if (lengthSquared(throughputScale) > 0.0f)
    {
        float distanceSQR = lengthSquared(surfaceToLight);

        //float G = cosThetaLight * cosThetaSurface/distanceSQR;
        float area = 0.5f * length(cross(bpos - apos, cpos - apos));

        light_pdf = distanceSQR / (cosThetaLight * lightNum * area);
        float3 Le = f3(l.emission);
        float3 f_val;
        float3 wi_local;
        toLocal(wi, intersect.normal, wi_local);
        surfaceToLight_local = wi_local;

        // wo is the incoming direction (passed to this function)
        // wi_local is the computed outgoing direction to the light
        f_eval(materials, intersect.materialID, textures, wo, wi_local, etaI, etaT, f_val, intersect.uv);

        contribution = f_val * Le * cosThetaSurface / light_pdf;
        contribution *= throughputScale;
    }
}

__global__ void __launch_bounds__(256, 2) Li_naive_unidirectional (
    RNGState* rngStates,
    Camera camera,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    bool useMIS,
    int w, int h,
    float4* __restrict__ colors,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = camera.generateCameraRay(localState, x, y);
    float3 Li = f3();
    float3 beta = f3(1.0f);
    for (int depth = 0; depth < maxDepth; depth++)
    {
        Intersection intersect;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            Li += beta * sampleSky(r.direction);
            break;
        }

        float3 toSurface_local;
        float3 toNext_local;
        toLocal(r.direction, intersect.normal, toSurface_local);

        float3 f_val;
        float pdf;
        sample_f_eval(localState, materials, intersect.materialID, textures, toSurface_local, 1.0f, 1.0f, intersect.backface, toNext_local, f_val, pdf, intersect.uv);

        if (pdf <= 0.0f || lengthSquared(f_val) < EPSILON) break;

        Li += intersect.backface ? f3(0.0f) : intersect.emission * beta;

        beta *= f_val * fabsf(toNext_local.z) / pdf;

        float3 toNext_world;
        toWorld(toNext_local, intersect.normal, toNext_world);

        r.origin = intersect.point + ((toNext_local.z > 0.0f) ? (intersect.normal * RAY_EPSILON) : (-intersect.normal * RAY_EPSILON));
        r.direction = toNext_world;
    }
    colors[pixelIdx] += f4(Li);
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__host__ void launch_naive_unidirectional(
    int maxDepth,
    Camera camera,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    bool useMIS,
    int w, int h,
    float4* __restrict__ colors
)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((w+15)/16, (h+15)/16);

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    // Allocate device buffers for the image formatting kernel
    float4* d_finalOutput;
    float4* d_overlay;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));
    cudaMalloc(&d_overlay, w * h * sizeof(float4));
    cudaMemset(d_overlay, 0, w * h * sizeof(float4)); // Zero out the dummy overlay

    cudaDeviceSynchronize();

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    int saveIntervalSamples = 1500;
    Image image = Image(w, h);
    std::vector<float4> h_finalOutput(w * h);

    std::cout << "Running Kernels" << std::endl;

    // Start total timer for average ms calculation
    auto renderStartTime = std::chrono::steady_clock::now();

    for (int currSample = 0; currSample < numSample; currSample++)
    {
        Li_naive_unidirectional<<<gridSize, blockSize>>>(d_rngStates, camera, materials, textures, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum,
            lights, lightNum, numSample, useMIS, w, h, colors, currSample);

        if ((currSample % saveIntervalSamples == 0 || currSample == numSample-1) && DO_PROGRESSIVERENDER)
        {
            // Launch the formatting kernel to handle averaging, NaNs, and Infs on the GPU
            cleanAndFormatImage<<<gridSize, blockSize>>>(
                colors, d_overlay, d_finalOutput, w, h, currSample
            );

            // Copy the finalized buffer back to the host
            cudaMemcpy(h_finalOutput.data(), d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            // Clean OpenMP loop simply maps the formatted colors to the image
            #pragma omp parallel for
            for (int i = 0; i < w * h; i++)
            {
                int x = i % w;
                int y = i / w;
                image.setColor(x, y, h_finalOutput[i]);
            }

            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);

            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            // Reset the dummy overlay just like in the wavefront version
            cudaMemset(d_overlay, 0, w * h * sizeof(float4));
        }
    }

    printf("\n");
    cudaDeviceSynchronize();

    // Free the newly added device buffers
    cudaFree(d_rngStates);
    cudaFree(d_finalOutput);
    cudaFree(d_overlay);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}

__global__ void __launch_bounds__(256, 2) Li_unidirectional (
    RNGState* rngStates,
    Camera camera,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    bool useMIS,
    int w, int h,
    float4* __restrict__ colors,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = Ray();
    float3 beta = f3(1.0f, 1.0f, 1.0f);
    float3 Li = f3();
    float3 wi_local = f3();
    float3 wo_local = f3();
    float3 wi_world = f3();
    Intersection previousintersectREAL = Intersection(); // last REAL intersection (true hit)

    // for nested dielectrics
    int mediumStack[16];
    int stackTop = 0;
    mediumStack[stackTop++] = 0; //index of AIR (IOR 1.0f, Priority 99)

    r = camera.generateCameraRay(localState, x, y);

    float pdf = EPSILON;
    float etaI = EPSILON;
    float etaT = EPSILON;

    bool hitFirstnonSpecular = false;

    for (int depth = 0; depth < maxDepth; depth++)
    {

        Intersection intersect = Intersection();
        intersect.valid = false;
        //BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f, false, previousintersectANY.triIDX);
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect, 999999.0f);

        if (!intersect.valid)
        {
            Li += beta * sampleSky(r.direction);
            break;
        }

        int materialID = intersect.materialID;

        //float3 old_wi_local = wi_local;

        toLocal(r.direction, intersect.normal, wi_local); // assign new wi_local
        wi_world = normalize(r.direction);

        bool isSpecular = false;
        if (materials[materialID].isSpecular)
        {
            isSpecular = true;
        }

        bool trueHit = true;

        int minPrior = materials[mediumStack[0]].priority;
        int minPriorID = mediumStack[0];
        for (int i = 1; i < stackTop; i++)
        {
            if (materials[mediumStack[i]].priority < minPrior)
            {
                minPrior = materials[mediumStack[i]].priority;
                minPriorID = mediumStack[i];
            }
        }

        float3 absorption_coeff = f3(materials[minPriorID].absorption);
        float distanceTraveled = intersect.dist;

        if (distanceTraveled > EPSILON)
        {
            float3 attenuation = f3(
                exp(-absorption_coeff.x * distanceTraveled),
                exp(-absorption_coeff.y * distanceTraveled),
                exp(-absorption_coeff.z * distanceTraveled)
            );
            beta *= attenuation;
        }

        if (materials[materialID].boundary) // if this material is a boundary between media
        {

            if (materials[materialID].priority <= minPrior) // new material has lower or equal priority to the minimum priority
            {
                // true hit, continue with shading
                if (materials[materialID].type == MAT_SMOOTHDIELECTRIC)
                {
                    etaI = materials[minPriorID].ior; //the dominating current medium
                    if (!intersect.backface) //entering surface
                    {
                        etaT = materials[materialID].ior; //is later added iff we actually refract
                    }
                    else // exiting dominant surface
                    {
                        if (stackTop == 1)
                        {
                            if (materials[mediumStack[0]].priority != 99)
                                printf("error: single medium in stack that isnt air\n");
                            etaT = 1.0f;
                        }
                        else
                        {
                            // the new material is a true hit, so it must appear somewhere in the stack, since we are exiting it
                            minPrior = 99;
                            int secondLowest = mediumStack[0];
                            for (int i = 0; i < stackTop; i++)
                            {   //printf("%d\n", materials[mediumStack[i]].priority);
                                if (materials[mediumStack[i]].priority)
                                {
                                    // checks for the dominant medium in the absence of the one we are exiting, defaults to air
                                    if (minPrior > materials[mediumStack[i]].priority && mediumStack[i] != materialID)
                                    {
                                        secondLowest = mediumStack[i];
                                        minPrior = materials[mediumStack[i]].priority;
                                    }
                                }
                            }
                            // this is the dominant medium if we pretend like the one we just exited isnt there
                            // we KNOW that we are exiting the DOMINANT medium since it is a true hit
                            etaT = materials[secondLowest].ior;
                        }

                    }
                }
            }
            else // false hit, ignore intersection
            {
                trueHit = false;
                if (!intersect.backface) //entering non dominant surface
                {
                    mediumStack[stackTop++] = intersect.materialID; //push new medium

                }
                else //exiting non dominant surface
                {
                    removeMaterialFromStack(mediumStack, &stackTop, materialID);
                }
            }
        }
        else // if this isnt a boundary event. We still need to know the current medium ior for thin walled events
            etaI = materials[minPriorID].ior;

        if (trueHit)
        {
            if (lengthSquared(intersect.emission) > EPSILON)
            {
                if (depth == 0 || !hitFirstnonSpecular)
                {
                    Li += intersect.backface ? f3(0.0f) : beta * intersect.emission;
                }
                else if (useMIS && !isSpecular) // found light using BSDF sampling, weigh against NEE
                {
                    float light_pdf = EPSILON;

                    neePDF(vertices, scene, lightNum, intersect.triIDX, previousintersectREAL, light_pdf, etaI, etaT, &intersect);
                    if (light_pdf > EPSILON)
                    {
                        float bsdfWeight = pdf * pdf / (light_pdf * light_pdf
                        + pdf * pdf);
                        Li += beta * intersect.emission * bsdfWeight;
                    }
                }
            }

            if (useMIS && lengthSquared(intersect.emission) < EPSILON && !isSpecular) // using nee mainly, weigh against BSDF pdf
            {
                float3 nee;
                float light_pdf = EPSILON;
                // we get wo_local, the direction from surface to sampled light, to evaluate the bsdf pdf,
                // and store it in wo_local
                nextEventEstimation(localState, materials, textures, BVH, BVHindices, vertices, scene, lights, lightNum, intersect,
                    wi_local, light_pdf, nee, wo_local, etaI, etaT);

                if (light_pdf > EPSILON)
                {
                    // to calculate the bsdf pdf
                    pdf_eval(materials, materialID, textures, wi_local, wo_local, etaI, etaT, pdf, intersect.uv); // stores the bsdf pdf val in pdf
                    float neeWeight = light_pdf * light_pdf / (pdf * pdf + light_pdf * light_pdf);

                    Li += beta * nee * neeWeight;
                }

            }
            float3 f_val = f3();
            sample_f_eval(localState, materials, materialID, textures, wi_local, etaI, etaT, intersect.backface, wo_local, f_val, pdf, intersect.uv);

            float3 wo_world= f3();
            toWorld(wo_local, intersect.normal, wo_world);

            pdf = fmaxf(pdf, 0.01);


            if (trueHit)
            {
                if (wo_local.z < 0.0f) // refracted
                {
                    if (!intersect.backface) // entering new surface (is dominant)
                    {
                        mediumStack[stackTop++] = intersect.materialID;
                    }
                    else // exiting dominant surface (materialID garunteed to be dominant)
                    {
                        removeMaterialFromStack(mediumStack, &stackTop, materialID);
                    }
                }
            }


            beta *= (f_val * fabsf(wo_local.z) / pdf);
            //beta = fminf3(beta, f3(10.0f));

            if (wo_local.z > 0) // reflected
                r.origin = intersect.point + intersect.normal * RAY_EPSILON;
            else //refracted
                r.origin = intersect.point - intersect.normal * RAY_EPSILON;

            r.direction = normalize(wo_world);
            previousintersectREAL = intersect;


        }
        else
        {
            //float3 wo_world = normalize(r.direction);
            toLocal(r.direction, intersect.normal, wo_local);
            //r.origin = intersect.point - intersect.normal * EPSILON * 1.0F; // needs to go through, so offset on other side of normal
            r.origin = intersect.point + r.direction * RAY_EPSILON; // needs to go through, so offset on other side of normal
        }

        {
            float lum = luminance(beta);
            float p = clamp(lum, 0.05f, 1.0f);

            if (rand(&localState) > p)   // survive with probability p
                break;

            beta /= p;  // compensate for the survival probability
        }

        if (!isSpecular)
        {
            hitFirstnonSpecular = true;
        }

    }
    colors[pixelIdx] += f4(Li);
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__host__ void launch_unidirectional(
    int maxDepth,
    Camera camera,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    bool useMIS,
    int w, int h,
    float4* __restrict__ colors
)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((w+15)/16, (h+15)/16);

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    cudaDeviceSynchronize();

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    int saveIntervalSamples = 10000; // Aligned with wavefront logic
    Image image = Image(w, h);

    std::cout << "Running Kernels Unidirectional" << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();

    for (int currSample = 0; currSample < numSample; currSample++)
    {
        Li_unidirectional<<<gridSize, blockSize>>>(d_rngStates, camera, materials, textures, BVH, BVHindices, maxDepth, vertices, vertNum, scene, triNum,
            lights, lightNum, numSample, useMIS, w, h, colors, currSample);

        cudaDeviceSynchronize();

        // Save and print progress based on sample count
        if (currSample % saveIntervalSamples == 0)
        {
            std::vector<float4> h_colors(w * h);
            cudaMemcpy(h_colors.data(), colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for // Added openmp pragma here like in your wavefront post-process if you want speed
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (isnan(h_colors[i].x) || isnan(h_colors[i].y) || isnan(h_colors[i].z)) {
                        h_colors[i] = f4(1.0f, 0.0f, 1.0f); // Bright Pink for NaN
                    }
                    if (isinf(h_colors[i].x) || isinf(h_colors[i].y) || isinf(h_colors[i].z)) {
                        h_colors[i] = f4(0.0f, 1.0f, 0.0f); // Bright Green for Inf
                    }
                    if (h_colors[image.toIndex(i, j)].x < 0 || h_colors[image.toIndex(i, j)].y < 0 || h_colors[image.toIndex(i, j)].z < 0)
                        std::cout << "\n" << i << ", " << j << " Negative color written: <" << h_colors[image.toIndex(i, j)].x << ", " << h_colors[image.toIndex(i, j)].y << ", "
                        << h_colors[image.toIndex(i, j)].z << ">" << std::endl;

                    image.setColor(i, j, h_colors[image.toIndex(i, j)] / (float)(currSample + 1));
                }
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);

            // Calculate timing and print inline progress
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);
        }
    }

    printf("\n"); // Move to a new line when the render loop finishes completely
    cudaDeviceSynchronize();
    cudaFree(d_rngStates);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}

__device__ bool BDPTnextEventEstimation(
    RNGState& localState,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    const Triangle* __restrict__ scene,
    const Triangle* __restrict__ lights,
    int lightNum,
    int materialID,
    float3 shadingPos,
    const float3 toShadingPos_local,
    const float3 shadingPos_normal,
    const float2 uv,
    float& light_pdf,
    float3& contribution,
    float3& shadingPos_to_lightPos,
    int& lightInd,
    float& cosLight,
    float& pdf_emit,
    float etaI, float etaT
)
{
    int totalLightNum = SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum; // +1 for the sky
    lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(rand(&localState) * (totalLightNum)), totalLightNum - 1) - 1) :
        (min(static_cast<int>(rand(&localState) * (lightNum)), lightNum - 1));

    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));

    if (lightInd != -1)
    {
        Triangle l = lights[lightInd];
        float3 apos = f3(vertices->positions[l.aInd]);
        float3 bpos = f3(vertices->positions[l.bInd]);
        float3 cpos = f3(vertices->positions[l.cInd]);

        float3 anorm = f3(vertices->normals[l.naInd]);
        float3 bnorm = f3(vertices->normals[l.nbInd]);
        float3 cnorm = f3(vertices->normals[l.ncInd]);

        float u = sqrtf(rand(&localState));
        float v = rand(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        float3 p = w0 * apos + w1 * bpos + w2 * cpos;
        float3 lightNormal = normalize(w0 * anorm + w1 * bnorm + w2 * cnorm);

        float3 surfaceToLight = p-shadingPos;
        shadingPos_to_lightPos = surfaceToLight;
        float3 surfaceToLight_unit = normalize(surfaceToLight);

        float distanceSQR = lengthSquared(surfaceToLight);
        distanceSQR = fmaxf(distanceSQR, RAY_EPSILON);

        Ray r = Ray(shadingPos + shadingPos_normal * RAY_EPSILON, surfaceToLight_unit);

        float distance = sqrtf(distanceSQR);

        float3 throughputScale = f3(1.0f);
        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, distance - EPSILON, l.triInd);

        if (lengthSquared(throughputScale) > 0.0f)
        {
            float cosThetaLight = dot(lightNormal, -surfaceToLight_unit);
            cosLight = cosThetaLight;

            if (cosThetaLight < EPSILON) {
                contribution = f3(0.0f);
                return true;
            }
            float cosThetaSurface = fabsf(dot(shadingPos_normal, surfaceToLight_unit));

            float G = cosThetaLight * cosThetaSurface / distanceSQR;

            float maxG = 15.0f;
            if (G > maxG) {
                G = maxG;
            }

            float area = 0.5f * length(cross(bpos - apos, cpos - apos));

            float pdf_choosePoint_area = 1.0f / area;
            float pdf_chooseLightPoint_area = pdf_choosePoint_area * pdf_chooseLight;
            light_pdf = pdf_chooseLightPoint_area; // nee pdf
            pdf_emit = cosThetaLight / PI; // emit pdf, directional

            //float pdf_chooseLightPoint_solidAngle = pdf_chooseLightPoint_area * distanceSQR / cosThetaLight;
            //light_pdf = pdf_chooseLightPoint_solidAngle;

            float3 towardLight_local;
            toLocal(surfaceToLight_unit, shadingPos_normal, towardLight_local);

            float3 f_val;
            f_eval(materials, materialID, textures, toShadingPos_local, towardLight_local, etaI, etaT, f_val, uv);

            contribution = throughputScale * f_val * f3(l.emission) * G / light_pdf; // unweighted
            return false;
        }
    }
    else
    {
        float3 dir_to_sky = sampleSphere(localState, 1.0f);
        float pdf_dir_solidAngle = 1.0f / (4.0f * PI);

        float3 surfaceToLight_unit = dir_to_sky;
        shadingPos_to_lightPos = surfaceToLight_unit;
        Ray r = Ray(shadingPos + shadingPos_normal * RAY_EPSILON, surfaceToLight_unit);

        float3 throughputScale = f3(1.0f);

        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, SKY_RADIUS, -1);

        if (lengthSquared(throughputScale) > 0.0f)
        {
            //printf("Sky visible from <%f,%f,%f> with normal <%f,%f,%f> and direction <%f,%f,%f>\n",
            //    r.origin.x, r.origin.y, r.origin.z, shadingPos_normal.x, shadingPos_normal.y, shadingPos_normal.z,
            //    surfaceToLight_unit.x, surfaceToLight_unit.y, surfaceToLight_unit.z);
            float cosThetaSurface = fabsf(dot(shadingPos_normal, surfaceToLight_unit));
            cosLight = -69.420; // not used in calculations for sky

            float3 Le = sampleSky(surfaceToLight_unit);

            float pdf_chooseLightPoint_solidAngle = pdf_chooseLight * pdf_dir_solidAngle;
            pdf_emit = pdf_chooseLightPoint_solidAngle; // emit pdf, directional
            //use the same disk sampling used in the light vertex generation
            //float pdf_pos_area = 1.0f / (PI * sceneRadius * sceneRadius);
            light_pdf = pdf_chooseLightPoint_solidAngle;

            float3 towardLight_local;
            toLocal(surfaceToLight_unit, shadingPos_normal, towardLight_local);

            float3 f_val;
            f_eval(materials, materialID, textures, toShadingPos_local, towardLight_local, etaI, etaT, f_val, uv);

            // Unweighted Contribution
            // For infinite lights: Le * f_r * cosTheta / pdf_solidAngle
            contribution = throughputScale * f_val * Le * cosThetaSurface / pdf_chooseLightPoint_solidAngle;
            return false;
        }
    }
    return true;
}

// populates the section of eyePath buffer specified by the arguments. Also asigns the length of the path
__device__ void generateEyePath(
    RNGState& localState,
    Camera camera,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int w, int h, int x, int y,
    PathVertices* eyePath,
    int& pathLength
)
{
    pathLength = 1;
    Ray r = camera.generateCameraRay(localState, x, y);
    int firstIdx = pathBufferIdx(w, h, x, y, 0);

    float3 currThroughput = f3(1.0f);

    eyePath->pt[firstIdx] = f4(r.origin);
    eyePath->n[firstIdx] = f4(camera.getForwardVector());
    eyePath->beta[firstIdx] = f4(currThroughput);
    eyePath->isDelta[firstIdx] = true; // it is delta meaning the probability of a light path hitting it randomly is zero

    eyePath->lightInd[firstIdx] = -51;

    eyePath->misWeight[firstIdx] = 0.0f;
    eyePath->uv[firstIdx] = f2(0.0f);

    float aspect = (float)w / (float)h;
    float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;

    float cosAtCamera = fabsf(dot(camera.getForwardVector(), r.direction)); // r.direction should be normalized already

    float prevPDF_solidAngle; // outgoing pdf from scattering functions
    float prev_cosine; // the previous cosine between the normal and the outgoing ray

    prevPDF_solidAngle = 1.0f / (imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);
    prev_cosine = cosAtCamera;

    // these shouldnt be needed for the first vertex
    eyePath->d_vc[firstIdx] = 0.0f;
    eyePath->d_vcm[firstIdx] = 0.0f;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;

    float pdf_onebeforePrevRev_SA = -1.0f;
    bool prevWasDelta = false;

    for (int depth = 1; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = pathBufferIdx(w, h, x, y, depth-1);
        Intersection intersect = Intersection();
        intersect.valid = false;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid) // treat this as an endpoint
        {
            /* NOT IMPLEMENTED CORRECTLY FOR VCM STYLE YET*/
            return;
        }
        float3 geomN = intersect.normal;
        //bool doubleSided = materials[intersect.materialID].type == MAT_SMOOTHDIELECTRIC || materials[intersect.materialID].type == MAT_LEAF; // or check flag
        eyePath->uv[currIdx] = intersect.uv;
        eyePath->beta[currIdx] = f4(currThroughput);

        eyePath->materialID[currIdx] = intersect.materialID;
        eyePath->pt[currIdx] = f4(intersect.point);

        bool currDelta = materials[eyePath->materialID[currIdx]].isSpecular;
        eyePath->isDelta[currIdx] = currDelta;

        if (intersect.backface)
        {
            eyePath->backface[currIdx] = true;
        }
        else
            eyePath->backface[currIdx] = false;

        eyePath->n[currIdx] = f4(geomN);

        eyePath->wo[currIdx] = f4(normalize(-r.direction));

        float3 wo_world = intersect.point - f3(eyePath->pt[prevIdx]); // the incoming direction, pointing at the new surface
        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, intersect.normal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;
        eyePath->pdfFwd[currIdx] = pdfFwd_area;

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //okay apparently wi is the outgoing direction now wtf

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, intersect.materialID, textures, wo_local, etaI, etaT, intersect.backface, wi_local,
            f_val, pdfFwd_solidAngle, intersect.uv, TRANSPORTMODE_RADIANCE);

        //radiance is conserved through dielectric boundaries, so we dont need to apply a correction like we did for the light path

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        float3 nextToCurrent_local = -wi_local;
        float3 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;

        pdf_eval(materials, intersect.materialID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT, pdfRev_solidAngle, intersect.uv);

        if (currDelta)
            pdfRev_solidAngle = pdfFwd_solidAngle;
        // pdfRev is not stored

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Wrapping it up, self explanatory
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        if (depth == 1) {
            float pdf_connect = 1.0f;
            float pdf_trace = 1.0f;
            float numLightSample = (float) w * (float) h;
            numLightSample = 1.0f;

            float vcm = (pdf_connect * numLightSample) / (pdf_trace * pdfFwd_area);
            float vc = 0.0f;

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta)
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 0.0f;
            float vc = (G / pdfFwd_area) * (pdf_onebeforePrevRev_SA * prev_d_vc);

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 1.0f / pdfFwd_area;
            float vc = (G / pdfFwd_area) * (prev_d_vcm + pdf_onebeforePrevRev_SA * prev_d_vc);

            eyePath->d_vcm[currIdx] = vcm;
            eyePath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        if (lengthSquared(f3(scene[intersect.triIDX].emission)) > EPSILON)
            eyePath->lightInd[currIdx] = scene[intersect.triIDX].lightInd;
        else
            eyePath->lightInd[currIdx] = -51; // -1 is reserved for the sun

        if (pdfFwd_solidAngle < EPSILON)
            break;

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        bool transmitting = dot(wi_world, f3(eyePath->n[currIdx])) < 0.0f;

        if (transmitting)
            r.origin = intersect.point - f3(eyePath->n[currIdx]) * RAY_EPSILON;
        else
            r.origin = intersect.point + f3(eyePath->n[currIdx]) * RAY_EPSILON;

        r.origin = intersect.point + (transmitting ? (-f3(eyePath->n[currIdx]) * RAY_EPSILON) : (f3(eyePath->n[currIdx]) * RAY_EPSILON));
        r.direction = normalize(wi_world);

        pathLength++;

        // update the previous state
        prev_cosine = fabsf(wi_local.z);
        prevWasDelta = currDelta;
        prevPDF_solidAngle = pdfFwd_solidAngle;
    }
}

__device__ void generateFirstLightPathVertex(
    RNGState& localState,
    int maxDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int w, int h, int x, int y,
    PathVertices* lightPath,
    float& pdf_solidAngle,
    float& cosine,
    float3& out_wi
)
{
    int firstIdx = pathBufferIdx(w, h, x, y, 0);

    // the convention is that light index -1 is the environment, and that lightNum doesnt include the environment
    int lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(rand(&localState) * (lightNum + 1)), lightNum) - 1) :
        (min(static_cast<int>(rand(&localState) * (lightNum)), lightNum - 1));

    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));
    lightPath->uv[firstIdx] = f2(0.0f);
    lightPath->backface[firstIdx] = false;

    if (lightInd != -1)
    {
        Triangle l = lights[lightInd];
        float3 apos = f3(vertices->positions[l.aInd]);
        float3 bpos = f3(vertices->positions[l.bInd]);
        float3 cpos = f3(vertices->positions[l.cInd]);

        float3 anorm = f3(vertices->normals[l.naInd]);
        float3 bnorm = f3(vertices->normals[l.nbInd]);
        float3 cnorm = f3(vertices->normals[l.ncInd]);

        float u = sqrtf(rand(&localState));
        float v = rand(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        lightPath->materialID[firstIdx] = l.materialID;

        lightPath->pt[firstIdx] = f4(w0 * apos + w1 * bpos + w2 * cpos);
        lightPath->n[firstIdx] = f4(normalize(w0 * anorm + w1 * bnorm + w2 * cnorm));

        float3 dummy;
        float firstPDF_solidAngle;
        float3 outOfLight_world;

        float3 outOfLight_local;
        cosine_sample_f(localState, dummy, outOfLight_local, dummy, firstPDF_solidAngle);
        toWorld(outOfLight_local, f3(lightPath->n[firstIdx]) ,outOfLight_world);

        lightPath->wo[firstIdx] = f4(); // degenerate vector, does not exist
        out_wi = outOfLight_world; // Pass back locally

        float area = 0.5f * length(cross(bpos - apos, cpos - apos));
        float pdf_samplePoint = 1.0f / area;
        float pdfFwd_val = pdf_chooseLight * pdf_samplePoint;
        lightPath->pdfFwd[firstIdx] = pdfFwd_val;
        // pdfRev is 0.0f, no store needed

        float3 Le = f3(l.emission);
        lightPath->beta[firstIdx] = f4((Le * PI) / pdfFwd_val);
        //lightPath->beta[firstIdx] = f3(1.0f) / pdfFwd_val;

        lightPath->lightInd[firstIdx] = lightInd;
        lightPath->isDelta[firstIdx] = false;

        lightPath->misWeight[firstIdx] = 0.0f;

        pdf_solidAngle = pdfFwd_val * firstPDF_solidAngle;// spatial times directional pdf forms the complete solid angle pdf
        cosine = fabsf(outOfLight_local.z);
    }
    else
    {
        lightPath->materialID[firstIdx] = -1;
        float3 dir_in = -sampleSphere(localState, 1.0f);
        float pdf_dir = 1.0f / (4.0f * PI);

        // some dark magic idk
        float3 tangent;
        float3 bitangent;

        if (fabsf(dir_in.x) > 0.9f)
        {
            float3 worldY = f3(0.0f, 1.0f, 0.0f);
            tangent = normalize(cross(worldY, dir_in));
        }
        else
        {
            float3 worldX = f3(1.0f, 0.0f, 0.0f);
            tangent = normalize(cross(worldX, dir_in));
        }

        bitangent = normalize(cross(dir_in, tangent));

        float r1 = rand(&localState);
        float r2 = rand(&localState);

        float disk_r = sceneRadius * sqrtf(r1);
        float disk_phi = 2.0f * PI * r2;

        float u_offset = disk_r * cosf(disk_phi);
        float v_offset = disk_r * sinf(disk_phi);

        float3 p_disk = sceneCenter + (tangent * u_offset) + (bitangent * v_offset);
        float3 p_world = p_disk - (dir_in * SKY_RADIUS);
        float pdf_pos = 1.0f / (PI * sceneRadius * sceneRadius);

        lightPath->pt[firstIdx] = f4(p_world);
        lightPath->n[firstIdx] = f4(normalize(dir_in));

        lightPath->wo[firstIdx] = f4(); // degenerate vector, does not exist
        out_wi = dir_in;

        lightPath->lightInd[firstIdx] = -1; // the sky
        lightPath->isDelta[firstIdx] = false; // the sky

        float pdfFwd_full = pdf_chooseLight * pdf_pos * pdf_dir;
        float pdfFwd_val = pdf_chooseLight * pdf_pos;
        lightPath->pdfFwd[firstIdx] = pdfFwd_val;

        float3 Le = sampleSky(-dir_in);
        lightPath->beta[firstIdx] = f4(Le / pdfFwd_full);
        //lightPath->beta[firstIdx] = f3(1.0f) / pdfFwd_val;

        lightPath->misWeight[firstIdx] = 0.0f;

        // unused
        pdf_solidAngle = pdf_dir;
        cosine = 1.0f;
    }
}

__device__ void generateLightPath(
    RNGState& localState,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int w, int h, int x, int y,
    PathVertices* lightPath,
    int& pathLength
)
{
    pathLength = 1;
    int firstIdx = pathBufferIdx(w, h, x, y, 0);
    Ray r;
    float prevPDF_solidAngle; // outgoing pdf from scattering functions
    float prev_cosine; // the previous cosine between the normal and the outgoing ray
    float3 start_wi;

    float3 currThroughput = f3(1.0f);

    generateFirstLightPathVertex(localState, maxDepth, vertices, vertNum, scene, triNum, lights, lightNum, w, h, x, y, lightPath, prevPDF_solidAngle, prev_cosine, start_wi);

    currThroughput = f3(lightPath->beta[firstIdx]);

    r.origin = f3(lightPath->pt[firstIdx]) + f3(lightPath->n[firstIdx]) * RAY_EPSILON;
    r.direction = start_wi;

    prev_cosine = fabsf(dot(normalize(start_wi), f3(lightPath->n[firstIdx])));
    prevPDF_solidAngle = prev_cosine / PI; // emission pdf

    // these shouldnt be needed for the first vertex
    lightPath->d_vc[firstIdx] = 0.0f;
    lightPath->d_vcm[firstIdx] = 0.0f;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;

    float pdf_onebeforePrevRev_SA;
    bool prevWasDelta = false;

    for (int depth = 1; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = pathBufferIdx(w, h, x, y, depth-1);

        Intersection intersect = Intersection();
        intersect.valid = false;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            return;
        }

        lightPath->uv[currIdx] = intersect.uv;
        lightPath->beta[currIdx] = f4(currThroughput);
        float3 geomN = intersect.normal;

        lightPath->materialID[currIdx] = intersect.materialID;
        lightPath->pt[currIdx] = f4(intersect.point);

        bool currDelta = materials[lightPath->materialID[currIdx]].isSpecular;
        lightPath->isDelta[currIdx] = currDelta;

        if (intersect.backface)
        {
            lightPath->backface[currIdx] = true;
        }
        else
            lightPath->backface[currIdx] = false;

        lightPath->n[currIdx] = f4(geomN);

        lightPath->wo[currIdx] = f4(normalize(-r.direction));

        float3 wo_world = intersect.point - f3(lightPath->pt[prevIdx]); // the incoming direction, pointing at the new surface
        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, intersect.normal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area;

        pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;
        lightPath->pdfFwd[currIdx] = pdfFwd_area;


        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //okay apparently wi is the outgoing direction now wtf

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, intersect.materialID, textures, wo_local, etaI, etaT, intersect.backface,
            wi_local, f_val, pdfFwd_solidAngle, intersect.uv, TRANSPORTMODE_IMPORTANCE);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 nextToCurrent_local = -wi_local;
        float3 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;

        pdf_eval(materials, intersect.materialID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT, pdfRev_solidAngle, intersect.uv);

        if (currDelta)
            pdfRev_solidAngle = pdfFwd_solidAngle;
        // pdfRev is not stored

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Wrapping it up, self explanatory
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);
        // wi is not stored

        // we don't store the light index for a light path

        if (pdfFwd_solidAngle < EPSILON)
            break;

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        if (depth == 1) {

            /*the spatial probability of picking the starting light. This is equal to both
            the probability of connecting to the light vertex via nee, and also equal to the
            probability of starting a light path at the light vertex. This value is stored
            in the forward pdf of the light vertex
            */
            float pdf_sampleLight = lightPath->pdfFwd[firstIdx];

            // the pdf of connecting to the previous light via NEE
            float pdf_connect = pdf_sampleLight;

            // the pdf of starting a light path at the previous light
            float pdf_trace = pdf_sampleLight;

            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            // pdfFwdArea is the emission pdf of the light, converted to area density at the current vertex
            float vcm = (pdf_connect) / (pdf_trace * pdfFwd_area);
            float vc = G / (pdf_trace * pdfFwd_area);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta)
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex
            float vcm = 0.0f;
            float vc = (G / pdfFwd_area) * (pdf_onebeforePrevRev_SA * prev_d_vc);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            float vcm = 1.0f / pdfFwd_area;
            float vc = (G / pdfFwd_area) * (prev_d_vcm + pdf_onebeforePrevRev_SA * prev_d_vc);

            lightPath->d_vcm[currIdx] = vcm;
            lightPath->d_vc[currIdx] = vc;

            prev_d_vcm = vcm;
            prev_d_vc = vc;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }


        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool transmitting = dot(wi_world, f3(lightPath->n[currIdx])) < 0.0f;

        if (transmitting)
            r.origin = intersect.point - f3(lightPath->n[currIdx]) * RAY_EPSILON;
        else
            r.origin = intersect.point + f3(lightPath->n[currIdx]) * RAY_EPSILON;
        r.direction = wi_world;

        pathLength++;
        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevWasDelta = currDelta;
    }
}

// performs the randomwalk from a sampled light, and takes care of the vertex connection sttage where light is made to directly hit the camera lense.
__global__ void __launch_bounds__(256, 2) lightPathTracing (
    RNGState* rngStates,
    Camera camera,
    PathVertices* eyePath,
    PathVertices* lightPath,
    int* lightPathLengths,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int lightDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    int w, int h,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    int lightPathLength;
    generateLightPath(localState, materials, textures, BVH, BVHindices, lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, w, h, x, y, lightPath, lightPathLength);

    lightPathLengths[pixelIdx] = lightPathLength;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Perform special case of the connection: what if the light ray just connected straight to the camera?
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    for (int s = 1; (s <= lightPathLength) && (BDPT_LIGHTTRACE); s++)
    {
        int lightPathIDX = pathBufferIdx(w, h, x, y, s - 1);

        float2 pixelPos;
        if (!camera.worldToRaster(f3(lightPath->pt[lightPathIDX]), pixelPos))
            continue;

        int px = (int)pixelPos.x;
        int py = (int)pixelPos.y;
        int newPixelIndex = py * w + px;

        if (lightPath->isDelta[lightPathIDX])
            continue;

        float etaI = 1.0f;
        float etaT = 1.0f;

        float3 lightToCamera = camera.cameraOrigin - f3(lightPath->pt[lightPathIDX]);
        float3 lightToCamera_unit = normalize(lightToCamera);

        Ray r = Ray(f3(lightPath->pt[lightPathIDX]) + f3(lightPath->n[lightPathIDX]) * RAY_EPSILON, lightToCamera_unit);
        float3 throughputScale;

        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, length(lightToCamera) - RAY_EPSILON, -1);

        if (lengthSquared(throughputScale) > EPSILON)
        {
            int prevIdx = pathBufferIdx(w, h, x, y, s - 2);
            float cosAtLight = dot(f3(lightPath->n[lightPathIDX]), lightToCamera_unit);
            float cosAtCamera = fabsf(dot(camera.getForwardVector(), -lightToCamera_unit));

            if (cosAtLight <= EPSILON) continue;

            float3 lightNormal = f3(lightPath->n[lightPathIDX]);
            float3 currToPrev_world = f3(lightPath->wo[lightPathIDX]);
            float3 currToPrev_local;
            toLocal(currToPrev_world, lightNormal, currToPrev_local);

            float3 lightToCamera_local;
            toLocal(lightToCamera_unit, lightNormal, lightToCamera_local);

            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // Unweighted contribution calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------

            float3 light_f;
            if (s == 1)
            {
                light_f = f3(1.0f/PI); // since we initialized beta with a pi factor
            }
            else
            {
                f_eval(materials, lightPath->materialID[lightPathIDX], textures, -currToPrev_local, lightToCamera_local, etaI, etaT, light_f, lightPath->uv[lightPathIDX]);
            }

            float aspect = (float)w / (float)h;
            float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;

            float We = 1.0f / (imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera * cosAtCamera);

            float distanceSQR = fmaxf(lengthSquared(lightToCamera), RAY_EPSILON);
            float G = (cosAtLight * cosAtCamera) / distanceSQR;

            float3 contribution = f3(lightPath->beta[lightPathIDX]) * light_f * G * throughputScale * We; // unweighted

            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // MIS Weight Calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------
            float misWeight;

            if (s == 1)
            {
                float pdf_sample_y0 = lightPath->pdfFwd[lightPathIDX];
                float pdf_traceFromCamera = cosAtLight / (distanceSQR * imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);

                float wLight = pdf_traceFromCamera / pdf_sample_y0;

                misWeight = 1.0f / (1.0f + wLight);
            }
            else
            {
                // we dont consider randomwalks onto the eye lens
                float wEye = 0.0f;

                // the chance to begin a eye path at the eye vertex
                float pdf_trace = 1.0f;

                // the chance to connect a light vertex to the eye vertex
                float pdf_connect = 1.0f;

                float traceRatio = pdf_trace / pdf_connect;

                // the camera emission pdf of generating the current vertex
                float pdf_currRev_area = cosAtLight / (distanceSQR * imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);

                float numLightSample = (float) w * (float) h;
                numLightSample = 1.0f;

                float pdf_oneBeforePrevRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, -lightToCamera_local, currToPrev_local, etaI, etaT,
                        pdf_oneBeforePrevRev_SA, lightPath->uv[lightPathIDX]);

                float wLight = traceRatio * (pdf_currRev_area / numLightSample) *
                    (lightPath->d_vcm[lightPathIDX] + pdf_oneBeforePrevRev_SA * lightPath->d_vc[lightPathIDX]);

                misWeight = 1.0f / (1.0f + wLight + wEye);
            }

            float3 weightedContribution = contribution * misWeight;

            if (BDPT_PAINTWEIGHT)
                weightedContribution = f3(misWeight);
            if (!BDPT_DOMIS)
                weightedContribution = contribution;

            atomicAdd(&colors[newPixelIndex].x, weightedContribution.x);
            atomicAdd(&colors[newPixelIndex].y, weightedContribution.y);
            atomicAdd(&colors[newPixelIndex].z, weightedContribution.z);
        }
    }
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

/*
 This function is never called with t=1. That is reserved for the first kernel to deal with
 Returns the unweighted contribution and the mis weight

 The accumulated misWeight stored at a given path vertex vi does not know about the location of vi+1,
 so therefore it cannot hold the reverse/fwd pdf ratio at vi, since the reverse pdf at that point requires
 and incident direction corresponding to where vi+1 is (that is not known at path generation time).

 Therefore, we must always complete the partial mis weight by applying a recursive ratio corresponding
 to the reverse/fwd pdf of vi, in addition to any other terms representing other strategies.
 */
__device__ bool connectPath(
    RNGState& localState,
    int t, int s,
    int x, int y, int w, int h,
    Camera camera,
    int maxEyeDepth,
    int maxLightDepth,
    const Material* __restrict__ materials,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    const Triangle* __restrict__ scene,
    const Triangle* __restrict__ lights,
    int lightNum,
    TextureView textures,
    int eyePathLength,
    int lightPathLength,
    PathVertices* eyePath,
    PathVertices* lightPath,
    float3& contribution,
    float& misWeight)
{
    int eyePathIDX = pathBufferIdx(w, h, x, y, t - 1);
    int eyePathPREVIDX = pathBufferIdx(w, h, x, y, t - 2);
    int lightPathIDX;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Delta Case
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    if (s > 0)
    {
        lightPathIDX = pathBufferIdx(w, h, x, y, s - 1);
        if (eyePath->isDelta[eyePathIDX] || lightPath->isDelta[lightPathIDX])
            return true;
    }
    else
    {
        if (eyePath->isDelta[eyePathIDX])
            return true;
    }


    float etaI = 1.0f; // placeholders
    float etaT = 1.0f;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = k > 1, t = 1: Connect light directly to camera. This is handled in the lightpathtracing kernel in the first pass
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = 1, t = k > 1: NEE
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    if (s == 1 && t > 1 && BDPT_NEE)
    {
        //float eye_misWeight = eyePath->misWeight[eyePathIDX];
        float3 nee_contribution_unweighted; // assigned in nee
        float pdf_connect; // assigned in nee. in area measure for area light, and in SA for environment
        float3 eyeToLight; // assigned in nee
        int lightInd; // assigned in nee
        float cosLight; // assigned in nee
        float pdf_emit_SA; // the probability that the light was sampled to emit, decoupled from the nee probability

        float3 prevToCurr_local;
        float3 prevTocurr = -f3(eyePath->wo[eyePathIDX]);
        float3 prevTocurrUnit = normalize(prevTocurr);
        // shading function expects toShadingPos_local to face towards the surface, wo faces away
        toLocal(prevTocurr, f3(eyePath->n[eyePathIDX]), prevToCurr_local);

        // sets eyeToLight, lightPDF_area, lightInd, cosLight, neecontributionunweighted
        bool occluded = BDPTnextEventEstimation(localState, materials, textures, BVH, BVHindices, vertices, scene, lights, lightNum, eyePath->materialID[eyePathIDX],
            f3(eyePath->pt[eyePathIDX]), prevToCurr_local, f3(eyePath->n[eyePathIDX]), eyePath->uv[eyePathIDX], pdf_connect, nee_contribution_unweighted, eyeToLight,
            lightInd, cosLight, pdf_emit_SA, etaI, etaT);
        if (occluded)
        {
            return true;
        }
        float3 eyeToLight_unit = normalize(eyeToLight);
        float3 eyeToLight_local;
        toLocal(eyeToLight_unit, f3(eyePath->n[eyePathIDX]), eyeToLight_local);
        if (lightInd != -1)
        {
            float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);

            float wLight;
            float wEye;

            float pdf_eyeToLight_solidAngle;

            pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, prevToCurr_local, eyeToLight_local, etaI, etaT, pdf_eyeToLight_solidAngle, eyePath->uv[eyePathIDX]);
            float pdf_bsdf_area = pdf_eyeToLight_solidAngle * fabsf(cosLight) / distanceSQR;

            float bsdfRatio = pdf_bsdf_area / pdf_connect;
            wLight = bsdfRatio;

            // the probability to start a ligth path at the light, in this implementation its equal to nee probability
            float pdf_trace = pdf_connect;

            float traceRatio = pdf_trace / pdf_connect;

            // the probability of a light path emitting to the current vertex, converted to area density at the current vertex
            float pdf_currRev_area = pdf_emit_SA * fabsf(eyeToLight_local.z) / distanceSQR;

            float pdf_oneBeforePrevRev_SA;
            pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, -eyeToLight_local, -prevToCurr_local, etaI, etaT, pdf_oneBeforePrevRev_SA, eyePath->uv[eyePathIDX]);

            wEye = traceRatio * pdf_currRev_area * (eyePath->d_vcm[eyePathIDX] + pdf_oneBeforePrevRev_SA * eyePath->d_vc[eyePathIDX]);

            misWeight = 1.0f / (1.0f + wLight + wEye);

            contribution = nee_contribution_unweighted * f3(eyePath->beta[eyePathIDX]);
            //printf("wLight: %f, wEye %f\n", wLight, eyePath->d_vc[eyePathIDX]);
        }
        else // uh im not doing this rn
        {
            // environment lights currently unimplemented
        }


        return true;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // s = 0, t = k > 1: eye randomwalk randomly walked onto a light source.
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    if (s == 0 && t > 1)
    {
        if (!BDPT_NAIVE)
            return true;
        if (t == eyePathLength && (eyePath->lightInd[eyePathIDX] == -1)) // path terminated on the sky. We need to add the sky contribution (stored in beta)
        {
            // environment lights currently unimplemented
            return false;
        }
        else if (eyePath->lightInd[eyePathIDX] != -51 && !eyePath->backface[eyePathIDX]) // ie. we are on a light, and we are on the right side of it
        {
            float3 lightToPrev_unit = normalize(f3(eyePath->wo[eyePathIDX]));
            float3 lightNorm = f3(eyePath->n[eyePathIDX]);
            float cosThetaLight = fabsf(dot(lightNorm, lightToPrev_unit));
            float distanceSQR = lengthSquared(f3(eyePath->pt[eyePathIDX]) - f3(eyePath->pt[eyePathPREVIDX]));
            if (t == 2)
            {
                float cosAtCamera = fabsf(dot(f3(eyePath->n[eyePathPREVIDX]), -lightToPrev_unit));

                float aspect = (float)w / (float)h;
                float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;

                float3 light_f = f3(1.0f/PI);

                float pdf_traceFromCamera = cosThetaLight / (distanceSQR * imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);

                float pdf_chooseLight = 1.0f / (SAMPLE_ENVIRONMENT ? (lightNum + 1.0f) : lightNum);

                Triangle light = lights[eyePath->lightInd[eyePathIDX]];
                float3 apos = f3(vertices->positions[light.aInd]);
                float3 bpos = f3(vertices->positions[light.bInd]);
                float3 cpos = f3(vertices->positions[light.cInd]);

                float area = 0.5f * length(cross(bpos - apos, cpos - apos));

                float pdf_connect = pdf_chooseLight / area;

                float wEye = pdf_connect / pdf_traceFromCamera;

                misWeight = 1.0f / (1.0f + wEye);

                float3 Le = f3(lights[eyePath->lightInd[eyePathIDX]].emission);
                contribution = Le * f3(eyePath->beta[eyePathIDX]);
            }
            else
            {
                float3 Le = f3(lights[eyePath->lightInd[eyePathIDX]].emission);

                float wEye;
                float wLight = 0.0f;
                float pdf_connect;

                if (eyePath->isDelta[eyePathPREVIDX])
                {
                    pdf_connect = 0.0f;
                }
                else
                {
                    float pdf_chooseLight = 1.0f / (SAMPLE_ENVIRONMENT ? (lightNum + 1.0f) : lightNum);

                    Triangle light = lights[eyePath->lightInd[eyePathIDX]];
                    float3 apos = f3(vertices->positions[light.aInd]);
                    float3 bpos = f3(vertices->positions[light.bInd]);
                    float3 cpos = f3(vertices->positions[light.cInd]);

                    float area = 0.5f * length(cross(bpos - apos, cpos - apos));

                    pdf_connect = pdf_chooseLight / area;
                }

                // these happen to be equal, since pdf_trace is the spatial prob of starting light path at the current vertex
                float pdf_trace = pdf_connect;

                // purely the emission pdf of generating the previous vertex
                float pdf_oneBeforePrevRev_SA = cosThetaLight / PI;

                wEye = pdf_connect * eyePath->d_vcm[eyePathIDX] + pdf_trace * pdf_oneBeforePrevRev_SA * eyePath->d_vc[eyePathIDX];

                misWeight = 1.0f / (1.0f + wEye + wLight);

                contribution = Le * f3(eyePath->beta[eyePathIDX]);

                //printf("<%f, %f, %f> * <%f, %f, %f>\n", Le.x, Le.y, Le.z, f3(eyePath->beta[eyePathIDX]).x, f3(eyePath->beta[eyePathIDX]).y, f3(eyePath->beta[eyePathIDX]).z);


                float lum = luminance(contribution);
                if (lum > MAX_FIREFLY_LUM)
                {
                    contribution *= (MAX_FIREFLY_LUM / lum);
                }
            }

        }
        return true;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // General Case: s > 1, t > 1
    // Connect a vertex from the Eye Path (eyePathIDX) to a vertex from the Light Path (lightPathIDX)
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    if ( s > 1 && t > 1 && BDPT_CONNECTION)
    {
        float3 lightPos = f3(lightPath->pt[lightPathIDX]);
        float3 eyePos = f3(eyePath->pt[eyePathIDX]);
        float3 lightNorm = f3(lightPath->n[lightPathIDX]);
        float3 eyeNorm = f3(eyePath->n[eyePathIDX]);

        float3 eyeToLight = lightPos - eyePos;
        float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);
        float distance = length(eyeToLight);
        float3 eyetoLight_unit = eyeToLight / distance; // Normalized direction: Eye -> Light
        float3 lightToEye_unit = -eyetoLight_unit; // Normalized direction: Eye -> Light

        if (distanceSQR < RAY_EPSILON)
            return true;

        float cosLight = fabsf(dot(lightNorm, -eyetoLight_unit));
        float cosEye = fabsf(dot(eyeNorm, eyetoLight_unit));

        // If geometry allows connection
        if ((cosLight > EPSILON) && (cosEye > EPSILON))
        {
            // currently connections cannot happen through transmissive materials
            Ray r = Ray(eyePos + eyeNorm * RAY_EPSILON, eyetoLight_unit);

            float3 throughputScale;
            BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, distance - RAY_EPSILON, -1);

            if (lengthSquared(throughputScale) > EPSILON)
            {
                //-------------------------------------------------------
                // Calculate reverse pdf at curr eye index (area)
                //-------------------------------------------------------
                float3 lightToEye_localAtLight;
                toLocal(lightToEye_unit, lightNorm, lightToEye_localAtLight);

                float3 toLightFromPrev_localAtLight;
                toLocal(-f3(lightPath->wo[lightPathIDX]), lightNorm, toLightFromPrev_localAtLight);

                // bsdf evaluated at the light vertex, of scattering towards the eye vertex
                float pdf_eyeRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, toLightFromPrev_localAtLight,
                    lightToEye_localAtLight, etaI, etaT, pdf_eyeRev_SA, lightPath->uv[lightPathIDX]);

                // convert to area density around the eye vertex
                float pdf_eyeRev_area = pdf_eyeRev_SA * cosEye / distanceSQR;

                //-------------------------------------------------------
                // Calculate reverse pdf at prev eye index (SA)
                //-------------------------------------------------------

                float3 lightToEye_localAtEye;
                toLocal(lightToEye_unit, eyeNorm, lightToEye_localAtEye);

                float3 toPrevFromEye_localAtEye;
                toLocal(f3(eyePath->wo[eyePathIDX]), eyeNorm, toPrevFromEye_localAtEye);

                // pdf of generating the vertex before the eye vertex
                float pdf_oneBeforeEyeRev_SA;
                pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, lightToEye_localAtEye,
                    toPrevFromEye_localAtEye, etaI, etaT, pdf_oneBeforeEyeRev_SA, eyePath->uv[eyePathIDX]);

                //-------------------------------------------------------
                // Calculate reverse pdf at curr light index (area)
                //-------------------------------------------------------

                float3 toEyeFromPrev_localAtEye = -toPrevFromEye_localAtEye;
                float3 eyeToLight_localAtEye = -lightToEye_localAtEye;

                float pdf_lightRev_SA;
                pdf_eval(materials, eyePath->materialID[eyePathIDX], textures, toEyeFromPrev_localAtEye,
                    eyeToLight_localAtEye, etaI, etaT, pdf_lightRev_SA, eyePath->uv[eyePathIDX]);

                float pdf_lightRev_area = pdf_lightRev_SA * cosLight / distanceSQR;

                //-------------------------------------------------------
                // Calculate reverse pdf at prev light index (SA)
                //-------------------------------------------------------

                float3 eyeToLight_localAtLight = -lightToEye_localAtLight;
                float3 toPrevFromLight_localAtLight = -toLightFromPrev_localAtLight;

                float pdf_oneBeforeLightRev_SA;
                pdf_eval(materials, lightPath->materialID[lightPathIDX], textures, eyeToLight_localAtLight,
                    toPrevFromLight_localAtLight, etaI, etaT, pdf_oneBeforeLightRev_SA, lightPath->uv[lightPathIDX]);

                float wEye = pdf_eyeRev_area * (eyePath->d_vcm[eyePathIDX] + pdf_oneBeforeEyeRev_SA * eyePath->d_vc[eyePathIDX]);
                float wLight = pdf_lightRev_area * (lightPath->d_vcm[lightPathIDX] + pdf_oneBeforeLightRev_SA * lightPath->d_vc[lightPathIDX]);

                misWeight = 1.0f / (1.0f + wEye + wLight);

                float3 f_eye;
                f_eval(materials, eyePath->materialID[eyePathIDX], textures, lightToEye_localAtEye,
                    toPrevFromEye_localAtEye, etaI, etaT, f_eye, eyePath->uv[eyePathIDX]);

                float3 f_light;
                f_eval(materials, lightPath->materialID[lightPathIDX], textures, eyeToLight_localAtLight,
                    toPrevFromLight_localAtLight, etaI, etaT, f_light, lightPath->uv[lightPathIDX]);

                float G = fabsf(cosEye * cosLight) / distanceSQR;
                float maxG = 2.0f;
                if (G > maxG) {
                    G = maxG;
                }

                contribution = f3(eyePath->beta[eyePathIDX]) * f3(lightPath->beta[lightPathIDX]) * f_eye * f_light * G * throughputScale;

                return true;
            }
        }
    }
    return true;
}


__global__ void __launch_bounds__(256, 2) Li_bidirectional(
    RNGState* rngStates,
    Camera camera,
    PathVertices* eyePath,
    PathVertices* lightPath,
    int* lightPathLengths,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    int eyeDepth, int lightDepth,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum,
    int numSample,
    int w, int h,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    int eyePathLength = 0; // measures number of pathvertices, not segments
    int lightPathLength = lightPathLengths[pixelIdx]; // measures number of pathvertices, not segments

    // light path is already computed from the previous kernel
    generateEyePath(localState, camera, materials, textures, BVH, BVHindices, eyeDepth, vertices, vertNum, scene, triNum, lights,
        lightNum, w, h, x, y, eyePath, eyePathLength);

    float3 fullContribution = f3(0.0f);

    // using bdpt naming conventions with t and s
    for (int t = 2; t <= eyePathLength; t++)
    {
        for (int s = 0; s <= lightPathLength; s++)
        {
            float3 unweighted_contribution = f3(0.0f); // set in connect path
            float misWeight = 0.0f; // set in connect path

            if (!connectPath(localState, t, s, x, y, w, h, camera, eyeDepth, lightDepth, materials, BVH, BVHindices, vertices, scene, lights, lightNum,
                textures, eyePathLength, lightPathLength, eyePath, lightPath, unweighted_contribution, misWeight) && BDPT_DRAWPATH)
            {
                drawPath(overlay, eyePath, camera, x, y, w, eyePathLength, eyeDepth, f3(rand(&localState), rand(&localState), rand(&localState)));
            }

            float3 weightedContribution = unweighted_contribution * misWeight;

            if (BDPT_PAINTWEIGHT)
                fullContribution += f3(misWeight);
            else if (BDPT_DOMIS)
                fullContribution += weightedContribution;
            else
                fullContribution += unweighted_contribution;
        }
    }

    colors[pixelIdx] += f4(fullContribution);
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__host__ void launch_bidirectional(
    int eyeDepth,
    int lightDepth,
    Camera camera,
    PathVertices* eyePath,
    PathVertices* lightPath,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum, int numSample, int w, int h,
    float3 h_sceneCenter, float h_sceneRadius,
    float4* __restrict__ colors, float4* __restrict__ overlay,
    bool postProcess
)
{
    // --- SETUP ---
    dim3 blockSize(16, 16);
    dim3 gridSize((w + 15) / 16, (h + 15) / 16);

    // Create a dedicated CUDA Stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Push symbols asynchronously
    cudaMemcpyToSymbolAsync(sceneCenter, &(h_sceneCenter), sizeof(float3), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(sceneRadius, &(h_sceneRadius), sizeof(float), 0, cudaMemcpyHostToDevice, stream);

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    // Path Lengths
    int* d_pathLengths = nullptr;
    cudaMalloc(&d_pathLengths, w * h * sizeof(int));
    cudaMemsetAsync(d_pathLengths, 0, w * h * sizeof(int), stream);

    // Temporary buffer for saving images (holds the normalized, clean result)
    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));

    // Memory Info Print
    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n", freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));

    // Image Object (CPU) & Saving logic from SPPM
    int saveIntervalSamples = 30; // Matches SPPM logic
    Image image = Image(w, h);
    image.postProcess = postProcess;
    std::vector<float4> h_finalOutput(w * h);

    std::cout << "Starting BDPT Render..." << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();

    // --- MAIN RENDER LOOP ---
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        // Launch kernels on the stream (currSample fixes the RNG loop bug!)
        lightPathTracing<<<gridSize, blockSize, 0, stream>>>(
            d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices,
            lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, numSample, w, h,
            colors, overlay, currSample
        );

        Li_bidirectional<<<gridSize, blockSize, 0, stream>>>(
            d_rngStates, camera, eyePath, lightPath, d_pathLengths, materials, textures, BVH, BVHindices,
            eyeDepth, lightDepth, vertices, vertNum, scene, triNum, lights, lightNum, numSample, w, h,
            colors, overlay, currSample
        );

        // --- PROGRESSIVE SAVING LOGIC ---
        if (DO_PROGRESSIVERENDER && currSample % saveIntervalSamples == 0)
        {
            // Run the helper kernel (Handles NaNs, Normalization, Overlay)
            cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
                colors, overlay, d_finalOutput, w, h, currSample
            );

            // Copy the clean result to Host asynchronously
            cudaMemcpyAsync(h_finalOutput.data(), d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost, stream);

            // Wait ONLY for the memory copy to finish before the CPU touches h_finalOutput
            cudaStreamSynchronize(stream);

            #pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                int x = i % w;
                int y = i / w;
                image.setColor(x, y, h_finalOutput[i]);
            }

            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);

            // Time Tracking
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            // Clear the overlay buffer for the next interval
            cudaMemsetAsync(overlay, 0, w * h * sizeof(float4), stream);
        }
    }

    printf("\n"); // Clear the line after the progress bar finishes

    // Final catch-all sync
    cudaStreamSynchronize(stream);

    // Cleanup resources
    cudaStreamDestroy(stream);
    cudaFree(d_pathLengths);
    cudaFree(d_rngStates);
    cudaFree(d_finalOutput);

    // Error Checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Render executed successfully." << std::endl;
    }
}

__device__ void generateVCMLightPath(
    RNGState& localState,
    int x, int y,
    VCMPathVertices lightPath,
    Photons photons,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices, int vertNum,
    const Triangle* __restrict__ scene, int triNum,
    const Triangle* __restrict__ lights, int lightNum,
    float4* __restrict__ overlay,
    int& pathLength,
    int* globalPhotonIndex
)
{
    pathLength = 0;

    // the convention is that light index -1 is the environment, and that lightNum doesnt include the environment
    int lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(rand(&localState) * (lightNum + 1)), lightNum) - 1) :
        (min(static_cast<int>(rand(&localState) * (lightNum)), lightNum - 1));

    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));

    Ray r;
    float prevPDF_solidAngle = -1.0f; // outgoing pdf from scattering functions
    float prev_cosine = -1.0f; // the previous cosine between the normal and the outgoing ray
    float3 start_wi = f3();

    float pdf_chooseLightPos = -1.0f;
    float3 currThroughput = f3();

    float3 y0Pos;

    if (lightInd == -1) {return;}
    else
    {
        Triangle light = lights[lightInd];
        float3 apos = f3(vertices->positions[light.aInd]);
        float3 bpos = f3(vertices->positions[light.bInd]);
        float3 cpos = f3(vertices->positions[light.cInd]);

        float3 anorm = f3(vertices->normals[light.naInd]);
        float3 bnorm = f3(vertices->normals[light.nbInd]);
        float3 cnorm = f3(vertices->normals[light.ncInd]);

        float area = 0.5f * length(cross(bpos - apos, cpos - apos));

        // for depth 1, this is NOT a solid angle PDF, but we are just reusing the varible
        pdf_chooseLightPos = pdf_chooseLight / area;

        float u = sqrtf(rand(&localState));
        float v = rand(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        y0Pos = w0 * apos + w1 * bpos + w2 * cpos;
        float3 y0Norm = normalize(w0 * anorm + w1 * bnorm + w2 * cnorm);

        float3 wo_local;
        cosine_emit(localState, wo_local, prevPDF_solidAngle);
        toWorld(wo_local, y0Norm, start_wi);

        r.origin = y0Pos + y0Norm * RAY_EPSILON;
        r.direction = start_wi;

        currThroughput = f3(light.emission) * PI / pdf_chooseLightPos;

        prev_cosine = fabsf(dot(normalize(start_wi), y0Norm));
    }
    float3 prevPos = y0Pos;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;
    float prev_d_vm = -1.0f;

    float pdf_onebeforePrevRev_SA = -1.0f;
    bool prevWasDelta = false;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = (depth == 0) ? -1 : pathBufferIdx(w, h, x, y, depth-1);

        Intersection intersect;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            return;
        }
        float2 currUV = intersect.uv;
        float3 currBeta = currThroughput;
        float3 currNormal = intersect.normal;
        int currMatID = intersect.materialID;
        float3 currPos = intersect.point;

        bool currDelta = materials[currMatID].isSpecular;
        bool currBackface = intersect.backface;

        float3 currWo = normalize(-r.direction);

        float3 wo_world = currPos - prevPos; // the incoming direction, pointing at the new surface

        //if (lengthSquared(wo_world) < EPSILON)
        //    printf("Has not moved\n");
        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, currNormal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area;
        pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        // the NEW pdf forward (curr to next)
        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //direction to next vertex

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, currMatID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val,
            pdfFwd_solidAngle, currUV, TRANSPORTMODE_IMPORTANCE);

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 nextToCurrent_local = -wi_local;
        float3 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;
        pdf_eval(materials, currMatID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT,
            pdfRev_solidAngle, currUV);

        if (currDelta)
            pdfRev_solidAngle = pdfFwd_solidAngle; // probabilities of scattering fwd backward on the current delta surface

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Update running values
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        float curr_d_vcm = -1.0f;
        float curr_d_vc = -1.0f;
        float curr_d_vm = -1.0f;

        if (depth == 0) {

            /*the spatial probability of picking the starting light. This is equal to both
            the probability of connecting to the light vertex via nee, and also equal to the
            probability of starting a light path at the light vertex. This value is stored
            in the forward pdf of the light vertex
            */
            float pdf_sampleLight = pdf_chooseLightPos;

            // the pdf of connecting to the previous light via NEE
            float pdf_connect = pdf_sampleLight;

            // the pdf of starting a light path at the previous light
            float pdf_trace = pdf_sampleLight;

            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            // pdfFwdArea is the emission pdf of the light, converted to area density at the current vertex
            curr_d_vcm = (pdf_connect * pdf_connect) / (pdf_trace * pdf_trace * pdfFwd_area * pdfFwd_area);
            curr_d_vc = G * G / (pdf_trace * pdf_trace * pdfFwd_area * pdfFwd_area);
            curr_d_vm = G * G / (pdf_trace * pdf_trace * pdfFwd_area * pdfFwd_area * eta_vcm * eta_vcm);

            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta)
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex

            curr_d_vcm = 0.0f;
            curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
            curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);

            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            if (materials[currMatID].roughness < MERGE_ROUGHNESS_BOUND)
            {
                curr_d_vcm = 1.0f / (pdfFwd_area * pdfFwd_area);
                curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (prev_d_vcm + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
                curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (0.0f +
                    (prev_d_vcm) / (eta_vcm * eta_vcm) + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);
            }
            else
            {
                curr_d_vcm = 1.0f / (pdfFwd_area * pdfFwd_area);
                curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (eta_vcm * eta_vcm +
                    prev_d_vcm + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
                curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (1.0f +
                    (prev_d_vcm) / (eta_vcm * eta_vcm) + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);
            }

            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }

        //if (curr_d_vc < EPSILON || curr_d_vm < EPSILON)
        //if (depth == 0)
            //printf("in light pass at depth %d: dvcm: %f, dvc %f, dvm %f\n", depth, curr_d_vcm, curr_d_vc, curr_d_vm);

        //printf("LIGHT PATH dvcm %f, dvc %f, dvm %f\n", curr_d_vcm ,curr_d_vc, curr_d_vm);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Save Data. We use set functions because the light path struct is highly optimized for memory footprint and contains a ton of shenanigans
        //---------------------------------------------------------------------------------------------------------------------------------------------------


        // light path data (for connections)

        if (!DO_SPPM)
        {
            setPos(lightPath, currIdx, currPos);
            setNormal(lightPath, currIdx, currNormal);
            setWo(lightPath, currIdx, currWo);
            setBeta(lightPath, currIdx, currBeta);
            setUV(lightPath, currIdx, currUV);

            // the boolean flags and light index and material IDs are all packed into one uint. -2 is a flag to say no light
            setAllInfo(lightPath, currIdx, currDelta, currBackface, -2, currMatID);

            setD_vcm(lightPath, currIdx, curr_d_vcm);
            setD_vc(lightPath, currIdx, curr_d_vc);
        }

        // photon data (for merging)
        if (!currDelta && VCM_DOMERGE)
        {
            int photonInd = atomicAdd(globalPhotonIndex, 1);

            if (photonInd < w * h * maxDepth)
            {
                setPackedPosVM(photons, photonInd, currPos, curr_d_vm);
                setWi(photons, photonInd, currWo);
                setNormalInfo(photons, photonInd, currNormal, currBackface);
                setBeta(photons, photonInd, currBeta);

                setD_vcm(photons, photonInd, curr_d_vcm);

            }
        }



        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        r.origin = (wi_local.z < EPSILON) ? (currPos - currNormal * RAY_EPSILON) : (currPos + currNormal * RAY_EPSILON);
        r.direction = wi_world;

        pathLength++;
        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevWasDelta = currDelta;
        prevPos = currPos;
    }
}

__global__ void __launch_bounds__(256, 2) doLightPass(
    RNGState* rngStates,
    Camera camera,
    VCMPathVertices lightPath,
    Photons photons,
    int* lightPathLengths,
    const Material* __restrict__ materials, TextureView textures,
    const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices,
    int lightDepth,
    const Vertices* __restrict__ vertices, int vertNum,
    const Triangle* __restrict__ scene, int triNum,
    const Triangle* __restrict__ lights, int lightNum,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    int* globalPhotonIndex,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    int lightPathLength;
    generateVCMLightPath(
        localState,
        x, y,
        lightPath,
        photons,
        materials,
        textures,
        BVH, BVHindices,
        lightDepth,
        vertices, vertNum,
        scene, triNum,
        lights, lightNum,
        overlay,
        lightPathLength,
        globalPhotonIndex
    );

    //printf("end path gen\n");
    lightPathLengths[pixelIdx] = lightPathLength;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Perform special case of the connection: what if the light ray just connected straight to the camera?
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    for (int s = 2; (s <= lightPathLength + 1) && (BDPT_LIGHTTRACE); s++)
    {
        int lightPathIDX = pathBufferIdx(w, h, x, y, s - 2);

        int lightInd;
        int materialID;
        bool lightDelta;
        bool backface;
        getAllInfo(lightPath, lightPathIDX, lightDelta, backface, lightInd, materialID);

        float3 lightPos = getPos(lightPath, lightPathIDX);
        float3 lightNorm = getNormal(lightPath, lightPathIDX);

        float2 lightUV = getUV(lightPath, lightPathIDX);
        float3 lightBeta = getBeta(lightPath, lightPathIDX);

        float2 pixelPos;
        if (!camera.worldToRaster(lightPos, pixelPos))
            continue;

        int px = (int)pixelPos.x;
        int py = (int)pixelPos.y;
        int newPixelIndex = py * w + px;

        if (lightDelta)
            continue;

        float etaI = 1.0f;
        float etaT = 1.0f;

        float3 lightToCamera = camera.cameraOrigin - lightPos;
        float3 lightToCamera_unit = normalize(lightToCamera);

        Ray r = Ray(lightPos + lightNorm * RAY_EPSILON, lightToCamera_unit);
        float3 throughputScale;

        BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale, length(lightToCamera) - RAY_EPSILON, -1);

        if (lengthSquared(throughputScale) > EPSILON)
        {
            int prevIdx = pathBufferIdx(w, h, x, y, s - 2);
            float cosAtLight = dot(lightNorm, lightToCamera_unit);
            float cosAtCamera = fabsf(dot(camera.getForwardVector(), -lightToCamera_unit));

            if (cosAtLight <= EPSILON) continue;

            float3 lightNormal = lightNorm;
            float3 currToPrev_world = getWo(lightPath, lightPathIDX);
            float3 currToPrev_local;
            toLocal(currToPrev_world, lightNormal, currToPrev_local);

            float3 lightToCamera_local;
            toLocal(lightToCamera_unit, lightNormal, lightToCamera_local);

            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // Unweighted contribution calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------

            float3 light_f;

            f_eval(materials, materialID, textures, -currToPrev_local, lightToCamera_local, etaI, etaT, light_f, lightUV);

            float aspect = (float)w / (float)h;
            float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;

            float We = 1.0f / (imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera * cosAtCamera);

            float distanceSQR = fmaxf(lengthSquared(lightToCamera), RAY_EPSILON);
            float G = (cosAtLight * cosAtCamera) / distanceSQR;

            float3 contribution = lightBeta * light_f * G * throughputScale * We; // unweighted

            //---------------------------------------------------------------------------------------------------------------------------------------------------
            // MIS Weight Calculation
            //---------------------------------------------------------------------------------------------------------------------------------------------------
            float misWeight;

            // we dont consider randomwalks onto the eye lens
            float wEye = 0.0f;

            // the chance to begin a eye path at the eye vertex
            float pdf_trace = 1.0f;

            // the chance to connect a light vertex to the eye vertex
            float pdf_connect = 1.0f;

            float traceRatio = pdf_trace / pdf_connect;

            // the camera emission pdf of generating the current vertex
            float pdf_currRev_area = cosAtLight / (distanceSQR * imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);

            float numLightSample = 1.0f;

            float pdf_oneBeforePrevRev_SA;
            pdf_eval(materials, materialID, textures, -lightToCamera_local, currToPrev_local, etaI, etaT,
                    pdf_oneBeforePrevRev_SA, lightUV);

            float wLight = traceRatio * traceRatio * (pdf_currRev_area * pdf_currRev_area) / (numLightSample * numLightSample) * (
                (materials[materialID].roughness < MERGE_ROUGHNESS_BOUND) ? 0.0f : (eta_vcm * eta_vcm) +
                getD_vcm(lightPath, lightPathIDX) + pdf_oneBeforePrevRev_SA * pdf_oneBeforePrevRev_SA * getD_vc(lightPath, lightPathIDX));

            misWeight = 1.0f / (1.0f + wLight + wEye);

            float3 weightedContribution = contribution * misWeight;

            if (BDPT_PAINTWEIGHT)
                weightedContribution = f3(misWeight);
            if (!BDPT_DOMIS)
                weightedContribution = contribution;

            atomicAdd(&colors[newPixelIndex].x, weightedContribution.x);
            atomicAdd(&colors[newPixelIndex].y, weightedContribution.y);
            atomicAdd(&colors[newPixelIndex].z, weightedContribution.z);
        }
    }
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

/*
Performs the connection calculation of the implicit hit case. Only called when it hits the front side of an emissive surface.
*/
__device__ __noinline__ bool connectImplicitHit(
    float3 lightPos,
    float3 lightNorm,
    float3 throughput,
    int lightInd,
    float d_vc,
    float d_vcm,
    float3 prevPos,
    bool prevDelta,
    const Triangle* __restrict__ lights,
    int lightNum,
    const Vertices* __restrict__ vertices,
    float3& unweightedContribution,
    float& misWeight
)
{
    float3 lightToPrev = prevPos - lightPos;
    float distanceSQR = lengthSquared(lightToPrev);
    float3 lightToPrev_unit = normalize(lightToPrev);
    float cosLight = dot(lightNorm, lightToPrev_unit);

    if (lightInd == -1)
        return false;

    Triangle light = lights[lightInd];
    float3 Le = f3(light.emission);

    float pdf_connect;

    if (prevDelta)
    {
        pdf_connect = 0.0f;
    }
    else
    {
        float pdf_chooseLight = 1.0f / (SAMPLE_ENVIRONMENT ? (lightNum + 1.0f) : lightNum);

        float3 apos = f3(vertices->positions[light.aInd]);
        float3 bpos = f3(vertices->positions[light.bInd]);
        float3 cpos = f3(vertices->positions[light.cInd]);

        float area = 0.5f * length(cross(bpos - apos, cpos - apos));

        pdf_connect = pdf_chooseLight / area;
    }

    // these happen to be equal, since pdf_trace is the spatial prob of starting light path at the current vertex
    float pdf_trace = pdf_connect;
    float pdf_oneBeforePrevRev_SA = cosLight / PI;

    float wEye = pdf_connect * pdf_connect * d_vcm +
        pdf_trace * pdf_trace * pdf_oneBeforePrevRev_SA * pdf_oneBeforePrevRev_SA *
        d_vc;

    misWeight = 1.0f / (1.0f + wEye);
    unweightedContribution = Le * throughput;

    return true;
}

__device__ __noinline__ bool connectNEE(
    RNGState& localState,
    float3 eyePos,
    float3 eyeNorm,
    float3 eyethroughput,
    float2 eyeUV,
    int eyeMatID,
    bool eyeBackface,
    float d_vc,
    float d_vcm,
    float3 prevPos,
    const Triangle* __restrict__ lights,
    int lightNum,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    const Triangle* __restrict__ scene,
    float3& unweightedContribution,
    float& misWeight
)
{
    float3 nee_contribution_unweighted; // assigned in nee
    float pdf_connect; // assigned in nee. in area measure for area light, and in SA for environment
    float3 eyeToLight; // assigned in nee
    int lightInd; // assigned in nee
    float cosLight; // assigned in nee
    float pdf_emit_SA; // the probability that the light was sampled to emit to the eye vertex

    float3 prevToCurr_local;
    float3 prevTocurr = eyePos - prevPos;
    float3 prevTocurrUnit = normalize(prevTocurr);
    // shading function expects toShadingPos_local to face towards the surface, wo faces away
    toLocal(prevTocurr, eyeNorm, prevToCurr_local);

    //placeholders
    float etaI = 1.0f;
    float etaT = 1.0f;

    bool occluded = BDPTnextEventEstimation(localState, materials, textures, BVH, BVHindices,
        vertices, scene, lights, lightNum, eyeMatID, eyePos, prevToCurr_local, eyeNorm,
        eyeUV, pdf_connect, nee_contribution_unweighted, eyeToLight, lightInd, cosLight,
        pdf_emit_SA, etaI, etaT);

    if (occluded)
        return false;

    if (lightInd == -1)
        return false;

    float3 eyeToLight_unit = normalize(eyeToLight);
    float3 eyeToLight_local;
    toLocal(eyeToLight_unit, eyeNorm, eyeToLight_local);

    float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);

    float pdf_eyeToLight_solidAngle;

    pdf_eval(materials, eyeMatID, textures, prevToCurr_local, eyeToLight_local, etaI, etaT,
        pdf_eyeToLight_solidAngle, eyeUV);
    float pdf_bsdf_area = pdf_eyeToLight_solidAngle * fabsf(cosLight) / distanceSQR;

    float bsdfRatio = pdf_bsdf_area / pdf_connect;
    float wLight = bsdfRatio * bsdfRatio;

    float pdf_trace = pdf_connect;
    float traceRatio = pdf_trace / pdf_connect;

    float pdf_currRev_area = pdf_emit_SA * fabsf(eyeToLight_local.z) / distanceSQR;

    float pdf_oneBeforePrevRev_SA;
    pdf_eval(materials, eyeMatID, textures, -eyeToLight_local, -prevToCurr_local, etaI, etaT,
        pdf_oneBeforePrevRev_SA, eyeUV);

    float wEye = traceRatio * traceRatio * pdf_currRev_area * pdf_currRev_area * (
        (materials[eyeMatID].roughness < MERGE_ROUGHNESS_BOUND) ? 0.0f : (eta_vcm * eta_vcm) +
        d_vcm + pdf_oneBeforePrevRev_SA * pdf_oneBeforePrevRev_SA * d_vc);

    misWeight = 1.0f / (1.0f + wLight + wEye);
    unweightedContribution = nee_contribution_unweighted * eyethroughput;

    return true;
}

// assumes that both surfaces are not delta
__device__ __noinline__ bool connectGeneral(
    RNGState& localState,
    float3 eyePos,
    float3 eyeNorm,
    float3 eyeThroughput,
    float2 eyeUV,
    int eyeMatID,
    float eyeD_vc,
    float eyeD_vcm,
    float3 eyePrevPos,
    float3 lightPos,
    float3 lightNorm,
    float3 lightThroughput,
    float3 lightWo,
    float2 lightUV,
    int lightMatID,
    float lightD_vc,
    float lightD_vcm,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    const Triangle* __restrict__ scene,
    float3& unweightedContribution,
    float& misWeight
)
{
    float3 eyeToLight = lightPos - eyePos;
    float distanceSQR = fmaxf(lengthSquared(eyeToLight), RAY_EPSILON);
    float distance = length(eyeToLight);
    float3 eyetoLight_unit = eyeToLight / distance; // Normalized direction: Eye -> Light
    float3 lightToEye_unit = -eyetoLight_unit; // Normalized direction: Eye -> Light

    if (distanceSQR < RAY_EPSILON)
        return false;

    float cosLight = fabsf(dot(lightNorm, -eyetoLight_unit));
    float cosEye = fabsf(dot(eyeNorm, eyetoLight_unit));

    if ((cosLight < EPSILON) || (cosEye < EPSILON))
        return false;

    Ray r = Ray(eyePos + eyeNorm * RAY_EPSILON, eyetoLight_unit);

    float3 throughputScale;
    BVHShadowRay(r, BVH, BVHindices, vertices, scene, materials, throughputScale,
        distance - RAY_EPSILON, -1);

    if (lengthSquared(throughputScale) < EPSILON)
        return false;

    float etaI = 1.0f;
    float etaT = 1.0f;
    //-------------------------------------------------------
    // Calculate reverse pdf at curr eye index (area)
    //-------------------------------------------------------
    float3 lightToEye_localAtLight;
    toLocal(lightToEye_unit, lightNorm, lightToEye_localAtLight);

    float3 toLightFromPrev_localAtLight;
    toLocal(-lightWo, lightNorm, toLightFromPrev_localAtLight);

    // bsdf evaluated at the light vertex, of scattering towards the eye vertex
    float pdf_eyeRev_SA;
    pdf_eval(materials, lightMatID, textures, toLightFromPrev_localAtLight,
        lightToEye_localAtLight, etaI, etaT, pdf_eyeRev_SA, lightUV);

    // convert to area density around the eye vertex
    float pdf_eyeRev_area = pdf_eyeRev_SA * cosEye / distanceSQR;

    //-------------------------------------------------------
    // Calculate reverse pdf at prev eye index (SA)
    //-------------------------------------------------------

    float3 lightToEye_localAtEye;
    toLocal(lightToEye_unit, eyeNorm, lightToEye_localAtEye);

    float3 toPrevFromEye_localAtEye;
    toLocal(eyePrevPos - eyePos, eyeNorm, toPrevFromEye_localAtEye);

    // pdf of generating the vertex before the eye vertex
    float pdf_oneBeforeEyeRev_SA;
    pdf_eval(materials, eyeMatID, textures, lightToEye_localAtEye,
        toPrevFromEye_localAtEye, etaI, etaT, pdf_oneBeforeEyeRev_SA, eyeUV);

    //-------------------------------------------------------
    // Calculate reverse pdf at curr light index (area)
    //-------------------------------------------------------

    float3 toEyeFromPrev_localAtEye = -toPrevFromEye_localAtEye;
    float3 eyeToLight_localAtEye = -lightToEye_localAtEye;

    float pdf_lightRev_SA;
    pdf_eval(materials, eyeMatID, textures, toEyeFromPrev_localAtEye,
        eyeToLight_localAtEye, etaI, etaT, pdf_lightRev_SA, eyeUV);

    float pdf_lightRev_area = pdf_lightRev_SA * cosLight / distanceSQR;

    //-------------------------------------------------------
    // Calculate reverse pdf at prev light index (SA)
    //-------------------------------------------------------

    float3 eyeToLight_localAtLight = -lightToEye_localAtLight;
    float3 toPrevFromLight_localAtLight = -toLightFromPrev_localAtLight;

    float pdf_oneBeforeLightRev_SA;
    pdf_eval(materials, lightMatID, textures, eyeToLight_localAtLight,
        toPrevFromLight_localAtLight, etaI, etaT, pdf_oneBeforeLightRev_SA, lightUV);

    float wEye = pdf_eyeRev_area * pdf_eyeRev_area * (
        (materials[eyeMatID].roughness < MERGE_ROUGHNESS_BOUND) ? 0.0f : (eta_vcm * eta_vcm) +
        eyeD_vcm + pdf_oneBeforeEyeRev_SA * pdf_oneBeforeEyeRev_SA * eyeD_vc);

    float wLight = pdf_lightRev_area * pdf_lightRev_area * (
        (materials[lightMatID].roughness < MERGE_ROUGHNESS_BOUND) ? 0.0f : (eta_vcm * eta_vcm) +
        lightD_vcm + pdf_oneBeforeLightRev_SA * pdf_oneBeforeLightRev_SA * lightD_vc);

    float3 f_eye;
    f_eval(materials, eyeMatID, textures, lightToEye_localAtEye,
        toPrevFromEye_localAtEye, etaI, etaT, f_eye, eyeUV);

    float3 f_light;
    f_eval(materials, lightMatID, textures, eyeToLight_localAtLight,
        toPrevFromLight_localAtLight, etaI, etaT, f_light, lightUV);

    float G = fabsf(cosEye * cosLight) / distanceSQR;
    float maxG = 2.0f;
    if (G > maxG) {
        G = maxG;
    }

    misWeight = 1.0f / (1.0f + wEye + wLight);
    unweightedContribution = eyeThroughput * lightThroughput * f_eye * f_light * G * throughputScale;

    return true;
}

__global__ void __launch_bounds__(256, 2) doEyePass(
    RNGState* rngStates,
    Camera camera,
    const VCMPathVertices lightPath,
    const int* __restrict__ lightPathLengths,
    const Photons photons_sorted,
    const uint32_t* __restrict__ cell_start,
    const uint32_t* __restrict__ cell_end,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices, int vertNum,
    const Triangle* __restrict__ scene, int triNum,
    const Triangle* __restrict__ lights, int lightNum,
    int hashTableSize,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    int photonCount,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = camera.generateCameraRay(localState, x, y);

    float aspect = (float)w / (float)h;
    float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;
    float cosAtCamera = fabsf(dot(camera.getForwardVector(), r.direction));

    float prevPDF_solidAngle = 1.0f / (imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);
    float prev_cosine = cosAtCamera; // the previous cosine between the normal and the outgoing ray
    float3 start_wi = f3();

    float3 currThroughput = f3(1.0f);

    float3 prevPos = camera.cameraOrigin;

    float prev_d_vcm = -1.0f;
    float prev_d_vc = -1.0f;
    float prev_d_vm = -1.0f;

    float pdf_onebeforePrevRev_SA = -1.0f;
    bool prevWasDelta = true;

    float3 colorSum = f3();

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);

        Intersection intersect = Intersection();
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            break;
        }

        float2 currUV = intersect.uv;
        float3 currBeta = currThroughput;
        float3 currNormal = intersect.normal;
        int currMatID = intersect.materialID;
        float3 currPos = intersect.point;

        bool currDelta = materials[currMatID].isSpecular;
        bool currBackface = intersect.backface;

        float3 currWo = normalize(-r.direction);

        float3 wo_world = currPos - prevPos; // the incoming direction, pointing at the new surface
        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, currNormal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area;
        pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        // the NEW pdf forward (curr to next)
        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //direction to next vertex

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, currMatID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val,
            pdfFwd_solidAngle, currUV, TRANSPORTMODE_RADIANCE);

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 nextToCurrent_local = -wi_local;
        float3 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;
        pdf_eval(materials, currMatID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT,
            pdfRev_solidAngle, currUV);

        if (currDelta)
            pdfRev_solidAngle = pdfFwd_solidAngle; // probabilities of scattering fwd backward on the current delta surface

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Update running values
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        float curr_d_vcm = -1.0f;
        float curr_d_vc = -1.0f;
        float curr_d_vm = -1.0f;

        if (depth == 0) {
            float pdf_connect = 1.0f;
            float pdf_trace = 1.0f;
            float numLightSample = 1.0f;

            curr_d_vcm = (pdf_connect * pdf_connect * numLightSample * numLightSample) / (pdf_trace * pdf_trace * pdfFwd_area * pdfFwd_area);
            curr_d_vc = 0.0f;
            curr_d_vm = 0.0f;

            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else if (prevWasDelta)
        {
            float G = prev_cosine / distanceSQR; // distance to previous vertex

            curr_d_vcm = 0.0f;
            curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
            curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);

            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }
        else
        {
            // to convert to area density at previous vertex
            float G = prev_cosine / distanceSQR;

            if (materials[currMatID].roughness < MERGE_ROUGHNESS_BOUND)
            {
                curr_d_vcm = 1.0f / (pdfFwd_area * pdfFwd_area);
                curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (prev_d_vcm + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
                curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (0.0f +
                    (prev_d_vcm) / (eta_vcm * eta_vcm) + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);
            }
            else
            {
                curr_d_vcm = 1.0f / (pdfFwd_area * pdfFwd_area);
                curr_d_vc = (G * G / (pdfFwd_area * pdfFwd_area)) * (eta_vcm * eta_vcm +
                    prev_d_vcm + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vc);
                curr_d_vm = (G * G / (pdfFwd_area * pdfFwd_area)) * (1.0f +
                    (prev_d_vcm) / (eta_vcm * eta_vcm) + pdf_onebeforePrevRev_SA * pdf_onebeforePrevRev_SA * prev_d_vm);
            }


            prev_d_vcm = curr_d_vcm;
            prev_d_vc = curr_d_vc;
            prev_d_vm = curr_d_vm;
            pdf_onebeforePrevRev_SA = pdfRev_solidAngle;
        }

        //if (depth == 0)
            //printf("in eye pass at depth %d: dvcm: %f, dvc %f, dvm %f\n", depth, curr_d_vcm, curr_d_vc, curr_d_vm);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Perform Connection. (this may be slower, but cuts down on VRAM, which is really the problem)
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        if (BDPT_NAIVE) {
            float3 unweightedContribution = f3(0.0f);
            float misWeight = 0.0f;

            int currLightInd = -2;
            if (lengthSquared(f3(scene[intersect.triIDX].emission)) > EPSILON)
                currLightInd = scene[intersect.triIDX].lightInd;

            if ((currLightInd != -2) && !currBackface)
                connectImplicitHit(
                    currPos, currNormal, currBeta, currLightInd,
                    curr_d_vc, curr_d_vcm,
                    prevPos, prevWasDelta,
                    lights, lightNum, vertices,
                    unweightedContribution, misWeight
                );

            float3 contribution;
            if (BDPT_PAINTWEIGHT)
                contribution = f3(misWeight);
            else if (BDPT_DOMIS)
                contribution = unweightedContribution * misWeight;
            else
                contribution = unweightedContribution;

            float lum = luminance(contribution);
            if (lum > MAX_FIREFLY_LUM)
            {
                contribution *= (MAX_FIREFLY_LUM / lum);
            }
            colorSum += contribution;
        }

        if (!currDelta && BDPT_NEE) {
            float3 unweightedContribution = f3(0.0f);
            float misWeight = 0.0f;
            connectNEE(
                localState,
                currPos, currNormal, currBeta, currUV, currMatID, currBackface,
                curr_d_vc, curr_d_vcm,
                prevPos,
                lights, lightNum, materials, textures, BVH, BVHindices, vertices, scene,
                unweightedContribution, misWeight
            );
            float3 contribution;
            if (BDPT_PAINTWEIGHT)
                contribution = f3(misWeight);
            else if (BDPT_DOMIS)
                contribution = unweightedContribution * misWeight;
            else
                contribution = unweightedContribution;

            float lum = luminance(contribution);
            if (lum > MAX_FIREFLY_LUM)
            {
                contribution *= (MAX_FIREFLY_LUM / lum);
            }
            colorSum += contribution;
        }

        // run against every light vertex. There are lightPathLengths[pixelIdx] + 1 light vertices,
        // all but one of which (the one on the light) are actually stored in the buffer (indexed 0 to lightPathLengths[pixelIdx]-1)
        for (int s = 2; s <= lightPathLengths[pixelIdx] + 1; s++)
        {
            int lightPathIDX = pathBufferIdx(w, h, x, y, s - 2);

            if (currDelta || !BDPT_CONNECTION)
                break;

            float3 unweightedContribution = f3(0.0f);
            float misWeight = 0.0f;

            int lightLightInd;
            int lightMatID;
            bool lightDelta;
            bool lightBackface;
            getAllInfo(lightPath, lightPathIDX, lightDelta, lightBackface, lightLightInd, lightMatID);

            if (lightDelta)
                continue;

            float3 lightPos = getPos(lightPath, lightPathIDX);
            float3 lightNorm = getNormal(lightPath, lightPathIDX);
            float3 lightThroughput = getBeta(lightPath, lightPathIDX);
            float3 lightWo = getWo(lightPath, lightPathIDX);
            float2 lightUV = getUV(lightPath, lightPathIDX);

            float lightD_vc = getD_vc(lightPath, lightPathIDX);
            float lightD_vcm = getD_vcm(lightPath, lightPathIDX);

            connectGeneral(
                localState,
                currPos, currNormal, currBeta, currUV, currMatID,
                curr_d_vc, curr_d_vcm,
                prevPos,
                lightPos, lightNorm, lightThroughput, lightWo, lightUV, lightMatID,
                lightD_vc, lightD_vcm,
                materials, textures, BVH, BVHindices, vertices, scene,
                unweightedContribution, misWeight
            );

            float3 contribution;
            if (BDPT_PAINTWEIGHT)
                contribution = f3(misWeight);
            else if (BDPT_DOMIS)
                contribution = unweightedContribution * misWeight;
            else
                contribution = unweightedContribution;

            float lum = luminance(contribution);
            if (lum > MAX_FIREFLY_LUM)
            {
                contribution *= (MAX_FIREFLY_LUM / lum);
            }

            colorSum += contribution;
        }

        // for SPPM specifically (messy because i have sppm integrated into my vcm)
        if (DO_SPPM && lengthSquared(f3(scene[intersect.triIDX].emission)) > EPSILON)
        {
            colorSum += f3(scene[intersect.triIDX].emission) * currBeta;
            break;
        }
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Perform Merging.
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        //printf("etavcm: %f radius: %f", eta_vcm, eta_vcm_to_mergeRadius(eta_vcm, w * h));
        if (!currDelta && VCM_DOMERGE && materials[currMatID].roughness >= MERGE_ROUGHNESS_BOUND)
        {
            float mergeRadius = eta_vcm_to_mergeRadius(eta_vcm, w * h);
            int3 centerIndex = GetGridIndex(currPos, sceneMin, mergeRadius);
            float radiusSq = mergeRadius * mergeRadius;

            float3 totalContribution = f3();

            for (int z1 = -1; z1 <= 1; ++z1)
            {
                for (int y1 = -1; y1 <= 1; ++y1)
                {
                    for (int x1 = -1; x1 <= 1; ++x1)
                    {
                        int3 neighborIndex = make_int3(
                            centerIndex.x + x1,
                            centerIndex.y + y1,
                            centerIndex.z + z1
                        );

                        uint32_t hash = HashGridIndex(neighborIndex, hashTableSize);
                        uint32_t start = __ldg(&cell_start[hash]);
                        uint32_t end   = __ldg(&cell_end[hash]);

                        if (start == 0xFFFFFFFF) continue;

                        for (int i = start; i < end; ++i) {

                            float3 photonPos;
                            float lightD_vm;
                            getPosVM(photons_sorted, i, photonPos, lightD_vm);

                            float3 photonNorm;
                            bool photonBackface;
                            getNormalInfo(photons_sorted, i, photonNorm, photonBackface);

                            float distSq = lengthSquared(currPos - photonPos);

                            // Raw normals are always flipped so that it is on the same side as the incident ray, requiring the backface flag to
                            // indicate which side the incident ray is pointing towards
                            if (currBackface != photonBackface) // they are NOT on the same side. CANNOT merge
                                continue;

                            if (distSq <= radiusSq && dot(photonNorm, currNormal) > 0.9f) {
                                float lightD_vcm = getD_vcm(photons_sorted, i);
                                //float lightD_vm = getD_vm(photons_sorted, i);

                                float3 photonToPrev = getWi(photons_sorted, i);
                                float3 eyeToPrev = prevPos - currPos;

                                // need to calculate the pdf of scattering back to the previous eye, and previous light vertex

                                float3 eyeToPrev_local;
                                toLocal(eyeToPrev, currNormal, eyeToPrev_local);

                                float3 photonPrevToPhoton_local;
                                toLocal(-photonToPrev, currNormal, photonPrevToPhoton_local);

                                float eyeRevPDF_SA;
                                // photon's previous to photon and eye to eye's prev
                                pdf_eval(materials, currMatID, textures, photonPrevToPhoton_local, eyeToPrev_local, etaI, etaT, eyeRevPDF_SA, currUV);

                                float lightRevPDF_SA;
                                // eye's prev to eye, and photon to photon's prev
                                pdf_eval(materials, currMatID, textures, -eyeToPrev_local, -photonPrevToPhoton_local, etaI, etaT, lightRevPDF_SA, currUV);

                                float wEye = (curr_d_vcm) / (eta_vcm * eta_vcm) + eyeRevPDF_SA * eyeRevPDF_SA * curr_d_vm;
                                float wLight = (lightD_vcm) / (eta_vcm * eta_vcm) + lightRevPDF_SA * lightRevPDF_SA * lightD_vm;

                                float misWeight = 1.0f / (1.0f + wEye + wLight);

                                float3 f_val;
                                f_eval(materials, currMatID, textures, photonPrevToPhoton_local, eyeToPrev_local, etaI, etaT, f_val, currUV);

                                float3 unweightedContribution = getBeta(photons_sorted, i) * f_val * currBeta / (eta_vcm);

                                float3 contribution = unweightedContribution * misWeight;
                                //if (x == 100 && y == 100)
                                //    printf("in merging depth: %d light: dvcm: %f, dvm %f, eye: dvcm: %f, dvm %f, misWeight %f\n", depth, lightD_vcm, lightD_vm, curr_d_vcm, curr_d_vm, misWeight);

                                if (misWeight > 1.0f)
                                    printf("mis weight is: %f\n", misWeight);

                                if (BDPT_PAINTWEIGHT)
                                    totalContribution += f3(misWeight);
                                else if (BDPT_DOMIS)
                                    totalContribution += contribution;
                                else
                                    totalContribution += unweightedContribution;
                            }
                        }
                    }
                }
            }

            float lum = luminance(totalContribution);
            if (lum > MERGE_MAX_FIREFLY_LUM)
            {
                totalContribution *= (MERGE_MAX_FIREFLY_LUM / lum);
            }
            colorSum += totalContribution;
            if (DO_SPPM) // SPPM only gathers density one time
                break;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool transmitting = dot(wi_world, currNormal) < 0.0f;

        r.origin = transmitting ? (currPos - currNormal * RAY_EPSILON) : (currPos + currNormal * RAY_EPSILON);
        r.direction = wi_world;

        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevWasDelta = currDelta;
        prevPos = currPos;
    }
    colors[pixelIdx] += f4(colorSum);
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__global__ void computeHashes(
    Photons photons,
    int photonCount,
    uint32_t* d_hash_keys,
    uint32_t* d_indices,
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= photonCount) return;

    float3 p = getPos(photons, i);

    d_hash_keys[i] = ComputeGridHash(p, sceneMin, mergeRadius, hashTableSize);
    d_indices[i] = i;
}

__global__ void reorderPhotons(
    Photons photons,
    Photons photons_sorted,
    int photonCount,
    uint32_t* d_indices_out
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= photonCount) return;

    int src = d_indices_out[i];

    /*
    It is not neccesary to unpack the values, we just copy them over directly
    */
    photons_sorted.pos_plus_vm[i] = photons.pos_plus_vm[src];
    photons_sorted.beta_x[i] = photons.beta_x[src];
    photons_sorted.beta_y[i] = photons.beta_y[src];
    photons_sorted.beta_z[i] = photons.beta_z[src];
    photons_sorted.packedWi[i] = photons.packedWi[src];
    photons_sorted.packedNormal[i] = photons.packedNormal[src];
    //photons_sorted.d_vc[i] = photons.d_vc[src];
    photons_sorted.d_vcm[i] = photons.d_vcm[src];
    //photons_sorted.d_vm[i] = photons.d_vm[src];
}

__global__ void buildTable(
    uint32_t* d_hashes_sorted,
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    int numPhotons,
    int hashTableSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPhotons) return;

    uint32_t hash = d_hashes_sorted[i];

    if (hash >= hashTableSize) {
        printf("Error: Thread %d found invalid hash %u (Limit: %d)\n", i, hash, hashTableSize);
        return;
    }

    if (i == 0 || d_hashes_sorted[i - 1] != hash) {
        d_cell_start[hash] = i;
    }

    if (i == numPhotons - 1 || d_hashes_sorted[i + 1] != hash) {
        d_cell_end[hash] = i + 1;
    }
}

__host__ inline void buildHashGrid(
    Photons photons,
    Photons photons_sorted,
    int photonCount,
    uint32_t* d_hash_keys_in,
    uint32_t* d_hash_keys_out,
    uint32_t* d_indices_in,
    uint32_t* d_indices_out,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize
)
{
    int blockSize = 256;
    int numBlocks = (photonCount + blockSize - 1) / blockSize;

    computeHashes<<<numBlocks, blockSize>>>(
        photons,
        photonCount,
        d_hash_keys_in,
        d_indices_in,
        sceneMin,
        mergeRadius,
        hashTableSize
    );

    //checkCudaErrors("compute hashes");

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_hash_keys_in, d_hash_keys_out, d_indices_in, d_indices_out, photonCount);

    //checkCudaErrors("radix sort");

    reorderPhotons<<<numBlocks, blockSize>>>(
        photons,
        photons_sorted,
        photonCount,
        d_indices_out
    );

    //checkCudaErrors("reorder photons");

    cudaMemset(d_cell_start, 0xFF, hashTableSize * sizeof(uint32_t));
    cudaMemset(d_cell_end,   0xFF, hashTableSize * sizeof(uint32_t));

    buildTable<<<numBlocks, blockSize>>>(
        d_hash_keys_out,
        d_cell_start,
        d_cell_end,
        photonCount,
        hashTableSize
    );

    //checkCudaErrors("build table");
}

__global__ void paintPhotons(Photons photons, int numPhotons, float4* __restrict__ overlay, int w, int h, Camera camera, int* numPainted)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPhotons || i % 50 != 0) return;
    float2 pixelPos;
    float3 pos = getPos(photons, i);
    if (camera.worldToRaster(pos, pixelPos))
    {
        int px = (int)pixelPos.x;
        int py = (int)pixelPos.y;
        int newPixelIndex = py * w + px;
        atomicAdd(&overlay[newPixelIndex].x, 1.0f);
        atomicAdd(numPainted, 1);
    }
}

__global__ void paintGridBox(
    Photons photons_sorted,      // MUST use the sorted photon array
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    float3 queryPos,             // The 3D world position to inspect
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize,
    float4* __restrict__ overlay,
    int w, int h,
    Camera camera
)
{
    uint32_t hash = ComputeGridHash(queryPos, sceneMin, mergeRadius, hashTableSize);

    uint32_t start = d_cell_start[hash];
    uint32_t end = d_cell_end[hash];

    if (start == 0xFFFFFFFF || end == 0xFFFFFFFF || start >= end) {
        return; // Nothing in this grid cell
    }

    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        float3 pos = getPos(photons_sorted, i);
        float2 pixelPos;

        if (camera.worldToRaster(pos, pixelPos))
        {
            int px = (int)pixelPos.x;
            int py = (int)pixelPos.y;

            if (px >= 0 && px < w && py >= 0 && py < h) {
                int pixelIndex = py * w + px;
                atomicAdd(&overlay[pixelIndex].x, 0.0000002f);
            }
        }
    }
}

__host__ void launch_VCM(
    int eyeDepth,
    int lightDepth,
    Camera camera,
    VCMPathVertices* lightPath,
    Photons* photons,
    Photons* photons_sorted,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum, int numSample,
    int h_w, int h_h,
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    bool postProcess,
    float mergeRadiusPower,
    float initialRadiusMultiplier
)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((h_w+15)/16, (h_h+15)/16);

    // Constants setup
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float3));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float3));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));
    cudaMemcpyToSymbol(w, &(h_w), sizeof(int));
    cudaMemcpyToSymbol(h, &(h_h), sizeof(int));

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, h_w * h_h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif


    // set up device buffers for VCM and display
    int* d_pathLengths = nullptr;

    cudaMalloc(&d_pathLengths, h_w * h_h * sizeof(int));
    cudaMemset(d_pathLengths, 0, h_w * h_h * sizeof(int));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, h_w * h_h * sizeof(float4));

    // set up buffers used to create the hash table
    int maxPhotonCount = h_w * h_h * lightDepth;

    uint32_t* d_hash_keys_in;
    uint32_t* d_hash_keys_out;
    uint32_t* d_indices_in;
    uint32_t* d_indices_out;

    cudaMalloc(&d_hash_keys_in, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_hash_keys_out, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_indices_in, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_indices_out, maxPhotonCount * sizeof(uint32_t));

    int hashTableSize = GetNextPrime(maxPhotonCount * 2);

    uint32_t* d_cell_start;
    uint32_t* d_cell_end;

    cudaMalloc(&d_cell_start, hashTableSize * sizeof(uint32_t));
    cudaMalloc(&d_cell_end, hashTableSize * sizeof(uint32_t));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_hash_keys_in, d_hash_keys_out, d_indices_in, d_indices_out, maxPhotonCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int* d_global_photon_counter;
    cudaMalloc(&d_global_photon_counter, sizeof(int));

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    auto lastSaveTime = std::chrono::steady_clock::now();
    int saveIntervalSamples = 200;
    Image image = Image(h_w, h_h);
    image.postProcess = postProcess;
    std::vector<float4> h_finalOutput(h_w * h_h);

    std::cout << "Begin Render" << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();

    float mergeRadius;
    float h_eta_vcm;
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        mergeRadius = calculateMergeRadius(h_sceneRadius * initialRadiusMultiplier, mergeRadiusPower, currSample);
        h_eta_vcm = mergeRadius * mergeRadius * (h_w * h_h * h_PI);
        cudaMemcpyToSymbol(eta_vcm, &(h_eta_vcm), sizeof(float));

        cudaMemset(d_global_photon_counter, 0, sizeof(int));
        doLightPass<<<gridSize, blockSize>>>(
            d_rngStates,
            camera,
            *lightPath,
            *photons,
            d_pathLengths,
            materials, textures,
            BVH, BVHindices,
            lightDepth,
            vertices, vertNum,
            scene, triNum,
            lights, lightNum,
            colors, overlay,
            d_global_photon_counter,
            currSample
        );

        int photonCount;
        cudaMemcpy(&photonCount, d_global_photon_counter, sizeof(int), cudaMemcpyDeviceToHost);

        buildHashGrid(
            *photons,
            *photons_sorted,
            photonCount,
            d_hash_keys_in,
            d_hash_keys_out,
            d_indices_in,
            d_indices_out,
            d_temp_storage,
            temp_storage_bytes,
            d_cell_start,
            d_cell_end,
            h_sceneMin,
            mergeRadius,
            hashTableSize
        );

        doEyePass<<<gridSize, blockSize>>>(
            d_rngStates,
            camera,
            *lightPath, d_pathLengths,
            *photons_sorted, d_cell_start, d_cell_end,
            materials, textures,
            BVH, BVHindices,
            eyeDepth,
            vertices, vertNum, scene, triNum, lights, lightNum,
            hashTableSize,
            colors, overlay,
            photonCount,
            currSample
        );

        if (DO_PROGRESSIVERENDER)
            cudaDeviceSynchronize();

        if (currSample % saveIntervalSamples == 0 && DO_PROGRESSIVERENDER)
        {
            cleanAndFormatImage<<<gridSize, blockSize>>>(
                colors, overlay, d_finalOutput, h_w, h_h, currSample
            );

            cudaMemcpy(h_finalOutput.data(), d_finalOutput, h_w * h_h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < h_w * h_h; i++) {
                int x = i % h_w;
                int y = i / h_w;
                image.setColor(x, y, h_finalOutput[i]);
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);


            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            cudaMemset(overlay, 0, h_w * h_h * sizeof(float4));
        }
    }

    printf("\n"); // Move to a new line when the render loop finishes completely

    cudaDeviceSynchronize();
    cudaFree(d_pathLengths);
    cudaFree(d_rngStates);
    cudaFree(d_finalOutput);

    cudaFree(d_hash_keys_in);
    cudaFree(d_hash_keys_out);
    cudaFree(d_indices_in);
    cudaFree(d_indices_out);

    cudaFree(d_cell_start);
    cudaFree(d_cell_end);

    cudaFree(d_temp_storage);
    cudaFree(d_global_photon_counter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}

__global__ void tracePhotons(
    RNGState* rngStates,
    int w, int h,
    Photons photons,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices, int vertNum,
    const Triangle* __restrict__ scene, int triNum,
    const Triangle* __restrict__ lights, int lightNum,
    int* globalPhotonIndex,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    // the convention is that light index -1 is the environment, and that lightNum doesnt include the environment
    int lightInd = SAMPLE_ENVIRONMENT ? (min(static_cast<int>(rand(&localState) * (lightNum + 1)), lightNum) - 1) :
        (min(static_cast<int>(rand(&localState) * (lightNum)), lightNum - 1));

    float pdf_chooseLight = 1.0f / ((float) (SAMPLE_ENVIRONMENT ? (lightNum + 1) : lightNum));

    Ray r;
    float prevPDF_solidAngle = -1.0f; // outgoing pdf from scattering functions
    float prev_cosine = -1.0f; // the previous cosine between the normal and the outgoing ray
    float3 start_wi = f3();

    float pdf_chooseLightPos = -1.0f;
    float3 currThroughput = f3();

    float3 y0Pos;

    if (lightInd == -1) {return;}
    else
    {
        Triangle light = lights[lightInd];
        float3 apos = f3(vertices->positions[light.aInd]);
        float3 bpos = f3(vertices->positions[light.bInd]);
        float3 cpos = f3(vertices->positions[light.cInd]);

        float3 anorm = f3(vertices->normals[light.naInd]);
        float3 bnorm = f3(vertices->normals[light.nbInd]);
        float3 cnorm = f3(vertices->normals[light.ncInd]);

        float area = 0.5f * length(cross(bpos - apos, cpos - apos));

        // for depth 1, this is NOT a solid angle PDF, but we are just reusing the varible
        pdf_chooseLightPos = pdf_chooseLight / area;

        float u = sqrtf(rand(&localState));
        float v = rand(&localState);

        float w0 = (1.0f - u);
        float w1 = u * (1.0f - v);
        float w2 = u * v;

        y0Pos = w0 * apos + w1 * bpos + w2 * cpos;
        float3 y0Norm = normalize(w0 * anorm + w1 * bnorm + w2 * cnorm);

        float3 wo_local;
        cosine_emit(localState, wo_local, prevPDF_solidAngle);
        toWorld(wo_local, y0Norm, start_wi);

        r.origin = y0Pos + y0Norm * RAY_EPSILON;
        r.direction = start_wi;

        currThroughput = f3(light.emission) * PI / pdf_chooseLightPos;

        prev_cosine = fabsf(dot(normalize(start_wi), y0Norm));
    }
    float3 prevPos = y0Pos;

    bool prevWasDelta = false;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);
        int prevIdx = (depth == 0) ? -1 : pathBufferIdx(w, h, x, y, depth-1);

        Intersection intersect;
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            return;
        }
        float2 currUV = intersect.uv;
        float3 currBeta = currThroughput;
        float3 currNormal = intersect.normal;
        int currMatID = intersect.materialID;
        float3 currPos = intersect.point;

        bool currDelta = materials[currMatID].isSpecular;
        bool currBackface = intersect.backface;

        float3 currWo = normalize(-r.direction);

        float3 wo_world = currPos - prevPos; // the incoming direction, pointing at the new surface

        //if (lengthSquared(wo_world) < EPSILON)
        //    printf("Has not moved\n");
        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, currNormal, wo_local);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate forward pdf (previous vertex to current)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float distanceSQR = fmaxf(lengthSquared(wo_world), RAY_EPSILON);

        // previous pdf (solid angle) * abs of dot product of current normal with incoming direction into the current surface divided by distance squared
        float pdfFwd_area;
        pdfFwd_area = prevPDF_solidAngle * fabsf(wo_local.z) / distanceSQR;

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        // the NEW pdf forward (curr to next)
        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //direction to next vertex

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, currMatID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val,
            pdfFwd_solidAngle, currUV, TRANSPORTMODE_IMPORTANCE);

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // calculate backwards pdf (current vertex to previous)
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        float3 nextToCurrent_local = -wi_local;
        float3 currentToPrev_local = -wo_local;

        float pdfRev_solidAngle;
        pdf_eval(materials, currMatID, textures, nextToCurrent_local, currentToPrev_local, etaI, etaT,
            pdfRev_solidAngle, currUV);

        if (currDelta)
            pdfRev_solidAngle = pdfFwd_solidAngle; // probabilities of scattering fwd backward on the current delta surface

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Update running values
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Save Data. We use set functions because the light path struct is highly optimized for memory footprint and contains a ton of shenanigans
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        // photon data (for merging)
        if (!currDelta)
        {
            int photonInd = atomicAdd(globalPhotonIndex, 1);

            if (photonInd < w * h * maxDepth)
            {
                setPackedPosVM(photons, photonInd, currPos, 1.0f);
                setWi(photons, photonInd, currWo);
                setNormalInfo(photons, photonInd, currNormal, currBackface);
                setBeta(photons, photonInd, currBeta);
            }
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        r.origin = (wi_local.z < EPSILON) ? (currPos - currNormal * RAY_EPSILON) : (currPos + currNormal * RAY_EPSILON);
        r.direction = wi_world;

        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevWasDelta = currDelta;
        prevPos = currPos;
    }

    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__global__ void traceEyePaths(
    RNGState* rngStates,
    Camera camera,
    const Photons photons_sorted,
    const uint32_t* __restrict__ cell_start,
    const uint32_t* __restrict__ cell_end,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH, const int2* __restrict__ BVHindices,
    int maxDepth,
    const Vertices* __restrict__ vertices, int vertNum,
    const Triangle* __restrict__ scene, int triNum,
    const Triangle* __restrict__ lights, int lightNum,
    int w, int h,
    int hashTableSize,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    int photonCount,
    int frameNum
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    //RNGState localState = rngStates[pixelIdx];
    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = camera.generateCameraRay(localState, x, y);

    float aspect = (float)w / (float)h;
    float imagePlaneArea = 4.0f * aspect * camera.fovScale * camera.fovScale;
    float cosAtCamera = fabsf(dot(camera.getForwardVector(), r.direction));

    float prevPDF_solidAngle = 1.0f / (imagePlaneArea * cosAtCamera * cosAtCamera * cosAtCamera);
    float prev_cosine = cosAtCamera; // the previous cosine between the normal and the outgoing ray
    float3 start_wi = f3();

    float3 currThroughput = f3(1.0f);

    float3 prevPos = camera.cameraOrigin;

    float3 colorSum = f3();

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int currIdx = pathBufferIdx(w, h, x, y, depth);

        Intersection intersect = Intersection();
        BVHSceneIntersect(r, BVH, BVHindices, vertices, scene, intersect);

        if (!intersect.valid)
        {
            break;
        }

        float3 wo_local; // the incoming direction to the current path vertex. we use this for the cosine in the pdf conversion
        toLocal(r.direction, intersect.normal, wo_local);


        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Scatter to next vertex
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        // the NEW pdf forward (curr to next)
        float pdfFwd_solidAngle;
        float3 f_val;
        float3 wi_local; //direction to next vertex

        float etaI = 1.0f; // TEMPORARY, CHANGE AFTER IMPLEMENTING PRIORITY NESTED DIELECTRICS
        float etaT = 1.0f;

        sample_f_eval(localState, materials, intersect.materialID, textures, wo_local, etaI, etaT, intersect.backface, wi_local, f_val,
            pdfFwd_solidAngle, intersect.uv, TRANSPORTMODE_RADIANCE);

        float3 wi_world;
        toWorld(wi_local, intersect.normal, wi_world);

        // for SPPM specifically (messy because i have sppm integrated into my vcm)
        if (lengthSquared(f3(scene[intersect.triIDX].emission)) > EPSILON)
        {
            colorSum += f3(scene[intersect.triIDX].emission) * currThroughput;
            break;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Perform Merging.
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        //printf("etavcm: %f radius: %f", eta_vcm, eta_vcm_to_mergeRadius(eta_vcm, w * h));
        if (materials[intersect.materialID].roughness > MERGE_ROUGHNESS_BOUND)
        {
            float mergeRadius = eta_vcm_to_mergeRadius(eta_vcm, w * h);
            int3 centerIndex = GetGridIndex(intersect.point, sceneMin, mergeRadius);
            float radiusSq = mergeRadius * mergeRadius;

            float3 totalContribution = f3();

            for (int z1 = -1; z1 <= 1; ++z1)
            {
                for (int y1 = -1; y1 <= 1; ++y1)
                {
                    for (int x1 = -1; x1 <= 1; ++x1)
                    {
                        int3 neighborIndex = make_int3(
                            centerIndex.x + x1,
                            centerIndex.y + y1,
                            centerIndex.z + z1
                        );

                        uint32_t hash = HashGridIndex(neighborIndex, hashTableSize);
                        uint32_t start = __ldg(&cell_start[hash]);
                        uint32_t end   = __ldg(&cell_end[hash]);

                        if (start == 0xFFFFFFFF) continue;

                        for (int i = start; i < end; ++i) {

                            float3 photonPos = getPos(photons_sorted, i);

                            float3 photonNorm;
                            bool photonBackface;
                            getNormalInfo(photons_sorted, i, photonNorm, photonBackface);

                            float distSq = lengthSquared(intersect.point - photonPos);

                            // Raw normals are always flipped so that it is on the same side as the incident ray, requiring the backface flag to
                            // indicate which side the incident ray is pointing towards
                            if (intersect.backface != photonBackface) // they are NOT on the same side. CANNOT merge
                                continue;

                            if (distSq <= radiusSq && dot(photonNorm, intersect.normal) > 0.9f) {
                                float lightD_vcm = getD_vcm(photons_sorted, i);
                                //float lightD_vm = getD_vm(photons_sorted, i);

                                float3 photonToPrev = getWi(photons_sorted, i);
                                float3 eyeToPrev = prevPos - intersect.point;

                                // need to calculate the pdf of scattering back to the previous eye, and previous light vertex

                                float3 eyeToPrev_local;
                                toLocal(eyeToPrev, intersect.normal, eyeToPrev_local);

                                float3 photonPrevToPhoton_local;
                                toLocal(-photonToPrev, intersect.normal, photonPrevToPhoton_local);

                                float3 f_val;
                                f_eval(materials, intersect.materialID, textures, photonPrevToPhoton_local,
                                    eyeToPrev_local, etaI, etaT, f_val, intersect.uv);

                                float3 unweightedContribution = getBeta(photons_sorted, i) * f_val * currThroughput / (eta_vcm);


                                if (isnan(unweightedContribution.x) || isnan(unweightedContribution.y) || isnan(unweightedContribution.z)) {
                                    printf("nan\n");
                                }
                                else if (isinf(unweightedContribution.x) || isinf(unweightedContribution.y) || isinf(unweightedContribution.z)) {
                                    printf("inf\n");
                                }
                                else if (unweightedContribution.x <= 0 || unweightedContribution.y <= 0 || unweightedContribution.z <= 0) {
                                    printf("neg/zero\n");
                                }

                                totalContribution += unweightedContribution;
                            }
                        }
                    }
                }
            }

            float lum = luminance(totalContribution);
            if (lum > MERGE_MAX_FIREFLY_LUM)
            {
                totalContribution *= (MERGE_MAX_FIREFLY_LUM / lum);
            }
            colorSum += totalContribution;

            // SPPM only gathers density one time
            break;
        }
        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Set up next interaction
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool transmitting = dot(wi_world, intersect.normal) < 0.0f;

        currThroughput = currThroughput * f_val * fabsf(wi_local.z) / pdfFwd_solidAngle;

        r.origin = transmitting ?
            (intersect.point - intersect.normal * RAY_EPSILON) :
            (intersect.point + intersect.normal * RAY_EPSILON);
        r.direction = wi_world;

        prevPDF_solidAngle = pdfFwd_solidAngle; // update the prev pdf
        prev_cosine = fabsf(wi_local.z); // update the prev cosine
        prevPos = intersect.point;
    }
    colors[pixelIdx] += f4(colorSum);
    //rngStates[pixelIdx] = localState;
    save_rng(pixelIdx, &localState, rngStates);
}

__host__ void launch_SPPM(
    int eyeDepth,
    int lightDepth,
    Camera camera,
    Photons* photons,
    Photons* photons_sorted,
    const Material* __restrict__ materials,
    TextureView textures,
    const BVHnode* __restrict__ BVH,
    const int2* __restrict__ BVHindices,
    const Vertices* __restrict__ vertices,
    int vertNum,
    const Triangle* __restrict__ scene,
    int triNum,
    const Triangle* __restrict__ lights,
    int lightNum, int numSample,
    int w, int h,
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    bool postProcess,
    float mergeRadiusPower,
    float initialRadiusMultiplier
)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((w+15)/16, (h+15)/16);

    // Constants setup
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float3));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float3));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    // set up device buffers for VCM and display
    int* d_pathLengths = nullptr;

    cudaMalloc(&d_pathLengths, w * h * sizeof(int));
    cudaMemset(d_pathLengths, 0, w * h * sizeof(int));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));

    // set up buffers used to create the hash table
    int maxPhotonCount = w * h * lightDepth;

    uint32_t* d_hash_keys_in;
    uint32_t* d_hash_keys_out;
    uint32_t* d_indices_in;
    uint32_t* d_indices_out;

    cudaMalloc(&d_hash_keys_in, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_hash_keys_out, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_indices_in, maxPhotonCount * sizeof(uint32_t));
    cudaMalloc(&d_indices_out, maxPhotonCount * sizeof(uint32_t));

    int hashTableSize = GetNextPrime(maxPhotonCount * 2);

    uint32_t* d_cell_start;
    uint32_t* d_cell_end;

    cudaMalloc(&d_cell_start, hashTableSize * sizeof(uint32_t));
    cudaMalloc(&d_cell_end, hashTableSize * sizeof(uint32_t));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_hash_keys_in, d_hash_keys_out, d_indices_in, d_indices_out, maxPhotonCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int* d_global_photon_counter;
    cudaMalloc(&d_global_photon_counter, sizeof(int));

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    auto lastSaveTime = std::chrono::steady_clock::now();
    int saveIntervalSamples = 30;
    Image image = Image(w, h);
    image.postProcess = postProcess;
    std::vector<float4> h_finalOutput(w * h);

    std::cout << "Begin Render with SPPM" << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();

    float mergeRadius;
    float h_eta_vcm;
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        mergeRadius = calculateMergeRadius(h_sceneRadius * initialRadiusMultiplier, mergeRadiusPower, currSample);
        h_eta_vcm = mergeRadius * mergeRadius * (w * h * h_PI);
        cudaMemcpyToSymbol(eta_vcm, &(h_eta_vcm), sizeof(float));

        cudaMemset(d_global_photon_counter, 0, sizeof(int));
        tracePhotons<<<gridSize, blockSize>>>(
            d_rngStates,
            w, h,
            *photons,
            materials, textures,
            BVH, BVHindices,
            lightDepth,
            vertices, vertNum,
            scene, triNum,
            lights, lightNum,
            d_global_photon_counter,
            currSample
        );

        int photonCount;
        cudaMemcpy(&photonCount, d_global_photon_counter, sizeof(int), cudaMemcpyDeviceToHost);

        buildHashGrid(
            *photons,
            *photons_sorted,
            photonCount,
            d_hash_keys_in,
            d_hash_keys_out,
            d_indices_in,
            d_indices_out,
            d_temp_storage,
            temp_storage_bytes,
            d_cell_start,
            d_cell_end,
            h_sceneMin,
            mergeRadius,
            hashTableSize
        );

        traceEyePaths<<<gridSize, blockSize>>>(
            d_rngStates,
            camera,
            *photons_sorted, d_cell_start, d_cell_end,
            materials, textures,
            BVH, BVHindices,
            eyeDepth,
            vertices, vertNum, scene, triNum, lights, lightNum,
            w, h,
            hashTableSize,
            colors, overlay,
            photonCount,
            currSample
        );

        if (DO_PROGRESSIVERENDER)
            cudaDeviceSynchronize();

        if (currSample % saveIntervalSamples == 0 && DO_PROGRESSIVERENDER)
        {
            cleanAndFormatImage<<<gridSize, blockSize>>>(
                colors, overlay, d_finalOutput, w, h, currSample
            );

            cudaMemcpy(h_finalOutput.data(), d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                int x = i % w;
                int y = i / w;
                image.setColor(x, y, h_finalOutput[i]);
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);


            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            cudaMemset(overlay, 0, w * h * sizeof(float4));
        }
    }

    printf("\n"); // Move to a new line when the render loop finishes completely

    cudaDeviceSynchronize();
    cudaFree(d_pathLengths);
    cudaFree(d_rngStates);
    cudaFree(d_finalOutput);

    cudaFree(d_hash_keys_in);
    cudaFree(d_hash_keys_out);
    cudaFree(d_indices_in);
    cudaFree(d_indices_out);

    cudaFree(d_cell_start);
    cudaFree(d_cell_end);

    cudaFree(d_temp_storage);
    cudaFree(d_global_photon_counter);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}