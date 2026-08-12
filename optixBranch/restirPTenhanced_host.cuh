#pragma once
#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"

#include "restirPTenhancedShaders.cuh"
#include "restirPTobjects.cuh"
#include "hostSetup.cuh"
#include "animation.cuh"
#include "restirPTenhanced_kernels.cuh"
#include "restirPTenhanced_spatialReuseTextures.cuh"
#include "settings.cuh"
#include <filesystem>

__host__ void launch_restir (
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t frameCount
) {
    commonParams.camera.antiAliasJitterDist = 1.0f;
    Reservoir reservoir1, reservoir2;
    GBuffer gbuffer1;
    GBuffer gbuffer2;

    void* r1Memory = allocateReservoir(reservoir1, commonParams.w * commonParams.h);
    void* r2Memory = allocateReservoir(reservoir2, commonParams.w * commonParams.h);
    void* gb1Memory = allocateGBuffer(gbuffer1, commonParams.w * commonParams.h);
    void* gb2Memory = allocateGBuffer(gbuffer2, commonParams.w * commonParams.h);

    DenoiserGuides denoiserGuides;
    void* dgMemory = allocateDenoiserGuides(denoiserGuides, commonParams.w * commonParams.h);

    short2* reuseTexture1 = allocateReuseTexture(254, 16);
    short2* reuseTexture2 = allocateReuseTexture(230, 16);
    short2* reuseTexture3 = allocateReuseTexture(210, 16);

    ShiftResultBuffer shiftResultBuffer1;
    ShiftResultBuffer shiftResultBuffer2;
    ShiftResultBuffer shiftResultBuffer3;
    
    void* sr_bufferMemory_1 = allocateShiftResultBuffer(shiftResultBuffer1, commonParams.w * commonParams.h);
    void* sr_bufferMemory_2 = allocateShiftResultBuffer(shiftResultBuffer2, commonParams.w * commonParams.h);
    void* sr_bufferMemory_3 = allocateShiftResultBuffer(shiftResultBuffer3, commonParams.w * commonParams.h);
    

    PipelineParams allParams = {};
    allParams.common = commonParams;

    RestirCommonParams restirParams = {};
    restirParams.reservoir = reservoir1;
    restirParams.lastFrameReservoir = reservoir2;
    restirParams.gbuffer = gbuffer1;
    restirParams.prevGbuffer = gbuffer2;
    restirParams.reuseTextures[0] = reuseTexture1;
    restirParams.reuseTextures[1] = reuseTexture2;
    restirParams.reuseTextures[2] = reuseTexture3;
    restirParams.reuseTextureSizes[0] = 254;
    restirParams.reuseTextureSizes[1] = 230;
    restirParams.reuseTextureSizes[2] = 210;
    restirParams.shiftResultBuffer[0] = shiftResultBuffer1;
    restirParams.shiftResultBuffer[1] = shiftResultBuffer2;
    restirParams.shiftResultBuffer[2] = shiftResultBuffer3;
    restirParams.denoiserGuides = denoiserGuides;

    allParams.restir = restirParams;

#if VALIDATE_REUSE_TEXTURES == 1
    validateReuseTextures(
        restirParams.reuseTextures,
        restirParams.reuseTextureSizes,
        NUM_REUSE_TEXTURES,
        commonParams.w, commonParams.h,
        8,   // frames to sweep (covers all transpose/flip combinations)
        16   // n_sigma passed to allocateReuseTexture above
    );
#endif

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));

    CUstream stream;
    cudaStreamCreate(&stream);

#if USE_DENOISER == 1
    // ---- OptiX TEMPORAL denoiser (albedo + geometric-normal guides + flow) ----
    // NOTE: OptiX has drifted a couple of these fields across versions (e.g.
    // denoiseAlpha lives in OptixDenoiserOptions on 8/9, and was in
    // OptixDenoiserParams on 7.x). If a field errors, that's the culprit; flip
    // USE_DENOISER to 0 to fall straight back to the raw path.
    OptixDenoiser denoiser = nullptr;
    {
        OptixDenoiserOptions dopt = {};
        dopt.guideAlbedo = 1;
        dopt.guideNormal = 1;
        optixDenoiserCreate(engineState.context, OPTIX_DENOISER_MODEL_KIND_TEMPORAL, &dopt, &denoiser);
    }

    OptixDenoiserSizes dsizes = {};
    optixDenoiserComputeMemoryResources(denoiser, commonParams.w, commonParams.h, &dsizes);

    CUdeviceptr d_denoiserState = 0, d_denoiserScratch = 0, d_hdrIntensity = 0;
    cudaMalloc(reinterpret_cast<void**>(&d_denoiserState),   dsizes.stateSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_denoiserScratch), dsizes.withoutOverlapScratchSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_hdrIntensity),    sizeof(float));

    optixDenoiserSetup(denoiser, stream,
        commonParams.w, commonParams.h,
        d_denoiserState,   dsizes.stateSizeInBytes,
        d_denoiserScratch, dsizes.withoutOverlapScratchSizeInBytes);

    // Ping-pong denoised outputs: this frame's output is next frame's previousOutput.
    float4* d_denoiseOut  = nullptr;
    float4* d_denoisePrev = nullptr;
    cudaMalloc(&d_denoiseOut,  commonParams.w * commonParams.h * sizeof(float4));
    cudaMalloc(&d_denoisePrev, commonParams.w * commonParams.h * sizeof(float4));

    // Color/albedo/normal are tightly-packed float4; flow is tightly-packed float2.
    auto makeDenoiserImage = [&](CUdeviceptr ptr) -> OptixImage2D {
        OptixImage2D img = {};
        img.data              = ptr;
        img.width             = commonParams.w;
        img.height            = commonParams.h;
        img.rowStrideInBytes  = commonParams.w * sizeof(float4);
        img.pixelStrideInBytes= sizeof(float4);
        img.format            = OPTIX_PIXEL_FORMAT_FLOAT4;
        return img;
    };
    auto makeFlowImage = [&](CUdeviceptr ptr) -> OptixImage2D {
        OptixImage2D img = {};
        img.data              = ptr;
        img.width             = commonParams.w;
        img.height            = commonParams.h;
        img.rowStrideInBytes  = commonParams.w * sizeof(float2);
        img.pixelStrideInBytes= sizeof(float2);
        img.format            = OPTIX_PIXEL_FORMAT_FLOAT2;
        return img;
    };
#endif

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if CAMERA_MOVES == 0 
    TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.0f, 90.0f, 0.0f);
#else
    //TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.36f, 90.0f, 0.0f);
    //OrbitCameraAnimation animation = OrbitCameraAnimation(f3(0.0f, 1.0f, 0.0f), 2.5f, -0.5f, 90.0f, -20.0f);
#endif
    //LinearCameraAnimation animation = LinearCameraAnimation(commonParams.camera.cameraOrigin, f3(commonParams.camera.xRot, commonParams.camera.yRot, commonParams.camera.zRot), f3(0.00f, 0.02f, 0.0f) ,f3());
    

    float3 startOrigin = make_float3(commonParams.camera.cameraOrigin.x, 1.59222f, commonParams.camera.cameraOrigin.z);
    float3 startRotation = make_float3(32.0f, commonParams.camera.yRot, commonParams.camera.zRot);

    // Calculated per-frame deltas (for 1000 frames)
    float3 posDelta = make_float3(0.00f, 0.0129624f, 0.00f);
    float3 rotDelta = make_float3(-0.001118f, 0.00f, 0.00f);

    
    LinearCameraAnimation animation = LinearCameraAnimation(
        startOrigin,
        startRotation,
        posDelta,
        rotDelta
    );

    animation.update(allParams.common.camera, 0);

    dim3 blockSize(32, 8);  
    dim3 gridSize((commonParams.w+31)/32, (commonParams.h+7)/8);
    
    Image image = Image(commonParams.w, commonParams.h);
#if DEBUG_VISUALIZE_TYPE == 1
    image.postProcess = false;
#endif
    // Post-processing (exposure/tonemap/gamma) now happens in the
    // cleanFormatAndPostProcessImage kernel below rather than on the host.
    // Capture the decision once and disable Image::postProcess so
    // saveImageBMP() doesn't re-apply it on top of the already-processed data.
    const bool gpuPostProcess = image.postProcess;
    image.postProcess = false;
    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(d_finalOutput, 0, commonParams.w * commonParams.h * sizeof(float4));

    float4* d_overlay;
    cudaMalloc(&d_overlay, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(d_overlay, 0, commonParams.w * commonParams.h * sizeof(float4));
    float4* host_colors = new float4[commonParams.w * commonParams.h];

    uint8_t* d_duplication_map;
    cudaMalloc(&d_duplication_map, commonParams.w * commonParams.h * sizeof(uint8_t));
    cudaMemset(d_duplication_map, 0, commonParams.w * commonParams.h * sizeof(uint8_t));

    allParams.restir.duplication_map = d_duplication_map;
    allParams.common.overlay_buffer = d_overlay;

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

#if EQUAL_TIME_COMPARE == 1
    // --- Equal-time unidirectional PT comparison state ---------------------
    // Separate accumulation so the PT pass never touches ReSTIR's buffers.
    float4* d_pt_accum;
    float4* d_pt_final;
    cudaMalloc(&d_pt_accum, commonParams.w * commonParams.h * sizeof(float4));
    cudaMalloc(&d_pt_final, commonParams.w * commonParams.h * sizeof(float4));

    CUdeviceptr d_pt_params;
    cudaMalloc(reinterpret_cast<void**>(&d_pt_params), sizeof(PipelineParams));

    Image ptImage = Image(commonParams.w, commonParams.h);
    // Post-processing now happens in the cleanFormatAndPostProcessImage kernel
    // below rather than on the host; disable Image::postProcess so
    // saveImageBMP() doesn't re-apply it on top of the already-processed data.
    ptImage.postProcess = false;

    // Pinned ring of per-sample launch params (identical except the RNG seed).
    // Pinned memory + distinct slots let the PT loop fire its sample launches as
    // one un-synchronised burst with no host-buffer aliasing hazard. 64 distinct
    // seeds per frame is ample; beyond that seeds repeat (harmless).
    const int PT_SEED_RING = 64;
    PipelineParams* h_ptRing = nullptr;
    cudaMallocHost(&h_ptRing, PT_SEED_RING * sizeof(PipelineParams));

    // frameStart/frameStop bracket ONLY the ReSTIR render kernels; ptStart/ptMid
    // time the one calibration sample; ptStart/ptNow time the whole PT burst.
    cudaEvent_t frameStart, frameStop, ptStart, ptMid, ptNow;
    cudaEventCreate(&frameStart);
    cudaEventCreate(&frameStop);
    cudaEventCreate(&ptStart);
    cudaEventCreate(&ptMid);
    cudaEventCreate(&ptNow);

    std::filesystem::create_directories(ASSET_PATH("renders/unidirectional"));
#endif
    {
        uint32_t numPix = ((commonParams.w * commonParams.h) + 31) & ~31;
        cudaMemsetAsync(r1Memory,  0, (size_t)numPix * RESERVOIR_SIZE, stream);
        cudaMemsetAsync(r2Memory,  0, (size_t)numPix * RESERVOIR_SIZE, stream);
        cudaMemsetAsync(gb1Memory, 0, (size_t)numPix * GBUFFER_SIZE,   stream);
        cudaMemsetAsync(gb2Memory, 0, (size_t)numPix * GBUFFER_SIZE,   stream);
        cudaMemsetAsync(dgMemory,  0, (size_t)numPix * DENOISER_GUIDES_SIZE, stream);
        allParams.restir.lastFrameCamera = allParams.common.camera;
    }

    cudaEventRecord(start, stream);

    for (uint32_t frame = 0; frame < frameCount; frame++) {
        allParams.common.frame_index = frame;
        cudaMemcpyAsync(
            reinterpret_cast<void*>(d_params), 
            &allParams, 
            sizeof(PipelineParams), 
            cudaMemcpyHostToDevice, 
            stream
        );

        // Generate candidates, fill allParams.restir.reservoir

#if EQUAL_TIME_COMPARE == 1
        cudaEventRecord(frameStart, stream);
#endif

        optixLaunch(
            engineState.pipeline,
            stream,
            d_params,
            sizeof(PipelineParams),
            &engineState.sbt_restirCandidate,
            commonParams.w,                   // Launch X
            commonParams.h,                   // Launch Y
            1                       // Launch Z
        );

        computeDualMV<<<gridSize, blockSize, 0, stream>>>(
            allParams.restir.gbuffer, 
            commonParams.w, 
            commonParams.h
        );

#if USE_DUPLICATION_MAP
        computeDuplicationMapKernel<<<gridSize, blockSize, 0, stream>>>(
            allParams.restir.lastFrameReservoir,
            allParams.restir.duplication_map,
            commonParams.w,
            commonParams.h
        );
#endif
        

        if (frame > 0) {
            optixLaunch(
                engineState.pipeline,
                stream,
                d_params,
                sizeof(PipelineParams),
                &engineState.sbt_restirTemporal,
                commonParams.w,                   // Launch X
                commonParams.h,                   // Launch Y
                1                       // Launch Z
            );
        }

#if DO_SPATIAL_SHIFT == 1 
{
        optixLaunch(
            engineState.pipeline,
            stream,
            d_params,
            sizeof(PipelineParams), 
            &engineState.sbt_restirSpatial,                  
            commonParams.w,                   // Launch X
            commonParams.h,                   // Launch Y
            NUM_REUSE_TEXTURES                       // Launch Z
        );

        resolveSpatialReuse<<<gridSize, blockSize, 0, stream>>>(
            allParams
        );

        Reservoir temp = allParams.restir.lastFrameReservoir;
        allParams.restir.lastFrameReservoir = allParams.restir.reservoir;
        allParams.restir.reservoir = temp;
}
#else
        // Spatial disabled -> the resolve (and its shading section) never runs,
        // so fall back to the standalone display kernel. Note this path still
        // shades with F*W, so it retains the color noise the vector-weight
        // shading in resolveSpatialReuse is there to fix.
        displayWinningReservoirs<<<gridSize, blockSize, 0, stream>>>(allParams);
#endif

#if EQUAL_TIME_COMPARE == 1
        // End of the ReSTIR render work for this frame. Everything after this
        // (BMP save, host copy) is CPU/IO and is deliberately excluded from the
        // time budget handed to the PT below.
        cudaEventRecord(frameStop, stream);
#endif

#if EQUAL_TIME_COMPARE == 1
        // ---------------------------------------------------------------------
        // Equal-time unidirectional PT pass. Runs back-to-back with the ReSTIR
        // render -- NO host work (save/copy) in between, so the GPU stays boosted
        // and the timing is taken hot, not from an idle P-state. Uses THIS
        // frame's camera (animation.update hasn't advanced it yet) and fills the
        // ReSTIR frame's GPU-time budget with samples launched as un-synced
        // bursts (a per-sample sync would idle the GPU and inflate every number).
        // ---------------------------------------------------------------------
        cudaEventSynchronize(frameStop);
        float restirMs = 0.0f;
        cudaEventElapsedTime(&restirMs, frameStart, frameStop);

        // Fresh accumulation each video frame (no cross-frame accumulation;
        // matches ReSTIR's per-frame reconstruction). Sequential single-sample
        // launches accumulate with += safely -- they serialise on the stream.
        cudaMemsetAsync(d_pt_accum, 0, commonParams.w * commonParams.h * sizeof(float4), stream);

        // Refill the pinned seed ring for this frame: same params, one distinct
        // frame_index (RNG seed) per slot, pointed at the PT accumulation buffer.
        for (int i = 0; i < PT_SEED_RING; i++) {
            h_ptRing[i] = allParams;
            h_ptRing[i].common.accum_buffer = d_pt_accum;
            h_ptRing[i].common.frame_index  = frame * PT_SEED_RING + i;
        }

        auto ptLaunch = [&](int s) {
            cudaMemcpyAsync(reinterpret_cast<void*>(d_pt_params),
                            &h_ptRing[s % PT_SEED_RING],
                            sizeof(PipelineParams), cudaMemcpyHostToDevice, stream);
            optixLaunch(engineState.pipeline, stream, d_pt_params, sizeof(PipelineParams),
                        &engineState.sbt_unidirectional,
                        commonParams.w, commonParams.h, 1);
        };

        // Phase 1: one calibration sample to estimate per-sample GPU cost.
        cudaEventRecord(ptStart, stream);
        ptLaunch(0);
        cudaEventRecord(ptMid, stream);
        cudaEventSynchronize(ptMid);
        float calMs = 0.0f;
        cudaEventElapsedTime(&calMs, ptStart, ptMid);

        // Samples that fit in the ReSTIR budget (at least the one already fired).
        int ptSamples = (calMs > 0.0f) ? (int)(restirMs / calMs) : 1;
        if (ptSamples < 1) ptSamples = 1;

        // Phase 2: launch the remainder as one un-synced burst, then sync once.
        for (int s = 1; s < ptSamples; s++) ptLaunch(s);
        cudaEventRecord(ptNow, stream);
        cudaEventSynchronize(ptNow);
        float ptElapsed = 0.0f;
        cudaEventElapsedTime(&ptElapsed, ptStart, ptNow);

        // Divisor is currentSampleCount + 1, so pass ptSamples - 1 to divide by ptSamples.
        cleanFormatAndPostProcessImage<<<gridSize, blockSize, 0, stream>>>(
            d_pt_accum, nullptr, d_pt_final, commonParams.w, commonParams.h, ptSamples - 1,
            2.0f, true
        );
        cudaMemcpyAsync(host_colors, d_pt_final,
                        commonParams.w * commonParams.h * sizeof(float4),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        #pragma omp parallel for
        for (int i = 0; i < commonParams.w * commonParams.h; i++) {
            ptImage.setColor(i % commonParams.w, i / commonParams.w, host_colors[i]);
        }
        std::stringstream ptss;
        ptss << "renders/unidirectional/render" << std::setfill('0') << std::setw(4) << frame << ".bmp";
        ptImage.saveImageBMP(ASSET_PATH(ptss.str()));

        printf("frame %4u | ReSTIR %7.3f ms | PT %4d spp (%.3f ms)\n",
               frame, restirMs, ptSamples, ptElapsed);
        fflush(stdout);
#endif

#if SAVE_SEQUENCE == 1

        // Always normalize to linear HDR here — tonemapping (if any) happens
        // further down, after denoising, so the denoiser never sees
        // gamma-corrected/tonemapped data.
#if ACCUMULATE_FRAMES == 1
        cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
            allParams.common.accum_buffer, allParams.common.overlay_buffer, d_finalOutput, commonParams.w, commonParams.h, frame
        );
#else
        cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
            allParams.common.accum_buffer, allParams.common.overlay_buffer, d_finalOutput, commonParams.w, commonParams.h, 0
        );
#endif

#if USE_DENOISER == 1
        // Denoise the reconstructed linear-HDR frame (d_finalOutput) before it goes
        // to the host for tone-mapping/save. Guides are this frame's primary-hit
        // albedo + geometric normal; color input is normalized linear HDR.
        {
            OptixImage2D colorIn = makeDenoiserImage(reinterpret_cast<CUdeviceptr>(d_finalOutput));

            OptixDenoiserGuideLayer guideLayer = {};
            guideLayer.albedo = makeDenoiserImage(reinterpret_cast<CUdeviceptr>(allParams.restir.denoiserGuides.albedo));
            guideLayer.normal = makeDenoiserImage(reinterpret_cast<CUdeviceptr>(allParams.restir.denoiserGuides.normal));
            guideLayer.flow   = makeFlowImage    (reinterpret_cast<CUdeviceptr>(allParams.restir.denoiserGuides.flow));

            OptixDenoiserLayer layer = {};
            layer.input          = colorIn;
            layer.previousOutput = makeDenoiserImage(reinterpret_cast<CUdeviceptr>(d_denoisePrev));
            layer.output         = makeDenoiserImage(reinterpret_cast<CUdeviceptr>(d_denoiseOut));

            optixDenoiserComputeIntensity(denoiser, stream, &colorIn, d_hdrIntensity,
                d_denoiserScratch, dsizes.withoutOverlapScratchSizeInBytes);

            OptixDenoiserParams dprm = {};
            dprm.hdrIntensity = d_hdrIntensity;
            dprm.blendFactor  = 0.0f;
            dprm.temporalModeUsePreviousLayers = (frame == 0) ? 0u : 1u; // no history on frame 0

            optixDenoiserInvoke(denoiser, stream, &dprm,
                d_denoiserState, dsizes.stateSizeInBytes,
                &guideLayer, &layer, 1, 0, 0,
                d_denoiserScratch, dsizes.withoutOverlapScratchSizeInBytes);
        }
        { float4* tmp = d_denoiseOut; d_denoiseOut = d_denoisePrev; d_denoisePrev = tmp; } // this frame's output -> next frame's history
        // Tonemap/gamma only now, after denoising, on the linear denoised
        // result (now in d_denoisePrev post-swap). Write into d_finalOutput
        // (no longer needed as linear data) rather than in place, so the
        // buffer kept for next frame's temporal history stays linear.
        if (gpuPostProcess) {
            postProcessOnly<<<gridSize, blockSize, 0, stream>>>(
                d_denoisePrev, d_finalOutput, commonParams.w, commonParams.h, 2.0f, true
            );
            cudaMemcpyAsync(host_colors, d_finalOutput, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(host_colors, d_denoisePrev, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
        }
#else
        if (gpuPostProcess) {
            postProcessOnly<<<gridSize, blockSize, 0, stream>>>(
                d_finalOutput, d_finalOutput, commonParams.w, commonParams.h, 2.0f, true
            );
        }
        cudaMemcpyAsync(host_colors, d_finalOutput, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
#endif
        cudaStreamSynchronize(stream);

        #pragma omp parallel for
        for (int i = 0; i < commonParams.w * commonParams.h; i++) {
            int x = i % commonParams.w;
            int y = i / commonParams.w;
            image.setColor(x, y, host_colors[i]);
        }
        std::stringstream ss;
        
#if SAVE_FOR_VIDEO == 1
    #if DEBUG_VISUALIZE_TYPE == 1
        ss << "renders/restirDebug/render" << std::setfill('0') << std::setw(4) << frame << ".bmp";
    #elif DEBUG_VISUALIZE_TYPE == 0
        ss << "renders/restir/sponza/rende" << std::setfill('0') << std::setw(4) << frame << ".bmp";
    #endif  
#endif

        std::string filename = ASSET_PATH(ss.str());
        
        std::string filename2 = ASSET_PATH("renders/restir/render.bmp");
        image.saveImageBMP(filename);
        image.saveImageBMP(filename2);
        cudaMemsetAsync(d_overlay, 0, commonParams.w * commonParams.h * sizeof(float4), stream);
#endif

        // Swap reservoirs (changes baked into vram at start of loop)
        Reservoir temp = allParams.restir.lastFrameReservoir;
        allParams.restir.lastFrameReservoir = allParams.restir.reservoir;
        allParams.restir.reservoir = temp;
        
        GBuffer tempGB = allParams.restir.prevGbuffer;
        allParams.restir.prevGbuffer = allParams.restir.gbuffer;
        allParams.restir.gbuffer = tempGB;

        allParams.restir.lastFrameCamera = allParams.common.camera;
#if CAMERA_MOVES != 0
        animation.update(allParams.common.camera, frame + 1);
#endif

        /* 
        cudaMemsetAsync(allParams.restir.reservoir.F, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.W, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.initRandomSeed, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.pathFlags, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.rcVertexGeometry, 0, commonParams.w * commonParams.h * sizeof(float4), stream);
        cudaMemsetAsync(allParams.restir.reservoir.rcVertexRadiance, 0, commonParams.w * commonParams.h * sizeof(uint32_t), stream);
        cudaMemsetAsync(allParams.restir.reservoir.cachedJacobian, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.cachedNeePdf, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        */


        cudaStreamSynchronize(stream);
    }



    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "ReSTIR PT took: " << milliseconds/frameCount 
        << " ms per frame, or " << 1.0f / (milliseconds * 0.001f/frameCount) << " frames per second."<< std::endl;



    cudaFree(reinterpret_cast<void*>(d_params));
#if EQUAL_TIME_COMPARE == 1
    cudaFree(d_pt_accum);
    cudaFree(d_pt_final);
    cudaFree(reinterpret_cast<void*>(d_pt_params));
    cudaFreeHost(h_ptRing);
    cudaEventDestroy(frameStart);
    cudaEventDestroy(frameStop);
    cudaEventDestroy(ptStart);
    cudaEventDestroy(ptMid);
    cudaEventDestroy(ptNow);
#endif
    cudaFree(r1Memory);
    cudaFree(r2Memory);
#if USE_DENOISER == 1
    optixDenoiserDestroy(denoiser);
    cudaFree(reinterpret_cast<void*>(d_denoiserState));
    cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
    cudaFree(reinterpret_cast<void*>(d_hdrIntensity));
    cudaFree(d_denoiseOut);
    cudaFree(d_denoisePrev);
#endif
    cudaFree(gb1Memory);
    cudaFree(gb2Memory);
    cudaFree(dgMemory);
    cudaFree(d_finalOutput);
    cudaFree(d_overlay);
    cudaFree(d_duplication_map);
    cudaFree(reuseTexture1);
    cudaFree(reuseTexture2);
    cudaFree(reuseTexture3);
    cudaFree(sr_bufferMemory_1);
    cudaFree(sr_bufferMemory_2);
    cudaFree(sr_bufferMemory_3);
}



