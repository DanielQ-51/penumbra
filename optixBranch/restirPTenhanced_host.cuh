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
#include "configParser.cuh" // RenderConfig, threaded into launch_restir for host-side post-process (exposure)
#include "animation.cuh"
#include "restirPTenhanced_kernels.cuh"
#include "restirPTenhanced_spatialReuseTextures.cuh"
#include "settings.cuh"
#include <filesystem>

// ---------------------------------------------------------------------------
// Always-present ReSTIR device state. This is the buffer set both launch_restir
// and the profiler (launchProfile) need. It is a plain pointer bundle -- no
// methods -- holding the raw allocation handles for teardown plus d_params.
// The Reservoir/GBuffer/ShiftResultBuffer/DenoiserGuides structs themselves
// live in allParams.restir; only their backing memory handles live here.
// The launcher-specific buffers (OptiX denoiser, equal-time PT) are NOT here --
// they stay inline in launch_restir behind their #if gates.
// ---------------------------------------------------------------------------
struct RestirState {
    void* r1Memory  = nullptr;
    void* r2Memory  = nullptr;
    void* gb1Memory = nullptr;
    void* gb2Memory = nullptr;
    void* dgMemory  = nullptr;
    void* sr1Memory = nullptr;
    void* sr2Memory = nullptr;
    void* sr3Memory = nullptr;
    void* temporalFwdMemory = nullptr;
    short2* reuseTexture1 = nullptr;
    short2* reuseTexture2 = nullptr;
    short2* reuseTexture3 = nullptr;
    float4*  d_finalOutput     = nullptr;
    float4*  d_overlay         = nullptr;
    uint8_t* d_duplication_map = nullptr;
    CUdeviceptr d_params = 0;
};

// Allocate the ReSTIR buffer set and wire it into allParams.restir (+ the
// overlay/dup pointers on allParams.common). allParams.common must already be
// populated by the caller. Extracted verbatim from launch_restir's original
// inline setup so both the launcher and the profiler share one allocation path.
__host__ void setupRestirState(const CommonParams& commonParams, RestirState& s, PipelineParams& allParams) {
    Reservoir reservoir1, reservoir2;
    GBuffer   gbuffer1, gbuffer2;

    s.r1Memory  = allocateReservoir(reservoir1, commonParams.w * commonParams.h);
    s.r2Memory  = allocateReservoir(reservoir2, commonParams.w * commonParams.h);
    s.gb1Memory = allocateGBuffer(gbuffer1, commonParams.w * commonParams.h);
    s.gb2Memory = allocateGBuffer(gbuffer2, commonParams.w * commonParams.h);

    DenoiserGuides denoiserGuides;
    s.dgMemory = allocateDenoiserGuides(denoiserGuides, commonParams.w * commonParams.h);

    s.reuseTexture1 = allocateReuseTexture(254, 16);
    s.reuseTexture2 = allocateReuseTexture(230, 16);
    s.reuseTexture3 = allocateReuseTexture(210, 16);

    ShiftResultBuffer shiftResultBuffer1, shiftResultBuffer2, shiftResultBuffer3;
    s.sr1Memory = allocateShiftResultBuffer(shiftResultBuffer1, commonParams.w * commonParams.h);
    s.sr2Memory = allocateShiftResultBuffer(shiftResultBuffer2, commonParams.w * commonParams.h);
    s.sr3Memory = allocateShiftResultBuffer(shiftResultBuffer3, commonParams.w * commonParams.h);

    TemporalFwdBuffer temporalFwdBuffer;
    s.temporalFwdMemory = allocateTemporalFwdBuffer(temporalFwdBuffer, commonParams.w * commonParams.h);

    RestirCommonParams restirParams = {};
    restirParams.reservoir          = reservoir1;
    restirParams.lastFrameReservoir = reservoir2;
    restirParams.gbuffer            = gbuffer1;
    restirParams.prevGbuffer        = gbuffer2;
    restirParams.reuseTextures[0]   = s.reuseTexture1;
    restirParams.reuseTextures[1]   = s.reuseTexture2;
    restirParams.reuseTextures[2]   = s.reuseTexture3;
    restirParams.reuseTextureSizes[0] = 254;
    restirParams.reuseTextureSizes[1] = 230;
    restirParams.reuseTextureSizes[2] = 210;
    restirParams.shiftResultBuffer[0] = shiftResultBuffer1;
    restirParams.shiftResultBuffer[1] = shiftResultBuffer2;
    restirParams.shiftResultBuffer[2] = shiftResultBuffer3;
    restirParams.temporalFwd          = temporalFwdBuffer;
    restirParams.denoiserGuides       = denoiserGuides;

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

    cudaMalloc(reinterpret_cast<void**>(&s.d_params), sizeof(PipelineParams));

    cudaMalloc(&s.d_finalOutput, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(s.d_finalOutput, 0, commonParams.w * commonParams.h * sizeof(float4));
    cudaMalloc(&s.d_overlay, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(s.d_overlay, 0, commonParams.w * commonParams.h * sizeof(float4));
    cudaMalloc(&s.d_duplication_map, commonParams.w * commonParams.h * sizeof(uint8_t));
    cudaMemset(s.d_duplication_map, 0, commonParams.w * commonParams.h * sizeof(uint8_t));

    allParams.restir.duplication_map = s.d_duplication_map;
    allParams.common.overlay_buffer  = s.d_overlay;
}

__host__ void freeRestirState(RestirState& s) {
    cudaFree(reinterpret_cast<void*>(s.d_params));
    cudaFree(s.r1Memory);
    cudaFree(s.r2Memory);
    cudaFree(s.gb1Memory);
    cudaFree(s.gb2Memory);
    cudaFree(s.dgMemory);
    cudaFree(s.d_finalOutput);
    cudaFree(s.d_overlay);
    cudaFree(s.d_duplication_map);
    cudaFree(s.reuseTexture1);
    cudaFree(s.reuseTexture2);
    cudaFree(s.reuseTexture3);
    cudaFree(s.sr1Memory);
    cudaFree(s.temporalFwdMemory);
    cudaFree(s.sr2Memory);
    cudaFree(s.sr3Memory);
}

// ---------------------------------------------------------------------------
// One ReSTIR frame's GPU work + temporal-history advancement -- exactly the
// body of launch_restir's per-frame loop between recording frameStart and
// frameStop, plus the loop-bottom ping-pong swaps. Caller responsibilities
// (kept OUT of here so timing/output are unchanged): set frame_index +
// debugVersion, upload allParams -> d_params BEFORE calling, bracket the call
// with the timing events, and advance the camera afterward.
//
// allParams is taken by reference because the history ping-pong swaps mutate
// its reservoir/gbuffer pointers and must persist to the next frame.
//
// NOTE: the loop-bottom swaps now run here (before the equal-time PT pass in
// launch_restir) instead of after it. That is behavior-neutral: the PT pass
// reads only allParams.common.camera (not yet animated) and its own private
// accumulation buffer, and never touches the reservoir/gbuffer pointers.
// ---------------------------------------------------------------------------
__host__ void renderFrameRestir(
    OptixEngineState& engineState,
    PipelineParams&   allParams,
    CUdeviceptr       d_params,
    dim3 gridSize, dim3 blockSize,
    uint32_t frame,
    CUstream stream)
{
    const uint32_t w = allParams.common.w;
    const uint32_t h = allParams.common.h;

    // 1) Candidate generation -> fills allParams.restir.reservoir
    optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
                &engineState.sbt_restirCandidate, w, h, 1);

    computeDualMV<<<gridSize, blockSize, 0, stream>>>(allParams.restir.gbuffer, allParams.restir.prevGbuffer, w, h);

#if USE_DUPLICATION_MAP
    computeDuplicationMapKernel<<<gridSize, blockSize, 0, stream>>>(
        allParams.restir.lastFrameReservoir, allParams.restir.duplication_map, w, h);
#endif

    // 2) Temporal reuse (skipped on frame 0 -- no history yet). Two-launch split: forward
    // shift, then backward shift + MIS + resolve. Each launch inlines evaluateHybridShift
    // ONCE vs twice single-launch -> ~5% on sponza. The legacy single-launch kernel
    // (__raygen__restirTemporalReuse) and its SBT entry (sbt_restirTemporal) are kept but no
    // longer launched.
    if (frame > 0) {
        optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
                    &engineState.sbt_restirTemporalFwd, w, h, 1);
        optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
                    &engineState.sbt_restirTemporalBwd, w, h, 1);
    }

#if DO_SPATIAL_SHIFT == 1
    {
        // 3) Spatial reuse: launch Z = NUM_REUSE_TEXTURES
        optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
                    &engineState.sbt_restirSpatial, w, h, NUM_REUSE_TEXTURES);

        resolveSpatialReuse<<<gridSize, blockSize, 0, stream>>>(allParams);

        Reservoir temp = allParams.restir.lastFrameReservoir;
        allParams.restir.lastFrameReservoir = allParams.restir.reservoir;
        allParams.restir.reservoir = temp;
    }
#else
    // Spatial disabled -> the resolve (and its shading section) never runs, so
    // fall back to the standalone display kernel.
    displayWinningReservoirs<<<gridSize, blockSize, 0, stream>>>(allParams);
#endif

    // ---- temporal-history advancement (was the bottom of the frame loop) ----
    Reservoir temp = allParams.restir.lastFrameReservoir;
    allParams.restir.lastFrameReservoir = allParams.restir.reservoir;
    allParams.restir.reservoir = temp;

    GBuffer tempGB = allParams.restir.prevGbuffer;
    allParams.restir.prevGbuffer = allParams.restir.gbuffer;
    allParams.restir.gbuffer = tempGB;

    allParams.restir.lastFrameCamera = allParams.common.camera;
}

__host__ void launch_restir (
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t frameCount,
    const RenderConfig& config
) {
    commonParams.camera.antiAliasJitterDist = 0.0f;
    PipelineParams allParams = {};
    allParams.common = commonParams;

    
    RestirState state;
    setupRestirState(commonParams, state, allParams);

    void* r1Memory  = state.r1Memory;
    void* r2Memory  = state.r2Memory;
    void* gb1Memory = state.gb1Memory;
    void* gb2Memory = state.gb2Memory;
    void* dgMemory  = state.dgMemory;
    CUdeviceptr d_params = state.d_params;

    CUstream stream;
    cudaStreamCreate(&stream);

#if USE_DENOISER == 1
    // ---- OptiX TEMPORAL denoiser (albedo + geometric-normal guides + flow) ----
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
    //TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.0f, 90.0f, 0.0f);
#else
    //TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.36f, 90.0f, 0.0f);
#endif
    LinearCameraAnimation animation = LinearCameraAnimation(commonParams.camera.cameraOrigin, f3(commonParams.camera.xRot, commonParams.camera.yRot, commonParams.camera.zRot), f3(0, 0.00f, 3.00f / 30.0f) ,f3());
    

    float3 startOrigin = make_float3(commonParams.camera.cameraOrigin.x, 1.59222f, commonParams.camera.cameraOrigin.z);
    float3 startRotation = make_float3(32.0f, commonParams.camera.yRot, commonParams.camera.zRot);

    // Calculated per-frame deltas (for 1000 frames)
    float3 posDelta = make_float3(0.00f, 0.0129624f, 0.00f);
    float3 rotDelta = make_float3(-0.001118f, 0.00f, 0.00f);

    
    LinearCameraAnimation animation1 = LinearCameraAnimation(
        startOrigin,
        startRotation,
        posDelta,
        rotDelta
    );

    //OrbitCameraAnimation animation = OrbitCameraAnimation(f3(0.0f, 1.0f, 0.0f), 2.5f, -0.5f, 90.0f, -20.0f);

    //animation.update(allParams.common.camera, 0);

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

    // Output/overlay/dup buffers are owned by setupRestirState (and already
    // wired into allParams). Alias them so the save code below is unchanged.
    float4*  d_finalOutput     = state.d_finalOutput;
    float4*  d_overlay         = state.d_overlay;
    uint8_t* d_duplication_map = state.d_duplication_map;
    float4*  host_colors       = new float4[commonParams.w * commonParams.h];

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

        // One ReSTIR frame: candidate -> dual-MV (-> dup map) -> temporal ->
        // spatial+resolve (or display), plus temporal-history ping-pong. The
        // frameStart/frameStop events below bracket exactly this call.
        renderFrameRestir(engineState, allParams, d_params, gridSize, blockSize, frame, stream);

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
            config.exposure, true
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
                d_denoisePrev, d_finalOutput, commonParams.w, commonParams.h, config.exposure, true
            );
            cudaMemcpyAsync(host_colors, d_finalOutput, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(host_colors, d_denoisePrev, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
        }
#else
        if (gpuPostProcess) {
            postProcessOnly<<<gridSize, blockSize, 0, stream>>>(
                d_finalOutput, d_finalOutput, commonParams.w, commonParams.h, config.exposure, true
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

        // Reservoir/gbuffer ping-pong + lastFrameCamera now happen inside
        // renderFrameRestir. Only the camera animation advance stays here.
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
#if USE_DENOISER == 1
    optixDenoiserDestroy(denoiser);
    cudaFree(reinterpret_cast<void*>(d_denoiserState));
    cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
    cudaFree(reinterpret_cast<void*>(d_hdrIntensity));
    cudaFree(d_denoiseOut);
    cudaFree(d_denoisePrev);
#endif
    // All always-present ReSTIR buffers (reservoirs, gbuffers, denoiser guides,
    // reuse textures, shift buffers, output/overlay/dup, d_params) are freed here.
    freeRestirState(state);
    delete[] host_colors;
}



