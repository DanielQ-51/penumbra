#pragma once
// ---------------------------------------------------------------------------
// Technique-profiling suite (a SEPARATE option, not a replacement for the
// launchers). Compares N optimization variants of the SAME compile-time
// integrator, where each variant is selected at runtime by
// CommonParams::debugVersion. It drives the exact same per-frame render units
// the launchers use (renderFrameRestir / renderFrameUnidirectional), so the
// profiled work is identical to a real render.
//
// Wire device-side branches yourself, e.g. in the raygen:
//     if (params.common.debugVersion == 0) { ...baseline... }
//     else if (params.common.debugVersion == 1) { ...variant... }
// then add matching rows to PROFILE_ARMS below.
//
// Enable via `#define PROFILE_TECHNIQUES 1` in settings.cuh; initRender then
// calls launchProfile instead of the normal launch_* path.
// ---------------------------------------------------------------------------

#include "profilingStats.cuh"
#include "restirPTenhanced_host.cuh"   // renderFrameRestir, RestirState, setup/free
#include "unidirectional_host.cuh"     // renderFrameUnidirectional
#include "objects.cuh"                 // OPTIX_NORMAL / OPTIX_RESTIR_PT
#include "animation.cuh"               // LinearCameraAnimation
#include "settings.cuh"
#include <vector>
#include <algorithm>

// ---- configure the technique variants here --------------------------------
// One row per variant. `debugVersion` is written into CommonParams::debugVersion
// for that arm; `label` is shown in the report. Arm 0 is the baseline every
// other arm is compared against.
static const ProfileArm PROFILE_ARMS[] = {
    { "baseline",  0 },
    { "spatial combined 2-bit reorder",  1 },
    // { "variant 2", 2 },
};
static const int NUM_PROFILE_ARMS = (int)(sizeof(PROFILE_ARMS) / sizeof(PROFILE_ARMS[0]));

__host__ void profileRestir(OptixEngineState& engineState, CommonParams commonParams,
                            uint32_t warmup, uint32_t frames)
{
    const int nArms = NUM_PROFILE_ARMS;
    if (nArms == 0 || frames == 0) { printf("[profile] nothing to do\n"); return; }
    if (warmup >= frames) warmup = 0;

    std::vector<ProfileArm> arms(PROFILE_ARMS, PROFILE_ARMS + nArms);

    commonParams.camera.antiAliasJitterDist = 1.0f;

    dim3 blockSize(32, 8);
    dim3 gridSize((commonParams.w + 31) / 32, (commonParams.h + 7) / 8);

    // One independent ReSTIR buffer set + params block per arm.
    std::vector<PipelineParams> allParams(nArms);
    std::vector<RestirState>    state(nArms);
    for (int a = 0; a < nArms; ++a) {
        allParams[a] = {};
        allParams[a].common = commonParams;
        setupRestirState(commonParams, state[a], allParams[a]);
        allParams[a].common.debugVersion = arms[a].debugVersion;
    }

    CUstream stream;
    cudaStreamCreate(&stream);

    // Same camera animation as launch_restir. It is deterministic in the frame
    // index, so all arms share one trajectory; each keeps its own camera +
    // lastFrameCamera history inside its own allParams.
    float3 startOrigin   = make_float3(commonParams.camera.cameraOrigin.x, 1.59222f, commonParams.camera.cameraOrigin.z);
    float3 startRotation = make_float3(32.0f, commonParams.camera.yRot, commonParams.camera.zRot);
    float3 posDelta      = make_float3(0.00f, 0.0129624f, 0.00f);
    float3 rotDelta      = make_float3(-0.001118f, 0.00f, 0.00f);
    LinearCameraAnimation animation(startOrigin, startRotation, posDelta, rotDelta);

    // Prime frame-0 camera + zero every arm's history (mirrors launch_restir's
    // one-time init before the loop). After this, no arm's history is ever reset.
    const uint32_t numPix = ((commonParams.w * commonParams.h) + 31) & ~31;
    for (int a = 0; a < nArms; ++a) {
        cudaMemsetAsync(state[a].r1Memory,  0, (size_t)numPix * RESERVOIR_SIZE,       stream);
        cudaMemsetAsync(state[a].r2Memory,  0, (size_t)numPix * RESERVOIR_SIZE,       stream);
        cudaMemsetAsync(state[a].gb1Memory, 0, (size_t)numPix * GBUFFER_SIZE,         stream);
        cudaMemsetAsync(state[a].gb2Memory, 0, (size_t)numPix * GBUFFER_SIZE,         stream);
        cudaMemsetAsync(state[a].dgMemory,  0, (size_t)numPix * DENOISER_GUIDES_SIZE, stream);
        animation.update(allParams[a].common.camera, 0);
        allParams[a].restir.lastFrameCamera = allParams[a].common.camera;
    }

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    std::vector<std::vector<double>> rate(nArms);
    for (auto& v : rate) v.reserve(frames);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("\n[profile] ReSTIR: %d arms x %u continuous frames (+%u warmup, stats only)\n",
           nArms, frames, warmup);
    printf("[profile] %d independent buffer sets -- Free: %.2f MB of %.2f MB\n",
           nArms, freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));

    for (uint32_t frame = 0; frame < frames; ++frame) {
        for (int k = 0; k < nArms; ++k) {
            const int a = (k + (int)frame) % nArms;   // rotate arm order per frame

            allParams[a].common.frame_index = frame;
            cudaMemcpyAsync(reinterpret_cast<void*>(state[a].d_params), &allParams[a],
                            sizeof(PipelineParams), cudaMemcpyHostToDevice, stream);

            cudaEventRecord(evStart, stream);
            renderFrameRestir(engineState, allParams[a], state[a].d_params, gridSize, blockSize, frame, stream);
            cudaEventRecord(evStop, stream);
            cudaStreamSynchronize(stream);

            if (frame >= warmup) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, evStart, evStop);
                rate[a].push_back((double)ms);
            }
#if CAMERA_MOVES != 0
            animation.update(allParams[a].common.camera, frame + 1);
#endif
        }
        printf("\r[profile] frame %u / %u", frame + 1, frames);
        fflush(stdout);
    }

    reportProfileStats(arms, rate, "ms/frame");

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaStreamDestroy(stream);
    for (int a = 0; a < nArms; ++a) freeRestirState(state[a]);
}

// ===========================================================================
// Unidirectional: per-sample wave interleave.
//
// Samples are independent (shared accumulator, distinct RNG seed), so every arm
// can be measured within the same wave microseconds apart, rotating arm order
// per wave. Total samples per arm = frames. Timings are paired by wave. This
// drives the shared renderFrameUnidirectional unit; for the highest-fidelity
// 2-arm SER comparison, launch_unidirectional's own pinned device-ring path
// remains available.
// ===========================================================================
__host__ void profileUnidirectional(OptixEngineState& engineState, CommonParams commonParams,
                                    uint32_t warmup, uint32_t frames)
{
    const int nArms = NUM_PROFILE_ARMS;
    if (nArms == 0) { printf("[profile] no arms\n"); return; }

    const uint32_t w = commonParams.w, h = commonParams.h;
    const uint32_t sampleCount = frames;
    if (sampleCount == 0) { printf("[profile] nothing to do\n"); return; }

    std::vector<ProfileArm> arms(PROFILE_ARMS, PROFILE_ARMS + nArms);

    PipelineParams allParams = {};
    allParams.common = commonParams;
    allParams.common.camera.antiAliasJitterDist = 1.0f;

    const uint32_t RING = 64;   // launches per timed interval

    CUstream stream;
    cudaStreamCreate(&stream);

    // One device slot per ring entry so each launch reads its own params.
    CUdeviceptr d_ring;
    cudaMalloc(reinterpret_cast<void**>(&d_ring), (size_t)RING * sizeof(PipelineParams));

    cudaMemsetAsync(commonParams.accum_buffer, 0, (size_t)w * h * sizeof(float4), stream);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    std::vector<std::vector<double>> rate(nArms);
    std::vector<double> waveMs(nArms, 0.0);
    const uint32_t warmupSamples = (warmup >= sampleCount) ? 0 : warmup;

    printf("\n[profile] unidirectional: %d arms, %u spp per arm (+%u warmup)\n",
           nArms, sampleCount, warmupSamples);

    uint32_t wave = 0;
    for (uint32_t base = 0; base < sampleCount; base += RING, ++wave) {
        const uint32_t waveCount = std::min(RING, sampleCount - base);

        for (int k = 0; k < nArms; ++k) {
            const int a = (k + (int)wave) % nArms;   // rotate arm order per wave
            allParams.common.debugVersion = arms[a].debugVersion;

            cudaEventRecord(evStart, stream);
            for (uint32_t i = 0; i < waveCount; ++i) {
                renderFrameUnidirectional(engineState, allParams,
                    d_ring + (size_t)i * sizeof(PipelineParams), base + i, stream);
            }
            cudaEventRecord(evStop, stream);
            cudaStreamSynchronize(stream);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            waveMs[a] = (double)ms / waveCount;
        }

        if (base >= warmupSamples)
            for (int a = 0; a < nArms; ++a) rate[a].push_back(waveMs[a]);

        printf("\r[profile] %u / %u spp per arm", base + waveCount, sampleCount);
        fflush(stdout);
    }

    reportProfileStats(arms, rate, "ms/spp");

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(reinterpret_cast<void*>(d_ring));
    cudaStreamDestroy(stream);
}

// Dispatch on the compile-time-selected integrator. Never compares one
// integrator against another -- only variants of the one that is running.
// `frames` is the full sequence length: continuous ReSTIR frames, or total
// unidirectional samples per arm. `warmup` frames/samples are rendered but
// excluded from the statistics.
__host__ void launchProfile(OptixEngineState& engineState, CommonParams commonParams,
                            int integratorChoice,
                            uint32_t warmup, uint32_t frames)
{
    if (integratorChoice == OPTIX_RESTIR_PT) {
        profileRestir(engineState, commonParams, warmup, frames);
    } else if (integratorChoice == OPTIX_NORMAL) {
        profileUnidirectional(engineState, commonParams, warmup, frames);
    } else {
        printf("[profile] integrator %d not supported by the profiler\n", integratorChoice);
    }
}
