#pragma once
#include <optix.h>
#include <optix_stubs.h>
#include "optixStructs.cuh"
#include <cstdio>
#include <algorithm>

// One unidirectional sample: seed with sampleIndex, upload allParams into the
// given device slot, and fire a single sbt_unidirectional launch that
// accumulates into allParams.common.accum_buffer. This is the modular per-frame
// unit the profiler (launchProfile / profiling.cuh) drives; launch_unidirectional
// below keeps its own pinned host-ring burst for peak throughput.
__host__ void renderFrameUnidirectional(
    OptixEngineState& engineState,
    PipelineParams&   allParams,
    CUdeviceptr       d_paramSlot,
    uint32_t          sampleIndex,
    CUstream          stream)
{
    allParams.common.frame_index = sampleIndex;   // unique RNG seed per sample
    cudaMemcpyAsync(reinterpret_cast<void*>(d_paramSlot), &allParams,
                    sizeof(PipelineParams), cudaMemcpyHostToDevice, stream);
    optixLaunch(engineState.pipeline, stream, d_paramSlot, sizeof(PipelineParams),
                &engineState.sbt_unidirectional,
                allParams.common.w, allParams.common.h, 1);
}

__host__ void launch_unidirectional(
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t sampleCount
) {
    if (sampleCount == 0) return;

    const uint32_t w = commonParams.w;
    const uint32_t h = commonParams.h;

    PipelineParams allParams = {};
    allParams.common = commonParams;
    //allParams.common.camera.antiAliasJitterDist = 1.0f;

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));

    CUstream stream;
    cudaStreamCreate(&stream);

    // Start from a clean accumulator so the buffer holds exactly the sum of the
    // samples we fire here, regardless of prior state.
    cudaMemsetAsync(commonParams.accum_buffer, 0, (size_t)w * h * sizeof(float4), stream);

    const uint32_t RING = 64;
    PipelineParams* h_ring = nullptr;
    cudaMallocHost(&h_ring, RING * sizeof(PipelineParams));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n", freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));

    cudaEventRecord(start, stream);

    for (uint32_t base = 0; base < sampleCount; base += RING) {
        uint32_t waveCount = std::min(RING, sampleCount - base);

        // Prefill this wave's slots (no launches in flight are reading these yet:
        // the previous wave was synced before we got here).
        for (uint32_t i = 0; i < waveCount; i++) {
            h_ring[i] = allParams;
            h_ring[i].common.frame_index = base + i; // unique RNG seed per sample
        }

        // Fire the wave as an un-synced burst.
        for (uint32_t i = 0; i < waveCount; i++) {
            cudaMemcpyAsync(
                reinterpret_cast<void*>(d_params),
                &h_ring[i],
                sizeof(PipelineParams),
                cudaMemcpyHostToDevice,
                stream
            );

            optixLaunch(
                engineState.pipeline,
                stream,
                d_params,
                sizeof(PipelineParams),
                &engineState.sbt_unidirectional,
                w,   // Launch X
                h,   // Launch Y
                1    // Launch Z
            );
        }

        // Wave barrier: everything above must finish reading h_ring before we
        // overwrite it for the next wave. Doubles as the progress throttle.
        cudaStreamSynchronize(stream);
        printf("\rUnidirectional: %u / %u spp", base + waveCount, sampleCount);
        fflush(stdout);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nUnidirectional PT: %u spp in %.2f ms (%.4f ms/spp)\n",
           sampleCount, ms, ms / sampleCount);

    cudaFreeHost(h_ring);
    cudaFree(reinterpret_cast<void*>(d_params));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
