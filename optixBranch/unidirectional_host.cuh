#pragma once
#include <optix.h>
#include <optix_stubs.h>
#include "optixStructs.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Interleaved A/B profiler for the unidirectional path tracer.
//
// Each wave is run once per arm, back to back, with the arm order rotated every
// wave. Both arms therefore see the same clocks, the same thermal state, the
// same memory layout and the same driver state -- which is what lets you
// resolve a sub-1% difference that run-to-run comparison buries.
//
// sampleCount is PER ARM. Total launches = sampleCount * NUM_ARMS.
// The accumulator is shared across arms, so the image output of this run is not
// meaningful. Timing only.
// ---------------------------------------------------------------------------

struct SerArm {
    const char* label;
    bool        newVersion;
};

// ---- configure the arms here ----------------------------------------------
// To compare more than two variants, widen CommonParams::newVersion to a
// uint32_t mode, change the field below to match, and add rows.
static const SerArm SER_ARMS[] = {
    { "Old Version", false },
    { "New Version", true  },
};
static const int NUM_ARMS = (int)(sizeof(SER_ARMS) / sizeof(SER_ARMS[0]));
// ---------------------------------------------------------------------------

__host__ void launch_unidirectional(
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t sampleCount
) {
    if (sampleCount == 0) return;

    const uint32_t w = commonParams.w;
    const uint32_t h = commonParams.h;

    const uint32_t RING   = 64;   // launches per timed interval
    const uint32_t WARMUP = 64;   // samples discarded per arm (OptiX first-launch init, first-touch)

    PipelineParams allParams = {};
    allParams.common = commonParams;
    allParams.common.camera.antiAliasJitterDist = 1.0f;

    float* d_padSink = nullptr;
    cudaMalloc(&d_padSink, sizeof(float));
    cudaMemset(d_padSink, 0, sizeof(float));

    commonParams.padSentinel = 1e30f;
    commonParams.padSink     = d_padSink;

    CUstream stream;
    cudaStreamCreate(&stream);

    // Pinned staging ring: keeps the host from overwriting a slot whose async
    // copy may still be in flight.
    PipelineParams* h_ring = nullptr;
    cudaMallocHost(&h_ring, RING * sizeof(PipelineParams));

    // One DEVICE slot per ring entry, so a wave uploads in a single copy and
    // each launch reads its own slot. A shared d_params would force
    // memcpy(i) -> launch(i) -> memcpy(i+1) serialization.
    // cudaMalloc is 256B aligned and sizeof(T) is a multiple of alignof(T), so
    // d_ring + i*sizeof(PipelineParams) stays correctly aligned.
    CUdeviceptr d_ring;
    cudaMalloc(reinterpret_cast<void**>(&d_ring), (size_t)RING * sizeof(PipelineParams));

    cudaMemsetAsync(commonParams.accum_buffer, 0, (size_t)w * h * sizeof(float4), stream);

    cudaEvent_t wStart, wStop;
    cudaEventCreate(&wStart);
    cudaEventCreate(&wStop);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n", freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));

    std::vector<std::vector<double>> rate(NUM_ARMS);   // ms/spp per wave, per arm
    uint32_t counted = 0;
    const uint32_t warmup = (WARMUP >= sampleCount) ? 0 : WARMUP;

    uint32_t waveIndex = 0;
    for (uint32_t base = 0; base < sampleCount; base += RING, ++waveIndex) {
        const uint32_t waveCount = std::min(RING, sampleCount - base);
        double waveMs[16] = {};

        for (int k = 0; k < NUM_ARMS; ++k) {
            // Rotate arm order every wave so no arm permanently inherits
            // another's cache state.
            const int a = (k + (int)waveIndex) % NUM_ARMS;

            for (uint32_t i = 0; i < waveCount; i++) {
                h_ring[i] = allParams;
                h_ring[i].common.frame_index = base + i;                    // unique RNG seed per sample
                h_ring[i].common.newVersion  = SER_ARMS[a].newVersion;      // <-- the only arm difference
            }

            cudaMemcpyAsync(reinterpret_cast<void*>(d_ring), h_ring,
                            (size_t)waveCount * sizeof(PipelineParams),
                            cudaMemcpyHostToDevice, stream);

            // --- timed region: launches only -----------------------------
            // cudaEventElapsedTime measures wall-clock between when the events
            // execute on the stream, so anything leaving the stream idle in
            // this window is charged to the render. Sync, printf and ring
            // refill all sit outside it.
            cudaEventRecord(wStart, stream);
            for (uint32_t i = 0; i < waveCount; i++) {
                OptixResult r = optixLaunch(
                    engineState.pipeline,
                    stream,
                    d_ring + (size_t)i * sizeof(PipelineParams),
                    sizeof(PipelineParams),
                    &engineState.sbt_unidirectional,
                    w, h, 1);
                if (r != OPTIX_SUCCESS) { printf("\noptixLaunch failed: %d\n", (int)r); return; }
            }
            cudaEventRecord(wStop, stream);
            // --- end timed region ----------------------------------------

            cudaStreamSynchronize(stream);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, wStart, wStop);
            waveMs[a] = (double)ms / waveCount;
        }

        if (base >= warmup) {
            for (int a = 0; a < NUM_ARMS; ++a) rate[a].push_back(waveMs[a]);
            counted += waveCount;
        }

        printf("\r%u / %u spp per arm", base + waveCount, sampleCount);
        for (int a = 0; a < NUM_ARMS; ++a)
            printf("   %s %.4f", SER_ARMS[a].label, waveMs[a]);
        printf("%s   ", base < warmup ? "  [warmup]" : "");
        fflush(stdout);
    }

    // ---- summary ----------------------------------------------------------
    auto median_of = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    printf("\n\n%u spp per arm, %zu waves counted\n", counted, rate[0].size());

    double med[16] = {}, sd[16] = {};
    for (int a = 0; a < NUM_ARMS; ++a) {
        const std::vector<double>& v = rate[a];
        if (v.empty()) continue;

        double sum = 0.0;
        for (double x : v) sum += x;
        const double mean = sum / v.size();

        double acc = 0.0;
        for (double x : v) acc += (x - mean) * (x - mean);

        med[a] = median_of(v);
        sd[a]  = std::sqrt(acc / v.size());

        printf("  %-12s median %.4f   mean %.4f   min %.4f   max %.4f   sd %.4f ms/spp\n",
               SER_ARMS[a].label, med[a], mean,
               *std::min_element(v.begin(), v.end()),
               *std::max_element(v.begin(), v.end()), sd[a]);
    }

    // ---- paired per-wave comparison ---------------------------------------
    // The unpaired numbers above are useless for the decision: their spread is
    // dominated by thermal drift over the run, not by measurement noise, so a
    // sub-1% effect gets compared against the width of the throttling curve.
    //
    // Each wave is its own controlled experiment -- both arms ran microseconds
    // apart at the same clock and the same temperature -- so drift cancels
    // exactly inside every pair. Two independent tests on those pairs:
    //
    //   1. Paired mean difference vs its standard error. Answers "how big".
    //   2. Sign test on the per-wave winner. Answers "is it consistent", and is
    //      insensitive to magnitude, outliers and drift entirely. With ~800
    //      waves a true win rate past ~54% is conclusive even if the effect is
    //      a tenth of a percent.
    //
    // Note: arm order alternates by wave, so any "second arm runs warmer"
    // effect lands on each arm half the time. It widens the paired sd but does
    // not bias the mean.
    for (int a = 1; a < NUM_ARMS; ++a) {
        if (med[0] <= 0.0) break;
        const std::vector<double>& va = rate[a];
        const std::vector<double>& v0 = rate[0];
        if (va.size() != v0.size() || va.empty()) continue;

        std::vector<double> d;
        d.reserve(va.size());
        int wins = 0, ties = 0;
        for (size_t i = 0; i < va.size(); ++i) {
            const double diff = va[i] - v0[i];          // negative => arm a faster
            d.push_back(diff);
            if      (diff <  0.0) ++wins;
            else if (diff == 0.0) ++ties;
        }

        const double n = (double)d.size();
        double m = 0.0;
        for (double x : d) m += x;
        m /= n;
        double acc = 0.0;
        for (double x : d) acc += (x - m) * (x - m);
        const double dSd   = std::sqrt(acc / n);
        const double dSem  = dSd / std::sqrt(n);
        const double dMed  = median_of(d);
        const double pooled = std::sqrt(sd[0] * sd[0] + sd[a] * sd[a]);

        // Sign test: under "no difference" each wave is a fair coin flip.
        const double winRate = wins / n;
        const double zSign   = (winRate - 0.5) / (0.5 / std::sqrt(n));

        const bool bigEnough  = std::fabs(m)     > 3.0 * dSem;
        const bool consistent = std::fabs(zSign) > 3.0;

        printf("\n  PAIRED  %s vs %s   (%.0f wave pairs)\n",
               SER_ARMS[a].label, SER_ARMS[0].label, n);
        printf("    mean diff    %+.4f +/- %.4f ms/spp (sem)   %+.3f%%\n",
               m, dSem, 100.0 * m / med[0]);
        printf("    median diff  %+.4f ms/spp   %+.3f%%\n",
               dMed, 100.0 * dMed / med[0]);
        printf("    paired sd    %.4f ms/spp   (unpaired pooled %.4f -- %.1fx wider)\n",
               dSd, pooled, dSd > 0.0 ? pooled / dSd : 0.0);
        printf("    sign test    %s faster in %d / %.0f waves = %.1f%%  (z = %+.1f, %d ties)\n",
               SER_ARMS[a].label, wins, n, 100.0 * winRate, zSign, ties);
        printf("    verdict      %s\n",
               (bigEnough && consistent) ? "REAL - consistent and above noise"
             : (consistent)              ? "REAL but tiny - consistent sign, magnitude near noise"
             : (bigEnough)               ? "magnitude above noise but sign inconsistent - suspect an outlier"
                                         : "no detectable difference");
    }

    cudaFreeHost(h_ring);
    cudaFree(reinterpret_cast<void*>(d_ring));
    cudaEventDestroy(wStart);
    cudaEventDestroy(wStop);
    cudaStreamDestroy(stream);
}