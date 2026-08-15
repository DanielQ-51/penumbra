#pragma once
// ---------------------------------------------------------------------------
// Shared statistics for the technique-profiling suite.
//
// A "pair" is one controlled experiment in which every arm was measured
// microseconds apart under the same clock / thermal / driver state (a wave of
// launches for the unidirectional integrator, or a (round, frame) slot for
// ReSTIR). Because every arm appears once per pair, thermal drift over the run
// cancels exactly inside each pair -- which is what lets a sub-1% effect be
// resolved. reportProfileStats() consumes those paired samples and prints, per
// arm, the unpaired distribution plus, for every arm vs arm 0, a paired
// mean-difference test and a sign test.
//
// This math was generalized from the 2-arm SER A/B block that used to live
// inline in unidirectional_host.cuh; it now serves any number of arms.
// ---------------------------------------------------------------------------

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

struct ProfileArm {
    const char* label;        // human-readable name shown in the report
    uint32_t    debugVersion; // value written into CommonParams::debugVersion for this arm
};

// rate[a] = per-pair timings (e.g. ms/spp or ms/frame) for arm a. Every arm
// must have the same number of samples (they are paired by index). `unit` is a
// display string only ("ms/spp", "ms/frame", ...).
__host__ inline void reportProfileStats(
    const std::vector<ProfileArm>&           arms,
    const std::vector<std::vector<double>>&  rate,
    const char*                              unit = "ms")
{
    const int nArms = (int)arms.size();
    if (nArms == 0 || rate.empty() || rate[0].empty()) {
        printf("\n[profile] no samples collected\n");
        return;
    }

    auto median_of = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    printf("\n\n%zu paired samples per arm\n", rate[0].size());

    std::vector<double> med(nArms, 0.0), sd(nArms, 0.0);
    for (int a = 0; a < nArms; ++a) {
        const std::vector<double>& v = rate[a];
        if (v.empty()) continue;

        double sum = 0.0;
        for (double x : v) sum += x;
        const double mean = sum / v.size();

        double acc = 0.0;
        for (double x : v) acc += (x - mean) * (x - mean);

        med[a] = median_of(v);
        sd[a]  = std::sqrt(acc / v.size());

        printf("  %-14s median %.4f   mean %.4f   min %.4f   max %.4f   sd %.4f %s\n",
               arms[a].label, med[a], mean,
               *std::min_element(v.begin(), v.end()),
               *std::max_element(v.begin(), v.end()), sd[a], unit);
    }

    // ---- paired per-pair comparison, every arm vs arm 0 --------------------
    // The unpaired numbers above are dominated by thermal drift over the run,
    // not measurement noise, so a sub-1% effect gets buried under the width of
    // the throttling curve. Each pair is its own controlled experiment, so
    // drift cancels inside it. Two independent tests on the per-pair diffs:
    //   1. Paired mean difference vs its standard error  -> "how big".
    //   2. Sign test on the per-pair winner              -> "how consistent"
    //      (insensitive to magnitude, outliers and drift entirely).
    for (int a = 1; a < nArms; ++a) {
        if (med[0] <= 0.0) break;
        const std::vector<double>& va = rate[a];
        const std::vector<double>& v0 = rate[0];
        if (va.size() != v0.size() || va.empty()) continue;

        std::vector<double> d;
        d.reserve(va.size());
        int wins = 0, ties = 0;
        for (size_t i = 0; i < va.size(); ++i) {
            const double diff = va[i] - v0[i];   // negative => arm a faster
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
        const double dSd    = std::sqrt(acc / n);
        const double dSem   = dSd / std::sqrt(n);
        const double dMed   = median_of(d);
        const double pooled = std::sqrt(sd[0] * sd[0] + sd[a] * sd[a]);

        // Sign test: under "no difference" each pair is a fair coin flip.
        const double winRate = wins / n;
        const double zSign   = (winRate - 0.5) / (0.5 / std::sqrt(n));

        const bool bigEnough  = std::fabs(m)     > 3.0 * dSem;
        const bool consistent = std::fabs(zSign) > 3.0;

        printf("\n  PAIRED  %s vs %s   (%.0f pairs)\n",
               arms[a].label, arms[0].label, n);
        printf("    mean diff    %+.4f +/- %.4f %s (sem)   %+.3f%%\n",
               m, dSem, unit, 100.0 * m / med[0]);
        printf("    median diff  %+.4f %s   %+.3f%%\n",
               dMed, unit, 100.0 * dMed / med[0]);
        printf("    paired sd    %.4f %s   (unpaired pooled %.4f -- %.1fx wider)\n",
               dSd, unit, pooled, dSd > 0.0 ? pooled / dSd : 0.0);
        printf("    sign test    %s faster in %d / %.0f pairs = %.1f%%  (z = %+.1f, %d ties)\n",
               arms[a].label, wins, n, 100.0 * winRate, zSign, ties);
        printf("    verdict      %s\n",
               (bigEnough && consistent) ? "REAL - consistent and above noise"
             : (consistent)              ? "REAL but tiny - consistent sign, magnitude near noise"
             : (bigEnough)               ? "magnitude above noise but sign inconsistent - suspect an outlier"
                                         : "no detectable difference");
    }
}
