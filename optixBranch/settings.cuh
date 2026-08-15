#pragma once

#ifndef USE_RESTIR_PT
#define USE_RESTIR_PT 1
#endif

// Reservoir memory layout. 0 = legacy SOA (one global buffer per field).
// 1 = AOS split, tuned for SER (which destroys cross-thread coalescing anyway):
#ifndef RESERVOIR_LAYOUT_AOS
#define RESERVOIR_LAYOUT_AOS 1
#endif

// Equal-time comparison harness. When 1, launch_restir also runs the
// unidirectional path tracer back-to-back each frame: it measures the ReSTIR
// frame's GPU time with CUDA events, then adaptively fills exactly that time
// budget with PT samples on the same camera, emitting a parallel BMP sequence
// under renders/unidirectional/. 0 = clean ReSTIR-only build (no PT code, no
// per-frame timing overhead compiled in).
#ifndef EQUAL_TIME_COMPARE
#define EQUAL_TIME_COMPARE 0
#endif

// Master switch for device side debug instrumentation.
// Host side prints (timings, memory usage) are deliberately not gated.
#ifndef DEBUG_MODE
#define DEBUG_MODE 0
#endif

// Technique-profiling suite. When 1, initRender routes to launchProfile instead
// of the normal launch_* path: it sweeps CommonParams::debugVersion across the
// PROFILE_ARMS table (profiling.cuh) for the compile-time-selected integrator
// and reports paired per-variant timing statistics. 0 = normal render.
#ifndef PROFILE_TECHNIQUES
#define PROFILE_TECHNIQUES 1
#endif

#ifndef SAVE_SEQUENCE
#define SAVE_SEQUENCE 0
#endif

#ifndef SAVE_FOR_VIDEO
#define SAVE_FOR_VIDEO 0
#endif

#ifndef ACCUMULATE_FRAMES
#define ACCUMULATE_FRAMES 0
#endif

#ifndef DEBUG_VISUALIZE_TYPE
#define DEBUG_VISUALIZE_TYPE 0
#endif

#ifndef TEMPORAL_SKIP_REVERSE_SHIFT
#define TEMPORAL_SKIP_REVERSE_SHIFT 1
#endif

#ifndef CAMERA_MOVES
#define CAMERA_MOVES 0
#endif

#ifndef LERP_MCAP
#define LERP_MCAP 300.0f
#endif

#ifndef RECON_FOOTPRINT_C_CONSTANT
#define RECON_FOOTPRINT_C_CONSTANT 0.02f
#endif

// Master enable. 0 = every texture read stays at mip 0 (original behavior); the
// cone registers still compile but getDataGeoLOD returns lod = 0.
#ifndef USE_RAY_CONES
#define USE_RAY_CONES 1
#endif

// Global mip bias. + = blurrier / cheaper / less texture aliasing; - = sharper
// (and more aliasing). The single "overall aggressiveness" dial (like mipLodBias).
#ifndef RAYCONE_LOD_BIAS
#define RAYCONE_LOD_BIAS 0.0f
#endif

// How fast the cone widens per bounce, scaled by surface roughness. Larger =
// secondary/GI bounces defocus onto coarser mips sooner (cheaper, less noise).
#ifndef RAYCONE_ROUGHNESS_SPREAD
#define RAYCONE_ROUGHNESS_SPREAD 0.40f
#endif

#ifndef DEBUG_TEST_PIXEL_X
#define DEBUG_TEST_PIXEL_X 750
#endif

#ifndef DEBUG_TEST_PIXEL_Y
#define DEBUG_TEST_PIXEL_Y (1400 - 1070)
#endif

#ifndef NUM_REUSE_TEXTURES
#define NUM_REUSE_TEXTURES 3
#endif

#ifndef DO_SPATIAL_SHIFT
#define DO_SPATIAL_SHIFT 1
#endif

// Run the OptiX denoiser (HDR model, albedo + geometric-normal guides) on the
// reconstructed linear-HDR frame before tone-mapping. 0 = save the raw noisy frame.
#ifndef USE_DENOISER
#define USE_DENOISER 0
#endif

#ifndef TEMPORAL_USE_DUAL_MV
#define TEMPORAL_USE_DUAL_MV 1
#endif

// 1 = adaptive cCap reduction via the sample duplication map (Enhanced §5).
// 0 = disable it and fall back to the original hard M-cap of Lin 2022
//     (cCap = LERP_MCAP constant). Also skips the per-frame duplication-map
//     kernel (a 17x17 neighborhood scan per pixel) on the host side.
#ifndef USE_DUPLICATION_MAP
#define USE_DUPLICATION_MAP 1
#endif

#ifndef TEMPORAL_SER_SORT_MORTON_CODE
#define TEMPORAL_SER_SORT_MORTON_CODE 0
#endif

// Runs the reuse texture self tests once at startup and prints a report.
// Costs a few ms, only meant to be on while debugging the pairing. At 0 the
// validation kernels are not compiled at all.
#ifndef VALIDATE_REUSE_TEXTURES
#define VALIDATE_REUSE_TEXTURES 0
#endif

// ---------------------------------------------------------------------------
// Debug instrumentation macros. Gated on DEBUG_MODE above.
//
// Use IS_DEBUG_PIXEL instead of comparing against DEBUG_TEST_PIXEL_X/Y by hand,
// and DEBUG_PRINTF / DEBUG_DRAWLINE / DEBUG_PRINT_PIXEL instead of calling
// printf / drawLine / printPixelData directly from device code, so a
// DEBUG_MODE 0 build carries none of it.
// ---------------------------------------------------------------------------
#if DEBUG_MODE == 1
    #define IS_DEBUG_PIXEL(px, py)  ((px) == DEBUG_TEST_PIXEL_X && (py) == DEBUG_TEST_PIXEL_Y)
    #define DEBUG_PRINTF(...)       printf(__VA_ARGS__)
    #define DEBUG_DRAWLINE(...)     drawLine(__VA_ARGS__)
    #define DEBUG_PRINT_PIXEL(...)  printPixelData(__VA_ARGS__)
#else
    #define IS_DEBUG_PIXEL(px, py)  (false)
    #define DEBUG_PRINTF(...)       ((void)0)
    #define DEBUG_DRAWLINE(...)     ((void)0)
    #define DEBUG_PRINT_PIXEL(...)  ((void)0)
#endif

#ifndef NORMAL_REJECTION_THRESHOLD
#define NORMAL_REJECTION_THRESHOLD 0.98f
#endif

#ifndef PLANAR_DIST_REJECTION_THRESHOLD
#define PLANAR_DIST_REJECTION_THRESHOLD 0.005f
#endif

#ifndef TRUE_DIST_REJECTION_THRESHOLD
#define TRUE_DIST_REJECTION_THRESHOLD 30.0f
#endif