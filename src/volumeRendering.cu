#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "volumeRendering.cuh"
#include "volumeUtils.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include "helpers.cuh"
#include "sceneContexts.cuh"
#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

__device__ __constant__ float sceneRadius;
__device__ __constant__ float3 sceneCenter;
__device__ __constant__ float3 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

using leaf_t = nanovdb::LeafNode<float>;

__device__ void buildOrthonormalBasis(const nanovdb::Vec3f& n, nanovdb::Vec3f& b1, nanovdb::Vec3f& b2) {
    // A simple and robust way to generate two orthogonal vectors.
    // If n is pointing too close to the X-axis, use the Y-axis to cross, otherwise use X-axis.
    if (abs(n[0]) > 0.9f) {
        b1 = nanovdb::Vec3f(0.0f, 1.0f, 0.0f).cross(n);
    } else {
        b1 = nanovdb::Vec3f(1.0f, 0.0f, 0.0f).cross(n);
    }
    b1.normalize();
    b2 = n.cross(b1);
    b2.normalize();
}

__device__ nanovdb::Vec3f sample_HG(const nanovdb::Vec3f& incoming_dir, float g, float u1, float u2) {
    float cos_theta;

    if (abs(g) < 1e-3f) {
        // Isotropic edge case (g is effectively 0)
        cos_theta = 1.0f - 2.0f * u2;
    } else {
        // Anisotropic Henyey-Greenstein math
        float sqrTerm = (1.0f - g * g) / (1.0f - g + 2.0f * g * u2);
        cos_theta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
    }

    float sin_theta = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    // Azimuth angle (perfectly uniform around the ray)
    float phi = 2.0f * PI * u1;

    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    // --- PHASE 2: Build the Local Vector ---
    // This vector assumes the incoming ray was pointing perfectly down the Z-axis (0, 0, 1)
    nanovdb::Vec3f local_dir(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    );

    // --- PHASE 3: Rotate to World Space ---
    // Create an Orthonormal Basis (Tangent, Bi-tangent, Normal) around the incoming ray
    nanovdb::Vec3f tangent, bitangent;
    buildOrthonormalBasis(incoming_dir, tangent, bitangent);

    // Multiply the local vector by the basis to transform it into World Space
    nanovdb::Vec3f world_dir =
        tangent * local_dir[0] +
        bitangent * local_dir[1] +
        incoming_dir * local_dir[2];

    world_dir.normalize(); // Ensure perfect unit length to prevent floating point drift

    return world_dir;
}

__device__ inline float evaluate_HG(const float3& incoming_dir, const float3& outgoing_dir, float g) {
    // Get the cosine of the angle between the two directions
    float cos_theta = dot(incoming_dir, outgoing_dir);

    // Isotropic fast-path (matches the edge case in your sample_HG)
    if (fabsf(g) < 1e-3f) {
        return 1.0f / (4.0f * PI);
    }

    // Anisotropic Henyey-Greenstein math
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;

    // Safety clamp to prevent NaN if denom somehow hits exactly 0
    denom = fmaxf(denom, 1e-7f);

    // Note: (denom * sqrtf(denom)) is computationally faster on the GPU than powf(denom, 1.5f)
    return (1.0f / (4.0f * PI)) * (1.0f - g2) / (denom * sqrtf(denom));
}

__global__ void render_volume(
    RNGState* rngStates,
    Camera camera,
    const SceneContext sceneContext,
    float4* __restrict__ colors,
    int maxDepth,
    int frameNum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = camera.generateCameraRay(localState, x, y);

    nanovdb::Ray<float> worldRay(
        nanovdb::Vec3f(r.origin.x, r.origin.y, r.origin.z),
        nanovdb::Vec3f(r.direction.x, r.direction.y, r.direction.z)
    );

    auto accessor = sceneContext.volumes->density_pointer->getAccessor();
    auto sampler = nanovdb::math::createSampler<1>(accessor);

    float3 throughput = f3(1.0f);
    float3 pixelColor = f3();
    float density_scale = 3.0f;
    // --- 1. BOUNCE LOOP ---
    for (int depth = 0; depth < maxDepth; depth++) {
        localState = load_rng(pixelIdx, frameNum, depth+1, rngStates);
        nanovdb::Ray<float> indexRay = worldRay.worldToIndexF(*sceneContext.volumes->density_pointer);
        nanovdb::math::TreeMarcher<leaf_t, nanovdb::Ray<float>, decltype(accessor)> marcher(accessor);

        if (!marcher.init(indexRay)) {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sceneContext.lightSampler.envMap.sampleDir(f3(nv_dir[0], nv_dir[1], nv_dir[2]));
            break;
        }

        const leaf_t* leaf = nullptr;
        float t0, t1;

        bool hit_particle = false;
        float t_hit = 0.0f;

        while (marcher.step(&leaf, t0, t1)) {
            float local_majorant = leaf->maximum() * density_scale;
            if (local_majorant <= 0.0f) continue;

            float t_current = t0;

            while (true) {
                float step = -log(rand(&localState)) / local_majorant;
                t_current += step;

                if (t_current >= t1) break; // Exited the node

                float true_density = sampler(indexRay(t_current)) * density_scale;

                if (rand(&localState) < (true_density / local_majorant)) {
                    hit_particle = true;
                    t_hit = t_current; // Save the exact hit distance
                    break;
                }
            }

            if (hit_particle) break;
        }

        if (hit_particle) {
            throughput *= f3(sceneContext.volumes->albedo);

            nanovdb::Vec3f index_hit_pos = indexRay(t_hit); // Use the saved t_hit
            nanovdb::Vec3f world_hit_pos = sceneContext.volumes->density_pointer->indexToWorldF(index_hit_pos);

            nanovdb::Vec3f new_world_dir = sample_HG(worldRay.dir(), 0.6f, rand(&localState), rand(&localState));

            worldRay = nanovdb::Ray<float>(world_hit_pos, new_world_dir);

        } else {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sceneContext.lightSampler.envMap.sampleDir(f3(nv_dir[0], nv_dir[1], nv_dir[2]));
            break; // Break the depth loop
        }
    }

    colors[pixelIdx] += f4(pixelColor);
    save_rng(pixelIdx, &localState, rngStates);
}

__device__ float estimate_volume_transmittance(
    const Ray& worldRay,
    const BVHContext& bvhContext,
    const VolumeInterval* volHits,
    int num_volHits,
    RNGState* localState
) {
    float global_transmittance = 1.0f;

    // Convert custom Ray to NanoVDB Ray once
    nanovdb::Vec3f world_origin(worldRay.origin.x, worldRay.origin.y, worldRay.origin.z);
    nanovdb::Vec3f world_dir(worldRay.direction.x, worldRay.direction.y, worldRay.direction.z);
    nanovdb::Ray<float> nanoWorldRay(world_origin, world_dir);

    for (int v = 0; v < num_volHits; ++v) {
        const VolumeInterval& hit = volHits[v];
        const Volume& vol = bvhContext.volumes[hit.volume_ID];

        const nanovdb::NanoGrid<float>* densityGrid = vol.density_pointer;
        float densityScale = vol.densityScale;

        auto accessor = densityGrid->getAccessor();
        auto sampler = nanovdb::math::createSampler<1>(accessor);

        // 1. Transform ray to index space
        nanovdb::Ray<float> indexRay = nanoWorldRay.worldToIndexF(*densityGrid);
        nanovdb::Vec3f index_origin = indexRay.start();

        // Calculate specific start and end points in INDEX space
        nanovdb::Vec3f index_start_pos = densityGrid->worldToIndexF(world_origin + world_dir * hit.t_min);
        nanovdb::Vec3f index_end_pos = densityGrid->worldToIndexF(world_origin + world_dir * hit.t_max);

        float index_start_t = (index_start_pos - index_origin).length();
        float index_end_t = (index_end_pos - index_origin).length();

        index_start_t = fmaxf(index_start_t, 1e-5f);
        index_end_t = fmaxf(index_end_t, index_start_t + 1e-5f); // Ensure t1 is strictly > t0

        // Apply tight bounds directly to the index ray using the correct API
        indexRay.setTimes(index_start_t, index_end_t);

        // 2. Initialize the TreeMarcher using leaf_t
        nanovdb::math::TreeMarcher<leaf_t, nanovdb::Ray<float>, decltype(accessor)> marcher(accessor);

        // If the shadow segment misses the active volume entirely, skip to the next volume
        if (!marcher.init(indexRay)) {
            continue;
        }

        const leaf_t* leaf = nullptr;
        float t0, t1;

        // 3. March through the active nodes
        while (marcher.step(&leaf, t0, t1)) {

            // Clamp the node bounds to our specific intersection interval
            float node_start_t = fmaxf(t0, index_start_t);
            float node_end_t = fminf(t1, index_end_t);

            if (node_start_t >= node_end_t) continue;

            float local_majorant = leaf->maximum() * densityScale;
            if (local_majorant <= 0.0f) continue; // Empty space

            float t_current = node_start_t;

            // 4. Ratio Tracking Loop
            while (true) {
                float step = -log(rand(localState)) / local_majorant;
                t_current += step;

                if (t_current >= node_end_t) break;

                float true_density = sampler(indexRay(t_current)) * densityScale;
                float prob_null_collision = 1.0f - (true_density / local_majorant);

                global_transmittance *= prob_null_collision;

                // 5. Russian Roulette
                if (global_transmittance < 0.1f) {
                    float termination_prob = fmaxf(0.05f, 1.0f - global_transmittance);
                    if (rand(localState) < termination_prob) {
                        return 0.0f; // Ray was absorbed
                    }
                    global_transmittance /= (1.0f - termination_prob);
                }
            }
        }
    }

    return global_transmittance;
}


__global__ void render_volume_surface_integrated(
    RNGState* rngStates,
    Camera camera,
    const SceneContext sceneContext,
    const BVHContext bvhContext,
    float4* __restrict__ colors,
    int maxDepth,
    int frameNum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);

    Ray r = camera.generateCameraRay(localState, x, y);

    float3 throughput = f3(1.0f);
    float3 pixelColor = f3();

    float lastPDF = -1.0f;
    bool prevDelta = true;

    // upper level loop, that runs once per bvh query
    for (int depth = 0; depth < maxDepth; depth++) {

        float3 bary;
        float min_t_surface;
        float min_t_volume;
        int primID_surface;
        int primID_volume;
        BVHSceneIntersect_volume(r, bvhContext, bary, min_t_surface, min_t_volume, primID_volume, primID_surface);

        bool hit_particle = false;
        bool hit_surface = false;

        // case 1, a real surface is interseceted before any volume
        if (min_t_surface < min_t_volume && primID_surface != -1) {
            hit_surface = true;
        // volume aabb hit before any surface (internally handles whether its actually a surface or volume hit)
        } else if (min_t_surface > min_t_volume && primID_volume != -1) {
            nanovdb::NanoGrid<float>* densityGrid = sceneContext.volumes[primID_volume].density_pointer;

            auto accessor = densityGrid->getAccessor();
            auto sampler = nanovdb::math::createSampler<1>(accessor);

            nanovdb::Ray<float> worldRay = toNanoVDB(r);

            nanovdb::Ray<float> indexRay = worldRay.worldToIndexF(*densityGrid);

            // the t val of the surface intersection, in index space
            float index_t_surface = 1e30f;
            if (primID_surface != -1) {
                nanovdb::Vec3f world_surf_pos = worldRay(min_t_surface);
                nanovdb::Vec3f index_surf_pos = densityGrid->worldToIndexF(world_surf_pos);
                index_t_surface = (index_surf_pos - indexRay.start()).length() / indexRay.dir().length();
            }

            nanovdb::math::TreeMarcher<leaf_t, nanovdb::Ray<float>, decltype(accessor)> marcher(accessor);

            if (marcher.init(indexRay)) {
                const leaf_t* leaf = nullptr;
                float t0, t1;

                float t_hit = 0.0f;

                // this one steps through the voxel grid voxel by voxel
                // Double check: This is a very pure, no importance sampling loop,
                // its purpose is soley to break down the volume into more specfic discrete majorants,
                // for the inner delta tracking loop to figure out
                while (marcher.step(&leaf, t0, t1)) {
                    if (t0 >= index_t_surface) {
                        hit_surface = true;
                        break;
                    }

                    // not neccesary?
                    float node_exit_t = fminf(t1, index_t_surface);

                    float local_majorant = leaf->maximum() * sceneContext.volumes[primID_volume].densityScale;
                    if (local_majorant <= 0.0f) continue;

                    float t_current = t0;

                    // this one steps through the voxel itself, using the majorant sampling to determine whether to stop or continue
                    while (true) {
                        float step = -log(rand(&localState)) / local_majorant;
                        t_current += step;

                        if (t_current >= index_t_surface) {
                            hit_surface = true;
                            break;
                        }

                        if (t_current >= t1) break; // Exited the node

                        float true_density = sampler(indexRay(t_current)) * sceneContext.volumes[primID_volume].densityScale;

                        if (rand(&localState) < (true_density / local_majorant)) {
                            hit_particle = true;
                            t_hit = t_current; // Save the exact hit distance
                            break;
                        }
                    }

                    if (hit_particle) break;
                }

/** hit_particle and hit_Surface cannot BOTH be true.
 *
 * If hit_particle: apply heyney greenstein and perform volume NEE
 *  (with f_val calculaed using modified formula to account for the lack of a bsdf)
 *
 * If hit_surface: apply bsdf and also perform volume NEE
 *
 * If neither: sample sky.
 */
                if (hit_particle) {

                    float3 lightNormal;
                    float3 emission;
                    float3 shadingPosToLightNormalized;
                    float3 shadingPos;
                    float t_max;
                    float pdf;
                    {
                        nanovdb::Vec3f index_hit_pos = indexRay(t_hit); // Use the saved t_hit
                        nanovdb::Vec3f world_hit_pos = densityGrid->indexToWorldF(index_hit_pos);
                        shadingPos = toNovum(world_hit_pos);
                    }

                    bool sampledEnv = sample(
                        sceneContext.lightSampler,
                        rand(&localState), rand4(&localState),
                        shadingPos,
                        sceneContext.vertices,
                        emission,
                        shadingPosToLightNormalized,
                        lightNormal,
                        t_max,
                        pdf
                    );

                    bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);

                    if (!lightBackface) {
                        VolumeInterval volHits[2];
                        int num_volHits = 0;
                        float3 throughputScale = f3(1.0f);

                        if (!BVHShadow_volume(
                            Ray(shadingPos, shadingPosToLightNormalized),
                            bvhContext,
                            t_max,
                            volHits,
                            2,
                            num_volHits)
                        ) {
                            if (num_volHits > 0) {
                                throughputScale *= estimate_volume_transmittance(
                                    Ray(shadingPos, shadingPosToLightNormalized),
                                    bvhContext,
                                    volHits,
                                    num_volHits,
                                    &localState
                                );
                            }
                        } else {
                            throughputScale = f3(0.0f);
                        }

                        if (lengthSquared(throughputScale) > EPSILON) {
                            float phaseval = evaluate_HG(
                                r.direction,
                                shadingPosToLightNormalized,
                                sceneContext.volumes[primID_volume].anisotropy
                            );

                            float phasepdf = phaseval;

                            float3 contribution;
                            float misWeight;

                            if (sampledEnv) {
                                contribution = throughput * phaseval * emission * throughputScale / pdf;

                                misWeight = powerHeuristicTwoStrategy(
                                    pdf,        // Env map PDF is already in Solid Angle
                                    phasepdf   // Alternate strategy: Phase function PDF
                                );

                            } else {
                                float cosLight = dot(-shadingPosToLightNormalized, lightNormal);

                                if (cosLight > 0.0f) {
                                    contribution = throughput * phaseval * emission * cosLight * throughputScale /
                                        (pdf * t_max * t_max);

                                    misWeight = powerHeuristicTwoStrategy(
                                        (t_max * t_max) * pdf / cosLight, // Convert area pdf to Solid Angle
                                        phasepdf                         // Alternate strategy: Phase function PDF
                                    );
                                } else {
                                    contribution = f3(0.0f);
                                    misWeight = 0.0f;
                                }

                            }
                            // VOLUME NEE from volume hit
                            pixelColor += fireflyClamp(contribution * misWeight);
                        }
                    }


                    nanovdb::Vec3f index_hit_pos = indexRay(t_hit); // Use the saved t_hit
                    nanovdb::Vec3f world_hit_pos = densityGrid->indexToWorldF(index_hit_pos);

                    nanovdb::Vec3f new_world_dir =
                        sample_HG(worldRay.dir(),
                                    sceneContext.volumes[primID_volume].anisotropy,
                                    rand(&localState),
                                    rand(&localState)
                        );

                    nanovdb::Vec3f offset_pos = world_hit_pos + new_world_dir * RAY_EPSILON;
                    worldRay = nanovdb::Ray<float>(offset_pos, new_world_dir);
                    r = toNovumRay(worldRay);

                    lastPDF = evaluate_HG(toNovum(worldRay.dir()), toNovum(new_world_dir), sceneContext.volumes[primID_volume].anisotropy);
                    prevDelta = false;

                    throughput *= f3(sceneContext.volumes[primID_volume].albedo);
                } else if (primID_surface != -1) {
                    hit_surface = true;
                }
            }
        }


        if (hit_surface) {
            const Triangle& tri = sceneContext.scene[primID_surface];
            int materialID;
            float2 uv;
            bool backface;
            float3 normal;
            float3 shadingPos;

            {
                materialID = tri.materialID;

                uv = __ldg(&sceneContext.vertices->uvs[tri.uvaInd]) * (1.0f - bary.x - bary.y) +
                    __ldg(&sceneContext.vertices->uvs[tri.uvbInd]) * bary.x +
                    __ldg(&sceneContext.vertices->uvs[tri.uvcInd]) * bary.y;

                float3 apos = f3(__ldg(&sceneContext.vertices->positions[tri.aInd]));
                float3 bpos = f3(__ldg(&sceneContext.vertices->positions[tri.bInd]));
                float3 cpos = f3(__ldg(&sceneContext.vertices->positions[tri.cInd]));

                shadingPos = (1.0f - bary.x - bary.y) * apos + bary.x * bpos + bary.y * cpos;

                float3 a_n = f3(__ldg(&sceneContext.vertices->normals[tri.naInd]));
                float3 b_n = f3(__ldg(&sceneContext.vertices->normals[tri.nbInd]));
                float3 c_n = f3(__ldg(&sceneContext.vertices->normals[tri.ncInd]));

                normal = (1.0f - bary.x - bary.y) * a_n + bary.x * b_n + bary.y * c_n;
                backface = dot(normal, r.direction) > 0.0f;
                normal = backface ? -normal : normal;
            }

            float3 incomingDir;
            toLocal(r.direction, normal, incomingDir);


            if (!sceneContext.materials[materialID].isSpecular) {
                float3 lightNormal;
                float3 emission;
                float3 shadingPosToLightNormalized;
                float t_max;
                float pdf;

                bool sampledEnv = sample(
                    sceneContext.lightSampler,
                    rand(&localState), rand4(&localState),
                    shadingPos,
                    sceneContext.vertices,
                    emission,
                    shadingPosToLightNormalized,
                    lightNormal,
                    t_max,
                    pdf
                );

                float3 shadingPosToLightLocal;
                toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

                bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
                bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);

                if (!surfaceBackface && !lightBackface) {
                    VolumeInterval volHits[2];
                    int num_volHits = 0;
                    float3 throughputScale = f3(1.0f);

                    if (!BVHShadow_volume(
                        Ray(shadingPos + normal * RAY_EPSILON, shadingPosToLightNormalized),
                        bvhContext,
                        t_max,
                        volHits,
                        2,
                        num_volHits)
                    ) {
                        if (num_volHits > 0) {
                            throughputScale *= estimate_volume_transmittance(
                                Ray(shadingPos, shadingPosToLightNormalized),
                                bvhContext,
                                volHits,
                                num_volHits,
                                &localState
                            );
                        }
                    } else {
                        throughputScale = f3(0.0f);
                    }


                    if (lengthSquared(throughputScale) > EPSILON) {
                        float bsdfPDF;

                        pdf_eval(
                            sceneContext.materials,
                            materialID,
                            sceneContext.textures,
                            incomingDir,
                            shadingPosToLightLocal,
                            1.5f, // change later when medium stack integrated
                            1.5f, // change later
                            bsdfPDF,
                            uv
                        );

                        float3 f_val;
                        f_eval(
                            sceneContext.materials,
                            materialID,
                            sceneContext.textures,
                            incomingDir,
                            shadingPosToLightLocal,
                            1.5f, // change later when medium stack integrated
                            1.5f, // change later
                            f_val,
                            uv
                        );

                        float3 contribution;
                        float misWeight;

                        if (sampledEnv) {
                            float cosSurface = dot(normal, shadingPosToLightNormalized);
                            contribution = throughput * f_val * emission * cosSurface / pdf;
                            misWeight = powerHeuristicTwoStrategy(
                                pdf,
                                bsdfPDF
                            );
                        } else {
                            float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                            float cosSurface = dot(normal, shadingPosToLightNormalized);

                            contribution = throughput * // throughput
                                f_val * emission * cosLight * // NEE contribution
                                cosSurface / (pdf * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                            misWeight = powerHeuristicTwoStrategy(
                                (t_max * t_max) * pdf / cosLight, // convert area pdf to SA
                                bsdfPDF // alt strat
                            );
                        }

                        // Volume NEE from a surface
                        pixelColor += fireflyClamp(contribution * misWeight * throughputScale);
                    }
                }
            }


            float3 implicitContribution = backface ? f3(0.0f) : f3(tri.emission) * throughput;

            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                (lengthSquared(r.origin - shadingPos) * sceneContext.lightSampler.evaluateMeshPdf(tri) / (fabsf(incomingDir.z))) // alternate strategy
            );

            pixelColor += fireflyClamp(misWeight * implicitContribution);

            float3 outgoing;
            float3 f_val;
            float pdf;

            sample_f_eval(
                localState,
                sceneContext.materials,
                materialID,
                sceneContext.textures,
                incomingDir,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                backface,
                outgoing,
                f_val,
                pdf,
                uv,
                TRANSPORTMODE_RADIANCE
            );

            if (pdf < EPSILON) { break; }

            throughput *= f_val * fabsf(outgoing.z) / pdf;
            toWorld(outgoing, normal, outgoing);

            r.direction = outgoing;
            r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;

            lastPDF = pdf;
            prevDelta = sceneContext.materials[materialID].isSpecular;
        }

        if (!hit_particle && !hit_surface) {

            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                sceneContext.lightSampler.evaluateEnvPdf(r.direction) // alternate strategy (already in solid angle)
            );

            // hitting env
            pixelColor += fireflyClamp(throughput * sceneContext.lightSampler.envMap.sampleDir(r.direction) * misWeight);
            break; // We flew out into the sky, kill the bounce loop.
        }

    }
    colors[pixelIdx] += f4(pixelColor);
    save_rng(pixelIdx, &localState, rngStates);
}

__host__ void launch_simple_volume(
    Camera camera,
    const SceneContext sceneContext,
    int numSample, int maxDepth,
    int h_w, int h_h,
    float3 h_sceneCenter, float h_sceneRadius, float3 h_sceneMin,
    float4* __restrict__ colors,
    float4* __restrict__ overlay,
    bool postProcess,
    float exposure
)
{
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float3));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float3));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));
    cudaMemcpyToSymbol(w, &(h_w), sizeof(int));
    cudaMemcpyToSymbol(h, &(h_h), sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((h_w+15)/16, (h_h+15)/16);

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif

    cudaDeviceSynchronize();

    float4* d_finalOutput;
    float4* d_overlay;
    cudaMalloc(&d_finalOutput, h_w * h_h * sizeof(float4));
    cudaMalloc(&d_overlay, h_w * h_h * sizeof(float4));
    cudaMemset(d_overlay, 0, h_w * h_h * sizeof(float4)); // Zero out the dummy overlay


    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    // Image Object (CPU) & Saving logic from SPPM
    int saveIntervalSamples = 300; // Matches SPPM logic
    Image image = Image(h_w, h_h);
    // Post-processing (exposure/tonemap/gamma) now happens on the GPU in
    // cleanFormatAndPostProcessImage, so saveImageBMP() must not re-apply it.
    image.postProcess = false;
    std::vector<float4> h_finalOutput(h_w * h_h);

    std::cout << "Running Kernels volume" << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();

    for (int currSample = 0; currSample < numSample; currSample++)
    {

        render_volume_surface_integrated<<<gridSize, blockSize>>>(
            d_rngStates,
            camera,
            sceneContext,
            getBVHContext(sceneContext),
            colors,
            maxDepth,
            currSample
        );
        cudaDeviceSynchronize();

        if ((currSample % saveIntervalSamples == 0 || currSample == numSample-1) && DO_PROGRESSIVERENDER)
        {
            // Launch the formatting kernel to handle averaging, NaNs, and Infs on the GPU
            if (postProcess) {
                cleanFormatAndPostProcessImage<<<gridSize, blockSize>>>(
                    colors, d_overlay, d_finalOutput, h_w, h_h, currSample, exposure, image.use_fitted_aces
                );
            } else {
                cleanAndFormatImage<<<gridSize, blockSize>>>(
                    colors, d_overlay, d_finalOutput, h_w, h_h, currSample
                );
            }

            // Copy the finalized buffer back to the host
            cudaMemcpy(h_finalOutput.data(), d_finalOutput, h_w * h_h * sizeof(float4), cudaMemcpyDeviceToHost);

            // Clean OpenMP loop simply maps the formatted colors to the image
            #pragma omp parallel for
            for (int i = 0; i < h_w * h_h; i++)
            {
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

            // Reset the dummy overlay just like in the wavefront version
            cudaMemset(d_overlay, 0, h_w * h_h * sizeof(float4));
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