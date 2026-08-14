#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"

extern "C" {
    __constant__ PipelineParams allParams;
}

extern "C" __global__ void __closesthit__gather() {
    //float2 uvs = optixGetTriangleBarycentrics();

    //optixSetPayload_0(__float_as_uint(uvs.x));
    //optixSetPayload_1(__float_as_uint(uvs.y));
}

extern "C" __global__ void __raygen__unidirectional() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 0, nullptr);

    Ray r = params.camera.generateCameraRay(localState, x, y);

    float3 Li = f3(0.0f);
    float3 throughput = f3(1.0f);
    bool prevDelta = false;
    float lastPDF = 0.0f;
    for (int depth = 0; depth < params.max_depth; depth++)
    {   
        SurfaceHit hitData;

        if (params.newVersion) {
            hitData = traceClosestHitObjSER(params, r);
        } else {
            hitData = traceClosestNoSER(params, r);
        }
        
        if (!hitData.isHit)
        {
            float3 contribution = throughput * params.shadeContext.lightSampler.envMap.sampleDir(r.direction);
            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                params.shadeContext.lightSampler.evaluateEnvPdf(r.direction) // alternate strategy
            );
            Li += contribution * misWeight;
            break;
        }
        if (params.newVersion) {
            //optixReorder();
        } else {
            optixReorder(1u, 0u);
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 geoNormal;
        float3 normal;
        float3 emission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];
        getDataGeo(
            tri,
            params.shadeContext,
            hitData.barycentrics,
            r.direction,

            materialID,
            uv,
            shadingPos,
            geoNormal,
            normal,
            backface,
            emission
        );

        float3 contribution = backface ? f3(0.0f) : emission * throughput;

        float3 incomingDir;
        toLocal(r.direction, normal, incomingDir);

        if (luminance(contribution) > EPSILON) {
            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                (hitData.t * hitData.t * params.shadeContext.lightSampler.evaluateMeshPdf(tri) / (fabsf(incomingDir.z))) // alternate strategy
            );

            Li += contribution * misWeight;
        }

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;

        if (!currDelta) {
            float3 lightNormal;
            float3 emission;
            float3 shadingPosToLightNormalized;
            float t_max;
            float pdf;

            bool sampledEnv = sample(
                params.shadeContext.lightSampler,
                rand(&localState), rand4(&localState),
                shadingPos,
                params.shadeContext.vertices,
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
                float bsdfPDF;

                pdf_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
                    incomingDir,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    bsdfPDF,
                    uv
                );

                float3 f_val;
                f_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
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

                bool occluded = traceVisibility(
                    params,
                    Ray((shadingPos + shadingPosToLightNormalized * RAY_EPSILON), shadingPosToLightNormalized),
                    t_max * (1.0f - EPSILON3)
                );

                if (!occluded) {
                    Li += contribution * misWeight;
                }
            }
        }

        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);

        if (rand(&localState) > p)   // survive with probability p
        {
            save_rng(pixelIdx, &localState, nullptr);
            break;
        }
        throughput /= p;


        float3 outgoing;
        float3 f_val;
        float pdf;

        sample_f_eval(
            localState,
            params.shadeContext.materials,
            materialID,
            params.shadeContext.textures,
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

        if (pdf < EPSILON)
        {
            save_rng(pixelIdx, &localState, nullptr);
            break;
        }

        throughput *= f_val * fabsf(outgoing.z) / pdf;
        toWorld(outgoing, normal, outgoing);

        // write to next rayQueue
        r.origin = shadingPos + (dot(outgoing, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf;
        save_rng(pixelIdx, &localState, nullptr);
    }
    params.accum_buffer[pixelIdx] += f4(Li);
}