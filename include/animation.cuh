#pragma once

#include "objects.cuh"

struct __align__(32) LinearCameraAnimation {
    float3 origin;
    float3 initRot;
    float3 dxyz;
    float3 dxyzrot;

    LinearCameraAnimation(float3 o, float3 i, float3 dpos, float3 drot) : origin(o), initRot(i), dxyz(dpos), dxyzrot(drot) {}

    __host__ inline void update(Camera& cam, uint32_t frame) {
        cam.cameraOrigin = origin + dxyz * (float) frame;

        cam.xRot = initRot.x + dxyzrot.x * (float) frame;
        cam.yRot = initRot.y + dxyzrot.y * (float) frame;
        cam.zRot = initRot.z + dxyzrot.z * (float) frame;

        cam.preCompute();
    }
};
struct __align__(32) OrbitCameraAnimation {
    float3 target;        // Where the camera looks
    float radius;         // Distance from target (spherical radius)
    float orbitSpeed;     // Radians per frame
    float startAngle;     // Starting horizontal rotation
    float elevation;      // Vertical angle in radians (pitch)
    
    // Replaced 'camHeight' with 'elevationDeg'
    OrbitCameraAnimation(float3 targetPos, float orbitRadius, float speedDeg, float startAngleDeg, float elevationDeg) 
        : target(targetPos), radius(orbitRadius) 
    {
        orbitSpeed = speedDeg * (3.14159265f / 180.0f);
        startAngle = startAngleDeg * (3.14159265f / 180.0f);
        elevation = elevationDeg * (3.14159265f / 180.0f);
    }

    __host__ void update(Camera& cam, uint32_t frame) {
        // The horizontal rotation (azimuth)
        float angle = startAngle + (float)frame * orbitSpeed;

        // Spherical coordinate math:
        // Project the radius down onto the XZ plane based on the elevation.
        // If elevation is 90 degrees (straight up), horizontalRadius becomes 0.
        float horizontalRadius = radius * cosf(elevation);

        float3 pos = make_float3(
            target.x + horizontalRadius * cosf(angle),
            target.y + radius * sinf(elevation),       // Y uses sin() to go up/down 
            target.z + horizontalRadius * sinf(angle)
        );

        cam.cameraOrigin = pos;

        float3 dir = normalize(target - pos);

        // Your existing Euler extraction works perfectly with this
        cam.yRot = atan2f(-dir.x, -dir.z); 
        cam.xRot = asinf(dir.y);
        cam.zRot = 0.0f;
        
        cam.preCompute();
    }
};