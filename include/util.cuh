#pragma once

/*
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <curand_kernel.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <set>
#include <iomanip>
#include <chrono>
#include "imageUtil.cuh"
*/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <sstream>
#include "rng.cuh"
#include <tinyexr/tinyexr.h>

// need to set epsilon low for small scale scenes for some reason, likely due to rejection of min distance?
// ideally should be decoupled from ray epsilon type things; look into this
__device__ __constant__ float EPSILON = 0.0000001f;
__device__ __constant__ float EPSILON3 = 0.001f;
__device__ __constant__ float EPSILON2 = 0.001f;
__device__ __constant__ float RAY_EPSILON = 0.00001f;
__device__ __constant__ float VIS_EPSILON = 0.00001f;
__device__ __constant__ float PI = 3.141592f;
__device__ __constant__ float INVPI = 0.3183098f;


__device__ __constant__ float SKY_RADIUS = 100.0f;
__device__ __constant__ float MAX_FIREFLY_LUM = 50.0f;
__device__ __constant__ float MERGE_MAX_FIREFLY_LUM = 350.0f;
__device__ __constant__ float MERGE_ROUGHNESS_BOUND = 0.0f;

constexpr float h_PI = 3.141592f;

constexpr bool DO_PROGRESSIVERENDER = true;

#ifndef DO_FIREFLY_CLAMP
#define DO_FIREFLY_CLAMP 1
#endif


inline __host__ __device__ __forceinline__ float3 f3(float4 x) {
    return make_float3(x.x, x.y, x.z);
}

inline __host__ __device__ __forceinline__ float4 f4(float3 x) {
    return make_float4(x.x, x.y, x.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 f4(float x, float y, float z, float w = 0.0f) {
    return make_float4(x, y, z, w);
}

inline __host__ __device__ __forceinline__ float4 f4() {return make_float4(0,0,0,0);}

inline __host__ __device__ __forceinline__ float4 f4(float a) {return make_float4(a,a,a,0);}

inline __host__ __device__ __forceinline__ float3 f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

inline __host__ __device__ __forceinline__ float3 f3() {return make_float3(0,0,0);}

inline __host__ __device__ __forceinline__ float3 f3(float a) {return make_float3(a,a,a);}

inline __host__ __device__ __forceinline__ float2 f2(float x, float y) {return make_float2(x, y);}

inline __host__ __device__ __forceinline__ float2 f2() {return make_float2(0,0);}

inline __host__ __device__ __forceinline__ float2 f2(float a) {return make_float2(a,a);}

inline __host__ __device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4 &a, float t) {
    return make_float4(a.x * t, a.y * t, a.z * t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(float t, const float4 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4 &a, float t) {
    return make_float4(a.x / t, a.y / t, a.z / t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4& operator+=(float4 &a, const float4 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator*=(float4 &a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}

inline __host__ __device__  __forceinline__ float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator/=(float4 &a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4& a, const float4& b)
{
    return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4& v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

inline __host__ __device__ __forceinline__ float dot(const float4 &a, const float4 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ __forceinline__ float length(const float4 &v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ __forceinline__ float lengthSquared(const float4 &v) {
    return dot(v, v);
}

inline __host__ __device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ __forceinline__ float3 operator*(const float3 &a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}

inline __host__ __device__ __forceinline__ float3 operator*(float t, const float3 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float3 operator/(const float3 &a, float t) {
    return make_float3(a.x / t, a.y / t, a.z / t);
}

inline __host__ __device__ __forceinline__ float3& operator+=(float3 &a, const float3 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float3& operator*=(float3 &a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}

inline __host__ __device__  __forceinline__ float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float3& operator/=(float3 &a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

inline __host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __host__ __device__ __forceinline__ float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ __forceinline__ float length(const float3 &v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ __forceinline__ float lengthSquared(const float3 &v) {
    return dot(v, v);
}

inline __host__ __device__ __forceinline__ float3 normalize(const float3 &v) {
    float invLen = rsqrtf(dot(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

inline __host__ __device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}


inline __host__ __device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ __forceinline__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ __forceinline__ float2 operator*(const float2 &a, float t) {
    return make_float2(a.x * t, a.y * t);
}

inline __host__ __device__ __forceinline__ float2 operator*(float t, const float2 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float2 operator/(const float2 &a, float t) {
    return make_float2(a.x / t, a.y / t);
}

inline __host__ __device__ __forceinline__ float2& operator+=(float2 &a, const float2 &b) {
    a.x += b.x; a.y += b.y;
    return a;
}

inline __host__ __device__ __forceinline__ float2& operator*=(float2 &a, float t) {
    a.x *= t; a.y *= t;
    return a;
}

inline __host__ __device__  __forceinline__ float2& operator*=(float2& a, const float2& b) {
    a.x *= b.x; a.y *= b.y;
    return a;
}

inline __host__ __device__ __forceinline__ float2& operator/=(float2 &a, float t) {
    a.x /= t; a.y /= t;
    return a;
}

inline __host__ __device__ __forceinline__ float2 operator*(const float2& a, const float2& b)
{
    return make_float2(a.x*b.x, a.y*b.y);
}

inline __host__ __device__ __forceinline__ float2 operator/(const float2& a, const float2& b)
{
    return make_float2(a.x/b.x, a.y/b.y);
}

inline __host__ __device__ __forceinline__ float2 operator-(const float2& v)
{
    return make_float2(-v.x, -v.y);
}

inline __host__ __device__ __forceinline__ float clamp(float x, float minVal, float maxVal) {
    if (x < minVal) return minVal;
    if (x > maxVal) return maxVal;
    return x;
}

inline __host__ __device__ __forceinline__ float4 clampf4(float4 v, float minVal, float maxVal) {
    return make_float4(
        clamp(v.x, minVal, maxVal),
        clamp(v.y, minVal, maxVal),
        clamp(v.z, minVal, maxVal),
        0.0f
    );
}

inline __host__ std::ostream& operator<< (std::ostream& out, const float4& v )
{
    out << "<" << v.x << " " << v.y << " " << v.z << " " << v.w << ">";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream& out, const float3& v )
{
    out << "<" << v.x << " " << v.y << " " << v.z << ">";
    return out;
}

inline __device__ __forceinline__ void toWorld(const float3& wo_local, const float3& n, float3& wo_world)
{
    float3 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f3(-n.y, n.x, 0.0f));
    else
        t = normalize(f3(0.0f, -n.z, n.y));

    b = cross(n, t);
    wo_world = wo_local.x * t + wo_local.y * b + wo_local.z * n;
}

inline __device__ __forceinline__ float3 toWorld(const float3& wo_local, const float3& n)
{
    float3 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f3(-n.y, n.x, 0.0f));
    else
        t = normalize(f3(0.0f, -n.z, n.y));

    b = cross(n, t);
    return wo_local.x * t + wo_local.y * b + wo_local.z * n;
}


inline __device__ __forceinline__ void toLocal(const float3& wo_world, const float3& n, float3& wo_local)
{
    float3 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f3(-n.y, n.x, 0.0f));
    else
        t = normalize(f3(0.0f, -n.z, n.y));

    b = cross(n, t);
    wo_local = f3(dot(wo_world, t), dot(wo_world, b), dot(wo_world, n));
}

inline __device__ __forceinline__ float3 toLocal(const float3& wo_world, const float3& n)
{
    float3 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f3(-n.y, n.x, 0.0f));
    else
        t = normalize(f3(0.0f, -n.z, n.y));

    b = cross(n, t);
    return f3(dot(wo_world, t), dot(wo_world, b), dot(wo_world, n));
}

// Component-wise min of two float4s
inline __host__ __device__ __forceinline__ float4 fminf4(const float4 &a, const float4 &b) {
    return f4(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

// Component-wise max of two float4s
inline __host__ __device__ __forceinline__ float4 fmaxf4(const float4 &a, const float4 &b) {
    return f4(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

// Component-wise min of two float3s
inline __host__ __device__ __forceinline__ float3 fminf3(const float3 &a, const float3 &b) {
    return f3(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

// Component-wise max of two float3s
inline __host__ __device__ __forceinline__ float3 fmaxf3(const float3 &a, const float3 &b) {
    return f3(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

inline __host__ __device__ __forceinline__ float getFloat4Component(const float4 &v, int i) {
    switch(i) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
        default: return 0.0f; // or handle error
    }
}

inline __host__ __device__ __forceinline__ void setFloat4Component(float4 &v, int i, float value) {
    switch(i) {
        case 0: v.x = value; break;
        case 1: v.y = value; break;
        case 2: v.z = value; break;
        case 3: v.w = value; break;
    }
}

inline __host__ float surfaceArea(const float4& min, const float4& max)
{
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return 2.0f * (dx * dy + dx * dz + dy * dz);
}

__host__ __device__ __forceinline__ float3 sqrtf3(const float3& v) {
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

__host__ __device__ __forceinline__ float3 rotateX(const float3& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f3(
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float3 rotateY(const float3& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f3(
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float3 rotateZ(const float3& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f3(
        v.x * c - v.y * s,
        v.x * s + v.y * c,
        v.z
    );
}

__device__ __forceinline__ float3 sampleSphere(RNGState& localState, float R)
{
    float u = rand(&localState);
    float v = rand(&localState);

    float z = 1.0f - 2.0f * u;          // cos(theta) uniformly distributed
    float r = sqrtf(max(0.f, 1.f - z*z));

    float phi = 2.0f * 3.141592f * v;

    float x = r * cosf(phi);
    float y = r * sinf(phi);

    return f3(R * x, R * y, R * z);
}

__device__ __host__ inline float luminance(float4 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ __host__ inline float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ __forceinline__ uint32_t toRGB9E5(float3 c)
{
    float r = fmaxf(0.0f, c.x);
    float g = fmaxf(0.0f, c.y);
    float b = fmaxf(0.0f, c.z);

    float maxRGB = fmaxf(r, fmaxf(g, b));

    if (maxRGB > 65000.0f) {
        float scaleDown = 65000.0f / maxRGB;
        r *= scaleDown;
        g *= scaleDown;
        b *= scaleDown;
        maxRGB = 65000.0f;
    }

    int expShared = (maxRGB < 1.5258789e-05f) ? 0 : (int)floorf(log2f(maxRGB)) + 17;
    expShared = clamp(expShared, 0, 31);

    float denom = exp2f((float)(expShared - 15 - 9));

    int Ri = (int)roundf(r / denom);
    int Gi = (int)roundf(g / denom);
    int Bi = (int)roundf(b / denom);

    if (Ri > 511 || Gi > 511 || Bi > 511) {
        expShared++;
        denom *= 2.0f;
        Ri = (int)roundf(r / denom);
        Gi = (int)roundf(g / denom);
        Bi = (int)roundf(b / denom);
    }

    return (uint32_t)((expShared << 27) | ((Bi & 0x1FF) << 18) | ((Gi & 0x1FF) << 9) | (Ri & 0x1FF));
}

__device__ __forceinline__ float3 fromRGB9E5(uint32_t packed)
{
    int exponent = (int)(packed >> 27) - 15 - 9;

    float scale = __powf(2.0f, (float)exponent);

    float r = (float)(packed & 0x1FF) * scale;
    float g = (float)((packed >> 9) & 0x1FF) * scale;
    float b = (float)((packed >> 18) & 0x1FF) * scale;

    return make_float3(r, g, b);
}

// Encodes a unit vector into 32 bits (2x 16-bit snorm) with minimal error.
__device__ __forceinline__ float signNotZero(float k) { return (k >= 0.0f) ? 1.0f : -1.0f; }

__device__ __forceinline__ uint32_t packOct(float3 v)
{
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float invL1 = (l1norm > 0.0f) ? (1.0f / l1norm) : 0.0f;

    float2 res;
    res.x = v.x * invL1;
    res.y = v.y * invL1;

    if (v.z < 0.0f) {
        float tempX = res.x;
        float tempY = res.y;
        res.x = (1.0f - fabsf(tempY)) * signNotZero(tempX);
        res.y = (1.0f - fabsf(tempX)) * signNotZero(tempY);
    }

    int16_t x = (int16_t)__float2int_rn(fminf(fmaxf(res.x, -1.0f), 1.0f) * 32767.0f);
    int16_t y = (int16_t)__float2int_rn(fminf(fmaxf(res.y, -1.0f), 1.0f) * 32767.0f);

    return ((uint32_t)(uint16_t)x) | (((uint32_t)(uint16_t)y) << 16);
}

__device__ __forceinline__ float3 unpackOct(uint32_t p) {
    int16_t packedX = (int16_t)(p & 0xFFFF);
    int16_t packedY = (int16_t)(p >> 16);

    float x = fmaxf(-1.0f, packedX / 32767.0f);
    float y = fmaxf(-1.0f, packedY / 32767.0f);

    float3 v;
    v.x = x;
    v.y = y;
    v.z = 1.0f - (fabsf(x) + fabsf(y));

    float t = fmaxf(-v.z, 0.0f);
    v.x += (v.x >= 0.0f) ? -t : t;
    v.y += (v.y >= 0.0f) ? -t : t;

    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __forceinline__ uint32_t packOctFlags(float3 v, bool flag1, bool flag2)
{
    // 1. Octahedral Projection
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float invL1 = (l1norm > 0.0f) ? (1.0f / l1norm) : 0.0f;

    float2 res;
    res.x = v.x * invL1;
    res.y = v.y * invL1;

    // 2. Reflect folds if in lower hemisphere
    if (v.z < 0.0f) {
        float tempX = res.x;
        float tempY = res.y;
        res.x = (1.0f - fabsf(tempY)) * signNotZero(tempX);
        res.y = (1.0f - fabsf(tempX)) * signNotZero(tempY);
    }

    // 3. SNORM Quantization (15-bit)
    // Map [-1, 1] -> [-16383, 16383]
    // We use 16383.0f because that is the max value for a signed 15-bit integer (2^14 - 1).
    int32_t x = __float2int_rn(fminf(fmaxf(res.x, -1.0f), 1.0f) * 16383.0f);
    int32_t y = __float2int_rn(fminf(fmaxf(res.y, -1.0f), 1.0f) * 16383.0f);

    // 4. Bit Packing
    // Mask x and y to ensure they fit in 15 bits (0x7FFF), then shift.
    uint32_t packedX = (uint32_t)(x & 0x7FFF);
    uint32_t packedY = (uint32_t)(y & 0x7FFF);

    // Shift flags to positions 30 and 31
    uint32_t packedF1 = (uint32_t)flag1 << 30;
    uint32_t packedF2 = (uint32_t)flag2 << 31;

    return packedX | (packedY << 15) | packedF1 | packedF2;
}

/**
 * Unpacks the 3D vector and writes the flags to the provided pointers.
 */
__device__ __forceinline__ float3 unpackOctFlags(uint32_t p, bool* flag1, bool* flag2) {
    // 1. Extract Flags
    if (flag1) *flag1 = (p >> 30) & 1;
    if (flag2) *flag2 = (p >> 31) & 1;

    int32_t packedX = ((int32_t)(p << 17)) >> 17;

    int32_t packedY = ((int32_t)(p << 2)) >> 17;

    float x = fmaxf(-1.0f, packedX / 16383.0f);
    float y = fmaxf(-1.0f, packedY / 16383.0f);

    float3 v;
    v.x = x;
    v.y = y;
    v.z = 1.0f - (fabsf(x) + fabsf(y));

    float t = fmaxf(-v.z, 0.0f);
    v.x += (v.x >= 0.0f) ? -t : t;
    v.y += (v.y >= 0.0f) ? -t : t;

    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __forceinline__ uint32_t packFloat2ToSnorm16(float2 v)
{
    float cx = fmaxf(-1.0f, fminf(v.x, 1.0f));
    float cy = fmaxf(-1.0f, fminf(v.y, 1.0f));

    int ix = (int)(cx >= 0.0f ? (cx * 32767.0f + 0.5f) : (cx * 32767.0f - 0.5f));
    int iy = (int)(cy >= 0.0f ? (cy * 32767.0f + 0.5f) : (cy * 32767.0f - 0.5f));

    uint32_t packed = ((uint32_t)ix & 0x0000FFFF) | ((uint32_t)iy << 16);

    return packed;
}

inline __forceinline__ __device__ float2 unpackSnorm16ToFloat2(uint32_t packed)
{
    int ix = (int)(packed << 16) >> 16;
    int iy = (int)packed >> 16;

    float fx = (float)ix / 32767.0f;
    float fy = (float)iy / 32767.0f;

    fx = fmaxf(fx, -1.0f);
    fy = fmaxf(fy, -1.0f);

    return make_float2(fx, fy);
}

__device__ __forceinline__ uint32_t packFloat2ToUnorm16(float2 v)
{
    float cx = fmaxf(0.0f, fminf(v.x, 1.0f));
    float cy = fmaxf(0.0f, fminf(v.y, 1.0f));

    uint32_t ix = (uint32_t)(cx * 65535.0f + 0.5f);
    uint32_t iy = (uint32_t)(cy * 65535.0f + 0.5f);

    uint32_t packed = (ix & 0x0000FFFF) | (iy << 16);

    return packed;
}

__device__ __forceinline__ float2 unpackUnorm16ToFloat2(uint32_t packed)
{
    uint32_t ix = packed & 0x0000FFFF;
    uint32_t iy = packed >> 16;

    float fx = (float)ix / 65535.0f;
    float fy = (float)iy / 65535.0f;

    return make_float2(fx, fy);
}

__device__ __forceinline__ uint32_t packRGBA8(float3 albedo, float roughness) {
    // __saturatef clamps to [0, 1] before multiplying
    uint32_t r = __float2uint_rn(__saturatef(albedo.x) * 255.0f);
    uint32_t g = __float2uint_rn(__saturatef(albedo.y) * 255.0f);
    uint32_t b = __float2uint_rn(__saturatef(albedo.z) * 255.0f);
    uint32_t a = __float2uint_rn(__saturatef(roughness) * 255.0f);

    return (a << 24) | (b << 16) | (g << 8) | r;
}

__device__ __forceinline__ void unpackRGBA8(uint32_t packed, float3& albedo, float& roughness) {
    albedo.x  = (packed & 0xFF) / 255.0f;
    albedo.y  = ((packed >> 8) & 0xFF) / 255.0f;
    albedo.z  = ((packed >> 16) & 0xFF) / 255.0f;
    roughness = ((packed >> 24) & 0xFF) / 255.0f;
}

__device__ __forceinline__ uint32_t packRGB10A2(float3 albedo, uint32_t matTypeFlag) {
    uint32_t r = __float2uint_rn(__saturatef(albedo.x) * 1023.0f);
    uint32_t g = __float2uint_rn(__saturatef(albedo.y) * 1023.0f);
    uint32_t b = __float2uint_rn(__saturatef(albedo.z) * 1023.0f);

    uint32_t a = matTypeFlag & 0x3;

    return (a << 30) | (b << 20) | (g << 10) | r;
}

__device__ __forceinline__ void unpackRGB10A2(uint32_t packed, float3& albedo, uint32_t& matTypeFlag) {
    albedo.x    = (packed & 0x3FF) / 1023.0f;
    albedo.y    = ((packed >> 10) & 0x3FF) / 1023.0f;
    albedo.z    = ((packed >> 20) & 0x3FF) / 1023.0f;
    matTypeFlag = (packed >> 30) & 0x3;
}

__host__ __device__ inline bool IsPrime(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

__host__ __device__ inline int GetNextPrime(int n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;

    while (!IsPrime(n)) {
        n += 2;
    }
    return n;
}

__host__ inline float calculateMergeRadius(float initialRadius, float alpha, int currSample)
{
    return initialRadius * sqrtf( 1.0f / powf(currSample + 1, alpha));
}

__device__ __forceinline__ inline float eta_vcm_to_mergeRadius(float eta_vcm, int numPhotonLightPath)
{
    // eta_vcm = mergeRadius * mergeRadius * PI * numPhotonLightPath
    return sqrtf(eta_vcm / (PI * numPhotonLightPath));
}

__device__ __forceinline__ inline void accumulateOutput(float4* colors, float3 contribution, int idx)
{
    if (colors[idx].x < 0.0f || colors[idx].y < 0.0f || colors[idx].z < 0.0f) {
        printf("NEGATIVE NUMBER ADDED\n");
    }

#if DO_FIREFLY_CLAMP == 1
    float lum = luminance(contribution);
    if (lum > MAX_FIREFLY_LUM)
    {
        contribution *= (MAX_FIREFLY_LUM / lum);
    }
#endif
    atomicAdd(&colors[idx].x, contribution.x);
    atomicAdd(&colors[idx].y, contribution.y);
    atomicAdd(&colors[idx].z, contribution.z);
}

__device__ __forceinline__ inline float3 fireflyClamp(float3 contribution) {
#if DO_FIREFLY_CLAMP == 1
    float lum = luminance(contribution);
    if (lum > MAX_FIREFLY_LUM)
    {
        contribution *= (MAX_FIREFLY_LUM / lum);
    }
#endif
    return contribution;
}

__device__ __forceinline__ inline float powerHeuristicTwoStrategy(float primary, float other)
{
    return 1.0f / (1.0f + (other/primary) * (other/primary));
}

__device__ __forceinline__ inline float balanceHeuristicTwoStrategy(float primary, float other)
{
    return 1.0f / (1.0f + (other/primary));
}

__device__ __forceinline__ int BurnCycles(int iterations) {
    float f = 1.00f;
    float a = 0.99f;
    float b = 0.01f;
    for (int i = 0; i < iterations; i++) {
        #ifndef __INTELLISENSE__
            asm volatile (
                "fma.rn.f32 %0, %0, %1, %2;"
                "fma.rn.f32 %0, %0, %1, %2;"
                "fma.rn.f32 %0, %0, %1, %2;"
                "fma.rn.f32 %0, %0, %1, %2;"
                : "+f"(f)
                : "f"(a), "f"(b)
            );
        #endif
    }

    return f;
}

__device__ inline float2 dirToUV(float3 dir) {
    // atan2f(Z, X) perfectly reverses the outDir generation
    float phi = atan2f(dir.z, dir.x);

    // Wrap negative angles back around the circle
    if (phi < 0.0f) {
        phi += 2.0f * PI;
    }

    float u = phi / (2.0f * PI);

    // Clamp Y to prevent NaNs if dir.y floats slightly out of [-1, 1] bounds
    float v = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y))) / PI;

    return make_float2(u, v);
}

__host__ __device__ __forceinline__ inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ __forceinline__ float targetFunction(float3 contribution) {
    return luminance(fireflyClamp(contribution));
    //return luminance(contribution);
}
