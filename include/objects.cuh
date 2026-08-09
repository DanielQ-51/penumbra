#pragma once

#include "util.cuh"
#include "material.cuh"
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cuda_fp16.h>
#include <nanovdb/NanoVDB.h>

struct Volume
{
    float4 aabbMIN;
    float4 aabbMAX;
    nanovdb::NanoGrid<float>* density_pointer;
    nanovdb::NanoGrid<float>* temperature_pointer;
    float densityScale;
    float4 albedo;
    float anisotropy;

    __host__ Volume(
        float4 min,
        float4 max,
        nanovdb::NanoGrid<float>* dp,
        nanovdb::NanoGrid<float>* tp,
        float ds,
        float ani,
        float4 alb
    ) : aabbMIN(min), aabbMAX(max), density_pointer(dp), temperature_pointer(tp), densityScale(ds), albedo(alb), anisotropy(ani) {}
};

enum PrimitiveType {
    TYPE_INNER = 0,
    TYPE_TRIANGLE = 1, // Leaf node pointing to Triangle array
    TYPE_VOLUME = 2    // Leaf node pointing to Volume array
};

struct __align__(64) BVHnode
{
    float4 aabbMIN;
    float4 aabbMAX;
    int left;
    int right;
    int first;
    int primCount;

    __host__ BVHnode() {}

    __host__ BVHnode(float4 min, float4 max, int l, int r, int f, int ct)
        : aabbMIN(min), aabbMAX(max), left(l), right(r), first(f), primCount(ct) {} 
};

inline void traverseBVH(const std::vector<BVHnode>& bvh, int nodeIdx, int level,
                       int& totalNodes, 
                       std::vector<int>& leafDepths, 
                       std::vector<int>& leafSizes) 
{
    if (nodeIdx < 0 || nodeIdx >= (int)bvh.size()) return;

    const BVHnode& node = bvh[nodeIdx];
    totalNodes++;

    // Check if it's a leaf node
    if (node.left == -1 && node.right == -1) {
        leafDepths.push_back(level);
        leafSizes.push_back(node.primCount);
        return; 
    }

    // Recurse into children
    if (node.left != -1)  traverseBVH(bvh, node.left,  level + 1, totalNodes, leafDepths, leafSizes);
    if (node.right != -1) traverseBVH(bvh, node.right, level + 1, totalNodes, leafDepths, leafSizes);
}

inline void printBVHSummary(const std::vector<BVHnode>& bvh) 
{
    if (bvh.empty()) {
        std::cout << "BVH is empty.\n";
        return;
    }

    // 1. Setup local tracking variables (Allocated on stack, no leaks!)
    int totalNodes = 0;
    std::vector<int> leafDepths;
    std::vector<int> leafSizes;

    // 2. Perform the traversal
    traverseBVH(bvh, 0, 0, totalNodes, leafDepths, leafSizes);

    // 3. Perform Calculations
    int leafCount = (int)leafDepths.size();
    if (leafCount == 0) return;

    int maxDepth = *std::max_element(leafDepths.begin(), leafDepths.end());
    int largestLeaf = *std::max_element(leafSizes.begin(), leafSizes.end());

    auto mean = [](const std::vector<int>& v) {
        return v.empty() ? 0.0 : static_cast<double>(std::accumulate(v.begin(), v.end(), 0LL)) / v.size();
    };

    auto stddev = [mean](const std::vector<int>& v) {
        if (v.empty()) return 0.0;
        double m = mean(v);
        double sumSq = 0.0;
        for (int x : v) sumSq += (x - m) * (x - m);
        return std::sqrt(sumSq / v.size());
    };

    auto median = [](std::vector<int> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return n % 2 == 0 ? 0.5 * (v[n/2 - 1] + v[n/2]) : (double)v[n/2];
    };

    // 4. Print Summary
    std::cout << "\n================= BVH Summary =================\n";
    std::cout << "Total nodes:       " << totalNodes << "\n";
    std::cout << "Leaf nodes:        " << leafCount << "\n";
    std::cout << "Internal nodes:    " << (totalNodes - leafCount) << "\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "Max leaf depth:    " << maxDepth << "\n";
    std::cout << "Average depth:     " << mean(leafDepths) << "\n";
    std::cout << "Median depth:      " << median(leafDepths) << "\n";
    std::cout << "Depth stddev:      " << stddev(leafDepths) << "\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "Largest leaf:      " << largestLeaf << " primitives\n";
    std::cout << "Average leaf size: " << mean(leafSizes) << " primitives\n";
    std::cout << "Median leaf size:  " << median(leafSizes) << " primitives\n";
    std::cout << "Leaf size stddev:  " << stddev(leafSizes) << "\n";

    // Print top 10 largest leaf sizes
    std::vector<int> sortedLeafSizes = leafSizes;
    std::sort(sortedLeafSizes.begin(), sortedLeafSizes.end(), std::greater<int>());
    std::cout << "Top 10 largest:    ";
    for (size_t i = 0; i < std::min<size_t>(10, sortedLeafSizes.size()); ++i)
        std::cout << sortedLeafSizes[i] << " ";
    
    std::cout << "\n===============================================\n\n";
}

struct VolumeNodeData {
    int nodeIndex;
    int depth;
    int leafSize;
    int originalVolumeID;
    float3 aabbMIN;
    float3 aabbMAX;
};


struct Vertices
{
    float4* positions;
    float4* normals;
    //float4* colors;
    float2* uvs;
};

struct Triangle
{
    int aInd;
    int bInd;
    int cInd;
    int naInd, nbInd, ncInd; // Normal indices

    int uvaInd, uvbInd, uvcInd;
    int materialID;
    float4 emission;

    int lightInd; // you might hit a emissive mesh but you need to know what LIGHT it corresponds to
    int triInd; // used when drawing shadow rays and needing to ignore the target light

    // Ray-cone texture LOD base term: 0.5 * log2(texelArea / worldParallelogramArea),
    // baked from the base-color texture's dimensions + the triangle's UV/world areas.
    // Runtime LOD is  lodDelta + log2(coneWidth / |n·d|)  (+ optional global bias).
    // 0 means "not computed" (no UVs / no base-color texture) -> full-res mip 0.
    // NOTE: baked in the space `positions` were stored in. buildFlattened bakes world
    // space, so this is exact there; an instanced (object-space) path would add a
    // per-instance -log2(scale) correction at runtime.
    float lodDelta = 0.0f;

    __device__ inline float area(const Vertices* vertices) {
        float3 apos = f3(__ldg(&vertices->positions[aInd]));
        return 0.5f * length(cross(
            f3(__ldg(&vertices->positions[bInd])) - apos,
            f3(__ldg(&vertices->positions[cInd])) - apos));
    }

    __device__ __forceinline__ float3 getNormal(const Vertices* vertices, float u, float v) const {
        float w = 1.0f - u - v;

        float3 nA = f3(__ldg(&vertices->normals[naInd]));
        float3 nB = f3(__ldg(&vertices->normals[nbInd]));
        float3 nC = f3(__ldg(&vertices->normals[ncInd]));

        return normalize(nA * w + nB * u + nC * v);
    }

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e, int lind, int tind)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e), lightInd(lind), triInd(tind){}
};

struct Ray
{
    float3 origin;
    float3 direction;

    __host__ __device__ Ray() {}

    __host__ __device__ Ray(float3 o, float3 d) : origin(o), direction(d) {}

    __host__ __device__ float3 at(float t) const {return origin + t*direction;}

};

struct Camera
{
    float3 cameraOrigin;
    int w;
    int h;

    float xRot; // in radians
    float yRot; // in radians
    float zRot; // in radians

    float aperture;
    float focalDist;
    //float imagePlaneDist; // for FOV
    float fovScale;

    float antiAliasJitterDist;

    float3 forward;
    float3 right;
    float3 up;

    __host__ static Camera Pinhole(const float3& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aajitter = 0.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = 0.000001f;
        c.focalDist = 1.0f/FOV;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();

        return c;
    }

    __host__ static Camera NotPinhole(const float3& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aperture, float focalDist, float aajitter = 2.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = aperture;
        c.focalDist = focalDist;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();
        return c;
    }


    // returns a camera ray with normalized direction
    __device__ __forceinline__ Ray generateCameraRay(RNGState& localState, int x, int y) const
    {
        Ray r;
        float aspect = (float)w / (float)h;

        float jitterX = 0.0f;
        float jitterY = 0.0f;
        if (antiAliasJitterDist != 0.0f) {
            jitterX = (rand(&localState) - 0.5f) * antiAliasJitterDist;
            jitterY = (rand(&localState) - 0.5f) * antiAliasJitterDist;
        }

        float u = (2.0f * ((x + jitterX) / (float)w) - 1.0f) * aspect * fovScale;
        float v = (2.0f * ((y + jitterY) / (float)h) - 1.0f) * fovScale;

        // 2. Calculate Focal Point using Precomputed Basis Vectors
        // This automatically handles X, Y, and Z rotation correctly.
        // Note: 'forward' corresponds to local (0,0,-1), so we move positive along 'forward'
        // to go deeper into the scene.
        float3 focalPoint = cameraOrigin + (right * (u * focalDist)) + (up * (v * focalDist)) + (forward * focalDist);

        // 3. Sample the Lens (Aperture)
        /*
        float3 lensOffset = f3(0.0f, 0.0f, 0.0f);

        int blades = 6;             // 6 = Hexagon, 5 = Pentagon, etc.
        float rotation = 0.0f;      // Rotate the bokeh shape (in radians)
        // ---------------------

        float r1 = rand(&localState); 
        float r2 = rand(&localState);

        // 1. Pick which "slice" (blade) of the aperture we are in
        float bladeIdx = floorf(r1 * (float)blades);
        
        // 2. Map r1 to the range [0, 1] within that specific slice
        // This effectively re-uses the random number to save a generator call
        float r1_local = r1 * (float)blades - bladeIdx;

        // 3. Calculate the angles for the two vertices of this triangle slice
        float twoPi = 6.283185f;
        float theta1 = ((bladeIdx) / (float)blades) * twoPi + rotation;
        float theta2 = ((bladeIdx + 1.0f) / (float)blades) * twoPi + rotation;

        // 4. Calculate the vertex positions on the aperture rim
        // We use aperture radius as the scale
        float2 v1 = make_float2(cosf(theta1), sinf(theta1));
        float2 v2 = make_float2(cosf(theta2), sinf(theta2));

        // 5. Uniformly sample the triangle defined by Center(0,0), V1, V2
        // sqrtf(r2) ensures we sample area uniformly (less density near center)
        float radiusScale = aperture * sqrtf(r2);
        
        // Linear interpolation between the two rim vertices
        float2 p = (v1 * (1.0f - r1_local) + v2 * r1_local) * radiusScale;

        float lensU = p.x;
        float lensV = p.y;

        // Apply to camera basis vectors
        lensOffset = (right * lensU) + (up * lensV);


        lensOffset = aperture > 0.0f ? f3() : lensOffset;
        */

        // 4. Final Ray Construction
        //r.origin = cameraOrigin + lensOffset;
        r.origin = cameraOrigin;
        r.direction = normalize(focalPoint - r.origin);

        return r;
    }

    __device__ __forceinline__ Ray generateCameraRayRecordOffset(RNGState& localState, int x, int y, half2& jitter) const
    {
        Ray r;
        float aspect = (float)w / (float)h;

        float jitterX = 0.0f;
        float jitterY = 0.0f;

        if (antiAliasJitterDist != 0.0f) {
            jitterX = (rand(&localState) - 0.5f) * antiAliasJitterDist;
            jitterY = (rand(&localState) - 0.5f) * antiAliasJitterDist;
        }

        jitter.x = __float2half(jitterX);
        jitter.y = __float2half(jitterY);

        float u = (2.0f * ((x + jitterX) / (float)w) - 1.0f) * aspect * fovScale;
        float v = (2.0f * ((y + jitterY) / (float)h) - 1.0f) * fovScale;

        float3 focalPoint = cameraOrigin + (right * (u * focalDist)) + (up * (v * focalDist)) + (forward * focalDist);

        r.origin = cameraOrigin;
        r.direction = normalize(focalPoint - r.origin);

        return r;
    }

    __host__ void preCompute()
    {
        float3 localForward = f3(0.0f, 0.0f, -1.0f);

        float3 worldForward = rotateX(localForward, xRot);
        worldForward = rotateY(worldForward, yRot);
        worldForward = rotateZ(worldForward, zRot);

        forward = normalize(worldForward);

        float3 localRight = f3(1.0f, 0.0f, 0.0f);
        right = normalize(rotateZ(rotateY(rotateX(localRight, xRot), yRot), zRot));

        float3 localUp = f3(0.0f, 1.0f, 0.0f);
        up = normalize(rotateZ(rotateY(rotateX(localUp, xRot), yRot), zRot));

    }

    __host__ __device__ __forceinline__ float3 getForwardVector() const
    {
        return forward;
    }

    __host__ __device__ __forceinline__ float3 getRightVector() const
    {
        return right;
    }

    __host__ __device__ __forceinline__ float3 getUpVector() const
    {
        return up;
    }

    // dark magic
    __device__ __forceinline__ bool worldToRaster(const float3& pointWorld, float2& pixelPos) const
    {
        float3 dir = pointWorld - cameraOrigin;

        float3 fwd = getForwardVector();
        float3 right = getRightVector();
        float3 up = getUpVector();

        float distZ = dot(dir, fwd);

        if (distZ <= 0.001f) return false; 

        float distX = dot(dir, right);
        float distY = dot(dir, up);

        float slopeX = distX / distZ;
        float slopeY = distY / distZ;
        
        float aspect = (float)w / (float)h;

        float ndcX = slopeX / (aspect * fovScale);
        float ndcY = slopeY / fovScale;

        if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f) {
            return false;
        }
        
        pixelPos.x = (ndcX + 1.0f) * 0.5f * (float)w;
        pixelPos.y = (ndcY + 1.0f) * 0.5f * (float)h;

        return true;
    }

    __host__ __device__ __forceinline__ inline float getInitialRayFootprint() const
    {
        return atanf(fovScale / (float)h);
    }


};

__device__ __forceinline__ void drawLine(float4* overlay, Camera camera, float3 p1, float3 p2, float3 color, int thickness)
{
    float nearClip = 0.002f;
    float3 camPos = camera.cameraOrigin;
    float3 camFwd = camera.forward;

    float d1 = dot(p1 - camPos, camFwd) - nearClip;
    float d2 = dot(p2 - camPos, camFwd) - nearClip;

    if (d1 < 0.0f && d2 < 0.0f) return;

    if (d1 < 0.0f) {
        float t = d1 / (d1 - d2);
        p1 = p1 + (p2 - p1) * t;
    } 
    else if (d2 < 0.0f) {
        float t = d2 / (d2 - d1);
        p2 = p2 + (p1 - p2) * t;
    }

    float2 pxf1, pxf2;

    if (!camera.worldToRaster(p1, pxf1) || !camera.worldToRaster(p2, pxf2))
        return;
    
    int x0 = (int)pxf1.x;
    int y0 = (int)pxf1.y;

    int x1 = (int)pxf2.x;
    int y1 = (int)pxf2.y;

    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; 

    // --- THICKNESS LOGIC ---
    // 1. Determine the major axis (is the line more vertical or horizontal?)
    bool isSteep = abs(y1 - y0) > abs(x1 - x0);
    
    // 2. Calculate the span offsets based on thickness
    int startW = -(thickness / 2);
    int endW = startW + thickness - 1;

    while (true) {
        // 3. Draw the perpendicular span instead of a single point
        for (int w = startW; w <= endW; ++w) {
            // Offset X if steep, offset Y if shallow
            int px = isSteep ? (x0 + w) : x0;
            int py = isSteep ? y0 : (y0 + w);

            if (px >= 0 && px < camera.w && py >= 0 && py < camera.h) {
                int pixelIndex = py * camera.w + px;
                
                atomicExch(&overlay[pixelIndex].x, color.x);
                atomicExch(&overlay[pixelIndex].y, color.y);
                atomicExch(&overlay[pixelIndex].z, color.z);
            }
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { 
            err += dy; 
            x0 += sx; 
        }
        if (e2 <= dx) { 
            err += dx; 
            y0 += sy; 
        }
    }
}

// Helper function to determine where a point lies relative to the screen bounds
__device__ __forceinline__ int computeOutCode(float x, float y, float w, float h) {
    int code = 0;
    if (x < 0.0f) code |= 1;      // Left
    else if (x >= w) code |= 2;   // Right
    if (y < 0.0f) code |= 4;      // Top (assuming 0 is top)
    else if (y >= h) code |= 8;   // Bottom
    return code;
}

__device__ __forceinline__ void drawRay(float4* overlay, Camera camera, float3 origin, float3 dir, float3 color, int thickness)
{
    // 1. Extend the ray to a distant point far beyond the scene bounds
    float3 p1 = origin;
    float3 p2 = origin + (dir * 100000.0f); 

    float nearClip = 0.002f;
    float3 camPos = camera.cameraOrigin;
    float3 camFwd = camera.forward;

    // 2. 3D Near-Plane Clipping (Prevents behind-camera projection issues)
    float d1 = dot(p1 - camPos, camFwd) - nearClip;
    float d2 = dot(p2 - camPos, camFwd) - nearClip;

    if (d1 < 0.0f && d2 < 0.0f) return;

    if (d1 < 0.0f) {
        float t = d1 / (d1 - d2);
        p1 = p1 + (p2 - p1) * t;
    } 
    else if (d2 < 0.0f) {
        float t = d2 / (d2 - d1);
        p2 = p2 + (p1 - p2) * t;
    }

    float2 pxf1, pxf2;
    if (!camera.worldToRaster(p1, pxf1) || !camera.worldToRaster(p2, pxf2))
        return;
    
    // 3. Cohen-Sutherland 2D Screen Clipping
    // CRITICAL: This stops the GPU from iterating thousands of times off-screen.
    float x0 = pxf1.x, y0 = pxf1.y;
    float x1 = pxf2.x, y1 = pxf2.y;
    float w = (float)camera.w;
    float h = (float)camera.h;

    int outcode0 = computeOutCode(x0, y0, w, h);
    int outcode1 = computeOutCode(x1, y1, w, h);
    bool accept = false;

    while (true) {
        if (!(outcode0 | outcode1)) {
            // Both points inside screen bounds
            accept = true;
            break;
        } else if (outcode0 & outcode1) {
            // Both points share an outside zone (e.g., both above screen), line is invisible
            break;
        } else {
            // Calculate intersection with the screen edge
            float x, y;
            int outcodeOut = outcode0 ? outcode0 : outcode1;

            if (outcodeOut & 8) {        // Bottom edge
                x = x0 + (x1 - x0) * (h - 1.0f - y0) / (y1 - y0);
                y = h - 1.0f;
            } else if (outcodeOut & 4) { // Top edge
                x = x0 + (x1 - x0) * (0.0f - y0) / (y1 - y0);
                y = 0.0f;
            } else if (outcodeOut & 2) { // Right edge
                y = y0 + (y1 - y0) * (w - 1.0f - x0) / (x1 - x0);
                x = w - 1.0f;
            } else if (outcodeOut & 1) { // Left edge
                y = y0 + (y1 - y0) * (0.0f - x0) / (x1 - x0);
                x = 0.0f;
            }

            // Move the outside point to the intersection
            if (outcodeOut == outcode0) {
                x0 = x; y0 = y;
                outcode0 = computeOutCode(x0, y0, w, h);
            } else {
                x1 = x; y1 = y;
                outcode1 = computeOutCode(x1, y1, w, h);
            }
        }
    }

    if (!accept) return; // Ray does not intersect the screen at all

    // 4. Bresenham Line Drawing (Now strictly bounded to the screen)
    int ix0 = (int)x0; int iy0 = (int)y0;
    int ix1 = (int)x1; int iy1 = (int)y1;

    int dx = abs(ix1 - ix0);
    int dy = -abs(iy1 - iy0);
    int sx = ix0 < ix1 ? 1 : -1;
    int sy = iy0 < iy1 ? 1 : -1;
    int err = dx + dy; 

    bool isSteep = abs(iy1 - iy0) > abs(ix1 - ix0);
    int startW = -(thickness / 2);
    int endW = startW + thickness - 1;

    while (true) {
        for (int w_offset = startW; w_offset <= endW; ++w_offset) {
            int px = isSteep ? (ix0 + w_offset) : ix0;
            int py = isSteep ? iy0 : (iy0 + w_offset);

            // We still keep the bounds check here to handle the thickness offsets
            if (px >= 0 && px < camera.w && py >= 0 && py < camera.h) {
                int pixelIndex = py * camera.w + px;
                atomicExch(&overlay[pixelIndex].x, color.x);
                atomicExch(&overlay[pixelIndex].y, color.y);
                atomicExch(&overlay[pixelIndex].z, color.z);
            }
        }

        if (ix0 == ix1 && iy0 == iy1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; ix0 += sx; }
        if (e2 <= dx) { err += dx; iy0 += sy; }
    }
}

struct PathVertices {
    // material ID at this vertex
    int* materialID;
    
    // location of this vertex
    float4* pt;

    // normal at this vertex
    float4* n;

    // direction from this vertex to the previous
    float4* wo;

    // uv coordinates of this vertex
    float2* uv;

    //float4* wi; storing is not neccesary

    // throughput
    float4* beta;

    // pdf of going from previous vertex to this in area measure
    float* pdfFwd;
    
    // pdf of going from current vertex to previous vertex in area measure
    //float* pdfRev;
    
    // accumulated weight up this this vertex
    float* misWeight; 
    
    /*
    Equal to (for intermediate vertices):

    G - Geometry factor to convert a reverse pdf into area measure around the previous vertex (cos at prev, distanceSQR from previous)
    pdfFwd - Forward pdf of generating the current vertex
    PREVpdfRev - solid angle pdf of generating the vertex before the previous vertex
    d_vcm for the previous vertex
    d_vc for the previous vertex
    */
    float* d_vc;

    // stores the forward pdf for the previous vertex (area measure)
    float* d_vcm;
    bool* isDelta;
    
    int* lightInd;
    bool* backface;
};

inline __device__ __forceinline__ int pathBufferIdx(int w, int h, int x, int y, int depth)
{
    return (depth * w * h) + y * w + x;
}

// rasterizes a path defined by the pathvertices. Used for debugging and visualization
__device__ __forceinline__ void drawPath(float4* overlay, PathVertices* path, Camera camera, int x, int y, int w,
    int depth, int maxDepth, float3 color)
{
    for (int i = 0; i < depth - 1; i++)
    {
        //float ratio = (float)(i+1) / float(depth);
        int pathIDX1 = pathBufferIdx(w, x, y, i, maxDepth);
        int pathIDX2 = pathBufferIdx(w, x, y, i+1, maxDepth);
        drawLine(overlay, camera, f3(path->pt[pathIDX1]), f3(path->pt[pathIDX2]), color, 3);
    }
}

static __device__ __forceinline__ void debugPrintPath(
    int w, int h, int x, int y, int maxDepth,
    const PathVertices& PV)
{
    //const int pixelIndex = y * w + x;

    // Print header first
    printf("=== PATH DUMP pixel (%d, %d) Length: %d ===\n", x, y, maxDepth);

    // Print each depth entry with its own single printf
    // (but each line is atomic so it will not mix)
    for (int depth = 0; depth < maxDepth; depth++)
    {
        int idx = pathBufferIdx(w, h, x, y, depth);

        printf(
            "Depth %d\n"
            "  materialID: %d\n"
            "  pt:   (%.3f, %.3f, %.3f)\n"
            "  n:    (%.3f, %.3f, %.3f)\n"
            "  wo:   (%.3f, %.3f, %.3f)\n"
            "  uv:   (%.3f, %.3f)\n"
            "  beta: (%.3f, %.3f, %.3f)\n"
            "  d_vc: %.6f\n"
            "  d_vcm: %.6f\n"
            "  isDelta: %d\n"
            "  lightInd: %d\n"
            "  backface: %d\n\n",

            depth,
            PV.materialID[idx],
            PV.pt[idx].x, PV.pt[idx].y, PV.pt[idx].z,
            PV.n[idx].x, PV.n[idx].y, PV.n[idx].z,
            PV.wo[idx].x, PV.wo[idx].y, PV.wo[idx].z,
            PV.uv[idx].x, PV.uv[idx].y,
            PV.beta[idx].x, PV.beta[idx].y, PV.beta[idx].z,
            PV.d_vc[idx],
            PV.d_vcm[idx],
            PV.isDelta[idx],
            PV.lightInd[idx],
            PV.backface[idx]
        );
    }
}


struct Intersection
{
    float3 point;
    float3 normal;
    float3 emission;

    float2 uv;
    //Ray ray;
    //Triangle tri;
    int triIDX;
    int materialID;
    bool valid;
    bool backface;

    float dist;

    __device__ Intersection() {valid = false; uv = f2(-1.0f);};
};

enum IntegratorChoice {
    UNIDIRECTIONAL = 0,
    BIDIRECTIONAL = 1,
    NAIVE_UNIDIRECTIONAL = 2,
    VCM = 3,
    SPPM = 4,
    WAVEFRONT_UNIDIRECTIONAL = 5,
    VOLUME_SIMPLE = 6,
    OPTIX_NORMAL = 7,
    OPTIX_RESTIR_PT = 8
};

enum TransportMode {
    TRANSPORTMODE_IMPORTANCE = 0,
    TRANSPORTMODE_RADIANCE = 1
};

__host__ inline int matchIntegrator(std::string name)
{
    if (name == "UNIDIRECTIONAL") return 0;
    else if (name == "BIDIRECTIONAL" || name == "BDPT") return 1;
    else if (name == "NAIVE_UNIDIRECTIONAL") return 2;
    else if (name == "VCM") return 3;
    else if (name == "SPPM") return 4;
    else if (name == "UNIDIRFAST") return 5;
    else if (name == "SIMPLEVOL") return 6;
    else if (name == "OPTIXNORMAL") return 7;
    else if (name == "RESTIRPT") return 8;

    std::cerr << "Invalid Integrator Choice!\n";
    return -1;
}

struct Medium {
    float ior;
    int priority;
    float4 simpleAbsorption;
};

#define MASK_DELTA      (0x1)
#define MASK_BACKFACE   (0x2)

#define MASK_LIGHT      0xFFFFFu // 20 bits
#define MASK_MAT        0x3FFu   // 10 bits

// The raw unsigned values stored in the bitfield
#define RAW_LIGHT_ENV  0xFFFFFu // Maps to -1
#define RAW_LIGHT_NONE   0xFFFFEu // Maps to -2

/*
A highly optimized data structure to acommodate for the large spatial complexity of VCM. Please hire me
*/
struct VCMPathVertices
{
    /* To avoid the extra space of the 4th part of a float4, but also to avoid the packing issues of a float3
    separate floats are used instead of packing inside a uint because precision is important for positions.
    */
    float* pos_x; 
    float* pos_y; 
    float* pos_z;

    /*
    These are decoded into float3's in the kernels.
    */
    uint32_t* packedNormal;
    uint32_t* packedWo;

    half* beta_x;
    half* beta_y;
    half* beta_z;

    half2* packedUV;

    /*
    bit 0: isDelta
    bit 1: backface
    bit 2-21: light index
    bit 22-31: material ID
    */
    uint32_t* packedInfo;
    
    float* d_vc;
    float* d_vcm;
};

__device__ __forceinline__ void setAllInfo(VCMPathVertices& x, int idx, bool isDelta, bool isBackface, int lightID, int matID) 
{
    uint32_t info = 0;
    info |= (isDelta ? 1u : 0u);
    info |= (isBackface ? 1u : 0u) << 1;
    
    // (-1 & MASK) becomes 0xFFFFF automatically
    // (-2 & MASK) becomes 0xFFFFE automatically
    info |= (uint32_t)(lightID & MASK_LIGHT) << 2; 
    
    info |= (uint32_t)(matID & MASK_MAT) << 22;
    x.packedInfo[idx] = info;
}

__device__ __forceinline__ void getAllInfo(
    const VCMPathVertices& x, 
    int idx, 
    bool& isDelta, 
    bool& isBackface, 
    int& lightID, 
    int& matID
) 
{
    uint32_t info = __ldg(&x.packedInfo[idx]);

    // Bit 0: isDelta
    isDelta = (info & 1u);

    // Bit 1: isBackface
    isBackface = (info >> 1) & 1u;

    // Bits 2-21: lightID (20 bits)
    uint32_t rawLight = (info >> 2) & MASK_LIGHT;
    
    // Decode special flags
    if (rawLight == RAW_LIGHT_NONE) {
        lightID = -1;
    } else if (rawLight == RAW_LIGHT_ENV) {
        lightID = -2;
    } else {
        lightID = (int)rawLight;
    }

    // Bits 22-31: matID (10 bits)
    matID = (info >> 22) & MASK_MAT;
}

__device__ __forceinline__ int getLightIndex(const VCMPathVertices& x, int idx) {
    uint32_t rawLight = (__ldg(&x.packedInfo[idx]) >> 2) & MASK_LIGHT;

    if (rawLight == RAW_LIGHT_NONE) return -1;
    if (rawLight == RAW_LIGHT_ENV)  return -2;
    return (int)rawLight;
}

__device__ __forceinline__ void setLightIndex(VCMPathVertices& x, int idx, int val) {
    uint32_t current = x.packedInfo[idx];
    
    // Clear the current light bits (Bits 2-21)
    // ~(0xFFFFF << 2) 
    current &= ~(MASK_LIGHT << 2); 
    
    // Apply new value (implicitly handles -1/-2 via masking)
    current |= (uint32_t)(val & MASK_LIGHT) << 2;
    
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ int getMaterialID(const VCMPathVertices& x, int idx) {
    return (__ldg(&x.packedInfo[idx]) >> 22) & MASK_MAT;
}

__device__ __forceinline__ void setMaterialID(VCMPathVertices& x, int idx, int val) {
    uint32_t current = __ldg(&x.packedInfo[idx]);
    current &= ~(MASK_MAT << 22); 
    
    current |= (uint32_t)(val & MASK_MAT) << 22;
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ bool getIsDelta(const VCMPathVertices& x, int idx) {
    return (__ldg(&x.packedInfo[idx]) & 1u);
}

__device__ __forceinline__ bool getIsBackface(const VCMPathVertices& x, int idx) {
    return (__ldg(&x.packedInfo[idx]) >> 1) & 1u;
}

__device__ __forceinline__ void setIsDelta(VCMPathVertices& x, int idx, bool val) {
    uint32_t current = __ldg(&x.packedInfo[idx]);
    current &= ~MASK_DELTA; // Clear bit
    current |= (val ? 1u : 0u);
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ void setIsBackface(VCMPathVertices& x, int idx, bool val) {
    uint32_t current = __ldg(&x.packedInfo[idx]);
    current &= ~MASK_BACKFACE;
    current |= (val ? 1u : 0u) << 1;
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ float3 getPos(const VCMPathVertices& verts, int idx)
{
    return f3(
        __ldg(&verts.pos_x[idx]),
        __ldg(&verts.pos_y[idx]),
        __ldg(&verts.pos_z[idx]));
}

__device__ __forceinline__ void setPos(VCMPathVertices& verts, int idx, float3 p)
{
    verts.pos_x[idx] = p.x;
    verts.pos_y[idx] = p.y;
    verts.pos_z[idx] = p.z;
}

__device__ __forceinline__ float3 getNormal(const VCMPathVertices& x, int idx) {
    return unpackOct(__ldg(&x.packedNormal[idx]));
}

__device__ __forceinline__ void setNormal(VCMPathVertices& x, int idx, float3 n) {
    x.packedNormal[idx] = packOct(n);
}

__device__ __forceinline__ float3 getWo(const VCMPathVertices& x, int idx) {
    return unpackOct(__ldg(&x.packedWo[idx]));
}

__device__ __forceinline__ void setWo(VCMPathVertices& x, int idx, float3 wo) {
    x.packedWo[idx] = packOct(wo);
}

__device__ __forceinline__ float3 getBeta(const VCMPathVertices& x, int idx) {
    //return fromRGB9E5(x.packedBeta[idx]);
    return f3(
        __half2float(__ldg(&x.beta_x[idx])),
        __half2float(__ldg(&x.beta_y[idx])),
        __half2float(__ldg(&x.beta_z[idx])));
}

__device__ __forceinline__ void setBeta(VCMPathVertices& x, int idx, float3 b) {
    //x.packedBeta[idx] = toRGB9E5(b);
    x.beta_x[idx] = __float2half(b.x);
    x.beta_y[idx] = __float2half(b.y);
    x.beta_z[idx] = __float2half(b.z);
}

__device__ __forceinline__ float2 getUV(const VCMPathVertices& x, int idx) {
    return __half22float2(__ldg(&x.packedUV[idx]));
}

__device__ inline void setUV(VCMPathVertices& x, int idx, float2 uv) {
    x.packedUV[idx] = __float22half2_rn(uv);
}

__device__ __forceinline__ float getD_vc(const VCMPathVertices& x, int idx) {
    return __ldg(&x.d_vc[idx]);
}

__device__ __forceinline__ void setD_vc(VCMPathVertices& x, int idx, float val) {
    x.d_vc[idx] = val;
}

__device__ __forceinline__ float getD_vcm(const VCMPathVertices& x, int idx) {
    return __ldg(&x.d_vcm[idx]);
}

__device__ __forceinline__ void setD_vcm(VCMPathVertices& x, int idx, float val) {
    x.d_vcm[idx] = val;
}

/*
Struct containing photon data for vcm.
*/
struct Photons
{
    float4* pos_plus_vm;

    uint32_t* packedWi;
    uint32_t* packedNormal;
    
    half* beta_x;
    half* beta_y;
    half* beta_z;

    float* d_vcm;
};

__device__ __forceinline__ float3 getPos(const Photons& ps, int idx)
{
    return f3(__ldg(&ps.pos_plus_vm[idx]));
}

__device__ __forceinline__ void getPosVM(const Photons& ps, int idx, float3& pos, float& vm)
{
    float4 packed = __ldg(&ps.pos_plus_vm[idx]);
    pos = f3(packed);
    vm = packed.w;
}

__device__ __forceinline__ void setPos(Photons& ps, int idx, float3 p)
{
    ps.pos_plus_vm[idx].x = p.x;
    ps.pos_plus_vm[idx].y = p.y;
    ps.pos_plus_vm[idx].z = p.z;
}

__device__ __forceinline__ void setPackedPosVM(Photons& ps, int idx, float3 p, float vm)
{
    ps.pos_plus_vm[idx] = make_float4(p.x, p.y, p.z, vm);
}

__device__ __forceinline__ void getNormalInfo(const Photons& x, int idx, float3& normal, bool& backface) {
    normal = unpackOctFlags(__ldg(&x.packedNormal[idx]), &backface, nullptr);
}

__device__ __forceinline__ void setNormalInfo(Photons& x, int idx, float3 n, bool backface) {
    x.packedNormal[idx] = packOctFlags(n, backface, false);
}

__device__ __forceinline__ float3 getWi(const Photons& x, int idx) {
    return unpackOct(__ldg(&x.packedWi[idx]));
}

__device__ __forceinline__ void setWi(Photons& x, int idx, float3 wi) {
    x.packedWi[idx] = packOct(wi);
}

__device__ __forceinline__ float3 getBeta(const Photons& x, int idx) {
    return f3(
        __half2float(__ldg(&x.beta_x[idx])),
        __half2float(__ldg(&x.beta_y[idx])),
        __half2float(__ldg(&x.beta_z[idx])));
}

__device__ __forceinline__ void setBeta(Photons& x, int idx, float3 b) {
    x.beta_x[idx] = __float2half(b.x);
    x.beta_y[idx] = __float2half(b.y);
    x.beta_z[idx] = __float2half(b.z);
}

// dont call this function
__device__ __forceinline__ float getD_vm(const Photons& x, int idx) {
    return __ldg(&x.pos_plus_vm[idx]).w;
}

__device__ __forceinline__ void setD_vm(Photons& x, int idx, float val) {
    x.pos_plus_vm[idx].w = val;
}

__device__ __forceinline__ float getD_vcm(const Photons& x, int idx) {
    return __ldg(&x.d_vcm[idx]);
}

__device__ __forceinline__ void setD_vcm(Photons& x, int idx, float val) {
    x.d_vcm[idx] = val;
}