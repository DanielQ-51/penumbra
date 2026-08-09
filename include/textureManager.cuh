#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "textureView.cuh" // TextureView + sampleTex (device-facing)
#include <stb_image/stb_image.h>

enum TexColorSpace {
    TEX_LINEAR = 0, // metallic-roughness, occlusion, normal maps  (raw data)
    TEX_SRGB   = 1  // baseColor, emissive                         (color)
};

class TextureManager {
private:
    std::vector<cudaMipmappedArray_t> mipArrays;
    std::vector<cudaTextureObject_t>  texObjects;
    std::vector<int2>                 texDims;   // level-0 (post-cap) width/height per texture
    cudaTextureObject_t* d_handles = nullptr;
    bool dirty = true;   // d_handles needs re-upload after the last add()
    int  maxDim_ = 2048; // cap on texture width/height (VRAM control); halved past this

    // Box-filter downsample of a uchar4 level to the next-smaller mip level.
    // NOTE: for sRGB textures this averages in sRGB byte space rather than
    // linear. That is a common, acceptable approximation; revisit if strong
    // color gradients show mip banding.
    static std::vector<uchar4> downsample(const std::vector<uchar4>& src,
                                          int w, int h, int& outW, int& outH)
    {
        outW = std::max(1, w / 2);
        outH = std::max(1, h / 2);
        std::vector<uchar4> dst(outW * outH);

        auto avg = [](int a, int b, int c, int d) -> unsigned char {
            return (unsigned char)((a + b + c + d + 2) / 4);
        };

        for (int y = 0; y < outH; ++y) {
            int y0 = std::min(2 * y,     h - 1);
            int y1 = std::min(2 * y + 1, h - 1);
            for (int x = 0; x < outW; ++x) {
                int x0 = std::min(2 * x,     w - 1);
                int x1 = std::min(2 * x + 1, w - 1);

                uchar4 a = src[y0 * w + x0];
                uchar4 b = src[y0 * w + x1];
                uchar4 c = src[y1 * w + x0];
                uchar4 d = src[y1 * w + x1];

                dst[y * outW + x] = make_uchar4(
                    avg(a.x, b.x, c.x, d.x),
                    avg(a.y, b.y, c.y, d.y),
                    avg(a.z, b.z, c.z, d.z),
                    avg(a.w, b.w, c.w, d.w));
            }
        }
        return dst;
    }

public:
    TextureManager() {}

    ~TextureManager() {
        for (cudaTextureObject_t o : texObjects)
            if (o) cudaDestroyTextureObject(o);
        for (cudaMipmappedArray_t m : mipArrays)
            if (m) cudaFreeMipmappedArray(m);
        if (d_handles) cudaFree(d_handles);
    }

    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    int count() const { return (int)texObjects.size(); }

    // Level-0 (post-cap) dimensions of a texture, for ray-cone LOD baking.
    // Returns {0,0} for an out-of-range index.
    int2 dims(int idx) const {
        if (idx < 0 || idx >= (int)texDims.size()) return make_int2(0, 0);
        return texDims[idx];
    }

    // Cap texture resolution (width & height). Any larger image is box-filtered
    // down until it fits before upload. Set before loading textures.
    void setMaxDimension(int d) { maxDim_ = (d > 0) ? d : 1; }

    int addFromMemory(const unsigned char* rgba, int w, int h, TexColorSpace cs)
    {
        if (!rgba || w <= 0 || h <= 0) {
            std::cerr << "TextureManager::addFromMemory: invalid image\n";
            return -1;
        }

        std::vector<uchar4> cur(w * h);
        std::memcpy(cur.data(), rgba, (size_t)w * h * 4);

        // Resolution cap: halve until within maxDim_ (each level costs w*h*4 B in
        // VRAM, so a 4K->2K cap is a 4x saving, 4K->1K a 16x saving).
        while (w > maxDim_ || h > maxDim_) {
            int nw, nh;
            cur = downsample(cur, w, h, nw, nh);
            w = nw; h = nh;
        }

        int numLevels = 1 + (int)std::floor(std::log2((float)std::max(w, h)));

        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        cudaExtent extent = make_cudaExtent(w, h, 0); // 2D: depth == 0

        cudaMipmappedArray_t mipArray = nullptr;
        cudaMallocMipmappedArray(&mipArray, &ch, extent, numLevels);

        // Upload level 0, then generate + upload each smaller level.
        int lw = w, lh = h;
        for (int level = 0; level < numLevels; ++level) {
            cudaArray_t levelArray;
            cudaGetMipmappedArrayLevel(&levelArray, mipArray, level);

            size_t pitch = (size_t)lw * sizeof(uchar4);
            cudaMemcpy2DToArray(levelArray, 0, 0, cur.data(),
                                pitch, pitch, lh, cudaMemcpyHostToDevice);

            if (level + 1 < numLevels) {
                int nw, nh;
                cur = downsample(cur, lw, lh, nw, nh);
                lw = nw; lh = nh;
            }
        }

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeMipmappedArray;
        resDesc.res.mipmap.mipmap = mipArray;

        cudaTextureDesc texDesc = {};
        texDesc.normalizedCoords   = 1;
        texDesc.filterMode         = cudaFilterModeLinear;
        texDesc.mipmapFilterMode   = cudaFilterModeLinear;
        texDesc.readMode           = cudaReadModeNormalizedFloat;
        texDesc.addressMode[0]     = cudaAddressModeWrap;
        texDesc.addressMode[1]     = cudaAddressModeWrap;
        texDesc.sRGB               = (cs == TEX_SRGB) ? 1 : 0;
        texDesc.maxMipmapLevelClamp = (float)(numLevels - 1);

        cudaTextureObject_t obj = 0;
        cudaCreateTextureObject(&obj, &resDesc, &texDesc, nullptr);

        mipArrays.push_back(mipArray);
        texObjects.push_back(obj);
        texDims.push_back(make_int2(w, h)); // w,h are the final level-0 dims (post-cap)
        dirty = true;
        return (int)texObjects.size() - 1;
    }

    // Load a texture from a file path via stb_image (forces 4 channels).
    // Returns the texture index, or -1 on failure.
    int addFromFile(const std::string& path, TexColorSpace cs)
    {
        int w, h, n;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 4);
        if (!data) {
            std::cerr << "TextureManager::addFromFile: failed to load " << path
                      << " (" << stbi_failure_reason() << ")\n";
            return -1;
        }
        int idx = addFromMemory(data, w, h, cs);
        stbi_image_free(data);
        return idx;
    }

    // Decode an in-memory encoded image (PNG/JPG bytes) via stb_image and add it.
    // This is the path for glTF-embedded (GLB) images, whose encoded bytes live
    // in a bufferView. Returns the texture index, or -1 on failure.
    int addFromEncodedMemory(const unsigned char* encoded, size_t len, TexColorSpace cs)
    {
        int w, h, n;
        unsigned char* data = stbi_load_from_memory(encoded, (int)len, &w, &h, &n, 4);
        if (!data) {
            std::cerr << "TextureManager::addFromEncodedMemory: decode failed ("
                      << stbi_failure_reason() << ")\n";
            return -1;
        }
        int idx = addFromMemory(data, w, h, cs);
        stbi_image_free(data);
        return idx;
    }

    // Uploads the handle table if needed and returns the device view.
    TextureView getView()
    {
        if (dirty) {
            if (d_handles) cudaFree(d_handles);
            size_t bytes = texObjects.size() * sizeof(cudaTextureObject_t);
            cudaMalloc(&d_handles, bytes);
            cudaMemcpy(d_handles, texObjects.data(), bytes, cudaMemcpyHostToDevice);
            dirty = false;
        }
        return TextureView{ d_handles, (int)texObjects.size() };
    }
};
