#pragma once

#include "util.cuh"
#include <vector>

class Image 
{
private:
    float4 toneMap(float4 c);
    float4 aces_fitted(float4 c);
    float4 gammaCorrect(float4 c);
    std::vector<float4> postProcessImage();
public:
    bool postProcess;
    bool use_fitted_aces = true;
    Image(int w, int h);
    ~Image();

    void setColor(int x, int y, float4 c);
    float4 getColor(int x, int y);
    void saveImageBMP(std::string fileName);
    void saveImageCSV();
    void saveImageCSV_MONO(int choice);
    

    int toIndex(int x, int y);
    std::pair<int,int> fromIndex(int i);

    std::vector<float4> data();

    const int width, height;
    std::vector<float4> pixels;
};

Image loadBMPToImage(const std::string &filename, bool isData);

__global__ void cleanAndFormatImage(
    float4* accumulationBuffer,
    float4* overlayBuffer,
    float4* outputBuffer,
    int w, int h, 
    int currentSampleCount
);

__global__ void cleanAndFormatImageNoOverlay(
    float4* accumulationBuffer, // Your raw 'colors' buffer (Sum of samples)
    float4* outputBuffer,       // A temporary buffer to store the result for saving
    int w, int h, 
    int currentSampleCount);

__global__ void cleanFormatAndPostProcessImage(
    float4* accumulationBuffer,
    float4* overlayBuffer,
    float4* outputBuffer,
    int w, int h,
    int currentSampleCount,
    float exposure,
    bool use_fitted_aces);

// Applies exposure/tonemap/gamma to an already-normalized linear buffer
// (e.g. post-denoise output), writing into a separate output buffer so the
// linear input (which may be reused as next frame's denoiser history) is
// left untouched. inputBuffer == outputBuffer is fine when the input has no
// further use. Kept separate from cleanFormatAndPostProcessImage so
// denoising can happen on linear data, with tonemapping applied only
// afterward.
__global__ void postProcessOnly(
    float4* inputBuffer,
    float4* outputBuffer,
    int w, int h,
    float exposure,
    bool use_fitted_aces);