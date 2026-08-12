/*
Handles the image writing.

Uses a 1d vector of pixels instead of 2d for minor optimization.

*/

#include <cstdint>
#include <utility>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <array>

#include "imageUtil.cuh"
#include "util.cuh"


#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)


void createBMPHeaders(int width, int height, BMPFileHeader &fileHeader, BMPInfoHeader &infoHeader);

Image::Image(int w, int h) : width(w), height(h), pixels(std::vector<float4>(w * h)), postProcess(true) {}

Image::~Image() {}

int Image::toIndex(int x, int y) {
    return y * width + x;
}

std::pair<int,int> Image::fromIndex(int i) {
    int y = i / width; // integer division
    int x = i % width; // remainder
    return {x, y};
}

void Image::setColor(int x, int y, float4 c) {
    pixels[toIndex(x, y)] = c;
}

float4 Image::getColor(int x, int y) {
    return pixels[toIndex(x, y)];
}

void Image::saveImageBMP(std::string fileName) {

    std::vector<float4> data = postProcess ? postProcessImage() : pixels;

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    createBMPHeaders(width, height, fileHeader, infoHeader);

    std::ofstream out(fileName, std::ios::binary);
    out.write((char*)&fileHeader, sizeof(fileHeader));
    out.write((char*)&infoHeader, sizeof(infoHeader));

    int rowSize = (3 * width + 3) & (~3); // each row padded to multiple of 4 bytes

    float4 c;
    std::vector<unsigned char> row(rowSize);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            c = data[toIndex(x, y)];

            row[x*3 + 0] = static_cast<unsigned char>(clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[x*3 + 1] = static_cast<unsigned char>(clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[x*3 + 2] = static_cast<unsigned char>(clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f);

        }

        out.write(reinterpret_cast<char*>(row.data()), rowSize);
    }

    out.close();
}

void Image::saveImageCSV() 
{
    std::ofstream csvOut("renderCSV.csv");
    csvOut << std::scientific << std::setprecision(3);

    float4 c;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            c = getColor(x, y);

            csvOut << "\"(" << c.x << ", " << c.y << ", " << c.z << ")\"";

            if (x < width - 1) {
                csvOut << ",";
            }
        }
        csvOut << "\n";
    }
    csvOut.close();
}

void Image::saveImageCSV_MONO(int choice) 
{
    std::ofstream csvOut("renderCSV.csv");
    csvOut << std::scientific << std::setprecision(3);

    float4 c;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            c = getColor(x, y);

            csvOut << getFloat4Component(c, choice);

            if (x < width - 1) {
                csvOut << ",";
            }
        }
        csvOut << "\n";
    }
    csvOut.close();
}

Image loadBMPToImage(const std::string &filename, bool isData) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open BMP: " << filename << "\n";
        return Image(0,0);
    }

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    in.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    in.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (fileHeader.bfType != 0x4D42) {
        std::cerr << "Not a BMP file: " << filename << "\n";
        return Image(0,0);
    }

    if (infoHeader.biBitCount != 24) {
        std::cerr << "Only 24-bit BMP supported: " << filename << "\n";
        return Image(0,0);
    }

    int width = infoHeader.biWidth;
    int height = infoHeader.biHeight;

    Image img = Image(width, height); // resize your Image

    int rowSize = (3 * width + 3) & (~3); // each row padded to multiple of 4
    std::vector<unsigned char> row(rowSize);

    for (int y = 0; y < height; y++) {
        in.read(reinterpret_cast<char*>(row.data()), rowSize);
        for (int x = 0; x < width; x++) {
            float b = row[x*3 + 0] / 255.0f;
            float g = row[x*3 + 1] / 255.0f;
            float r = row[x*3 + 2] / 255.0f;

            if (!isData)
            {
                r = powf(r, 2.2f);
                g = powf(g, 2.2f);
                b = powf(b, 2.2f);
            }

            img.setColor(x, height - 1 - y, make_float4(r, g, b, 1.0f)); // flip y
        }
    }

    in.close();
    return img;
}

std::vector<float4> Image::data()
{
    return pixels;
}

static const std::array<float4, 3> aces_input_matrix =
{
    f4(0.59719f, 0.35458f, 0.04823f),
    f4(0.07600f, 0.90834f, 0.01566f),
    f4(0.02840f, 0.13383f, 0.83777f)
};

static const std::array<float4, 3> aces_output_matrix =
{
    f4( 1.60475f, -0.53108f, -0.07367f),
    f4(-0.10208f,  1.10813f, -0.00605f),
    f4(-0.00327f, -0.07276f,  1.07602f)
};

float4 mul(const std::array<float4, 3>& m, const float4& v)
{
    float x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
    float y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
    float z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;
    return f4(x, y, z);
}

float4 rtt_and_odt_fit(float4 v)
{
    float4 a = v * (v + f4(0.0245786f)) - f4(0.000090537f);
    float4 b = v * (0.983729f * v + f4(0.4329510f)) + f4(0.238081f);
    return a / b;
}

float4 Image::aces_fitted(float4 c)
{
    c = mul(aces_input_matrix, c);
    c = rtt_and_odt_fit(c);
    return mul(aces_output_matrix, c);
}

float4 Image::toneMap(float4 color)
{
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    return clampf4((color * (A * color + f4(B))) / (color * (C * color + f4(D)) + f4(E)), 0.0f, 1.0f);
}

float4 Image::gammaCorrect(float4 c)
{
    float invGamma = 1.0f / 2.2f;
    return f4(
        powf(c.x, invGamma),
        powf(c.y, invGamma),
        powf(c.z, invGamma),
        0.0f
    );
}

std::vector<float4> Image::postProcessImage()
{
    int totalPixels = width * height;
    
    std::vector<float4> processed(totalPixels);
    
    float exposure = 5.0f;

    #pragma omp parallel for
    for (int i = 0; i < totalPixels; i++)
    {
        float4 exposedPixel = pixels[i] * f4(exposure, exposure, exposure, 1.0f);
        
        processed[i] = gammaCorrect(use_fitted_aces ? aces_fitted(exposedPixel) : toneMap(exposedPixel));
    }
    return processed;
}

void createBMPHeaders(int width, int height, BMPFileHeader &fileHeader, BMPInfoHeader &infoHeader) {
    int rowSize = (3 * width + 3) & (~3);
    int imageSize = rowSize * height;

    // File header
    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imageSize;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    // Info header
    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;
}

__global__ void cleanAndFormatImage(
    float4* accumulationBuffer, // Your raw 'colors' buffer (Sum of samples)
    float4* overlayBuffer,      // Your 'overlay' buffer
    float4* outputBuffer,       // A temporary buffer to store the result for saving
    int w, int h, 
    int currentSampleCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;

    int pixelIndex = idy * w + idx;

    // 1. Read the raw accumulated color
    float4 acc = accumulationBuffer[pixelIndex];
    float4 ov = overlayBuffer[pixelIndex];
    float4 finalColor;
    
    // 2. Check for NaNs/Infs BEFORE normalization
    if (isnan(acc.x) || isnan(acc.y) || isnan(acc.z)) {
        finalColor = f4(1.0f, 0.0f, 1.0f);
    } 
    else if (isinf(acc.x) || isinf(acc.y) || isinf(acc.z)) {
        finalColor = f4(0.0f, 1.0f, 0.0f);
    } 
    else if (acc.x < 0 || acc.y < 0 || acc.z < 0) {
        finalColor = f4(0.0f, 0.0f, 1.0f);
    } 
    else {
        // 3. Normalize (Average the samples)
        float scale = 1.0f / (float)(currentSampleCount + 1);
        finalColor = make_float4(acc.x * scale, acc.y * scale, acc.z * scale, 1.0f);
    }

    // 4. Apply Overlay (if present)
    // Assuming overlay logic: if overlay is NOT black, it overrides the render
    if (ov.x != 0.0f || ov.y != 0.0f || ov.z != 0.0f) {
        finalColor = ov;
    }

    // 5. Write to the output buffer
    outputBuffer[pixelIndex] = finalColor;
}

__global__ void cleanAndFormatImageNoOverlay(
    float4* accumulationBuffer, // Your raw 'colors' buffer (Sum of samples)
    float4* outputBuffer,       // A temporary buffer to store the result for saving
    int w, int h, 
    int currentSampleCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;

    int pixelIndex = idy * w + idx;

    // 1. Read the raw accumulated color
    float4 acc = accumulationBuffer[pixelIndex];
    float4 finalColor;

    // 2. Check for NaNs/Infs BEFORE normalization
    if (isnan(acc.x) || isnan(acc.y) || isnan(acc.z)) {
        finalColor = f4(1.0f, 0.0f, 1.0f);
    } 
    else if (isinf(acc.x) || isinf(acc.y) || isinf(acc.z)) {
        finalColor = f4(0.0f, 1.0f, 0.0f);
    } 
    else if (acc.x < 0 || acc.y < 0 || acc.z < 0) {
        finalColor = f4(0.0f, 0.0f, 1.0f);
    } 
    else {
        // 3. Normalize (Average the samples)
        float scale = 1.0f / (float)(currentSampleCount + 1);
        finalColor = make_float4(acc.x * scale, acc.y * scale, acc.z * scale, 1.0f);
    }

    outputBuffer[pixelIndex] = finalColor;
}

__constant__ float c_aces_input_matrix[9] = {
    0.59719f, 0.35458f, 0.04823f,
    0.07600f, 0.90834f, 0.01566f,
    0.02840f, 0.13383f, 0.83777f
};

__constant__ float c_aces_output_matrix[9] = {
     1.60475f, -0.53108f, -0.07367f,
    -0.10208f,  1.10813f, -0.00605f,
    -0.00327f, -0.07276f,  1.07602f
};

__device__ float3 mul_mat3(const float matrix[9], const float3& v) {
    return make_float3(
        matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z,
        matrix[3] * v.x + matrix[4] * v.y + matrix[5] * v.z,
        matrix[6] * v.x + matrix[7] * v.y + matrix[8] * v.z
    );
}

__device__ float3 rtt_and_odt_fit(float3 v) {
    float3 a = make_float3(
        v.x * (v.x + 0.0245786f) - 0.000090537f,
        v.y * (v.y + 0.0245786f) - 0.000090537f,
        v.z * (v.z + 0.0245786f) - 0.000090537f
    );
    float3 b = make_float3(
        v.x * (0.983729f * v.x + 0.4329510f) + 0.238081f,
        v.y * (0.983729f * v.y + 0.4329510f) + 0.238081f,
        v.z * (0.983729f * v.z + 0.4329510f) + 0.238081f
    );
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ float3 device_aces_fitted(float3 c) {
    c = mul_mat3(c_aces_input_matrix, c);
    c = rtt_and_odt_fit(c);
    c = mul_mat3(c_aces_output_matrix, c);
    return c;
}

__device__ float3 device_toneMap(float3 c) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    float3 num = make_float3(
        c.x * (A * c.x + B),
        c.y * (A * c.y + B),
        c.z * (A * c.z + B)
    );
    float3 den = make_float3(
        c.x * (C * c.x + D) + E,
        c.y * (C * c.y + D) + E,
        c.z * (C * c.z + D) + E
    );
    
    // clamp to 0-1
    return make_float3(
        fminf(fmaxf(num.x / den.x, 0.0f), 1.0f),
        fminf(fmaxf(num.y / den.y, 0.0f), 1.0f),
        fminf(fmaxf(num.z / den.z, 0.0f), 1.0f)
    );
}

__device__ float3 device_gammaCorrect(float3 c) {
    float invGamma = 1.0f / 2.2f;
    return make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
}

__global__ void cleanFormatAndPostProcessImage(
    float4* accumulationBuffer, 
    float4* overlayBuffer,      
    float4* outputBuffer,       
    int w, int h, 
    int currentSampleCount,
    float exposure,
    bool use_fitted_aces) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;
    int pixelIndex = idy * w + idx;

    float4 acc = accumulationBuffer[pixelIndex];
    float3 color;

    // 1. Check for NaNs/Infs
    if (isnan(acc.x) || isnan(acc.y) || isnan(acc.z)) {
        color = make_float3(1.0f, 0.0f, 1.0f);
    } 
    else if (isinf(acc.x) || isinf(acc.y) || isinf(acc.z)) {
        color = make_float3(0.0f, 1.0f, 0.0f);
    } 
    else if (acc.x < 0 || acc.y < 0 || acc.z < 0) {
        color = make_float3(0.0f, 0.0f, 1.0f);
    } 
    else {
        // 2. Normalize
        float scale = 1.0f / (float)(currentSampleCount + 1);
        color = make_float3(acc.x * scale, acc.y * scale, acc.z * scale);
        
        // 3. Post-Process (Exposure -> ToneMap -> Gamma)
        color = make_float3(color.x * exposure, color.y * exposure, color.z * exposure);
        
        if (use_fitted_aces) {
            color = device_aces_fitted(color);
        } else {
            color = device_toneMap(color);
        }
        
        color = device_gammaCorrect(color);
    }

    // 4. Overlay logic (Overwrites everything if present)
    if (overlayBuffer != nullptr) {
        float4 ov = overlayBuffer[pixelIndex];
        if (ov.x != 0.0f || ov.y != 0.0f || ov.z != 0.0f) {
            color = make_float3(ov.x, ov.y, ov.z);
        }
    }

    // Write final processed float4 to buffer
    outputBuffer[pixelIndex] = make_float4(color.x, color.y, color.z, 1.0f);
}

__global__ void postProcessOnly(
    float4* inputBuffer,
    float4* outputBuffer,
    int w, int h,
    float exposure,
    bool use_fitted_aces)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;
    int pixelIndex = idy * w + idx;

    float4 c = inputBuffer[pixelIndex];
    float3 color = make_float3(c.x * exposure, c.y * exposure, c.z * exposure);

    color = use_fitted_aces ? device_aces_fitted(color) : device_toneMap(color);
    color = device_gammaCorrect(color);

    outputBuffer[pixelIndex] = make_float4(color.x, color.y, color.z, 1.0f);
}