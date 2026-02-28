#include <cuda_runtime.h>
#include <iostream>

__global__ void quantize_kernel(const float* input, int8_t* output, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple linear quantization
        float val = input[idx] / scale;
        val = fmaxf(-127.0f, fminf(127.0f, roundf(val)));
        output[idx] = static_cast<int8_t>(val);
    }
}

void launchQuantizationKernel(float* input, int8_t* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Scale factor would be calculated based on dynamic range
    float scale = 0.01f; 

    quantize_kernel<<<numBlocks, blockSize>>>(input, output, size, scale);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}
