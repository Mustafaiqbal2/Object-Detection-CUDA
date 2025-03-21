#include <cuda_runtime.h>
#include <iostream>
#include "cuda_ops.h"

// Kernel function to perform a simple operation on the GPU
__global__ void simpleKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f; // Example operation: double the value
    }
}

// Function to launch the CUDA kernel
void launchSimpleKernel(float *data, int size) {
    float *d_data;
    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    simpleKernel<<<numBlocks, blockSize>>>(d_data, size);

    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}