#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <cuda_runtime.h>

// Function to perform a sample CUDA operation
__global__ void sampleCudaKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f; // Example operation: double the value
    }
}

// Function to launch the CUDA kernel
void launchSampleCudaKernel(float* data, int size) {
    float* d_data;
    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sampleCudaKernel<<<numBlocks, blockSize>>>(d_data, size);

    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

#endif // CUDA_OPS_H