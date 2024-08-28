#include <stdio.h>

// CUDA kernel function
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 10 threads
    helloFromGPU<<<1, 10>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t syncErr = cudaDeviceSynchronize();
    cudaError_t asyncErr = cudaGetLastError();

    if (syncErr != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(syncErr));
    }
    if (asyncErr != cudaSuccess) {
        printf("Async error: %s\n", cudaGetErrorString(asyncErr));
    }

    printf("Hello World from CPU!\n");

    return 0;
}
