#include <cstdio>
#include <cmath>

__global__ void set(int *A, int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        A[idx] = idx;
}

int main(void)
{
    const int N = 128;

    int *d_A; // Device memory
    int *h_A; // Host memory

    h_A = (int*) malloc(N*sizeof(int));

    // Allocate linear memory on the device
    cudaMalloc(&d_A, N*sizeof(int));
    // For 2D or 3D memory, use cudaMallocPitch or cudaMalloc3D respectively
    
    // Number of blocks, number of threads per block
    set<<<2, N / 2>>>(d_A, N);

    cudaMemcpy(h_A, d_A, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Check errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    cudaError_t asyncErr = cudaGetLastError();
    if (syncErr != cudaSuccess)
        printf("Sync error: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess)
        printf("Async error: %s\n", cudaGetErrorString(asyncErr));

    // Output
    for (int i = 0; i < N; i++)
        printf("%i ", h_A[i]);
    printf("\n");

    free(h_A);

    // Free device memory
    cudaFree(d_A);

    return 0;
}
