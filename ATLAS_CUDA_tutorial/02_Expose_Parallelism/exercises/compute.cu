#include <stdio.h>

const int DSIZE = 32*1048576;

// error checking macro
#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)

__global__ void kernel(float *x, int ds) {
    for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x) {
        float result = 1.0 + x[idx];

        #pragma unroll 1000               // Force the compiler to unroll the loop into 100 sequential statements
        for (int j = 1; j <= 1000; ++j) {
            result *= (float) j;
        }
        x[idx] = result;
    }
}

int main() {
    float *x;

    cudaMalloc(&x, DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemset(x, 0, DSIZE*sizeof(float)); // initialize array to zero;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Use maximum threads per block

    int blocks = 4096;  // modify this line for experimentation
    int threads = 1024; // modify this line for experimentation
    
    // int threads = prop.maxThreadsPerBlock;
    // int blocks = (DSIZE + threads - 1) / threads;
    
    kernel<<<blocks, threads>>>(x, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaDeviceSynchronize();
    cudaCheckErrors("kernel execution failure");

    return 0;
}
