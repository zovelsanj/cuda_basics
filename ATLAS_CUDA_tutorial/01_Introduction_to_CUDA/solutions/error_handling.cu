#include <cstdio>

__global__ void kernel (int* a) {
    a[0] = 1;
}

int main() {
    cudaError_t err;

    int* a;
    err = cudaMalloc(&a, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    kernel<<<1, 1>>>(a);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaFree(a);
    if (err != cudaSuccess) {
        printf("CUDA error %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
