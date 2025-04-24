#include <cstdio>

__global__ void kernel (int* a) {
    a[-1] = 1;
}

void check_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

int main() {
    int* a;
    cudaError_t err = cudaMalloc(&a, -sizeof(int));
    check_error(err);

    kernel<<<1, -1>>>(a);
    err = cudaGetLastError();
    check_error(err);

    err = cudaDeviceSynchronize();
    check_error(err);

    free(a);
}
