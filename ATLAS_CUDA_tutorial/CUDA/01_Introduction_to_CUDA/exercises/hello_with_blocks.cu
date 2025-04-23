#include <stdio.h>

__global__ void hello() {
    int block_id = blockIdx.x;
    printf("Hello from block: %u\n", block_id);
}

int main() {
    hello<<<8, 1>>>();
    cudaDeviceSynchronize();
}
