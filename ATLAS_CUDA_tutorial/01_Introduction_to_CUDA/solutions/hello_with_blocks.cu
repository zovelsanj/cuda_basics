#include <stdio.h>

__global__ void hello() {

    printf("Hello from block: %u\n", blockIdx.x);

}

int main() {

    hello<<<2,1>>>();
    cudaDeviceSynchronize();

}
