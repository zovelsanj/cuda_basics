#include <stdio.h>

__global__ void hello() {

    printf("Hello world\n");

}

int main() {

    hello<<<1,1>>>();
    cudaDeviceSynchronize();

}
