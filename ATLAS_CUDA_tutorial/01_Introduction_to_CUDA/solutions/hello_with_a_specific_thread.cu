#include <stdio.h>

__global__ void hello() {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 773) {
        printf("Hello from unique thread index: %u\n", index);
    }

}

int main(){

    hello<<<512, 2>>>();
    cudaDeviceSynchronize();

}
