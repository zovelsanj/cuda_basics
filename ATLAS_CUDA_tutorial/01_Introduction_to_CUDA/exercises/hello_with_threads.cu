#include <stdio.h>

__global__ void hello() {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    printf("Hello from block: %u, thread: %u\n", block_id, thread_id);
}

int main(){
    hello<<<4, 4>>>();
    cudaDeviceSynchronize();
}
