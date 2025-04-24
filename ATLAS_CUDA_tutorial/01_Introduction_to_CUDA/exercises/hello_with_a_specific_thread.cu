#include <stdio.h>

__global__ void hello(int index) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    if (thread_id == index) {
        printf("Hello from block: %u, unique thread: %u\n", block_id, thread_id);
    }
}

int main(){
    hello<<<2, 2>>>(1);
    cudaDeviceSynchronize();
}
