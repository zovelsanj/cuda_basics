#include <iostream>

__global__ void cuda_hierarchy_info(void)
{
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int thread_offset = block_offset + thread_id;
    
    int id = block_id + thread_id;

    printf("ID: %d | Block (%d, %d, %d): %d | Block Offset: %d | Thread (%d, %d, %d): %d | Thread Offset: %d\n", id, blockIdx.x, blockIdx.y, blockIdx.z, block_id, block_offset, threadIdx.x, threadIdx.y, threadIdx.z, thread_id, thread_offset);
}

int main() 
{
    const int block_x=2, block_y=3, block_z=4, thread_x=4, thread_y=4, thread_z=4;
    const int blocks_per_grid = block_x * block_y * block_z;
    const int threads_per_grid = thread_x * thread_y * thread_z;
    printf("Blocks per grid: %d, Threads per grid: %d\n", blocks_per_grid, threads_per_grid);

    cuda_hierarchy_info<<<blocks_per_grid, threads_per_grid>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) 
    {
        printf("Kernel launch failed with error \"%s\"\n", cudaGetErrorString(err));
    }
}
