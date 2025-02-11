#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

void cpu_vector_add(float *a, float *b, float *c, int N) 
{
    for (int i = 0; i < N; i++) 
    {
        c[i] = a[i] + b[i];
    }
}

//CUDA kernel (kernel is a function that is executed on the GPU by each thread)
__global__ void gpu_vector_add(float *a, float *b, float *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void gpu_vector_add_3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;  
    if (i < nx && j < ny && k < nz)
    {
        int idx = i * ny * nz + j * nz + k;
        c[idx] = a[idx] + b[idx];
    }
}

void initialize_arrays(float *vector, int N)
{
    for (int i = 0; i < N; i++)
    {
        vector[i] = (float)rand() / (float)(RAND_MAX);
    }
}

void allocate_memory(float **h_a, float **h_b, float **h_c, float **h_c_3d, 
                    float **d_a, float **d_b, float **d_c, float **d_c_3d, const int N)
{
    size_t size = N * sizeof(float);
    //Allocate memory on host (host = CPU)
    *h_a = (float*)malloc(size);
    *h_b = (float*)malloc(size);
    *h_c = (float*)malloc(size);
    *h_c_3d = (float*)malloc(size); //for 3D addition

    //Initialize host arrays
    initialize_arrays(*h_a, N);
    initialize_arrays(*h_b, N);
    initialize_arrays(*h_c, N);

    //Allocate memory on device (device = GPU)
    cudaMalloc(d_a, size);
    cudaMalloc(d_b, size);
    cudaMalloc(d_c, size);
    cudaMalloc(d_c_3d, size); //for 3D addition

    //Copy arrays from host to device
    cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, *h_b, size, cudaMemcpyHostToDevice);
}

bool check_results(float *d_c, float *h_c, int N)
{
    size_t size = N * sizeof(float);
    float *hc_gpu = (float*)malloc(size);  
    cudaMemcpy(hc_gpu, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        if (fabs(hc_gpu[i] - h_c[i]) > 1e-5)
        {
            printf("Error at position %d: %f != %f\n", i, hc_gpu[i], h_c[i]);
            return false;
        }
    }
    return true;
}

int main()
{ 
    const int N = 1000000;
    const int BLOCK_SIZE = 256;
    const int BLOCK_SIZE_3D_X = 8, BLOCK_SIZE_3D_Y = 8, BLOCK_SIZE_3D_Z = 8;
    const int nx = 100, ny = 100, nz = 100;

    float *h_a, *h_b, *h_c, *h_c_3d;
    float *d_a, *d_b, *d_c, *d_c_3d;
    
    allocate_memory(&h_a, &h_b, &h_c, &h_c_3d, &d_a, &d_b, &d_c, &d_c_3d, N);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // CPU timing using clock()
    clock_t cpu_start = clock();
    for (int i = 0; i < 1000; i++)
    {
        cpu_vector_add(h_a, h_b, h_c, N);
    }
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time);

    // GPU timing for 1D vector add
    cudaEvent_t start_1d, stop_1d;
    cudaEventCreate(&start_1d);
    cudaEventCreate(&stop_1d);
    
    cudaEventRecord(start_1d);
    for (int i = 0; i < 1000; i++)
    {
        gpu_vector_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop_1d);
    cudaEventSynchronize(stop_1d);
    
    float gpu_time_1d = 0;
    cudaEventElapsedTime(&gpu_time_1d, start_1d, stop_1d);
    printf("CPU | time: %f seconds, GPU 1D | time: %f miliseconds\n", cpu_time, gpu_time_1d);
    printf("Speedup: %f\n", cpu_time / gpu_time_1d * 1000);
    
    // GPU timing for 3D vector add
    cudaEvent_t start_3d, stop_3d;
    cudaEventCreate(&start_3d);
    cudaEventCreate(&stop_3d);
    
    cudaEventRecord(start_3d);
    for (int i = 0; i < 1000; i++)
    {
        gpu_vector_add_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
    }
    cudaEventRecord(stop_3d);
    cudaEventSynchronize(stop_3d);
    
    float gpu_time_3d = 0;
    cudaEventElapsedTime(&gpu_time_3d, start_3d, stop_3d);
    printf("CPU | time: %f seconds, GPU 3D | time: %f miliseconds\n", cpu_time, gpu_time_3d);
    printf("Speedup: %f\n", cpu_time / gpu_time_3d * 1000);

    // Clean up timing events
    cudaEventDestroy(start_1d);
    cudaEventDestroy(stop_1d);
    cudaEventDestroy(start_3d);
    cudaEventDestroy(stop_3d);

    printf("Results are %s\n", check_results(d_c, h_c, N) ? "correct" : "incorrect");

    //Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_3d);

    return 0;
}
