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

void initialize_arrays(float *vector, int N)
{
    for (int i = 0; i < N; i++)
    {
        vector[i] = (float)rand() / (float)(RAND_MAX);
    }
}

void allocate_memory(float **h_a, float **h_b, float **h_c, float **d_a, float **d_b, float **d_c, const int N)
{
    size_t size = N * sizeof(float);
    //Allocate memory on host (host = CPU)
    *h_a = (float*)malloc(size);
    *h_b = (float*)malloc(size);
    *h_c = (float*)malloc(size);

    //Initialize host arrays
    initialize_arrays(*h_a, N);
    initialize_arrays(*h_b, N);
    initialize_arrays(*h_c, N);

    //Allocate memory on device (device = GPU)
    cudaMalloc(d_a, size);
    cudaMalloc(d_b, size);
    cudaMalloc(d_c, size);

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
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    allocate_memory(&h_a, &h_b, &h_c, &d_a, &d_b, &d_c, N);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CPU timing using clock()
    clock_t cpu_start = clock();
    for (int i = 0; i < 1000; i++)
    {
        cpu_vector_add(h_a, h_b, h_c, N);
    }
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time);

    // GPU timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
    {
        gpu_vector_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU time: %f milliseconds\n", gpu_time);
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Results are %s\n", check_results(d_c, h_c, N) ? "correct" : "incorrect");

    //Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
