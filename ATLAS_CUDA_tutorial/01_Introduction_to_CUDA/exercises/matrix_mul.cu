#include <stdio.h>
#include <stdlib.h> // Required for exit(), rand(), RAND_MAX
#include <algorithm> // Required for std::min

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)

const int DSIZE = 4096;
const int block_size = 16;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 1.0f;
const float B_val = 2.0f;

void printMatrixCorner(const float* matrix, int size, int corner_size, const char* label) {
    printf("\n%s (Top-%dx%d Corner):\n", label, corner_size, corner_size);
    int print_dim = std::min(size, corner_size); // Determine actual print dimension

    for (int r = 0; r < print_dim; ++r) {
        for (int c = 0; c < print_dim; ++c) {
            printf("%8.4f ", matrix[r * size + c]); // Print with formatting
        }
        printf("\n");
    }
    printf("\n");
}

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

    if ((idx < ds) && (idy < ds)) {
        float temp = 0;
        for (int i = 0; i < ds; i++)
            temp += A[idy*ds+i] * B[i*ds+idx];   // dot product of row and column
        C[idy*ds+idx] = temp;                        // note that we're filling in C(idy,idx)
    }

}

int main() {
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // these are just for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;
    // start timing
    t0 = clock();

    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Print corners of matrices
    const int corner_print_size = 4; // Size of corner to print (e.g., 4x4)
    printMatrixCorner(h_A, DSIZE, corner_print_size, "Matrix A");
    printMatrixCorner(h_B, DSIZE, corner_print_size, "Matrix B");

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");


    // Launch kernel
    dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
    dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
    printf("Launching kernel with grid=(%u,%u) and block=(%u,%u)\n", grid.x, grid.y, block.x, block.y); // Print launch config
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Synchronize to ensure kernel completion before copying back
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel execution failure");

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure C");
    printMatrixCorner(h_C, DSIZE, corner_print_size, "Result Matrix C");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // Verify results
    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        if (h_C[i] != A_val*B_val*DSIZE) {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE);
            return -1;
        }
    }
    printf("Success!\n");

    // Print corner of result matrix
    printMatrixCorner(h_C, DSIZE, corner_print_size, "Result Matrix C");

    delete[] h_A; // free host memory using delete[] for new[]
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A); // free device memory
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nMatrix multiplication complete.\n");

    return 0;
}
