# Going through execrcises
Please go through [01_Introduction_to_CUDA.ipynb](01_Introduction_to_CUDA.ipynb) Jupyter notebook for more details on the exercises and CUDA basics. Here we will focus more on the output of the exercise and its working.
## Exercise 1: hello_world
```c
__global__ void hello() {
    int block_id = blockIdx.x;
    printf("Hello from block: %u\n", block_id);
}
```

This is an example of kernel (the execution of the function which runs on the device). 
Kernel is launched as follows:
```sh
kernelName<<<gridDim, blockDim>>>(arguments);
```
`gridDim` is the number of blocks in the grid. It is also called `blocks` as it gives the total number of blocks in a grid.

`blockDim` is the number of threads in the block. It is also called `threads` as it gives the total number of threads in a block.


The `<<<...>>>` syntax is mandatory because it's how CUDA knows how to distribute the work across the GPU. Without it, the compiler doesn't know how many parallel instances of the kernel to launch, hence the `"too few arguments"` error.

After launching a kernel, it's important to use `cudaDeviceSynchronize()` to wait for the kernel to complete, especially when you're printing or need the results before continuing.
For more details, see [01_Introduction_to_CUDA.ipynb - GPU Kernels: Device Code Section](01_Introduction_to_CUDA.ipynb#GPU-Kernels:-Device-Code).


## Exercise 2: hello_with_blocks
```c
__global__ void hello() {
    int block_id = blockIdx.x;
    printf("Hello from block: %u\n", block_id);
}
```
Its possible that you may get a pattern in the output, which may mislead if the GPU places the threads in an order. However, this is actually coincidental. The execution of GPU threads and blocks is indeed parallel and the order is not guaranteed. So, when the are blocks executed:
- They run in parallel (or as parallel as the GPU resources allow)
- The order of execution is not guaranteed
- The order of when their printf statements actually appear in the output is also not guaranteed

The reason you might see similar output patterns is because:
- The GPU scheduler might be using a similar scheduling pattern for these blocks in your specific runs
- The printf buffer might be getting flushed in a similar way across runs
- Your GPU might have enough resources to handle all blocks similarly across runs

**REMEMBER**: In CUDA programming, you should never rely on the execution order of blocks or threads unless you explicitly synchronize them. The parallel nature of GPU execution means the ordering can change based on many factors including:
- GPU hardware
- System load
- Driver version
- Number of concurrent processes
- Available resources

**NOTE:** Generally, block size is chosen a multiple of 32 (say, 256 / 32 = 8 warps per block). This ensures that all warps within the block are full, maximizing the efficiency of the SIMT execution on the SMs. Using block sizes that aren't multiples of 32 leads to partially filled warps and wasted execution resources.

## Exercise 3: hello_with_threads
Within a single block, threads are organized into groups called **warps** (typically 32 threads per warp on most NVIDIA GPUs). Threads within a warp execute in a pattern called SIMT (Single Instruction, Multiple Thread), which means they execute the same instruction at the same time.
```c
__global__ void hello() {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    printf("Hello from block: %u, thread: %u\n", block_id, thread_id);
}

int main(){
    hello<<<4, 4>>>();
    cudaDeviceSynchronize();
}
```

The blocks (`block_id`) can execute in any order - which is why you see block IDs appearing randomly as discussed in [Exercise: hello_with_blocks](#exercise-2-hello_with_blocks). However, within each block, the threads are executed in a more structured way:
Threads within the same warp execute in lockstep (together)
Since here are 4 threads per block, they're all part of the same warp
This is why you probably will see consistent order (0,1,2,3) within each block somethin like this:
```sh
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
Hello from block: 1, thread: 2
Hello from block: 1, thread: 3
...
```
These threads from block 1 execute together as part of the same warp, which is why they appear in sequence.
This is an important aspect of GPU architecture to understand:
- Blocks can execute in any order and in parallel (non-deterministic)
- Threads within a warp execute together (deterministic)
- The warp size (typically 32) is a hardware characteristic
- When you have fewer threads than a full warp (here 4), they still maintain this ordered execution pattern within their block

This **SIMT execution model** is one of the key characteristics that makes GPUs efficient for parallel processing, but it's also why you need to be careful when designing algorithms for GPU execution - threads within a warp that take different code paths (due to conditionals) can impact performance.

**NOTE:** However, synchronization can be done manually. See [06_Reductions_Redux.ipynb - Problem: Global Synchronization](../06_Reductions_Redux/06_Reductions_Redux.ipynb###Problem:-Global-Synchronization).


Another thing to point out here is that, the warp size (32) is not the limit for threads per block. The warp size is just the number of threads that execute together in SIMT fashion, but a block can contain multiple warps. So, if you lauch the kernel, for example as:
```c
hello<<<4, 64>>>();
```
the output will be something like this:
```sh
Hello from block: 3, thread: 0
Hello from block: 3, thread: 1
...
Hello from block: 3, thread: 30
Hello from block: 3, thread: 31
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
...
Hello from block: 0, thread: 30
Hello from block: 0, thread: 31
Hello from block: 3, thread: 32
Hello from block: 3, thread: 33
...
Hello from block: 3, thread: 62
Hello from block: 3, thread: 63
Hello from block: 0, thread: 32
Hello from block: 0, thread: 33
...
Hello from block: 0, thread: 62
Hello from block: 0, thread: 63
Hello from block: 2, thread: 32
Hello from block: 2, thread: 33
...
Hello from block: 2, thread: 62
Hello from block: 2, thread: 63
...
```

You can see a pattern where threads 0-31 print first, and then threads 32-63 print for each block. The GPU is processing these as two separate warps within the same block.

Now at this point, it is possible that one might get confused (well, I was confused when I just started to learn CUDA programming) with the blocks and warps as they both incorporate the threads. So lets get things cleared up. Think of it this way:
```sh
Block 0                          Block 1
├── Warp 0 (threads 0-31)       ├── Warp 0 (threads 0-31)
├── Warp 1 (threads 32-63)      ├── Warp 1 (threads 32-63)
└── Warp 2 (threads 64-95)      └── Warp 2 (threads 64-95)
and so on.
```
Here are the actual limits for CUDA threads (these may vary slightly by GPU architecture):
Maximum threads per block: typically 1024
Warp size: 32 threads
So a single block can contain up to 32 warps (1024/32 = 32 warps)

While the warp size of 32 determines how many threads execute instructions simultaneously, you can have multiple warps in a block. This is actually very common in practice, as having more threads per block (up to a point) can help hide memory latency and improve overall GPU utilization.

Just remember:
- Warp size (32) = number of threads that execute together
- Threads per block (up to 1024) = total threads in a block, automatically organized into warps
- Each block can have multiple warps
- The GPU scheduler manages the execution of multiple warps within each block


For more details see  [02_Expose_Parallelism.ipynb Execution Model](../02_Expose_Parallelism/02_Expose_Parallelism.ipynb##Execution-Model)

## Exercise 4: memory_management
You are probably familiar with `malloc()`, `free()` and `memcpy()` memory management functions in C. Their corresponding equivalent functions in CUDA are `cudaMalloc()`, `cudaFree()`, and `cudaMemcpy()`, respectively.

```c
cudaMalloc(&d_a, sizeof(int));
```
It allocates memory for one integer on the device if not initalized. Remember this memory is only accessible from the device. You must use `cudaFree()` to deallocate the memory when done.
It returns a `cudaError_t` to indicate success or failure, which is further discussed in the section [Exercise 5: error_handling](#exercise-5-error_handling). It returns cudaErrorInvalidValue if the size is 0 or negative.

```c
cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice)
```
This copies the initial value of `a` from host memory to device memory. It can be used the other way around as well. This is just setting up the initial state.

```c
__global__ void set_value (int* a) {
    *a = 2;
}

//existing code
set_value<<<1, 1>>>(d_a)
```

`set_value` kernel runs on the GPU to modify the value in device memory. It sets the value to 2 (in this case), regardless of what was previously there.

The sequence of operations in the script [memory_management.cu](exercises/memory_management.cu) is:
- Allocate host memory for a and set it to 1
- Allocate device memory for d_a
- Copy the value 1 from host to device
- Run a kernel that changes the device value to 2
- Copy the modified value (2) back to host
- Verify the value is now 2

For more details on the memory management, see section [01_Introduction_to_CUDA.ipynb - Memory Management](01_Introduction_to_CUDA.ipynb###Memory-Management).

## Exercise 5: error_handling
Every CUDA runtime API (such as `cudaMalloc()`, `cudaDeviceSynchronize()`, etc.) returns an error code of type `cudaError_t` i.e. an integer, which is `0` for `cudaSuccess`.

In the cuda script, [error_handling.cu](./exercises/error_handling.cu), without any error handling, it should show something like this:
```sh
error_handling.cu(11): warning #68-D: integer conversion resulted in a change of sign

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

error_handling.cu(11): warning #68-D: integer conversion resulted in a change of sign

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"
```

After error handling with the `check_error()` function:
```c
void check_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
```
It should show something like this:
```
Error: out of memory
```
Now, if you comment `exit()` line, you should see two errors but not the third one:
```
Error: out of memory
Error: invalid configuration argument
```
This is because the kernel never actually executes due to the invalid launch configuration in `kernel<<<1, -1>>>(a)`.

Breaking the errors down: 
- The first error comes from `cudaMalloc(&a, -sizeof(int))`. This fails because you can't allocate negative memory size. This is caught by the first error check.
- `kernel<<<1, -1>>>(a)` fails because the number of threads per block (-1) is invalid. The kernel launch itself fails before any kernel code runs. This is caught by `cudaGetLastError()`.
- The kernel code `a[-1] = 1` never actually runs because the kernel launch failed. If the kernel had launched successfully, this would have caused a memory access violation, but since the kernel never started, there's no error to catch in the `cudaDeviceSynchronize()` call.

For more details on error handling, see the section [01_Introduction_to_CUDA.ipynb - Error Handling](01_Introduction_to_CUDA.ipynb###Error-Handling).

## Exercise 6: vector_addition
Everything in the cuda script [vector_addition.cu](./exercises/vector_addition.cu) is already discussed, except for the arbitrary choice of the blocks and threads. Here the kernel is launched as
```c
vadd<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
```

Till now, we were using the total number of threads in a kernel launch as an integer multiple of the number of threads in a block, but typical problems are not exact multiples of **blockDim.x**. So the question is, how do you choose these values?

The most basic idea is to launch enough threads to cover the total amount of parallel work you need to do. If you have `N` independent items to process (e.g., elements in an array), you ideally want to launch at least `N` threads, so
`Total Threads = gridDim * blockDim`

You calculate the required number of blocks based on your chosen threads per block: 
```c
num_blocks = ceil(total_items / threads_per_block)
```
where `ceil()` is the ceiling division to ensure you have enough threads even if total_items isn't perfectly divisible.

Now in our case,
`(DSIZE + block_size - 1) / block_size` is a standard way to calculate the ceiling of `DSIZE / block_size` using integer division. We have
```c
const int DSIZE = 4096;
const int block_size = 256;
```
so,
```
(4096 + 256 - 1) / 256
= (4351) / 256
= 16 (using integer division)
```

Thus 16 blocks are needed. The total number of threads launched is `gridDim * blockDim = 16 * 256 = 4096`. In this specific case, the total number of threads (4096) exactly matches the data size (`DSIZE`). Try changing the `DSIZE` to a non-perfect multiple (e.g., 4097), then this calculation would result in `gridDim = 17`, launching `17 * 256 = 4352` threads. This ensures at least `DSIZE` threads are launched.

## Exercise 7: matrix_multiplication

Lets go through some basics that we haven't covered yet (lets ignore the non-relevant parts like timing, printing matrics, and error checks here). You will see, we have now initialized 2D blcoks and grods, because matrix multiplication is inherently a 2D problem. So, instead of organizing threads in a long 1D line (like in vector addition), it's more natural here to arrange them in 2D blocks.
```c
dim3 block(block_size, block_size);
dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
```

`dim3` is a simple struct containing `x`, `y`, and `z` integer members, so it can handle 1D, 2D, and 3D data uniformly.
