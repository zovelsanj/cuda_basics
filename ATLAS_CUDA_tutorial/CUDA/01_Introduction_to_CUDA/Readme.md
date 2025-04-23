# Going through execrcises
Please go through [01_Introduction_to_CUDA.ipynb](01_Introduction_to_CUDA.ipynb) Jupyter notebook for more details on the exercises and CUDA basics. Here we will focus more on the output of the exercise and its working.
## Exercise 1: hello_world
```
__global__ void hello() {
    int block_id = blockIdx.x;
    printf("Hello from block: %u\n", block_id);
}
```

This is an example of kernel (the execution of the function which runs on the device). 
Kernel is launched as follows:
```
kernelName<<<gridDim, blockDim>>>(arguments);
```
`gridDim` is the number of blocks in the grid. It is also called `blocks` as it gives the total number of blocks in a grid.

`blockDim` is the number of threads in the block. It is also called `threads` as it gives the total number of threads in a block.


The `<<<...>>>` syntax is mandatory because it's how CUDA knows how to distribute the work across the GPU. Without it, the compiler doesn't know how many parallel instances of the kernel to launch, hence the `"too few arguments"` error.

After launching a kernel, it's important to use `cudaDeviceSynchronize()` to wait for the kernel to complete, especially when you're printing or need the results before continuing.
For more details, see [01_Introduction_to_CUDA.ipynb - GPU Kernels: Device Code Section](01_Introduction_to_CUDA.ipynb#GPU-Kernels:-Device-Code).


## Exercise 2: hello_with_blocks
```
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

## Exercise 3: hello_with_threads
Within a single block, threads are organized into groups called **warps** (typically 32 threads per warp on most NVIDIA GPUs). Threads within a warp execute in a pattern called SIMT (Single Instruction, Multiple Thread), which means they execute the same instruction at the same time.
```
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
```
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

Another thing to point out here is that, the warp size (32) is not the limit for threads per block. The warp size is just the number of threads that execute together in SIMT fashion, but a block can contain multiple warps. So, if you lauch the kernel, for example as:
```
hello<<<4, 64>>>();
```
the output will be something like this:
```
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
```
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
