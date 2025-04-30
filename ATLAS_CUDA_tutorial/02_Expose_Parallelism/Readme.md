# Going through exercises
Please go through [02_Expose_Parallelism.ipynb](02_Expose_Parallelism.ipynb) Jupyter notebook for more details on this tutorial. Here we will focus more on the output of the exercise and its working.

The primary concept we focus on this tutorial is **in order to use GPUs effectively, you must expose massive parallelism** in your program. Specifically, **your program must launch a lot of threads**.

## Prerequisites
It is assumed that participants understand the fundamental principles of working with CUDA code, including:

- How to launch CUDA kernels that use both blocks and threads
- Basic memory management (`cudaMalloc()`, `cudaFree()`, `cudaMemcpy`)
- How to compile and run CUDA code

**NOTE:** The number of SMs (streaming multiprocessors) has been growing substantially over time, so we are forced to write code that is flexible enough to run efficiently even on massively parallel architectures. For example, the NVIDIA A100 GPU has 108 SMs compared to V100's 80, and well written CUDA code developed for V100 should be able to scale effectively to A100 without requiring anything other than recompiling.

## Exercise 1: vector_add
In my case I am using [Nsight compute profiler](https://developer.nvidia.com/tools-overview) for analyzing the performance. Its possible that you may require `sudo` access to run the `ncu` profiler. For example in my case, without sudo access, the error shows `The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0`:
```cpp
ncu ./test 
==PROF== Connected to process 837557 (/home/sanjay42/sanjay/cuda/ATLAS_CUDA_tutorial/CUDA/02_Expose_Parallelism/test)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
==PROF== Disconnected from process 837557
==WARNING== No kernels were profiled.
==WARNING== Profiling kernels launched by child processes requires the --target-processes all option.
```

But with sudo access, NVIDIA GPU Performance Counters are successfully accessed:
```cpp
sudo ncu --section SpeedOfLight ./test

==PROF== Connected to process 845145 (/home/sanjay42/sanjay/cuda/ATLAS_CUDA_tutorial/CUDA/02_Expose_Parallelism/test)
Current device: 0
Device name: NVIDIA GeForce RTX 4060 Ti
Total global memory: 16453568 KB
Total constant memory: 64 KB
Total shared memory per block: 48 KB
==PROF== Profiling "vadd" - 0: 0%....50%....100% - 9 passes
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
==PROF== Disconnected from process 845145
[845145] test@127.0.0.1
  vadd(const float *, const float *, float *, int) (816, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         8.68
    SM Frequency            cycle/nsecond         2.29
    Elapsed Cycles                  cycle    3,493,087
    Memory Throughput                   %        91.91
    DRAM Throughput                     %        91.91
    Duration                      msecond         1.52
    L1/TEX Cache Throughput             %        10.39
    L2 Cache Throughput                 %        18.06
    SM Active Cycles                cycle    3,449,776
    Compute (SM) Throughput             %         5.34
    ----------------------- ------------- ------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or 
          memory performance of the device. To further improve performance, work 
          will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              
```

See the details as above. This should now show you detailed GPU performance metrics. Lets go through each of them.

1. **DRAM Frequency (cycle/nsecond):** This shows the clock frequency of the GPU's DRAM memory. In the above case it's around 8.68 cycles per nanosecond. This is a measure of how fast the memory subsystem is running.

2. **SM Frequency (cycle/nsecond):** This shows the clock frequency of the Streaming Multiprocessors (SMs). Here, it's around 2.29 cycles per nanosecond. This is the core compute frequency of the GPU.

3. **Elapsed Cycles (cycle):** The total number of clock cycles taken to execute the kernel. This is a raw measure of how long the kernel took to run. Lower numbers indicate better performance.

4. **Memory Throughput (%):** Measures how much of the theoretical peak memory bandwidth you're achieving. 100% would mean you're using all available memory bandwidth. In this case, it's 91.91%, which is very good - almost saturating the memory bus.

5. **DRAM Throughput (%):** Similar to Memory Throughput, but specifically for DRAM operations. Also at 91.91%, indicating efficient memory access patterns.

6. **Duration (msecond):** The absolute time taken by the kernel in milliseconds. In this case it's 1.52 milliseconds. This is the most direct measure of performance.

7. **L1/TEX Cache Throughput (%):** Measures how effectively you're using the L1 cache and texture cache. At 10.39%, suggesting there's room for improvement in cache utilization. Lower numbers here might indicate cache misses or inefficient memory access patterns.

8. **L2 Cache Throughput (%):** Measures how effectively you're using the L2 cache. At 18.06%, indicating moderate L2 cache utilization. Higher numbers would suggest better cache reuse

9. **SM Active Cycles (cycle):** The number of cycles where the SMs were actively doing work. Here, it's 3,449,776 cycles. This helps understand how much of the time the compute units were busy

10. **Compute (SM) Throughput (%):** Measures how much of the theoretical peak compute performance you're achieving. At 5.34%, indicating that the kernel is not compute-bound. This suggests the kernel is likely memory-bound (which is typical for vector addition).

The profiler also provides an informative message:
```
INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit.      
Start by analyzing DRAM in the Memory Workload Analysis section.
```
This message indicates that the kernel is performing well (>80% utilization). However, the bottleneck appears to be in the memory subsystem (DRAM). To improve further, we might need to optimize memory access patterns or reduce memory operations. The kernel is spending most of its time moving data rather than doing computation, which is typical for memory-bound operations like vector addition.

**SEE** section [Exercise: Grid and Block Size Experimentation](02_Expose_Parallelism.ipynb##Exercise:-Grid-and-Block-Size-Experimentation
) of 02_Expose_Parallelism.ipynb, for more details on experimentation with `blocks` and `threads` in `vector_add.cu` script.

Remember we want to use as much parallel processing as possible so we don't want to use less number of threads per block, say 32. However, we should also take into account that the maximum numbe rof threads per block allowed by say NVIDIA GPUs is typically 1024. So if you exceed this number, say
```cpp
int blocks = 80;
int threads = 2048;
```

you should see the `kernel launch failure` error:
```cpp
==PROF== Connected to process 853231 (/home/sanjay42/sanjay/cuda/ATLAS_CUDA_tutorial/CUDA/02_Expose_Parallelism/test)
Fatal error: kernel launch failure (invalid configuration argument at exercises/compute.cu:45)
*** FAILED - ABORTING
==PROF== Disconnected from process 853231
==ERROR== The application returned an error code (1).
==WARNING== No kernels were profiled.
==WARNING== Profiling kernels launched by child processes requires the --target-processes all option.
```

So the question is, why does this limit exists? The reasons are:
- Each block needs to fit within the SM's resources (registers, shared memory)
- The hardware is designed to efficiently schedule and manage blocks of up to 1024 threads
- This limit ensures proper resource allocation and scheduling

What if you don't know how many SMs your GPU has? More importantly, what if you want to write code that is portable to all of the GPU architectures? In that case, you want to be able to programmatically determine the number of blocks to launch (and possibly the number of threads per block) so that you can make a good choice. To obtain information about the device we're running on, you can use the CUDA API `cudaGetDeviceProperties()`.

```cpp
int deviceid;
cudaGetDevice(&deviceid);
printf("Current device: %d\n", deviceid);
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, deviceid);
```
By default, the `deviceid` is 0. Then we can programatically try to optimize the performance by using
- all available SMs
- maximum number of blocks each SM can handle
- maximum number of threads per blocks

```cpp
int blocks = deviceProp.multiProcessorCount * deviceProp.maxBlocksPerMultiProcessor;
int threads = deviceProp.maxThreadsPerBlock;
```

While this configuration tries to maximize resource usage, it might not be optimal but it is a good balance for memory-bound operations.

## Launch Configuration

So, what does all of this mean for optimal kernel launch configuration? Recall that we defined the launch configuration as :

```cpp
kernel<<<N,M>>>();
```

where `N` is the number of blocks and `M` is the number of threads per block.

We've already suggested that the primary answer is that we want to launch many threads. (The total number of threads in a grid/kernel launch is simply `N * M`.) Why is it that the GPU needs to have a lot of threads from a performance perspective? Let's discuss about some relevant facts:
- Instructions are issued (warp-wide) in order
- A thread stalls when one of the operands (input data) isn’t ready
  - Note that a memory read by itself doesn’t generally stall execution (we can issue a read instruction, and then go on and do other work)
- Latency is hidden by switching threads
  - Global memory latency: >100 cycles  (varies by architecture/design)
  - Arithmetic latency: <100 cycles  (varies by architecture/design)


So, how many threads/blocks should we launch? The short answer is: **we need to launch enough threads to hide latency**. A typical place to start is **128 or 256 threads per block**, but you should always use whatever is best for your application.

**SEE** section [GPU Latency Hiding](02_Expose_Parallelism.ipynb##-GPU-Latency-Hiding) of 02_Expose_Parallelism.ipynb, for details on **Execution Model** and **optimal warp usage**.

## Exercise 2: compute-bound problem

The code [exercises/compute.cu](exercises/compute.cu) fills an array of values with the extended product of the integers from 1 to 100. The product is computed explicitly as an unrolled loop of 100 sequential floating point multiplies. Run this example with several different choices for grid and block size. How many threads are needed to achieve a reasonable fraction of the peak floating point compute throughput (`SM %`, as reported by Nsight Compute) of the GPU?

## Further Study
[Optimization in-depth](http://on-demand.gputechconf.com/gtc/2013/presentations/S3466-Programming-Guidelines-GPU-Architecture.pdf)

[Analysis-Driven Optimization](http://on-demand.gputechconf.com/gtc/2012/presentations/S0514-GTC2012-GPU-Performance-Analysis.pdf)

[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

[CUDA Tuning Guides](https://docs.nvidia.com/cuda/index.html#programming-guides) (Kepler/Maxwell/Pascal/Volta)
