# Introduction to Accelerated Computing
Introduction to basis of CUDA programming, including the ATLAS CUDA Workshop, April 2025.
For more details, refer to the [References](./ATLAS_CUDA_tutorial/References) section, which contains the basics on GPUs and CUDA programming.
Feel free to read them.

### Prerequisites

No GPU knowledge is assumed for this module. Basic knowledge of Fortran or C/C++ is assumed.

## Heterogeneous and Accelerated Computing

Heterogeneous computing involves using multiple different types of processors to accomplish a task. In this module we'll be focusing on how to use both CPUs and GPUs together to solve a problem.

![CPU and GPU](images/cpu_and_gpu.png)

In common accelerated computing parlance, we refer to the CPU as the **host** and the GPU as the **device**. GPUs have their own memory (RAM) which is distinct from the CPU's memory on most accelerated computing architectures. We refer to the CPU's memory as **host memory** and the GPU's memory as **device memory**.

![Accelerated Computing](images/accelerated_computing.png)

In accelerated computing we take the compute intensive parts of the application code, usually the part of the code where we are spending the most time, and parallelize that for execution on a GPU. The remainder of the code (which may be the vast majority of the number of lines of code) remain on the CPU. Ideally, the part of the code that remains on the CPU is traditional serial code such as input/output operations and high level control flow. The part of the code that makes the most sense to put on the GPU is the operations involving a substantial amount of compute work (typically integer or floating point mathematical operations). In particular, the work that goes onto the GPU must be highly parallelizable. This division of labor reflects the fact that CPUs have complex compute cores (that can execute many types of instructions) but relatively few of them, while GPUs have simple compute cores (that execute a more limited instruction set) but relatively many of them. A modern high end server class CPU typically has on the order of tens of cores, while a modern high end server class GPU has thousands of cores, so we really must adopt some method of parallel computing to effectively use the GPU.

![GPU Devotes More Transistors to Data Processing](images/gpu-devotes-more-transistors-to-data-processing.png)

The image above demonstrates the fundamental difference between a CPU and GPU: given the same amount of physical area on a computer chip (or, thought of another way, the number of transistors on the chip), GPUs preferentially devote their area to large amounts of fairly simplistic data processing components, while CPUs tend to devote more of their area to large caches and complex control flow components.

## NVIDIA HPC SDK

![NVIDIA HPC SDK](images/hpc-sdk.png)

The NVIDIA [HPC SDK](https://developer.nvidia.com/hpc-sdk) is a comprehensive suite of developer tools for accelerated HPC. It includes the C, C++, and Fortran compilers, libraries, and analysis tools necessary for developing HPC applications on the NVIDIA platform.

![NVIDIA HPC SDK 2](images/hpc-sdk-2.png)

Programming with the HPC SDK can involve one of several approaches, depending on your concern for productivity and performance. Options range from native acceleration in the languages -- [C++ parallel algorithms](https://docs.nvidia.com/hpc-sdk/compilers/c++-parallel-algorithms/index.html), starting with C++17, and Fortran `do concurrent` starting with Fortran 2008 -- to directive-based programming (hints provided to the compiler to accelerate a specific compute workload) and low-level programming in CUDA to achieve maximum performance.

![NVIDIA HPC SDK 3](images/hpc-sdk-3.png)

For many problems, off-the-shelf vendor libraries are an easy way to achieve acceleration, particularly with linear algebra problems, FFTs, and random number generation. NVIDIA provides several math libraries that offer drop-in acceleration on GPUs, often with similar or identical APIs to what you are familiar with on CPUs.

![NVIDIA HPC SDK 4](images/hpc-sdk-4.png)

## Review

In this module we have learned:

- The difference between host (CPU) and device (GPU) architectures
- Accelerated computing models offered by the NVIDIA HPC SDK

## Further Study

[NVIDIA HPC Developer Site](https://developer.nvidia.com/hpc)

## Lab Materials

You can download this notebook using the `File > Download as > Notebook (.ipnyb)` menu item. Source code files can be downloaded from the `File > Download` menu item after opening them.