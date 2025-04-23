## How CUDA works
In CUDA, when you launch a kernel (a function that runs on the GPU), you need to specify how many parallel instances of that kernel you want to run. This is done using the triple angle bracket syntax `<<<...>>>`.
```cpp
kernelName<<<gridDim, blockDim>>>(arguments);
```
`gridDim` is the number of blocks in the grid.
`blockDim` is the number of threads in the block.
arguments are the arguments to the kernel.

The `<<<...>>>` syntax is mandatory because it's how CUDA knows how to distribute the work across the GPU. Without it, the compiler doesn't know how many parallel instances of the kernel to launch, hence the `"too few arguments"` error.

**Note:** After launching a kernel, it's important to use `cudaDeviceSynchronize()` to wait for the kernel to complete, especially when you're printing or need the results before continuing.
