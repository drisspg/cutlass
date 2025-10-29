# Enabling Line Info for CuTe DSL Generated Kernels

This document explains how to enable debug information (lineinfo) for CUTLASS/CuTe DSL generated kernels in the Python interface.

## Background

The `-lineinfo` compiler flag enables generation of line-number information for device code. This is essential for:
- Profiling with Nsight Compute
- Profiling with nvprof
- Debugging kernel performance issues
- Correlating assembly code with source lines

## Usage

### Method 1: Using `enable_debug_info()` (Recommended)

The simplest way to enable lineinfo is to use the `enable_debug_info()` method:

```python
import cutlass
import numpy as np

# Enable debug information for all subsequent kernel compilations
cutlass.backend.compiler.enable_debug_info()

# Now compile and run your GEMM operation
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.ones((1024, 1024), dtype=np.float16) for i in range(4)]
plan.run(A, B, C, D)
```

### Method 2: Using `add_additional_options()`

For more flexibility, you can add multiple compiler flags:

```python
import cutlass

# Add -lineinfo flag along with other flags
cutlass.backend.compiler.add_additional_options(["-lineinfo", "-G"])

# Or add them one at a time
cutlass.backend.compiler.add_additional_options("-lineinfo")
```

### Method 3: Using `set_additional_options()`

To replace all additional options at once:

```python
import cutlass

# Set all additional options (replaces any previous options)
cutlass.backend.compiler.set_additional_options(["-lineinfo"])
```

## Disabling Debug Info

To disable debug information after enabling it:

```python
import cutlass

# Disable debug information
cutlass.backend.compiler.disable_debug_info()
```

## Checking Current Options

To see what additional compilation options are currently set:

```python
import cutlass

# Get the current list of additional compilation options
options = cutlass.backend.compiler.get_additional_options()
print(f"Current additional options: {options}")
```

## Important Notes

1. **Compilation Cache**: The compilation cache key includes the additional options, so kernels compiled with different options (e.g., with and without `-lineinfo`) will be cached separately.

2. **Performance Impact**: Debug information may slightly increase:
   - Binary size
   - Compilation time
   - In some cases, runtime performance (especially with `-G` which disables optimizations)

3. **Backend Support**: The lineinfo flag works with both `nvcc` and `nvrtc` backends.

4. **Global Setting**: The additional compilation options are global and affect all subsequent kernel compilations until changed.

## Example: Profiling with Nsight Compute

```python
import cutlass
import numpy as np

# Enable lineinfo for profiling
cutlass.backend.compiler.enable_debug_info()

# Create and compile the GEMM operation
plan = cutlass.op.Gemm(element=np.float32, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.random.randn(2048, 2048).astype(np.float32) for i in range(4)]

# Run the operation
plan.run(A, B, C, D)

# Now you can profile the kernel with:
# ncu --set full python your_script.py
```

## API Reference

### `enable_debug_info()`
Enable debug information in compiled kernels for profiling and debugging.

### `disable_debug_info()`
Disable debug information in compiled kernels.

### `set_additional_options(options)`
Set additional compilation options.
- **Parameters**: `options` (list of str or str) - Compiler flags to set

### `add_additional_options(options)`
Add additional compilation options to the existing set.
- **Parameters**: `options` (list of str or str) - Compiler flags to add

### `get_additional_options()`
Get the current list of additional compilation options.
- **Returns**: list of str - Current additional compiler flags

## Troubleshooting

### Issue: Lineinfo not appearing in profiler

**Solution**: Make sure you call `enable_debug_info()` before compiling the kernel (i.e., before calling `plan.run()` for the first time or before the operation's `compile()` method).

### Issue: Kernel recompilation every time

**Solution**: This is expected when debug options change. The cache key includes the compilation options, so different options result in different cached kernels.

## See Also

- [CUTLASS Python Interface Documentation](../README.md)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVCC Compiler Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
