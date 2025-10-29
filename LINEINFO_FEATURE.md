# Enabling Lineinfo for CUTLASS Python Kernels

## Quick Start

To enable line information (lineinfo) for profiling CUTLASS kernels with Nsight Compute or nvprof:

```python
import cutlass

# Enable lineinfo before compiling kernels
cutlass.backend.compiler.enable_debug_info()

# Now use CUTLASS as normal
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
plan.run(A, B, C, D)
```

## Why Enable Lineinfo?

The `-lineinfo` flag adds line-number information to the compiled CUDA kernel, which:
- Enables source-level profiling in Nsight Compute
- Shows which source lines correspond to hotspot instructions
- Helps identify performance bottlenecks in your kernels
- Is essential for effective GPU profiling and optimization

## Documentation

For detailed documentation, see [python/docs/lineinfo_guide.md](python/docs/lineinfo_guide.md)

## Example

See [examples/python/enable_lineinfo_example.py](examples/python/enable_lineinfo_example.py) for a complete example.

## API Summary

- `cutlass.backend.compiler.enable_debug_info()` - Enable lineinfo
- `cutlass.backend.compiler.disable_debug_info()` - Disable lineinfo
- `cutlass.backend.compiler.add_additional_options(flags)` - Add custom compiler flags
- `cutlass.backend.compiler.set_additional_options(flags)` - Set all custom flags
- `cutlass.backend.compiler.get_additional_options()` - Query current flags

## Note

This feature was added in CUTLASS 3.6.0 to address the need for profiling support in CuTe DSL generated kernels.
