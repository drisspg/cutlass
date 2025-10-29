#!/usr/bin/env python3
#################################################################################################
#
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Example: Profiling GEMM with lineinfo enabled

This example demonstrates how to enable lineinfo for profiling CUTLASS kernels
with Nsight Compute or nvprof.

Usage:
    # Run with profiling
    ncu --set full python gemm_with_lineinfo.py
    
    # Or with nvprof (deprecated but still works)
    nvprof --print-gpu-trace python gemm_with_lineinfo.py
"""

import cutlass
import numpy as np

def main():
    print("="*70)
    print("CUTLASS GEMM with Lineinfo Example")
    print("="*70)
    print()
    
    # Step 1: Enable debug info for profiling
    print("Step 1: Enabling lineinfo for profiling...")
    cutlass.backend.compiler.enable_debug_info()
    
    # Verify the flag is set
    options = cutlass.backend.compiler.get_additional_options()
    print(f"  Current compiler options: {options}")
    assert "-lineinfo" in options
    print("  ✓ Lineinfo enabled")
    print()
    
    # Step 2: Create a GEMM operation
    print("Step 2: Creating GEMM operation...")
    try:
        plan = cutlass.op.Gemm(
            element=np.float16, 
            layout=cutlass.LayoutType.RowMajor
        )
        print("  ✓ GEMM operation created")
        print()
    except Exception as e:
        print(f"  Note: Could not create GEMM (expected in non-CUDA environment)")
        print(f"  Error: {e}")
        print()
        print("To run this example with actual kernel compilation:")
        print("  1. Install CUDA Toolkit and cuda-python")
        print("  2. Run on a machine with NVIDIA GPU")
        print("  3. Profile with: ncu --set full python gemm_with_lineinfo.py")
        return
    
    # Step 3: Prepare input tensors
    print("Step 3: Preparing input tensors...")
    M, N, K = 2048, 2048, 2048
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    D = np.zeros((M, N), dtype=np.float16)
    print(f"  Matrix dimensions: A={A.shape}, B={B.shape}, C={C.shape}")
    print()
    
    # Step 4: Compile and run the kernel
    print("Step 4: Compiling and running GEMM...")
    print("  (This will compile the kernel with lineinfo enabled)")
    try:
        plan.run(A, B, C, D)
        print("  ✓ GEMM completed successfully")
        print()
    except Exception as e:
        print(f"  Error running GEMM: {e}")
        print()
        return
    
    # Step 5: Show profiling instructions
    print("Step 5: Profiling Instructions")
    print("-"*70)
    print("Now you can profile the kernel with source-level information:")
    print()
    print("With Nsight Compute:")
    print("  ncu --set full python gemm_with_lineinfo.py")
    print()
    print("The lineinfo will allow you to:")
    print("  - See which source lines correspond to performance bottlenecks")
    print("  - Correlate assembly instructions with CUDA C++ code")
    print("  - Better understand kernel performance characteristics")
    print()
    
    # Optional: Disable lineinfo for subsequent operations
    print("Step 6: Optional - Disable lineinfo")
    print("-"*70)
    print("You can disable lineinfo if you want to compile without debug info:")
    cutlass.backend.compiler.disable_debug_info()
    options_after = cutlass.backend.compiler.get_additional_options()
    print(f"  Compiler options after disable: {options_after}")
    print("  ✓ Lineinfo disabled")
    print()
    
    print("="*70)
    print("Example completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
