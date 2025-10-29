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
Example demonstrating how to enable lineinfo for CuTe DSL generated kernels
"""

print("Example: Enabling lineinfo for CUTLASS/CuTe kernels")
print("=" * 60)
print()

print("Method 1: Using enable_debug_info()")
print("-" * 60)
print("""
import cutlass

# Enable debug information (lineinfo) for all subsequent kernel compilations
cutlass.backend.compiler.enable_debug_info()

# Now compile and run your GEMM/Conv operation
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
plan.run(A, B, C, D)
""")

print()
print("Method 2: Using add_additional_options()")
print("-" * 60)
print("""
import cutlass

# Add -lineinfo flag (and any other flags you need)
cutlass.backend.compiler.add_additional_options(["-lineinfo", "-G"])

# Compile and run your operation
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
plan.run(A, B, C, D)
""")

print()
print("Method 3: Using set_additional_options()")
print("-" * 60)
print("""
import cutlass

# Set all additional options at once (replaces any previous options)
cutlass.backend.compiler.set_additional_options(["-lineinfo"])

# Compile and run your operation
plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
plan.run(A, B, C, D)
""")

print()
print("Disabling debug info:")
print("-" * 60)
print("""
import cutlass

# Disable debug information if needed
cutlass.backend.compiler.disable_debug_info()
""")

print()
print("Checking current options:")
print("-" * 60)
print("""
import cutlass

# Get the current list of additional compilation options
options = cutlass.backend.compiler.get_additional_options()
print(f"Current additional options: {options}")
""")

print()
print("=" * 60)
print("Notes:")
print("- The -lineinfo flag adds line-number information to device code")
print("- This is useful for profiling with Nsight Compute or nvprof")
print("- Debug info may slightly increase binary size and compile time")
print("- Options are preserved across compilation sessions")
print("=" * 60)
