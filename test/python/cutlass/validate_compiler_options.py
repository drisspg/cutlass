#!/usr/bin/env python3
"""
Validation script for lineinfo compiler options API.
This script validates the API without requiring CUDA runtime.
"""

import sys
import os

# Add python directory to path
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python'))
sys.path.insert(0, python_dir)

# Mock the cuda module to allow import without CUDA
class MockCuda:
    class cuda:
        class CUresult:
            CUDA_SUCCESS = 0
        def cuModuleLoadData(*args):
            return (CUresult.CUDA_SUCCESS, None)
        def cuModuleGetFunction(*args):
            return (CUresult.CUDA_SUCCESS, None)
        def cuDeviceGetCount():
            return (CUresult.CUDA_SUCCESS, 1)
    class cudart:
        class cudaError_t:
            cudaSuccess = 0
        def cudaFree(*args):
            return (cudaError_t.cudaSuccess,)
    class nvrtc:
        class nvrtcResult:
            NVRTC_SUCCESS = 0
    __version__ = "12.0.0"

sys.modules['cuda'] = MockCuda
sys.modules['cuda.cuda'] = MockCuda.cuda
sys.modules['cuda.cudart'] = MockCuda.cudart
sys.modules['cuda.nvrtc'] = MockCuda.nvrtc

# Now we can import the compiler
from cutlass.backend.compiler import ArtifactManager

def test_initialization():
    """Test that ArtifactManager initializes with empty additional options"""
    print("Test 1: Initialization")
    compiler = ArtifactManager()
    options = compiler.get_additional_options()
    assert options == [], f"Expected [], got {options}"
    print("  ✓ Passed: Initial options are empty")

def test_enable_debug_info():
    """Test enabling debug info"""
    print("\nTest 2: Enable debug info")
    compiler = ArtifactManager()
    compiler.enable_debug_info()
    options = compiler.get_additional_options()
    assert "-lineinfo" in options, f"Expected -lineinfo in {options}"
    print(f"  ✓ Passed: Options after enable_debug_info: {options}")

def test_enable_debug_info_idempotent():
    """Test that enabling debug info multiple times doesn't add duplicates"""
    print("\nTest 3: Enable debug info idempotent")
    compiler = ArtifactManager()
    compiler.enable_debug_info()
    compiler.enable_debug_info()
    options = compiler.get_additional_options()
    count = options.count("-lineinfo")
    assert count == 1, f"Expected 1 occurrence of -lineinfo, got {count}"
    print(f"  ✓ Passed: Only one -lineinfo flag: {options}")

def test_disable_debug_info():
    """Test disabling debug info"""
    print("\nTest 4: Disable debug info")
    compiler = ArtifactManager()
    compiler.enable_debug_info()
    compiler.disable_debug_info()
    options = compiler.get_additional_options()
    assert "-lineinfo" not in options, f"Expected -lineinfo not in {options}"
    print(f"  ✓ Passed: Options after disable_debug_info: {options}")

def test_set_additional_options():
    """Test setting additional compiler options"""
    print("\nTest 5: Set additional options")
    compiler = ArtifactManager()
    compiler.set_additional_options(["-lineinfo", "-G"])
    options = compiler.get_additional_options()
    assert options == ["-lineinfo", "-G"], f"Expected ['-lineinfo', '-G'], got {options}"
    print(f"  ✓ Passed: Options set correctly: {options}")

def test_add_additional_options():
    """Test adding additional compiler options"""
    print("\nTest 6: Add additional options")
    compiler = ArtifactManager()
    compiler.add_additional_options("-lineinfo")
    compiler.add_additional_options(["-G", "-O3"])
    options = compiler.get_additional_options()
    assert options == ["-lineinfo", "-G", "-O3"], f"Expected ['-lineinfo', '-G', '-O3'], got {options}"
    print(f"  ✓ Passed: Options added correctly: {options}")

def test_get_returns_copy():
    """Test that get_additional_options returns a copy"""
    print("\nTest 7: get_additional_options returns copy")
    compiler = ArtifactManager()
    compiler.add_additional_options("-lineinfo")
    options = compiler.get_additional_options()
    options.append("-G")
    # Original should not be modified
    original = compiler.get_additional_options()
    assert original == ["-lineinfo"], f"Expected ['-lineinfo'], got {original}"
    print(f"  ✓ Passed: Original options unchanged: {original}")

def test_backend_switch():
    """Test that options are preserved when switching backends"""
    print("\nTest 8: Options preserved across backend switch")
    compiler = ArtifactManager()
    compiler.enable_debug_info()
    compiler.nvrtc()
    options1 = compiler.get_additional_options()
    compiler.nvcc()
    options2 = compiler.get_additional_options()
    assert options1 == options2 == ["-lineinfo"], f"Options changed: {options1} vs {options2}"
    print(f"  ✓ Passed: Options preserved: {options2}")

def main():
    print("="*60)
    print("Validating Lineinfo Compiler Options API")
    print("="*60)
    
    try:
        test_initialization()
        test_enable_debug_info()
        test_enable_debug_info_idempotent()
        test_disable_debug_info()
        test_set_additional_options()
        test_add_additional_options()
        test_get_returns_copy()
        test_backend_switch()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
