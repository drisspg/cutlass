/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture.

    This example demonstrates all layout variants (NN, TN, NT, TT) for mixed precision
    blockscaled GEMM on the NVIDIA Blackwell SM100 architecture.

    Usage:

      $ ./examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm_all_layouts --m=2048 --n=2048 --k=2048
*/

#include <iostream>
#include <vector>
#include <string>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "helper.h"

using namespace cute;

struct Options {
  bool help;
  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle = 0;

  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10),
    swizzle(0)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "72c_blackwell_mixed_mxfp8_bf16_gemm_all_layouts\n\n"
      << "  Blackwell Mxfp8 x Mxfp4 GEMM - all layout variants.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./72c_blackwell_mixed_mxfp8_bf16_gemm_all_layouts" << " --m=1024 --n=512 --k=1024\n\n";

    return out;
  }

  double tflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * m * n * k;  // 2 ops per MAC (multiply + add)
    double tflop = double(flop) / double(1.0e12);  // Convert to trillion FLOPs
    return tflop / runtime_s;  // TFLOP/s
}
};

struct Result {
  double avg_runtime_ms;
  double tflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double tflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), tflops(tflops), status(status), error(error), passed(false)
  {}
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

// Macro to define GEMM runner for a specific layout combination
#define DEFINE_GEMM_RUNNER(NAME, LAYOUT_A, LAYOUT_B) \
struct GemmRunner##NAME { \
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>; \
  using LayoutATag = LAYOUT_A; \
  static constexpr int AlignmentA = 16; \
  \
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>; \
  using LayoutBTag = LAYOUT_B; \
  static constexpr int AlignmentB = 128; \
  \
  using ElementD = cutlass::bfloat16_t; \
  using ElementC = cutlass::bfloat16_t; \
  using LayoutCTag = cutlass::layout::RowMajor; \
  using LayoutDTag = cutlass::layout::RowMajor; \
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; \
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; \
  \
  using ElementAccumulator = float; \
  using ArchTag = cutlass::arch::Sm100; \
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; \
  \
  using MmaTileShape = Shape<_256,_256,_256>; \
  using ClusterShape = Shape<_2,_4,_1>; \
  \
  using CollectiveEpilogueType = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, \
      MmaTileShape, ClusterShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAccumulator, ElementAccumulator, \
      ElementC, LayoutCTag, AlignmentC, \
      ElementD, LayoutDTag, AlignmentD, \
      cutlass::epilogue::collective::EpilogueScheduleAuto \
    >::CollectiveOp; \
  \
  using CollectiveMainloopType = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, \
      ElementA, LayoutATag, AlignmentA, \
      ElementB, LayoutBTag, AlignmentB, \
      ElementAccumulator, \
      MmaTileShape, ClusterShape, \
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogueType::SharedStorage))>, \
      cutlass::gemm::collective::KernelScheduleAuto \
    >::CollectiveOp; \
  \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      Shape<int,int,int,int>, \
      CollectiveMainloopType, \
      CollectiveEpilogueType, \
      void>; \
  \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using LayoutA = decltype(cute::make_layout(make_shape(0,0,0), StrideA{})); \
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using LayoutB = decltype(cute::make_layout(make_shape(0,0,0), StrideB{})); \
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using LayoutC = decltype(cute::make_layout(make_shape(0,0,0), StrideC{})); \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
  using LayoutD = decltype(cute::make_layout(make_shape(0,0,0), StrideD{})); \
  \
  StrideA stride_A; \
  LayoutA layout_A; \
  LayoutSFA layout_SFA; \
  StrideB stride_B; \
  LayoutB layout_B; \
  LayoutSFB layout_SFB; \
  StrideC stride_C; \
  LayoutC layout_C; \
  StrideD stride_D; \
  LayoutD layout_D; \
  uint64_t seed = 0; \
  \
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A; \
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA; \
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B; \
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB; \
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C; \
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D; \
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D; \
  \
  template <typename T> \
  static auto make_iterator(T* ptr) { \
    return cute::recast_ptr<T>(ptr); \
  } \
  \
  void initialize(const Options &options) { \
    using namespace cute; \
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig; \
    \
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1}); \
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1}); \
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1}); \
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1}); \
    \
    layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A); \
    layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B); \
    layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C); \
    layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D); \
    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1)); \
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1)); \
    \
    block_A.reset(cutlass::make_Coord(size(layout_A))); \
    block_B.reset(cutlass::make_Coord(size(layout_B))); \
    block_C.reset(cutlass::make_Coord(size(layout_C))); \
    block_D.reset(cutlass::make_Coord(size(layout_D))); \
    block_reference_D.reset(cutlass::make_Coord(size(layout_D))); \
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA)))); \
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB)))); \
    \
    initialize_block(block_A.host_view(), seed + 2021); \
    initialize_block(block_B.host_view(), seed + 2022); \
    initialize_block(block_C.host_view(), seed + 2023); \
    initialize_block(block_SFA.host_view(), seed + 2024); \
    initialize_block(block_SFB.host_view(), seed + 2025); \
    \
    block_A.sync_device(); \
    block_B.sync_device(); \
    block_C.sync_device(); \
    block_SFA.sync_device(); \
    block_SFB.sync_device(); \
  } \
  \
  typename Gemm::Arguments args_from_options(const Options &options) { \
    typename Gemm::Arguments arguments { \
      cutlass::gemm::GemmUniversalMode::kGemm, \
      {options.m, options.n, options.k, 1}, \
      { \
        block_A.device_data(), stride_A, \
        block_B.device_data(), stride_B, \
        block_SFA.device_data(), layout_SFA, \
        block_SFB.device_data(), layout_SFB \
      }, \
      { \
        {options.alpha, options.beta}, \
        block_C.device_data(), stride_C, \
        block_D.device_data(), stride_D \
      } \
    }; \
    \
    arguments.scheduler.max_swizzle_size = options.swizzle; \
    return arguments; \
  } \
  \
  bool verify(const Options &options) { \
    using namespace cute; \
    Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A); \
    Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA); \
    Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B); \
    Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB); \
    \
    cutlass::reference::host::GettBlockScalingMainloopParams< \
        ElementAccumulator, \
        decltype(tensor_A), \
        decltype(tensor_SFA), \
        decltype(tensor_B), \
        decltype(tensor_SFB) \
      > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB}; \
    \
    auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C); \
    auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D); \
    \
    cutlass::reference::host::GettBlockScalingEpilogueParams< \
        ElementAccumulator, \
        ElementAccumulator, \
        ElementAccumulator, \
        decltype(tensor_C), \
        decltype(tensor_D) \
      > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D}; \
    \
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params); \
    \
    block_D.sync_host(); \
    bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view()); \
    passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0); \
    passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0); \
    \
    return passed; \
  } \
  \
  Result run(Options &options, const std::string& layout_name) { \
    initialize(options); \
    \
    Gemm gemm; \
    auto arguments = args_from_options(options); \
    size_t workspace_size = Gemm::get_workspace_size(arguments); \
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size); \
    \
    CUTLASS_CHECK(gemm.can_implement(arguments)); \
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get())); \
    CUTLASS_CHECK(gemm.run()); \
    \
    cudaDeviceSynchronize(); \
    \
    Result result; \
    result.passed = verify(options); \
    \
    std::cout << "=== Layout: " << layout_name << " ===" << std::endl; \
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl; \
    \
    if (!result.passed) { \
      std::cerr << "  ERROR: Verification failed!" << std::endl; \
      return result; \
    } \
    \
    if (options.iterations > 0) { \
      GpuTimer timer; \
      timer.start(); \
      for (int iter = 0; iter < options.iterations; ++iter) { \
        CUTLASS_CHECK(gemm.initialize(arguments, workspace.get())); \
        CUTLASS_CHECK(gemm.run()); \
      } \
      timer.stop(); \
      \
      float elapsed_ms = timer.elapsed_millis(); \
      result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations); \
      result.tflops = options.tflops(result.avg_runtime_ms / 1000.0); \
      \
      std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl; \
      std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl; \
      std::cout << "  TFLOPS: " << result.tflops << " (" << (result.tflops / 1000.0) << " TFLOPS)" << std::endl; \
    } \
    \
    return result; \
  } \
};

// Define all four layout combinations
DEFINE_GEMM_RUNNER(NN, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor)
DEFINE_GEMM_RUNNER(TN, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor)
DEFINE_GEMM_RUNNER(NT, cutlass::layout::RowMajor, cutlass::layout::RowMajor)
DEFINE_GEMM_RUNNER(TT, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor)

#endif

int main(int argc, char const **args) {
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
    return 0;
  }

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "\n========================================" << std::endl;
  std::cout << "Testing all layout variants" << std::endl;
  std::cout << "========================================\n" << std::endl;

  GemmRunnerNN gemm_nn;
  auto result_nn = gemm_nn.run(options, "NN (A:RowMajor, B:ColumnMajor)");
  std::cout << std::endl;

  GemmRunnerTN gemm_tn;
  auto result_tn = gemm_tn.run(options, "TN (A:ColumnMajor, B:ColumnMajor)");
  std::cout << std::endl;

  GemmRunnerNT gemm_nt;
  auto result_nt = gemm_nt.run(options, "NT (A:RowMajor, B:RowMajor)");
  std::cout << std::endl;

  GemmRunnerTT gemm_tt;
  auto result_tt = gemm_tt.run(options, "TT (A:ColumnMajor, B:RowMajor)");
  std::cout << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "NN: " << (result_nn.passed ? "PASS" : "FAIL") << " - " << result_nn.tflops / 1000.0 << " TFLOPS" << std::endl;
  std::cout << "TN: " << (result_tn.passed ? "PASS" : "FAIL") << " - " << result_tn.tflops / 1000.0 << " TFLOPS" << std::endl;
  std::cout << "NT: " << (result_nt.passed ? "PASS" : "FAIL") << " - " << result_nt.tflops / 1000.0 << " TFLOPS" << std::endl;
  std::cout << "TT: " << (result_tt.passed ? "PASS" : "FAIL") << " - " << result_tt.tflops / 1000.0 << " TFLOPS" << std::endl;
  std::cout << "========================================" << std::endl;

  if (!result_nn.passed || !result_tn.passed || !result_nt.passed || !result_tt.passed) {
    return -1;
  }
#endif

  return 0;
}
