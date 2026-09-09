// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_ERROR_CUH_INCLUDED
#define POLYARITH_CUDA_ERROR_CUH_INCLUDED

#include <stdexcept>
#include <string>

namespace polyarith::cuda {

void checkCudaErrors_impl(const cudaError_t result, const char *const func,
                          const char *const file, const int line) {
  if (result != cudaSuccess) {
    throw std::runtime_error("CUDA error at " + std::string(file) + ":" +
                             std::to_string(line) +
                             " code=" + std::to_string(result) + "(" +
                             cudaGetErrorName(result) + ") \"" + func + "\"");
  }
}

#define checkCudaErrors(value)                                                 \
  polyarith::cuda::checkCudaErrors_impl((value), #value, __FILE__, __LINE__)

} // namespace polyarith::cuda

#endif /* POLYARITH_CUDA_ERROR_CUH_INCLUDED */
