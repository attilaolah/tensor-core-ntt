// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_THREAD_CUH_INCLUDED
#define POLYARITH_CUDA_THREAD_CUH_INCLUDED

namespace polyarith {

namespace cuda {

template <class T> static __device__ T uniform_hint(const T value) {
  return __shfl_sync(UINT32_C(0xffff'ffff), value, 0);
}

[[maybe_unused]]
static __device__ int get_laneId(void) {
  int laneId;
  asm("mov.u32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

template <int dimensions>
[[maybe_unused]]
static __device__ int get_warpId(void)
  requires(dimensions == 1)
{
  return uniform_hint(threadIdx.x / 32);
}

template <int dimensions>
[[maybe_unused]]
static __device__ int get_warpId(void)
  requires(dimensions == 2)
{
  return uniform_hint((threadIdx.x + blockDim.x * threadIdx.y) / 32);
}

template <int dimensions>
[[maybe_unused]]
static __device__ int get_warpId(void)
  requires(dimensions == 3)
{
  return uniform_hint(
      (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) /
      32);
}

[[maybe_unused]]
static __device__ int get_nWarpId(void) {
  int nWarpId;
  asm("mov.u32 %0, %nwarpid;" : "=r"(nWarpId));
  return nWarpId;
}

} // namespace cuda

} // namespace polyarith

#endif /* POLYARITH_CUDA_THREAD_CUH_INCLUDED */
