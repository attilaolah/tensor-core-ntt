// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_MATRIX_CUH_INCLUDED
#define POLYARITH_CUDA_MATRIX_CUH_INCLUDED

#include <cstdint>
#include <cuda/std/utility>

#include "../unreduced_ratio.cuh"
#include "thread.cuh"



namespace polyarith::cuda {

template <class T>
static void load_matrix_16x16_packed4cols_n_lane(T a[8], const T *const ptr,
                                                 const int stride,
                                                 const int laneId) {
  const dim3 groupIdx(laneId % 4, laneId / 4);

  for (int i = 0; i < 8; ++i) {
    const int row = i / 4 * 8 + groupIdx.y;
    const int col = 4 * groupIdx.x + i % 4;
    a[i] = ptr[stride * row + col];
  }
}

template <class T>
static void load_matrix_16x16_packed2cols_n_lane(T a[8], const T *const ptr,
                                                 const int stride,
                                                 const int laneId) {
  const dim3 groupIdx(laneId % 4, laneId / 4);

  for (int i = 0; i < 8; ++i) {
    const int row = i / 2 % 2 * 8 + groupIdx.y;
    const int col = i / 4 * 8 + 2 * groupIdx.x + i % 2;
    a[i] = ptr[stride * row + col];
  }
}

template <class view_type>
static __host__ __device__ void
load_matrix2_packed4cols_n_lane(std::uint64_t *const a, view_type view,
                                const int laneId) {
  const dim3 groupIdx(laneId % 4, laneId / 4);

  for (int i = 0; i < view_type::per_warp_size; ++i) {
    const int row = i / 4 * 8 + groupIdx.y;
    const int col = 4 * groupIdx.x + i % 4;
    a[i] = view.at(row, col);
  }
}

template <class view_type>
static __device__ void load_matrix2_packed4cols_n(std::uint64_t *const a,
                                                  view_type view) {
  const int laneId = get_laneId();

  load_matrix2_packed4cols_n_lane(a, view, laneId);
}

template <class view_type>
static __host__ __device__ void
load_matrix2_packed4cols_t_lane(std::uint64_t *const a, view_type view,
                                const int laneId) {
  const dim3 groupIdx(laneId % 4, laneId / 4);

  if constexpr (view_type::per_warp_size % 2 == 0 && 0) {
    for (int i = 0; i < view_type::per_warp_size / 2; ++i) {
      const int row = 4 * groupIdx.x + i % 4;
      const int i0 = (laneId & 2) ? (i + 4) : i;
      const int i1 = (laneId & 2) ? i : (i + 4);
      const int col0 = i0 / 4 * 8 + groupIdx.y;
      const int col1 = i1 / 4 * 8 + groupIdx.y;
      a[i0] = view.at(row, col0);
      a[i1] = view.at(row, col1);
    }
  } else {
    for (int i = 0; i < view_type::per_warp_size; ++i) {
      const int row = 4 * groupIdx.x + i % 4;
      const int col = i / 4 * 8 + groupIdx.y;
      a[i] = view.at(row, col);
    }
  }
}

template <class view_type>
static __device__ void load_matrix2_packed4cols_t(std::uint64_t *const a,
                                                  view_type view) {
  const int laneId = get_laneId();

  load_matrix2_packed4cols_t_lane(a, view, laneId);
}

template <class view_type>
static __device__ void
store_matrix2_packed2cols_n(view_type view, const std::uint64_t *const a) {
  const int laneId = get_laneId();
  const dim3 groupIdx(laneId % 4, laneId / 4);

  if constexpr (std::same_as<typename view_type::col_stride,
                             UnreducedRatio<1>> &&
                view_type::per_warp_size % 2 == 0 && 1) {
    for (int i = 0; i < view_type::per_warp_size; i += 2) {
      const int row = i / 2 % 2 * 8 + groupIdx.y;
      const int col = i / 4 * 8 + 2 * groupIdx.x;
      *reinterpret_cast<ulong2 *>(&view.at(row, col)) =
          *reinterpret_cast<const ulong2 *>(&a[i]);
    }
  } else if constexpr (view_type::per_warp_size % 2 == 0 && 1) {
    for (int i = 0; i < view_type::per_warp_size; i += 2) {
      const int row = i / 2 % 2 * 8 + groupIdx.y;
      const int col = i / 4 * 8 + 2 * groupIdx.x;
      const int i0 = (laneId & 8) ? (i + 1) : i;
      const int i1 = (laneId & 8) ? i : (i + 1);
      const int col0 = (laneId & 8) ? (col + 1) : col;
      const int col1 = (laneId & 8) ? col : (col + 1);
      view.at(row, col0) = a[i0];
      view.at(row, col1) = a[i1];
    }
  } else {
    for (int i = 0; i < view_type::per_warp_size; ++i) {
      const int row = i / 2 % 2 * 8 + groupIdx.y;
      const int col = i / 4 * 8 + 2 * groupIdx.x + i % 2;
      view.at(row, col) = a[i];
    }
  }
}

template <class view_type>
static __device__ void
store_matrix2_packed2cols_t(view_type view, const std::uint64_t *const a) {
  const int laneId = get_laneId();
  const dim3 groupIdx(laneId % 4, laneId / 4);

  for (int i = 0; i < view_type::per_warp_size; ++i) {
    const int row = i / 4 * 8 + 2 * groupIdx.x + i % 2;
    const int col = i / 2 % 2 * 8 + groupIdx.y;
    view.at(row, col) = a[i];
  }
}

template <class view_type>
static __device__ void load_matrix_packed1col_n(std::uint64_t a[8],
                                                view_type view) {
  const int laneId = get_laneId();

  for (int i = 0; i < 8; ++i) {
    const int row = 2 * i + 1 * (laneId / 16);
    const int col = laneId % 16;
    a[i] = view.at(row, col);
  }
}

template <class view_type>
static __device__ void load_matrix_packed1col_t(std::uint64_t a[8],
                                                view_type view) {
  const int laneId = get_laneId();

  for (int i = 0; i < 8; ++i) {
    const int row = laneId % 16;
    const int col = 2 * i + 1 * (laneId / 16);
    a[i] = view.at(row, col);
  }
}

template <class view_type>
static __device__ void store_matrix_packed1col_n(view_type view,
                                                 const std::uint64_t a[8]) {
  const int laneId = get_laneId();

  for (int i = 0; i < 8; ++i) {
    const int row = 2 * i + 1 * (laneId / 16);
    const int col = laneId % 16;
    view.at(row, col) = a[i];
  }
}

template <class view_type>
static __device__ void store_matrix_packed1col_t(view_type view,
                                                 const std::uint64_t a[8]) {
  const int laneId = get_laneId();

  for (int i = 0; i < 8; ++i) {
    const int row = laneId % 16;
    const int col = 2 * i + 1 * (laneId / 16);
    view.at(row, col) = a[i];
  }
}

static __device__ void
matrix_8x16_packed2cols_to_packed4cols(std::uint64_t a[4]) {
  const int laneId = get_laneId();
  const dim3 groupIdx(laneId % 4, laneId / 4);

  const int src0 = ((laneId & 0x2) >> 1) | ((laneId & 0x1) << 1);
  const int src1 = src0 ^ 1;
  const std::uint32_t mask = UINT32_C(0xffff'ffff);
  std::uint64_t b[4];
  b[0] = __shfl_sync(mask, (laneId & 1) ? a[2] : a[0], src0, 4);
  b[1] = __shfl_sync(mask, (laneId & 1) ? a[3] : a[1], src0, 4);
  b[2] = __shfl_sync(mask, (laneId & 1) ? a[0] : a[2], src1, 4);
  b[3] = __shfl_sync(mask, (laneId & 1) ? a[1] : a[3], src1, 4);

  if (laneId & 2) {
    ::cuda::std::swap(b[0], b[2]);
    ::cuda::std::swap(b[1], b[3]);
  }
  for (int i = 0; i < 4; ++i) {
    a[i] = b[i];
  }
}

static __device__ void
matrix_16x16_packed2cols_to_packed4cols(std::uint64_t a[8]) {
  ::cuda::std::swap(a[2], a[4]);
  ::cuda::std::swap(a[3], a[5]);
  matrix_8x16_packed2cols_to_packed4cols(&a[0]);
  matrix_8x16_packed2cols_to_packed4cols(&a[4]);
}

template <int num_bits>
static __device__ void
matrix_16x16_decompose_u64_to_u8(std::uint32_t aa[8][2],
                                 const std::uint64_t a[8]) {
  aa[0][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 0, a[1] >> 0,
                                                   a[2] >> 0, a[3] >> 0);
  aa[1][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 8, a[1] >> 8,
                                                   a[2] >> 8, a[3] >> 8);
  aa[2][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 16, a[1] >> 16,
                                                   a[2] >> 16, a[3] >> 16);
  aa[3][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 24, a[1] >> 24,
                                                   a[2] >> 24, a[3] >> 24);
  aa[4][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 32, a[1] >> 32,
                                                   a[2] >> 32, a[3] >> 32);
  if constexpr (num_bits > 40) {
    aa[5][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 40, a[1] >> 40,
                                                     a[2] >> 40, a[3] >> 40);
  }
  if constexpr (num_bits > 48) {
    aa[6][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 48, a[1] >> 48,
                                                     a[2] >> 48, a[3] >> 48);
  }
  if constexpr (num_bits > 56) {
    aa[7][0] = arithmetic::concatenate<std::uint8_t>(a[0] >> 56, a[1] >> 56,
                                                     a[2] >> 56, a[3] >> 56);
  }

  aa[0][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 0, a[5] >> 0,
                                                   a[6] >> 0, a[7] >> 0);
  aa[1][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 8, a[5] >> 8,
                                                   a[6] >> 8, a[7] >> 8);
  aa[2][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 16, a[5] >> 16,
                                                   a[6] >> 16, a[7] >> 16);
  aa[3][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 24, a[5] >> 24,
                                                   a[6] >> 24, a[7] >> 24);
  aa[4][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 32, a[5] >> 32,
                                                   a[6] >> 32, a[7] >> 32);
  if constexpr (num_bits > 40) {
    aa[5][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 40, a[5] >> 40,
                                                     a[6] >> 40, a[7] >> 40);
  }
  if constexpr (num_bits > 48) {
    aa[6][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 48, a[5] >> 48,
                                                     a[6] >> 48, a[7] >> 48);
  }
  if constexpr (num_bits > 56) {
    aa[7][1] = arithmetic::concatenate<std::uint8_t>(a[4] >> 56, a[5] >> 56,
                                                     a[6] >> 56, a[7] >> 56);
  }
}

} // namespace polyarith::cuda



#endif /* POLYARITH_CUDA_MATRIX_CUH_INCLUDED */
