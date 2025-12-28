// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_MATRIX_VIEW_CUH_INCLUDED
#define POLYARITH_MATRIX_VIEW_CUH_INCLUDED

#include <cstdint>

#include "unreduced_ratio.cuh"

namespace polyarith {

template <int rows_, class row_stride_, int cols_, class col_stride_>
class MatrixView {

public:
  static constexpr int rows = rows_;
  using row_stride = row_stride_;
  static constexpr int cols = cols_;
  using col_stride = col_stride_;

  static_assert(is_unreduced_ratio<row_stride>);
  static_assert(is_unreduced_ratio<col_stride>);

  static_assert(rows >= 1);
  /* row_stride >= col_stride * cols */
  static_assert(row_stride::num * col_stride::den >=
                col_stride::num * cols * row_stride::den);
  static_assert(cols >= 1);
  /* col_stride >= 1 */
  static_assert(col_stride::num >= col_stride::den);

  static_assert(rows * cols % 32 == 0);
  static constexpr int per_warp_size = rows * cols / 32;

private:
  std::uint64_t *const ptr;

public:
  __host__ __device__ MatrixView(const std::uint64_t *const ptr)
      : ptr(const_cast<std::uint64_t *>(ptr)) {}

  __host__ __device__ std::uint64_t *data(void) { return ptr; }

  __host__ __device__ std::uint64_t &at(const int i, const int j) {
    return ptr[row_stride::multiply_floor(i) + col_stride::multiply_floor(j)];
  }
};

} // namespace polyarith

#endif /* POLYARITH_MATRIX_VIEW_CUH_INCLUDED */
