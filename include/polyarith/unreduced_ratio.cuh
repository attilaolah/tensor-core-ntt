// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_UNREDUCED_RATIO_CUH_INCLUDED
#define POLYARITH_UNREDUCED_RATIO_CUH_INCLUDED

#include <concepts>

namespace polyarith {

template <int num_, int den_ = 1> class UnreducedRatio {
public:
  static constexpr int num = num_;
  static constexpr int den = den_;

  template <class value_type>
  static __host__ __device__ auto multiply_floor(const value_type value) -> value_type {
    return value * num / den;
  }
};

template <class T>
concept is_unreduced_ratio = requires(T t) {
  { UnreducedRatio(t) } -> std::same_as<T>;
};

} // namespace polyarith

#endif /* POLYARITH_UNREDUCED_RATIO_CUH_INCLUDED */
