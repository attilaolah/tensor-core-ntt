// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_TWIDDLE_CUH_INCLUDED
#define POLYARITH_CUDA_TWIDDLE_CUH_INCLUDED

#include <bit>
#include <cstdint>
#include <vector>

#include "../arithmetic.cuh"
#include "../matrix_view.cuh"
#include "../modular.cuh"
#include "../modulus.cuh"
#include "../unreduced_ratio.cuh"
#include "matrix.cuh"
#include "thread.cuh"



namespace polyarith::cuda {

class NttForwardTwiddle16x16Coalesced {
  using multiplier_type = modular::MontgomeryFriendlyMultiplier64;

  struct {
    struct {
      multiplier_type values[1];
    } by_lane[32];
  } values[8];

public:
  NttForwardTwiddle16x16Coalesced() = default;

  explicit NttForwardTwiddle16x16Coalesced(const Modulus &modulus) {
    const std::uint64_t root = modulus.get_root_forward(16 * 16);

    std::uint64_t forward_twiddle[16][16];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        forward_twiddle[row][col] = modulus.power(
            root, arithmetic::bitswap_within<std::uint32_t>(row, 4) * col);
      }
    }

    for (int laneId = 0; laneId < 32; ++laneId) {
      std::uint64_t a[8];
      load_matrix_16x16_packed2cols_n_lane(a, &forward_twiddle[0][0], 16,
                                           laneId);
      for (int i = 0; i < 8; ++i) {
        values[i].by_lane[laneId].values[0] =
            multiplier_type(a[i], modulus.get_modulus());
      }
    }
  }

  __device__ void get(multiplier_type out[8]) const {
    const int laneId = get_laneId();

    for (int i = 0; i < 8; ++i) {
      out[i] = values[i].by_lane[laneId].values[0];
    }
  }

  __device__ void compute(std::uint64_t uu[8],
                          const std::uint64_t modulus) const {
    __align__(16) multiplier_type twiddle[8];
    get(twiddle);

    for (int i = 0; i < 8; ++i) {
      uu[i] = twiddle[i].multiply(uu[i], modulus);
    }
  }
};

class NttForwardTwiddleWmma16x16 {
  using multiplier_type = modular::MontgomeryFriendlyMultiplier64;

  multiplier_type values[16][16];

public:
  NttForwardTwiddleWmma16x16() = default;

  explicit NttForwardTwiddleWmma16x16(const Modulus &modulus) {
    const std::uint64_t root = modulus.get_root_forward(16 * 16);

    std::uint64_t forward_twiddle[16][16];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        forward_twiddle[row][col] = modulus.power(
            root, arithmetic::bitswap_within<std::uint32_t>(row, 4) * col);
      }
    }

    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        values[row][col] =
            multiplier_type(forward_twiddle[row][col], modulus.get_modulus());
      }
    }
  }

  __device__ void get(multiplier_type out[8]) const {
    const int laneId = get_laneId();

    static_assert(sizeof(multiplier_type) == sizeof(std::uint64_t));
    MatrixView<16, UnreducedRatio<16>, 16, UnreducedRatio<1>> view(
        reinterpret_cast<const std::uint64_t *>(&values[0][0]));

    load_matrix_packed1col_n(reinterpret_cast<std::uint64_t *>(out), view);
  }

  __device__ void compute(std::uint64_t uu[8],
                          const std::uint64_t modulus) const {
    __align__(16) multiplier_type twiddle[8];
    get(twiddle);

    for (int i = 0; i < 8; ++i) {
      uu[i] = twiddle[i].multiply(uu[i], modulus);
    }
  }
};

template <int n_> class NttForwardTwiddleIterativeCoalesced {
public:
  static constexpr int n = n_;

  static_assert(std::has_single_bit(static_cast<unsigned>(n)) &&
                std::countr_zero(static_cast<unsigned>(n)) % 4 == 0 &&
                n >= 16 * 16);

private:
  using multiplier_type = modular::MontgomeryFriendlyMultiplier64;

  struct {
    struct {
      struct {
        multiplier_type value;
      } by_lane[32];
    } subblock[8];
  } values[n / 256];

public:
  NttForwardTwiddleIterativeCoalesced() = default;

  explicit NttForwardTwiddleIterativeCoalesced(const Modulus &modulus) {
    const std::uint64_t root = modulus.get_root_forward(n);

    std::vector<std::uint64_t> forward_twiddle_data(n);
    MatrixView<16, UnreducedRatio<n / 16>, n / 16, UnreducedRatio<1>>
        forward_twiddle(forward_twiddle_data.data());
#pragma omp parallel for collapse(2)
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < n / 16; ++col) {
        forward_twiddle.at(row, col) = modulus.power(
            root, arithmetic::bitswap_within<std::uint32_t>(row, 4) * col);
      }
    }

#pragma omp parallel for
    for (int col = 0; col < n / 16; col += 16) {
      for (int laneId = 0; laneId < 32; ++laneId) {
        std::uint64_t a[8];
        load_matrix_16x16_packed2cols_n_lane(a, &forward_twiddle.at(0, col),
                                             n / 16, laneId);
        for (int i = 0; i < 8; ++i) {
          values[col / 16].subblock[i].by_lane[laneId].value =
              multiplier_type(a[i], modulus.get_modulus());
        }
      }
    }
  }

  __device__ void get(multiplier_type out[8], const int index) const {
    const int laneId = get_laneId();

    for (int i = 0; i < 8; ++i) {
      out[i] = values[index].subblock[i].by_lane[laneId].value;
    }
  }

  __device__ void compute(std::uint64_t uu[8], const int index,
                          const std::uint64_t modulus) const {
    __align__(16) multiplier_type twiddle[8];
    get(twiddle, index);

    for (int i = 0; i < 8; ++i) {
      uu[i] = twiddle[i].multiply(uu[i], modulus);
    }
  }
};

template <int n_> class NttForwardTwiddleIterativeWmma {
public:
  static constexpr int n = n_;

  static_assert(std::has_single_bit(static_cast<unsigned>(n)) &&
                std::countr_zero(static_cast<unsigned>(n)) % 4 == 0 &&
                n >= 16 * 16);

private:
  using multiplier_type = modular::MontgomeryFriendlyMultiplier64;

  multiplier_type values[n / 16 / 16][16][16];

public:
  NttForwardTwiddleIterativeWmma() = default;

  explicit NttForwardTwiddleIterativeWmma(const Modulus &modulus) {
    const std::uint64_t root = modulus.get_root_forward(n);

    std::vector<std::uint64_t> forward_twiddle_data(n);
    MatrixView<16, UnreducedRatio<n / 16>, n / 16, UnreducedRatio<1>>
        forward_twiddle(forward_twiddle_data.data());
#pragma omp parallel for collapse(2)
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < n / 16; ++col) {
        forward_twiddle.at(row, col) = modulus.power(
            root, arithmetic::bitswap_within<std::uint32_t>(row, 4) * col);
      }
    }

#pragma omp parallel for
    for (int col = 0; col < n / 16; col += 16) {
      for (int row = 0; row < 16; ++row) {
        for (int subcol = 0; subcol < 16; ++subcol) {
          values[col / 16][row][subcol] = multiplier_type(
              forward_twiddle.at(row, col + subcol), modulus.get_modulus());
        }
      }
    }
  }

  __device__ void compute(std::uint64_t uu[8], const int index,
                          const std::uint64_t modulus) const {
    static_assert(sizeof(multiplier_type) == sizeof(std::uint64_t));
    MatrixView<16, UnreducedRatio<16>, 16, UnreducedRatio<1>> view(
        reinterpret_cast<const std::uint64_t *>(&values[index][0][0]));

    __align__(16) multiplier_type twiddle[8];
    load_matrix_packed1col_n(reinterpret_cast<std::uint64_t *>(twiddle), view);

    for (int i = 0; i < 8; ++i) {
      uu[i] = twiddle[i].multiply(uu[i], modulus);
    }
  }
};

} // namespace polyarith::cuda



#endif /* POLYARITH_CUDA_TWIDDLE_CUH_INCLUDED */
