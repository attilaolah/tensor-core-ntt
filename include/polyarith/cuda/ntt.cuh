// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_NTT_CUH_INCLUDED
#define POLYARITH_CUDA_NTT_CUH_INCLUDED

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <mma.h>

#include "../arithmetic.cuh"
#include "../modular.cuh"
#include "../modulus.cuh"
#include "matrix.cuh"
#include "mma.cuh"
#include "thread.cuh"

namespace polyarith {

namespace cuda {

template <class reduction_type_> class NttForward16x16Coalesced {

public:
  using reduction_type = reduction_type_;

private:
  struct {
    struct {
      std::uint64_t value;
    } by_lane[32];
  } values[8];

public:
  NttForward16x16Coalesced(void) = default;

  NttForward16x16Coalesced(const Modulus &modulus) {
    const std::uint64_t factor = modulus.power(2, reduction_type::get_factor());
    const std::uint64_t root = modulus.get_root_forward(16);

    std::uint64_t forward_ntt[16][16];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        forward_ntt[row][col] = modulus.multiply(
            modulus.power(root,
                          polyarith::arithmetic::bitswap_within(row, 4) * col),
            factor);
      }
    }

    for (int laneId = 0; laneId < 32; ++laneId) {
      std::uint64_t a[8];
      polyarith::cuda::load_matrix_16x16_packed4cols_n_lane(
          a, &forward_ntt[0][0], 16, laneId);
      for (int pos = 0; pos < 8; ++pos) {
        std::uint8_t aa[8];
        for (int i = 0; i < 8; ++i) {
          aa[i] = a[i] >> (8 * pos);
        }
        std::memcpy(&values[pos].by_lane[laneId].value, aa, 8);
      }
    }
  }

  __device__ void get(void *const out) const {
    const int laneId = polyarith::cuda::get_laneId();

    for (int i = 0; i < (reduction_type::modulus_bits + 7) / 8; ++i) {
      reinterpret_cast<std::uint64_t *>(out)[i] =
          values[i].by_lane[laneId].value;
    }
  }

private:
  static __device__ void multiply_ntt_matrix(std::uint64_t uu[8],
                                             const std::uint32_t aa[8][2],
                                             const std::uint32_t bb[8][2],
                                             const std::uint64_t modulus,
                                             const reduction_type &reduction) {
    std::uint32_t ss[15][8]{};
    std::uint32_t tt[8][8]{};

    constexpr int num_digits = (reduction_type::modulus_bits + 7) / 8;
    for (int i = 0; i <= 2 * (num_digits - 1); ++i) {
      for (int j = 0; j < num_digits; ++j) {
        const int k = i - j;
        if (0 <= k && k < num_digits) {
          polyarith::cuda::mma_m16n16k16(ss[i], aa[j], bb[k], ss[i]);
        }
      }
    }

    for (int i = 0; i < num_digits - 1; ++i) {
      for (int j = 0; j < 8; ++j) {
        tt[i][j] = ss[2 * i + 0][j] + (ss[2 * i + 1][j] << 8);
      }
    }
    for (int j = 0; j < 8; ++j) {
      tt[num_digits - 1][j] = ss[2 * (num_digits - 1)][j];
    }

    for (int i = 0; i < 8; ++i) {
      uu[i] = reduction.reduce(tt[0][i], tt[1][i], tt[2][i], tt[3][i], tt[4][i],
                               tt[5][i], tt[6][i], tt[7][i], modulus);
    }
  }

public:
  __device__ void compute_2n(std::uint64_t a[8], const std::uint64_t modulus,
                             const reduction_type &reduction) const {
    const int laneId = polyarith::cuda::get_laneId();
    const dim3 groupIdx(laneId % 4, laneId / 4);

    std::uint32_t aa[8][2];
    polyarith::cuda::matrix_16x16_decompose_u64_to_u8<
        reduction_type::modulus_bits>(aa, a);

    __align__(16) std::uint32_t forward_ntt[8][2];
    get(forward_ntt);

    // mma(A, N^T) = A N
    // mma(A^T, N^T) = A^T N
    multiply_ntt_matrix(a, aa, forward_ntt, modulus, reduction);
  }

  __device__ void compute_2t(std::uint64_t a[8], const std::uint64_t modulus,
                             const reduction_type &reduction) const {
    const int laneId = polyarith::cuda::get_laneId();
    const dim3 groupIdx(laneId % 4, laneId / 4);

    std::uint32_t aa[8][2];
    polyarith::cuda::matrix_16x16_decompose_u64_to_u8<
        reduction_type::modulus_bits>(aa, a);

    __align__(16) std::uint32_t forward_ntt[8][2];
    get(forward_ntt);

    // mma(N^T, A) = N^T A^T = (A N)^T
    // mma(N^T, A^T) = N^T A = (A^T N)^T
    multiply_ntt_matrix(a, forward_ntt, aa, modulus, reduction);
  }
};

template <class reduction_type_> class NttForwardWmma16x16 {

public:
  using reduction_type = reduction_type_;

private:
  std::uint8_t values[8][16][16];

public:
  NttForwardWmma16x16(void) = default;

  NttForwardWmma16x16(const Modulus &modulus) {
    const std::uint64_t factor = modulus.power(2, reduction_type::get_factor());
    const std::uint64_t root = modulus.get_root_forward(16);

    std::uint64_t forward_ntt[16][16];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        forward_ntt[row][col] = modulus.multiply(
            modulus.power(root,
                          polyarith::arithmetic::bitswap_within(row, 4) * col),
            factor);
      }
    }

    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        for (int pos = 0; pos < 8; ++pos) {
          values[pos][row][col] = forward_ntt[row][col] >> (8 * pos);
        }
      }
    }
  }

private:
  static __device__ void multiply_ntt_matrix_wmma(
      std::uint64_t uu[8],
      const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                   std::uint8_t, nvcuda::wmma::row_major>
          aa[8],
      const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                   std::uint8_t, nvcuda::wmma::col_major>
          bb[8],
      const nvcuda::wmma::layout_t layout, const std::uint64_t modulus,
      const reduction_type &reduction) {
    const int laneId = polyarith::cuda::get_laneId();
    const int warpId = polyarith::cuda::get_warpId<2>();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, std::int32_t>
        ss[15], tt[8];
    for (int i = 0; i < 15; ++i) {
      nvcuda::wmma::fill_fragment(ss[i], 0);
    }

    constexpr int num_digits = (reduction_type::modulus_bits + 7) / 8;
    for (int i = 0; i <= 2 * (num_digits - 1); ++i) {
      for (int j = 0; j < num_digits; ++j) {
        const int k = i - j;
        if (0 <= k && k < num_digits) {
          nvcuda::wmma::mma_sync(ss[i], aa[j], bb[k], ss[i]);
        }
      }
    }

    for (int i = 0; i < num_digits - 1; ++i) {
      for (int j = 0; j < 8; ++j) {
        tt[i].x[j] = ss[2 * i + 0].x[j] + (ss[2 * i + 1].x[j] << 8);
      }
    }
    for (int j = 0; j < 8; ++j) {
      tt[num_digits - 1].x[j] = ss[2 * (num_digits - 1)].x[j];
    }

    extern __shared__ std::int32_t vv[][8][16][16];
    for (int i = 0; i < 8; ++i) {
      nvcuda::wmma::store_matrix_sync(&vv[warpId][i][0][0], tt[i], 16, layout);
    }

    __syncwarp();

    for (int i = 0; i < 8; ++i) {
      const int row = 2 * i + 1 * (laneId / 16);
      const int col = laneId % 16;
      uu[i] = reduction.reduce(vv[warpId][0][row][col], vv[warpId][1][row][col],
                               vv[warpId][2][row][col], vv[warpId][3][row][col],
                               vv[warpId][4][row][col], vv[warpId][5][row][col],
                               vv[warpId][6][row][col], vv[warpId][7][row][col],
                               modulus);
    }
  }

public:
  __device__ void compute(std::uint64_t a[8],
                          const nvcuda::wmma::layout_t layout,
                          const std::uint64_t modulus,
                          const reduction_type &reduction) const {
    const int laneId = polyarith::cuda::get_laneId();
    const int warpId = polyarith::cuda::get_warpId<2>();

    std::uint32_t aa[8][2];
    polyarith::cuda::matrix_16x16_decompose_u64_to_u8<
        reduction_type::modulus_bits>(aa, a);

    extern __shared__ std::uint8_t bb[][8][16][16];
    for (int pos = 0; pos < 8; ++pos) {
      for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + 1 * (laneId / 16);
        const int col = laneId % 16;
        // XXX
        bb[warpId * 4][pos][row][col] =
            aa[pos][(i < 4) ? 0 : 1] >> (8 * (i % 4));
      }
    }

    __syncwarp();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, std::uint8_t,
                           nvcuda::wmma::row_major>
        cc[8];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, std::uint8_t,
                           nvcuda::wmma::col_major>
        dd[8];
    for (int pos = 0; pos < 8; ++pos) {
      nvcuda::wmma::load_matrix_sync(cc[pos], &bb[warpId * 4][pos][0][0], 16);
      nvcuda::wmma::load_matrix_sync(dd[pos], &values[pos][0][0], 16);
    }

    // mma(A, N^T) = A N
    // mma(A^T, N^T) = A^T N
    multiply_ntt_matrix_wmma(a, cc, dd, layout, modulus, reduction);
  }
};

template <int n_, int radix_> class NttForwardScalarIterative {

public:
  static constexpr int n = n_;
  static constexpr int radix = radix_;

private:
  using multiplier_type = polyarith::modular::MontgomeryFriendlyMultiplier64;

  multiplier_type values[radix - 1][n / radix];

public:
  NttForwardScalarIterative(void) = default;

  NttForwardScalarIterative(const Modulus &modulus) {
    const std::uint64_t root = modulus.get_root_forward(n);

#pragma omp parallel for
    for (int i = 0; i < n / radix; ++i) {
      std::vector<int> indices;
      indices.reserve(radix - 1);
      for (int subradix = radix; subradix > 1; subradix /= 2) {
        for (int j = 0; j < subradix / 2; ++j) {
          indices.push_back(i * (radix / subradix) + n / subradix * j);
        }
      }
      if (indices.size() != radix - 1) {
        throw std::runtime_error("internal error");
      }
      for (int j = 0; j < radix - 1; ++j) {
        values[j][i] = multiplier_type(modulus.power(root, indices[j]),
                                       modulus.get_modulus());
      }
    }
  }

  __device__ void get(multiplier_type out[radix - 1], const int index) const {
    for (int i = 0; i < radix - 1; ++i) {
      out[i] = values[i][index];
    }
  }

private:
  static __device__ void montgomery_butterfly(
      std::uint64_t &a0, std::uint64_t &a1,
      const polyarith::modular::MontgomeryFriendlyMultiplier64 omega,
      const std::uint64_t N) {
    std::uint64_t b0;
    b0 = a0 + a1;
    if (b0 >= N * 2) {
      b0 -= N * 2;
    }

    std::uint64_t b1;
    b1 = a0 - a1 + N * 2;
    b1 = omega.multiply(b1, N);

    a0 = b0;
    a1 = b1;
  }

  template <int radix>
  static __device__ void montgomery_butterfly(
      std::uint64_t a[radix],
      const polyarith::modular::MontgomeryFriendlyMultiplier64 omega[radix - 1],
      const std::uint64_t N) {
    for (int i = 0; i < radix / 2; ++i) {
      montgomery_butterfly(a[radix / 2 * 0 + i], a[radix / 2 * 1 + i], omega[i],
                           N);
    }

    if constexpr (radix / 2 > 1) {
      montgomery_butterfly<radix / 2>(&a[radix / 2 * 0], &omega[radix / 2], N);
      montgomery_butterfly<radix / 2>(&a[radix / 2 * 1], &omega[radix / 2], N);
    }
  }

public:
  __device__ void compute(std::uint64_t a[radix], const int index,
                          const std::uint64_t modulus) const {
    multiplier_type omega[radix - 1];
    get(omega, index);

    montgomery_butterfly<radix>(a, omega, modulus);
  }
};

} // namespace cuda

} // namespace polyarith

#endif /* POLYARITH_CUDA_NTT_CUH_INCLUDED */
