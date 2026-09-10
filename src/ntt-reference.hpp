// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef NTT_REFERENCE_HPP_INCLUDED
#define NTT_REFERENCE_HPP_INCLUDED

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "polyarith/polyarith.cuh"

class NttReference {

  std::uint64_t log2m;
  polyarith::Modulus modulus;
  std::uint64_t omega_m, minv;

public:
  NttReference(void) = default;

  NttReference(const std::uint64_t m, const std::uint64_t N,
               const std::uint64_t generator) {
    if (!std::has_single_bit(m)) {
      throw std::invalid_argument(
          "Transform length must be a power of two for now");
    }

    log2m = std::countr_zero(m);
    modulus = polyarith::Modulus(N, generator);
    omega_m = modulus.get_root_forward(m);
    minv = modulus.invert(m);
  }

  void compute_forward(std::uint64_t *const dst,
                       const std::uint64_t *src) const {
    const std::uint64_t m = std::uint64_t{1} << log2m;

    std::uint64_t omega_2l = omega_m;
    std::vector<std::uint64_t> omega_2l_j_table;
    omega_2l_j_table.reserve(std::uint64_t(1) << (log2m - 1));

    for (std::uint64_t i = log2m - 1; i < log2m; --i) {
      const std::uint64_t l = std::uint64_t(1) << i;
      omega_2l_j_table.resize(l);

#pragma omp parallel for
      for (std::uint64_t j = 0; j < l; ++j) {
        omega_2l_j_table[j] = modulus.power(omega_2l, j);
      }

#pragma omp parallel for collapse(2)
      for (std::uint64_t j = 0; j < l; ++j) {
        for (std::uint64_t k = 0; k < m; k += l * 2) {
          const std::uint64_t x0 = src[j + k], x1 = src[j + k + l];
          dst[j + k] = modulus.add(x0, x1);
          dst[j + k + l] =
              modulus.multiply(modulus.subtract(x0, x1), omega_2l_j_table[j]);
        }
      }

      omega_2l = modulus.multiply(omega_2l, omega_2l);
      src = dst;
    }
  }

  void compute_inverse(std::uint64_t *const dst,
                       const std::uint64_t *const src) const {
    const std::uint64_t m = std::uint64_t{1} << log2m;

    for (std::uint64_t i{0}; i < m; ++i) {
      dst[i] = modulus.multiply(src[i], minv);
    }

    std::vector<std::uint64_t> omegainv_2l_j_table;
    omegainv_2l_j_table.reserve(std::uint64_t(1) << (log2m - 1));

    for (std::uint64_t i{0}; i < log2m; ++i) {
      const std::uint64_t l = std::uint64_t(1) << i;
      omegainv_2l_j_table.resize(l);

#pragma omp parallel for
      for (std::uint64_t j = 0; j < l; ++j) {
        omegainv_2l_j_table[j] =
            modulus.power(modulus.get_root_inverse(2 * l), j);
      }

#pragma omp parallel for collapse(2)
      for (std::uint64_t j = 0; j < l; ++j) {
        for (std::uint64_t k = j; k < m; k += l * 2) {
          const std::uint64_t x0 = dst[k],
                              x1 = modulus.multiply(dst[k + l],
                                                    omegainv_2l_j_table[j]);
          dst[k] = modulus.add(x0, x1);
          dst[k + l] = modulus.subtract(x0, x1);
        }
      }
    }
  }
};

#endif /* NTT_REFERENCE_HPP_INCLUDED */
