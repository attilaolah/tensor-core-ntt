// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_MODULUS_CUH_INCLUDED
#define POLYARITH_MODULUS_CUH_INCLUDED

#include <cstdint>

namespace polyarith {

class Modulus {
  std::uint64_t modulus, generator;

public:
  Modulus() = default;

  explicit Modulus(const std::uint64_t modulus, const std::uint64_t generator = {})
      : modulus(modulus), generator(generator) {}

  [[nodiscard]] auto get_modulus() const -> std::uint64_t { return modulus; }

  [[nodiscard]] auto get_generator() const -> std::uint64_t { return generator; }

  [[nodiscard]] auto add(const std::uint64_t a, const std::uint64_t b) const -> std::uint64_t {
    return (a < modulus - b) ? (a + b) : (a + b - modulus);
  }

  [[nodiscard]] auto subtract(const std::uint64_t a, const std::uint64_t b) const -> std::uint64_t {
    return (a >= b) ? (a - b) : (a - b + modulus);
  }

  [[nodiscard]] auto multiply(const std::uint64_t a, const std::uint64_t b) const -> std::uint64_t {
    return a * static_cast<unsigned __int128>(b) % modulus;
  }

  [[nodiscard]] auto power(std::uint64_t a, std::int64_t e) const -> std::uint64_t {
    if (e < 0) {
      return power(a, Modulus(modulus - 1).multiply(modulus - 2, -e));
    }

    std::uint64_t b = 1;
    for (; e; e >>= 1) {
      if (e & 1) {
        b = multiply(b, a);
      }
      a = multiply(a, a);
    }
    return b;
  }

  [[nodiscard]] auto invert(const std::uint64_t a) const -> std::uint64_t {
    return power(a, modulus - 2);
  }

  /* Note: Assuming the specified root exists. */
  [[nodiscard]] auto get_root_forward(const std::uint64_t order) const -> std::uint64_t {
    return power(generator, (modulus - 1) / order);
  }

  /* Note: Assuming the specified root exists. */
  [[nodiscard]] auto get_root_inverse(const std::uint64_t order) const -> std::uint64_t {
    return power(
        generator,
        Modulus(modulus - 1).multiply((modulus - 1) / order, modulus - 2));
  }
};

} // namespace polyarith

#endif /* POLYARITH_MODULUS_CUH_INCLUDED */
