// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_MODULAR_CUH_INCLUDED
#define POLYARITH_MODULAR_CUH_INCLUDED

#include <bit>
#include <cstdint>
#include <stdexcept>

#include "arithmetic.cuh"



namespace polyarith::modular {

static __host__ __device__ auto
calculate_montgomery_inverse(const std::uint32_t modulus) -> std::uint32_t {
  std::uint32_t num, den, t;
  num = (modulus * 3) ^ 2; /* 5 */
  den = num * modulus;

  t = 2 - den;
  num *= t; /* 10 */
  den *= t;

  t = 2 - den;
  num *= t; /* 20 */
  den *= t;

  t = 2 - den;
  num *= t; /* 32 */

  return num;
}

static __host__ __device__ auto
calculate_montgomery_inverse(const std::uint64_t modulus) -> std::uint64_t {
  const std::uint32_t inverse =
      calculate_montgomery_inverse(static_cast<std::uint32_t>(modulus));
  return inverse * (2 - modulus * inverse); /* 32 -> 64 */
}

template <int modulus_bits_> class MontgomeryFriendlyReductionBy64 {
public:
  constexpr static int modulus_bits = modulus_bits_;

private:
  __device__ auto normalize(const std::uint64_t u,
                                     const std::uint64_t modulus) const -> std::uint64_t {
    const auto N1 = static_cast<std::uint32_t>(modulus >> 32);
    if (static_cast<std::uint32_t>(u) >= 2 &&
        static_cast<std::uint32_t>(u >> 32) >= N1 * 2) {
      return u - modulus * 2;
    } else if (static_cast<std::uint32_t>(u) >= 1 &&
               static_cast<std::uint32_t>(u >> 32) >= N1) {
      return u - modulus;
    }
    return u;
  }

  /* Up to 40 bits. */
  __device__ auto
  reduce_impl(const std::uint32_t t0, const std::uint32_t t1,
              const std::uint32_t t2, const std::uint32_t t3,
              const std::uint32_t t4, const std::uint64_t modulus) const -> std::uint64_t {
    std::uint64_t u;

    asm("{\n\t\t"
        ".reg.u64 N;\n\t\t"
        ".reg.u32 N1;\n\t\t"
        "mov.b64 N, %1;\n\t\t"
        "mov.b64 {_, N1}, N;\n\t\t"

        ".reg.u32 s<8>;\n\t\t"
        "mov.b32 s0, %2;\n\t\t"
        "mov.b32 s1, %3;\n\t\t"
        "mov.b32 s2, %4;\n\t\t"
        "mov.b32 s3, %5;\n\t\t"
        "mov.b32 s4, %6;\n\t\t"

        ".reg.u64 c, t;\n\t\t"
        ".reg.u32 c0, c1, u;\n\t\t"

        "shl.b32 u, s1, 16;\n\t\t"
        "add.cc.u32 s0, s0, u;\n\t\t"
        "mul.wide.u32 c, s0, N1;\n\t\t"
        "sub.u64 c, N, c;\n\t\t"

        "shr.b32 u, s1, 16;\n\t\t"
        "addc.u32 s2, s2, u;\n\t\t"
        "shl.b32 u, s3, 16;\n\t\t"
        "add.cc.u32 s2, s2, u;\n\t\t"
        "mov.b64 t, {s2, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s3, 16;\n\t\t"
        "addc.u32 s4, s4, u;\n\t\t"
        "mov.b64 t, {s4, 0};\n\t\t"
        "add.u64 %0, c, t;\n\t\t"
        "}"
        : "=l"(u)
        : "l"(modulus), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(t4));

    return normalize(u, modulus);
  }

  /* Up to 48 bits. */
  __device__ auto
  reduce_impl(const std::uint32_t t0, const std::uint32_t t1,
              const std::uint32_t t2, const std::uint32_t t3,
              const std::uint32_t t4, const std::uint32_t t5,
              const std::uint64_t modulus) const -> std::uint64_t {
    std::uint64_t u;

    asm("{\n\t\t"
        ".reg.u64 N;\n\t\t"
        ".reg.u32 N1;\n\t\t"
        "mov.b64 N, %1;\n\t\t"
        "mov.b64 {_, N1}, N;\n\t\t"

        ".reg.u32 s<8>;\n\t\t"
        "mov.b32 s0, %2;\n\t\t"
        "mov.b32 s1, %3;\n\t\t"
        "mov.b32 s2, %4;\n\t\t"
        "mov.b32 s3, %5;\n\t\t"
        "mov.b32 s4, %6;\n\t\t"
        "mov.b32 s5, %7;\n\t\t"

        ".reg.u64 c, t;\n\t\t"
        ".reg.u32 c0, c1, u;\n\t\t"

        "shl.b32 u, s1, 16;\n\t\t"
        "add.cc.u32 s0, s0, u;\n\t\t"
        "mul.wide.u32 c, s0, N1;\n\t\t"
        "sub.u64 c, N, c;\n\t\t"

        "shr.b32 u, s1, 16;\n\t\t"
        "addc.u32 s2, s2, u;\n\t\t"
        "shl.b32 u, s3, 16;\n\t\t"
        "add.cc.u32 s2, s2, u;\n\t\t"
        "mov.b64 t, {s2, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s3, 16;\n\t\t"
        "addc.u32 s4, s4, u;\n\t\t"
        "shl.b32 u, s5, 16;\n\t\t"
        "add.cc.u32 s4, s4, u;\n\t\t"
        "mov.b64 t, {s4, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s5, 16;\n\t\t"
        "mov.b64 t, {u, 0};\n\t\t"
        "addc.u64 %0, c, t;\n\t\t"
        "}"
        : "=l"(u)
        : "l"(modulus), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(t4), "r"(t5));

    return normalize(u, modulus);
  }

  /* Up to 56 bits. */
  __device__ auto
  reduce_impl(const std::uint32_t t0, const std::uint32_t t1,
              const std::uint32_t t2, const std::uint32_t t3,
              const std::uint32_t t4, const std::uint32_t t5,
              const std::uint32_t t6, const std::uint64_t modulus) const -> std::uint64_t {
    std::uint64_t u;

    asm("{\n\t\t"
        ".reg.u64 N;\n\t\t"
        ".reg.u32 N1;\n\t\t"
        "mov.b64 N, %1;\n\t\t"
        "mov.b64 {_, N1}, N;\n\t\t"

        ".reg.u32 s<8>;\n\t\t"
        "mov.b32 s0, %2;\n\t\t"
        "mov.b32 s1, %3;\n\t\t"
        "mov.b32 s2, %4;\n\t\t"
        "mov.b32 s3, %5;\n\t\t"
        "mov.b32 s4, %6;\n\t\t"
        "mov.b32 s5, %7;\n\t\t"
        "mov.b32 s6, %8;\n\t\t"

        ".reg.u64 c, t;\n\t\t"
        ".reg.u32 c0, c1, u;\n\t\t"

        "shl.b32 u, s1, 16;\n\t\t"
        "add.cc.u32 s0, s0, u;\n\t\t"
        "mul.wide.u32 c, s0, N1;\n\t\t"
        "sub.u64 c, N, c;\n\t\t"

        "shr.b32 u, s1, 16;\n\t\t"
        "addc.u32 s2, s2, u;\n\t\t"
        "shl.b32 u, s3, 16;\n\t\t"
        "add.cc.u32 s2, s2, u;\n\t\t"
        "mov.b64 t, {s2, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s3, 16;\n\t\t"
        "addc.u32 s4, s4, u;\n\t\t"
        "shl.b32 u, s5, 16;\n\t\t"
        "add.cc.u32 s4, s4, u;\n\t\t"
        "mov.b64 t, {s4, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s5, 16;\n\t\t"
        "addc.u32 s6, s6, u;\n\t\t"
        "mov.b64 t, {s6, 0};\n\t\t"
        "add.u64 %0, c, t;\n\t\t"
        "}"
        : "=l"(u)
        : "l"(modulus), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(t4), "r"(t5),
          "r"(t6));

    return normalize(u, modulus);
  }

  /* Up to 62 bits. */
  __device__ auto
  reduce_impl(const std::uint32_t t0, const std::uint32_t t1,
              const std::uint32_t t2, const std::uint32_t t3,
              const std::uint32_t t4, const std::uint32_t t5,
              const std::uint32_t t6, std::uint32_t t7,
              const std::uint64_t modulus) const -> std::uint64_t {
    std::uint64_t u;

    asm("{\n\t\t"
        ".reg.u64 N;\n\t\t"
        ".reg.u32 N1;\n\t\t"
        "mov.b64 N, %1;\n\t\t"
        "mov.b64 {_, N1}, N;\n\t\t"

        ".reg.u32 s<8>;\n\t\t"
        "mov.b32 s0, %2;\n\t\t"
        "mov.b32 s1, %3;\n\t\t"
        "mov.b32 s2, %4;\n\t\t"
        "mov.b32 s3, %5;\n\t\t"
        "mov.b32 s4, %6;\n\t\t"
        "mov.b32 s5, %7;\n\t\t"
        "mov.b32 s6, %8;\n\t\t"
        "mov.b32 s7, %9;\n\t\t"

        ".reg.u64 c, t;\n\t\t"
        ".reg.u32 c0, c1, u;\n\t\t"

        "shl.b32 u, s1, 16;\n\t\t"
        "add.cc.u32 s0, s0, u;\n\t\t"
        "mul.wide.u32 c, s0, N1;\n\t\t"
        "sub.u64 c, N, c;\n\t\t"

        "shr.b32 u, s1, 16;\n\t\t"
        "addc.u32 s2, s2, u;\n\t\t"
        "shl.b32 u, s3, 16;\n\t\t"
        "add.cc.u32 s2, s2, u;\n\t\t"
        "mov.b64 t, {s2, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s3, 16;\n\t\t"
        "addc.u32 s4, s4, u;\n\t\t"
        "shl.b32 u, s5, 16;\n\t\t"
        "add.cc.u32 s4, s4, u;\n\t\t"
        "mov.b64 t, {s4, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "shr.b32 u, s5, 16;\n\t\t"
        "addc.u32 s6, s6, u;\n\t\t"
        "shl.b32 u, s7, 16;\n\t\t"
        "add.cc.u32 s6, s6, u;\n\t\t"
        "mov.b64 t, {s6, 0};\n\t\t"
        "add.u64 c, c, t;\n\t\t"
        "mov.b64 {c0, c1}, c;\n\t\t"
        "mul.wide.u32 c, c0, N1;\n\t\t"
        "mov.b64 t, {c1, 0};\n\t\t"
        "sub.u64 c, t, c;\n\t\t"
        "add.u64 c, c, N;\n\t\t"

        "mov.b64 %0, c;\n"
        "}"
        : "=l"(u)
        : "l"(modulus), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(t4), "r"(t5),
          "r"(t6), "r"(t7));

    return normalize(u, modulus);
  }

public:
  MontgomeryFriendlyReductionBy64() = default;

  explicit MontgomeryFriendlyReductionBy64(const std::uint64_t modulus) {
    if (std::bit_width(modulus) >= 63) {
      throw std::runtime_error("modulus needs to be within 62 bits for now");
    } else if (static_cast<std::uint32_t>(modulus) != 1) {
      throw std::runtime_error("modulus needs to be Montgomery friendly");
    } else if constexpr (modulus_bits != 64) {
      /* Check in two steps to supress compiler warning. */
      if ((modulus >> modulus_bits) != 0) {
        throw std::runtime_error("modulus exceeds the specified bit size");
      }
    }
  }

  static __host__ __device__ auto get_factor() -> int {
    if constexpr (modulus_bits <= 40) {
      return 32 * 2;
    } else if constexpr (modulus_bits <= 48) {
      return 32 * 3;
    } else if constexpr (modulus_bits <= 56) {
      return 32 * 3;
    }
    return 32 * 4;
  }

  __device__ auto
  reduce(const std::uint32_t t0, const std::uint32_t t1, const std::uint32_t t2,
         const std::uint32_t t3, const std::uint32_t t4, const std::uint32_t t5,
         const std::uint32_t t6, std::uint32_t t7,
         const std::uint64_t modulus) const -> std::uint64_t {
    if constexpr (modulus_bits <= 40) {
      return reduce_impl(t0, t1, t2, t3, t4, modulus);
    } else if constexpr (modulus_bits <= 48) {
      return reduce_impl(t0, t1, t2, t3, t4, t5, modulus);
    } else if constexpr (modulus_bits <= 56) {
      return reduce_impl(t0, t1, t2, t3, t4, t5, t6, modulus);
    }
    return reduce_impl(t0, t1, t2, t3, t4, t5, t6, t7, modulus);
  }
};

class MontgomeryMultiplier64 {
  std::uint64_t b;
  std::uint64_t bp;

public:
  MontgomeryMultiplier64() = default;

  MontgomeryMultiplier64(const std::uint64_t multiplier,
                         const std::uint64_t modulus) {
    b = arithmetic::multiply_wide(multiplier, -modulus) % modulus;
    bp = b * calculate_montgomery_inverse(modulus);
  }

  __host__ __device__ auto multiply(const std::uint64_t a,
                                             const std::uint64_t N) const -> std::uint64_t {
    const std::uint64_t ab1 = arithmetic::multiply_high(a, b);
    const std::uint64_t q = a * bp;
    const std::uint64_t qN1 = arithmetic::multiply_high(q, N);
    return arithmetic::subtract_and_add_if_borrows(ab1, qN1, N);
  }
};

class MontgomeryFriendlyMultiplier64 {
  std::uint64_t b;

public:
  MontgomeryFriendlyMultiplier64() = default;

  MontgomeryFriendlyMultiplier64(const std::uint64_t multiplier,
                                 const std::uint64_t modulus) {
    if (std::bit_width(modulus) >= 63) {
      throw std::runtime_error("modulus needs to be within 62 bits for now");
    } else if (static_cast<std::uint32_t>(modulus) != 1) {
      throw std::runtime_error("modulus needs to be Montgomery friendly");
    }

    b = arithmetic::multiply_wide(multiplier, -modulus) % modulus;
  }

  __host__ __device__ auto get_b() const -> std::uint64_t { return b; }

  __host__ __device__ auto multiply(const std::uint64_t a,
                                             const std::uint64_t N) const -> std::uint64_t {
    const std::uint32_t a0 = a;
    const std::uint32_t a1 = a >> 32;
    const std::uint32_t b0 = b;
    const std::uint32_t b1 = b >> 32;
    const std::uint32_t N1 = N >> 32;

    std::uint64_t c;
    c = arithmetic::multiply_wide(a0, b0) + (static_cast<std::uint64_t>(1) << 32);
    c = arithmetic::concatenate<std::uint32_t>(c >> 32, N1 + 1) -
        arithmetic::multiply_wide<std::uint32_t>(c, N1) +
        arithmetic::multiply_wide(a0, b1) + arithmetic::multiply_wide(a1, b0);
    c = arithmetic::concatenate<std::uint32_t>(c >> 32, N1) -
        arithmetic::multiply_wide<std::uint32_t>(c, N1) +
        arithmetic::multiply_wide(a1, b1);

    if (static_cast<std::uint32_t>(c) >= 1 &&
        static_cast<std::uint32_t>(c >> 32) >= N1) {
      c -= N;
    }

    return c;
  }
};

} // namespace polyarith::modular



#endif /* POLYARITH_MODULAR_CUH_INCLUDED */
