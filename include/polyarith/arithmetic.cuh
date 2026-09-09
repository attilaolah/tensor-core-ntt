// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_ARITHMETIC_CUH_INCLUDED
#define POLYARITH_ARITHMETIC_CUH_INCLUDED

#include <bit>
#include <concepts>
#include <cstdint>

namespace polyarith::arithmetic {

template <class T>
static constexpr auto bitswap_within(T value, const int width) -> std::uint32_t
  requires std::same_as<T, std::uint32_t>
{
  constexpr std::uint32_t c0 = UINT32_C(0x5555'5555),
                          c1 = UINT32_C(0x3333'3333),
                          c2 = UINT32_C(0x0f0f'0f0f),
                          c3 = UINT32_C(0x00ff'00ff);
  value = ((value & c0) << 1) | ((value >> 1) & c0);
  value = ((value & c1) << 2) | ((value >> 2) & c1);
  value = ((value & c2) << 4) | ((value >> 4) & c2);
  value = ((value & c3) << 8) | ((value >> 8) & c3);
  value = std::rotl(value, 16);
  return value >> (32 - width);
}

template <class T>
static __host__ __device__ auto concatenate(const T a0, const T a1, const T a2,
                                            const T a3) -> std::uint32_t
  requires std::same_as<T, std::uint8_t>
{
#if defined(__CUDA_ARCH__)
  std::uint32_t b;
  asm("{\n"
      ".reg.u8 a0, a1, a2, a3;\n"
      "mov.b16 {a0, _}, %1;\n"
      "mov.b16 {a1, _}, %2;\n"
      "mov.b16 {a2, _}, %3;\n"
      "mov.b16 {a3, _}, %4;\n"
      "mov.b32 %0, {a0, a1, a2, a3};"
      "}"
      : "=r"(b)
      : "h"(static_cast<std::uint16_t>(a0)),
        "h"(static_cast<std::uint16_t>(a1)),
        "h"(static_cast<std::uint16_t>(a2)),
        "h"(static_cast<std::uint16_t>(a3)));
  return b;
#else
  return a0 | (static_cast<std::uint32_t>(a1) << 8) |
         (static_cast<std::uint32_t>(a2) << 16) |
         (static_cast<std::uint32_t>(a3) << 24);
#endif
}

template <class T>
static __host__ __device__ auto concatenate(const T a0, const T a1)
    -> std::uint64_t
  requires std::same_as<T, std::uint32_t>
{
#if defined(__CUDA_ARCH__)
  std::uint64_t b;
  asm("mov.b64 %0, {%1, %2};" : "=l"(b) : "r"(a0), "r"(a1));
  return b;
#else
  return a0 | (static_cast<std::uint64_t>(a1) << 32);
#endif
}

/*
 * Addition/subtraction with subtraction/addition if carries/borrows.
 * We want to specify the carry/borrow flag as the predicate for the SEL
 * SASS instruction, but it looks like the PTX assembler cannot regard
 * carry/borrow and predicate as the same thing.
 * We first tried the SETP and the SLCT PTX instructions, but it ended up
 * with borrow predicate -> 0/1 mask by IMAD/IADD3 -> predicate generation
 * by ISETP -> selection by SEL.
 * Fow now, we use the LOP3 PTX instruction, which replaces this ISETP ->
 * SEL steps with simply the LOP3 SASS instruction.
 */

static __host__ __device__ auto
subtract_and_add_if_borrows(const std::uint64_t a, const std::uint64_t b,
                            const std::uint64_t c) -> std::uint64_t {
#if defined(__CUDA_ARCH__)
  std::uint64_t d;
  asm("{\n"
      ".reg.u64 d, e;\n"
      "sub.cc.u64 d, %1, %2;\n"
      "add.u64 e, d, %3;\n"
      ".reg.u32 mask;\n"
      "subc.u32 mask, 0, 0;\n"
      ".reg.u32 d0, d1, e0, e1;\n"
      "mov.b64 {d0, d1}, d;\n"
      "mov.b64 {e0, e1}, e;\n"
      "lop3.b32 d0, d0, e0, mask, (~0xaa & 0xf0) | (0xaa & 0xcc);\n"
      "lop3.b32 d1, d1, e1, mask, (~0xaa & 0xf0) | (0xaa & 0xcc);\n"
      "mov.b64 %0, {d0, d1};\n"
      "}"
      : "=l"(d)
      : "l"(a), "l"(b), "l"(c));
  return d;
#else
  return (a >= b) ? (a - b) : (a - b + c);
#endif
}

template <class T>
static __host__ __device__ auto multiply_high(const T a, const T b)
    -> std::uint32_t
  requires std::same_as<T, std::uint32_t>
{
#if defined(__CUDA_ARCH__)
  return __umulhi(a, b);
#else
  return (static_cast<std::uint64_t>(a) * b) >> 32;
#endif
}

template <class T>
static __host__ __device__ auto multiply_high(const T a, const T b)
    -> std::uint64_t
  requires std::same_as<T, std::uint64_t>
{
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#else
  return (static_cast<unsigned __int128>(a) * b) >> 64;
#endif
}

template <class T>
static __host__ __device__ auto multiply_wide(const T a, const T b)
    -> std::uint64_t
  requires std::same_as<T, std::uint32_t>
{
#if defined(__CUDA_ARCH__)
  std::uint64_t c;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(c) : "r"(a), "r"(b));
  return c;
#else
  return static_cast<std::uint64_t>(a) * b;
#endif
}

template <class T>
static __host__ __device__ auto multiply_wide(const T a, const T b)
    -> unsigned __int128
  requires std::same_as<T, std::uint64_t>
{
  return static_cast<unsigned __int128>(a) * b;
}

} // namespace polyarith::arithmetic

#endif /* POLYARITH_ARITHMETIC_CUH_INCLUDED */
