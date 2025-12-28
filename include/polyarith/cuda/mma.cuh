// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2026 Yukimasa Sugizaki

#ifndef POLYARITH_CUDA_MMA_CUH_INCLUDED
#define POLYARITH_CUDA_MMA_CUH_INCLUDED

#include <cstdint>

namespace polyarith {

namespace cuda {

[[maybe_unused]]
static __device__ void mma_m8n8k16(std::uint32_t d[2], const std::uint32_t a[1],
                                   const std::uint32_t b[1]) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0, %1}, {%2}, {%3}, "
      "{0, 0};"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(b[0]));
}

[[maybe_unused]]
static __device__ void mma_m8n8k16(std::uint32_t d[2], const std::uint32_t a[1],
                                   const std::uint32_t b[1],
                                   const std::uint32_t c[2]) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0, %1}, {%2}, {%3}, "
      "{%4, %5};"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
}

static __device__ void mma_m16n8k16(std::uint32_t d[4],
                                    const std::uint32_t a[2],
                                    const std::uint32_t b[1]) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, "
      "%5}, {%6}, {0, 0, 0, 0};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]));
#else
  mma_m8n8k16(&d[0], &a[0], &b[0]);
  mma_m8n8k16(&d[2], &a[1], &b[0]);
#endif
}

static __device__ void mma_m16n8k16(std::uint32_t d[4],
                                    const std::uint32_t a[2],
                                    const std::uint32_t b[1],
                                    const std::uint32_t c[4]) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, "
      "%5}, {%6}, {%7, %8, %9, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]),
        "r"(c[3]));
#else
  mma_m8n8k16(&d[0], &a[0], &b[0], &c[0]);
  mma_m8n8k16(&d[2], &a[1], &b[0], &c[2]);
#endif
}

static __device__ void mma_m16n16k16(std::uint32_t d[8],
                                     const std::uint32_t a[2],
                                     const std::uint32_t b[2]) {
  mma_m16n8k16(&d[0], &a[0], &b[0]);
  mma_m16n8k16(&d[4], &a[0], &b[1]);
}

static __device__ void mma_m16n16k16(std::uint32_t d[8],
                                     const std::uint32_t a[2],
                                     const std::uint32_t b[2],
                                     const std::uint32_t c[4]) {
  mma_m16n8k16(&d[0], &a[0], &b[0], &c[0]);
  mma_m16n8k16(&d[4], &a[0], &b[1], &c[4]);
}

} // namespace cuda

} // namespace polyarith

#endif /* POLYARITH_CUDA_MMA_CUH_INCLUDED */
