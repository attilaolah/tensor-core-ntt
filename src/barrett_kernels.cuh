#include "polyarith/polyarith.cuh"
#pragma once
#include <cstdint>
#include <utility>

__device__ __forceinline__ void device_warp_normalize(uint64_t *limbs,
                                                      uint64_t *s_direct_carry,
                                                      uint32_t *s_warp_G,
                                                      uint32_t *s_warp_P,
                                                      uint32_t &chunk_carry) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  uint64_t local_carry = 0;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint64_t value = limbs[i] + local_carry;
    limbs[i] = value & 0xFFFF;
    local_carry = value >> 16;
  }
  s_direct_carry[tid] = local_carry;
  __syncthreads();

  local_carry = tid == 0 ? chunk_carry : s_direct_carry[tid - 1];
  uint32_t block_propagates = 1;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint64_t value = limbs[i] + local_carry;
    limbs[i] = value & 0xFFFF;
    local_carry = value >> 16;
    block_propagates &= limbs[i] == 0xFFFF;
  }

  uint32_t val = (local_carry != 0 ? 0x10000U : 0U) |
                 (block_propagates != 0 ? 0xFFFFU : 0U);
  uint32_t G = val >= 0x10000;
  uint32_t P = (val & 0xFFFF) == 0xFFFF;

#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const uint32_t in_G = __shfl_up_sync(0xFFFFFFFF, G, offset);
    const uint32_t in_P = __shfl_up_sync(0xFFFFFFFF, P, offset);
    if (lane >= offset) {
      G |= P & in_G;
      P &= in_P;
    }
  }

  if (lane == 31) {
    s_warp_G[warp_id] = G;
    s_warp_P[warp_id] = P;
  }
  __syncthreads();

  if (warp_id == 0) {
    uint32_t wG = s_warp_G[lane];
    uint32_t wP = s_warp_P[lane];
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const uint32_t in_G = __shfl_up_sync(0xFFFFFFFF, wG, offset);
      const uint32_t in_P = __shfl_up_sync(0xFFFFFFFF, wP, offset);
      if (lane >= offset) {
        wG |= wP & in_G;
        wP &= in_P;
      }
    }
    s_warp_G[lane] = wG;
    s_warp_P[lane] = wP;
  }
  __syncthreads();

  const uint32_t carry_into_warp = warp_id > 0 ? s_warp_G[warp_id - 1] : 0;
  uint32_t carry_into_thread = carry_into_warp;
  const uint32_t G_prev = __shfl_up_sync(0xFFFFFFFF, G, 1);
  const uint32_t P_prev = __shfl_up_sync(0xFFFFFFFF, P, 1);
  if (lane > 0) {
    carry_into_thread = G_prev | (P_prev & carry_into_warp);
  }

  val = (val & 0xFFFF) + carry_into_thread;
  uint64_t carry = val >> 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint64_t value = limbs[i] + carry;
    limbs[i] = value & 0xFFFF;
    carry = value >> 16;
  }

  if (tid == 1023) {
    chunk_carry = static_cast<uint32_t>(s_direct_carry[tid] + carry);
  }
  __syncthreads();
}

__global__ void normalize_and_shift_right_kernel(uint64_t *out, uint64_t *in,
                                                 size_t shift, size_t n,
                                                 bool preserve_input) {
  __shared__ uint64_t s_normalized[4096];
  __shared__ uint64_t s_direct_carry[1024];
  __shared__ uint32_t s_warp_G[32];
  __shared__ uint32_t s_warp_P[32];
  __shared__ uint32_t chunk_carry;
  const int tid = threadIdx.x;

  if (tid == 0) {
    chunk_carry = 0;
  }
  __syncthreads();

  const int num_chunks = (n + 4095) / 4096;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int base = chunk * 4096;
    uint64_t limbs[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t source_idx = base + tid * 4 + i;
      limbs[i] = source_idx < n ? in[source_idx] : 0;
    }

    device_warp_normalize(limbs, s_direct_carry, s_warp_G, s_warp_P,
                          chunk_carry);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s_normalized[tid * 4 + i] = limbs[i];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int local_idx = tid + i * 1024;
      const size_t source_idx = base + local_idx;
      if (source_idx < n) {
        const uint64_t normalized = s_normalized[local_idx];
        if (preserve_input) {
          in[source_idx] = normalized;
        }
        if (source_idx >= shift) {
          out[source_idx - shift] = normalized;
        }
      }
    }
  }

  const size_t first_zero = n > shift ? n - shift : 0;
  for (size_t idx = first_zero + tid; idx < n; idx += blockDim.x) {
    out[idx] = 0;
  }
}

__global__ void sub_kernel_and_resolve(uint64_t *r, const uint64_t *t,
                                       const uint64_t *z2, size_t n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int64_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
      int64_t diff =
          static_cast<int64_t>(t[i]) - static_cast<int64_t>(z2[i]) - borrow;
      if (diff < 0) {
        diff += 65536;
        borrow = 1;
      } else {
        borrow = 0;
      }
      r[i] = diff;
    }
  }
}

__global__ void single_block_arbitrary_conditional_sub_p_kernel(
    uint64_t *r, const uint64_t *p, size_t d, size_t n, bool multiply_by_2) {
  __shared__ uint64_t s_r[4097];
  int tid = threadIdx.x;

  if (multiply_by_2) {
    int num_chunks = (n + 4095) / 4096;
    uint64_t carry_in = 0;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
      int base = chunk * 4096;
      for (int i = 0; i < 4; ++i) {
        int local_idx = tid + i * 1024;
        int global_idx = base + local_idx;
        if (global_idx < n) {
          s_r[local_idx] = r[global_idx];
        } else {
          s_r[local_idx] = 0;
        }
      }
      if (tid == 0) {
        s_r[4096] = carry_in;
      }
      __syncthreads();

      for (int i = 0; i < 4; ++i) {
        int local_idx = tid + i * 1024;
        int global_idx = base + local_idx;
        if (global_idx < n) {
          uint64_t val = s_r[local_idx];
          uint64_t prev = (local_idx > 0) ? s_r[local_idx - 1] : s_r[4096];
          r[global_idx] = ((val & 0x7FFF) << 1) | (prev >> 15);
        }
      }
      if (tid == 0) {
        carry_in = s_r[4095] >> 15;
      }
      __syncthreads();
    }
  }

  for (int iter = 0; iter < 2; iter++) {
    int max_diff_idx = -1;
    int max_diff_sign = 0;

    int num_chunks = (n + 4095) / 4096;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
      int base = chunk * 4096;
      int local_highest = -1;
      int local_sign = 0;

      for (int i = 3; i >= 0; --i) {
        int local_idx = tid + i * 1024;
        int global_idx = base + local_idx;
        if (global_idx < n) {
          uint64_t r_val = r[global_idx];
          uint64_t p_val = p[global_idx];
          if (r_val != p_val) {
            local_highest = global_idx;
            local_sign = (r_val > p_val) ? 1 : -1;
            break;
          }
        }
      }

      __shared__ int shared_warp_max[32];
      __shared__ int shared_max_sign;
      int lane = tid & 31;
      int warp = tid >> 5;
      int warp_max = local_highest;
      for (int offset = 16; offset > 0; offset /= 2) {
        warp_max =
            max(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
      }
      if (lane == 0) {
        shared_warp_max[warp] = warp_max;
      }
      __syncthreads();

      int block_max = -1;
      if (warp == 0) {
        block_max = shared_warp_max[lane];
        for (int offset = 16; offset > 0; offset /= 2) {
          block_max =
              max(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
          shared_warp_max[0] = block_max;
          shared_max_sign = 0;
        }
      }
      __syncthreads();

      int shared_max_idx = shared_warp_max[0];
      if (local_highest == shared_max_idx && local_highest != -1) {
        shared_max_sign = local_sign;
      }
      __syncthreads();

      if (shared_max_idx != -1) {
        max_diff_idx = shared_max_idx;
        max_diff_sign = shared_max_sign;
        break;
      }
    }

    bool geq = (max_diff_idx == -1) || (max_diff_sign == 1);
    if (!geq) {
      break;
    }

    int64_t borrow_in = 0;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
      int base = chunk * 4096;

      for (int i = 0; i < 4; ++i) {
        int local_idx = tid + i * 1024;
        int global_idx = base + local_idx;
        if (global_idx < n) {
          int64_t diff = static_cast<int64_t>(r[global_idx]) -
                         static_cast<int64_t>(p[global_idx]);
          s_r[local_idx] = static_cast<uint64_t>(diff);
        } else {
          s_r[local_idx] = 0;
        }
      }
      if (tid == 0) {
        s_r[4096] = 0;
        int64_t diff = static_cast<int64_t>(s_r[0]) - borrow_in;
        s_r[0] = static_cast<uint64_t>(diff);
      }
      __syncthreads();

      int changed = 1;
      while (changed) {
        changed = 0;
        for (int i = 0; i < 4; ++i) {
          int local_idx = tid + i * 1024;
          if (base + local_idx >= n) {
            continue;
          }

          auto val = static_cast<int64_t>(s_r[local_idx]);
          if (val < 0) {
            changed = 1;
            atomicAdd(reinterpret_cast<unsigned long long *>(&s_r[local_idx]),
                      65536ULL);
            atomicAdd(
                reinterpret_cast<unsigned long long *>(&s_r[local_idx + 1]),
                -1ULL);
          } else if (val >= 65536) {
            changed = 1;
            uint64_t carry = val >> 16;
            atomicAdd(reinterpret_cast<unsigned long long *>(&s_r[local_idx]),
                      -static_cast<unsigned long long>(carry << 16));
            atomicAdd(
                reinterpret_cast<unsigned long long *>(&s_r[local_idx + 1]),
                carry);
          }
        }
        changed = __syncthreads_or(changed);
      }

      borrow_in = -static_cast<int64_t>(s_r[4096]);
      __syncthreads();

      for (int i = 0; i < 4; ++i) {
        int local_idx = tid + i * 1024;
        int global_idx = base + local_idx;
        if (global_idx < n) {
          r[global_idx] = s_r[local_idx];
        }
      }
      __syncthreads();
    }
  }
}
