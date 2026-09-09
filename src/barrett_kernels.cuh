#pragma once
#include <cstdint>
#include <utility>

__global__ void shift_right_kernel(uint64_t *out, const uint64_t *in,
                                   size_t shift, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (idx + shift < n) {
      out[idx] = in[idx + shift];
    } else {
      out[idx] = 0;
    }
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

      __shared__ int shared_max_idx;
      __shared__ int shared_max_sign;
      if (tid == 0) {
        shared_max_idx = -1;
        shared_max_sign = 0;
      }
      __syncthreads();

      if (local_highest != -1) {
        atomicMax(&shared_max_idx, local_highest);
      }
      __syncthreads();

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
