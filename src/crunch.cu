#include <atomic>
#include <iomanip>
// test-fermat.cu
#include <cuda_runtime.h>
#include <gmp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cassert>
#include <chrono>

inline auto get_program_start() {
  static const auto start = std::chrono::steady_clock::now();
  return start;
}


struct ThreadState {
  uint64_t q_val = 0;
  int pct = 0;
  std::chrono::steady_clock::time_point cand_start;
  bool is_running = false;
};

inline std::vector<ThreadState>& get_global_thread_states() {
  static std::vector<ThreadState> states(4);
  return states;
}

inline std::mutex& get_state_mutex() {
  static std::mutex state_mutex;
  return state_mutex;
}

inline std::mutex &get_print_mutex() {
  static std::mutex print_mutex;
  return print_mutex;
}

#include <cstdint>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Include everything from test-ntt to reuse kernels and precomputation structs
#include "barrett_kernels.cuh"
#include "ntt.cu"

// -------------------------------------------------------------------------
// Constants and Primitives
// -------------------------------------------------------------------------
// constexpr size_t N = 4096;
constexpr int MODULUS_BITS = 64;

__global__ void mul_2_kernel(const uint64_t *in, uint64_t *out, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    uint64_t val = in[idx];
    uint64_t carry_in = (idx == 0) ? 0 : (in[idx - 1] >> 15);
    out[idx] = ((val & 0x7FFF) << 1) | carry_in;
  }
}

__global__ void pointwise_multiply_scaled(uint64_t *out, const uint64_t *a,
                                          const uint64_t *b, uint64_t inv_n,
                                          uint64_t modulus_val, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    unsigned __int128 p1 = static_cast<unsigned __int128>(a[idx]) * b[idx];
    out[idx] = p1 % modulus_val;
  }
}

__global__ void bit_reverse_permute(uint64_t *data, size_t n, int shift) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    uint32_t rev = __brev(idx) >> shift;
    if (idx < rev) {
      uint64_t tmp = data[idx];
      data[idx] = data[rev];
      data[rev] = tmp;
    }
  }
}

__global__ void reverse_and_scale(uint64_t *data, uint64_t inv_n,
                                  uint64_t modulus_val, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0 || idx == n / 2) {
    unsigned __int128 p = static_cast<unsigned __int128>(data[idx]) * inv_n;
    data[idx] = p % modulus_val;
  } else if (idx < n / 2) {
    size_t opp = n - idx;
    uint64_t val_idx = data[idx];
    uint64_t val_opp = data[opp];

    unsigned __int128 p1 = static_cast<unsigned __int128>(val_opp) * inv_n;
    unsigned __int128 p2 = static_cast<unsigned __int128>(val_idx) * inv_n;

    data[idx] = p1 % modulus_val;
    data[opp] = p2 % modulus_val;
  }
}

__global__ void single_block_arbitrary_resolve_carries(uint64_t *data,
                                                       size_t n) {
  __shared__ uint64_t smem[4097];
  int tid = threadIdx.x;

  uint64_t carry_in = 0;

  int num_chunks = (n + 4095) / 4096;
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    int base = chunk * 4096;

    for (int i = 0; i < 4; ++i) {
      int local_idx = tid + i * 1024;
      int global_idx = base + local_idx;
      if (global_idx < n) {
        smem[local_idx] = data[global_idx];
      } else {
        smem[local_idx] = 0;
      }
    }
    if (tid == 0) {
      smem[4096] = 0;
      smem[0] += carry_in;
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

        uint64_t val = smem[local_idx];
        uint64_t c = val >> 16;
        if (c > 0) {
          changed = 1;
          atomicAdd(reinterpret_cast<unsigned long long *>(&smem[local_idx]),
                    -static_cast<unsigned long long>(c << 16));
          atomicAdd(
              reinterpret_cast<unsigned long long *>(&smem[local_idx + 1]),
              static_cast<unsigned long long>(c));
        }
      }
      changed = __syncthreads_or(changed);
    }

    carry_in = smem[4096];
    __syncthreads();

    for (int i = 0; i < 4; ++i) {
      int local_idx = tid + i * 1024;
      int global_idx = base + local_idx;
      if (global_idx < n) {
        data[global_idx] = smem[local_idx];
      }
    }
    __syncthreads();
  }
}

void launch_resolve_carries(uint64_t *d_data, size_t n, cudaStream_t stream) {
  single_block_arbitrary_resolve_carries<<<1, 1024, 0, stream>>>(d_data, n);
}

__global__ void single_block_arbitrary_sub_kernel(uint64_t *r,
                                                  const uint64_t *t,
                                                  const uint64_t *z2,
                                                  size_t n) {
  __shared__ uint64_t s_r[4097];
  int tid = threadIdx.x;

  int64_t borrow_in = 0;

  int num_chunks = (n + 4095) / 4096;
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    int base = chunk * 4096;

    for (int i = 0; i < 4; ++i) {
      int local_idx = tid + i * 1024;
      int global_idx = base + local_idx;
      if (global_idx < n) {
        int64_t diff = static_cast<int64_t>(t[global_idx]) -
                       static_cast<int64_t>(z2[global_idx]);
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
          atomicAdd(reinterpret_cast<unsigned long long *>(&s_r[local_idx + 1]),
                    -1ULL);
        } else if (val >= 65536) {
          changed = 1;
          uint64_t carry = val >> 16;
          atomicAdd(reinterpret_cast<unsigned long long *>(&s_r[local_idx]),
                    -static_cast<unsigned long long>(carry << 16));
          atomicAdd(reinterpret_cast<unsigned long long *>(&s_r[local_idx + 1]),
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

void launch_sub_kernel(uint64_t *r, const uint64_t *t, const uint64_t *z2,
                       size_t n, cudaStream_t stream) {
  single_block_arbitrary_sub_kernel<<<1, 1024, 0, stream>>>(r, t, z2, n);
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=4096 or 65536)
// -------------------------------------------------------------------------
void launch_forward_ntt_fermat(
    size_t N_val, uint64_t *d_data,
    const precomputation::Precomputation<MODULUS_BITS> *precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>
        &constant_precomp,
    cudaStream_t stream) {
  const int smem_per_warp = 4 * 8 * 16 * 16;

  if (N_val == 4096) {
    constexpr int N = 4096;
    constexpr int n = 1 << 12; // 4096
    const dim3 block_dim(32 * 1, 1 * 1);
    const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
    run_forward_iterative_wmma_radix16<N, n>
        <<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32),
           stream>>>(d_data, precomp, constant_precomp);

    constexpr int n2 = 1 << 8; // 256
    const dim3 block_dim2(32, 32 / 32);
    const dim3 grid_dim2(N / 16 / 16 / block_dim2.y);
    run_forward_iterative_wmma_radix256<N, n2>
        <<<grid_dim2, block_dim2, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
  } else if (N_val == 65536) {
    constexpr int N = 65536;
    constexpr int n = 1 << 16;
    const dim3 block_dim(32 * 2, 1 * 1);
    const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
    run_forward_iterative_wmma_radix16<N, n>
        <<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32),
           stream>>>(d_data, precomp, constant_precomp);

    constexpr int n2 = 1 << 12;
    const dim3 block_dim2(32 * 1, 1 * 4);
    const dim3 grid_dim2(n2 / 16 / (block_dim2.x / 32 * 16),
                         N / n2 / block_dim2.y);
    run_forward_iterative_wmma_radix16<N, n2>
        <<<grid_dim2, block_dim2,
           smem_per_warp *(block_dim2.x * block_dim2.y / 32), stream>>>(
            d_data, precomp, constant_precomp);

    constexpr int n3 = 1 << 8;
    const dim3 block_dim3(32, 32 / 32);
    const dim3 grid_dim3(N / 16 / 16 / block_dim3.y);
    run_forward_iterative_wmma_radix256<N, n3>
        <<<grid_dim3, block_dim3, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
  }
}

__global__ void
pointwise_multiply_reverse_scale_kernel(uint64_t *out, const uint64_t *a,
                                        const uint64_t *b, uint64_t inv_n,
                                        uint64_t modulus_val, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0 || idx == n / 2) {
    unsigned __int128 p = static_cast<unsigned __int128>(a[idx]) * b[idx];
    p = (p % modulus_val) * inv_n;
    out[idx] = p % modulus_val;
  } else if (idx < n / 2) {
    size_t opp = n - idx;
    unsigned __int128 p_idx = static_cast<unsigned __int128>(a[idx]) * b[idx];
    unsigned __int128 p_opp = static_cast<unsigned __int128>(a[opp]) * b[opp];

    p_idx = (p_idx % modulus_val) * inv_n;
    p_opp = (p_opp % modulus_val) * inv_n;

    out[opp] = p_idx % modulus_val;
    out[idx] = p_opp % modulus_val;
  }
}

void launch_inverse_ntt_fermat_fused(
    size_t N_val, uint64_t *d_data,
    const precomputation::Precomputation<MODULUS_BITS> *precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>
        &constant_precomp,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (N_val + threads - 1) / threads;
  int shift = (N_val == 65536) ? 16 : 20;

  bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);
  launch_forward_ntt_fermat(N_val, d_data, precomp, constant_precomp, stream);
  bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);
}

void launch_inverse_ntt_fermat(
    size_t N_val, uint64_t *d_data, uint64_t inv_n, uint64_t modulus,
    const precomputation::Precomputation<MODULUS_BITS> *precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>
        &constant_precomp,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (N_val + threads - 1) / threads;

  int shift = (N_val == 65536) ? 16 : 20;

  bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);
  launch_forward_ntt_fermat(N_val, d_data, precomp, constant_precomp, stream);
  bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);

  reverse_and_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus,
                                                    N_val);
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=65536)
// -------------------------------------------------------------------------
auto read_prime(const char *file, size_t target_bits, size_t tolerance,
                mpz_t out) -> bool {
  std::ifstream f(file);
  if (!f.is_open()) {
    return false;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    mpz_t p;
    mpz_init_set_str(p, line.c_str(), 10);
    size_t bits = mpz_sizeinbase(p, 2);
    if (bits >= target_bits && bits <= target_bits + tolerance) {
      mpz_set(out, p);
      mpz_clear(p);
      return true;
    }
    mpz_clear(p);
  }
  return false;
}

// // Convert GMP to base 2^16 limbs
void gmp_to_limbs(mpz_t x, std::vector<uint64_t> &limbs, size_t N_val) {
  limbs.assign(N_val, 0);
  size_t count;
  std::vector<uint16_t> temp(N_val, 0);
  mpz_export(temp.data(), &count, -1, sizeof(uint16_t), 0, 0, x);
  for (size_t i = 0; i < count; ++i) {
    limbs[i] = temp[i];
  }
}

// Convert base 2^16 limbs to GMP
void limbs_to_gmp(const std::vector<uint64_t> &limbs, mpz_t x, size_t N_val) {
  std::vector<uint16_t> short_limbs(N_val);
  for (size_t i = 0; i < N_val; i++) {
    short_limbs[i] = static_cast<uint16_t>(limbs[i]);
  }
  mpz_import(x, N_val, -1, sizeof(uint16_t), 0, 0, short_limbs.data());
}

struct StreamContext {
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEvent_t ntt_start[9], ntt_end[9];
  cudaEvent_t pw_start[3], pw_end[3];
  cudaEvent_t carry_start[4], carry_end[4];

  thrust::device_vector<uint64_t> d_P;
  thrust::device_vector<uint64_t> d_P_freq;
  thrust::device_vector<uint64_t> d_mu_freq;
  thrust::device_vector<uint64_t> d_X;
  thrust::device_vector<uint64_t> d_T;
  thrust::device_vector<uint64_t> d_Q;
  thrust::device_vector<uint64_t> d_Z1;
  thrust::device_vector<uint64_t> d_Z2;

  bool initialized;

  StreamContext() : initialized(false) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 9; i++) {
      cudaEventCreate(&ntt_start[i]);
      cudaEventCreate(&ntt_end[i]);
    }
    for (int i = 0; i < 3; i++) {
      cudaEventCreate(&pw_start[i]);
      cudaEventCreate(&pw_end[i]);
    }
    for (int i = 0; i < 4; i++) {
      cudaEventCreate(&carry_start[i]);
      cudaEventCreate(&carry_end[i]);
    }
  }

  void init(size_t N_val) {
    if (!initialized || d_P.size() != N_val) {
      d_P.resize(N_val);
      d_P_freq.resize(N_val);
      d_mu_freq.resize(N_val);
      d_X.resize(N_val);
      d_T.resize(N_val);
      d_Q.resize(N_val);
      d_Z1.resize(N_val);
      d_Z2.resize(N_val);
      initialized = true;
    }
  }

  ~StreamContext() {
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

auto run_fermat_pipeline(
    mpz_t p, size_t bit_len, size_t d, size_t N_val, uint64_t inv_n,
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<MODULUS_BITS> *precomp_device_ptr,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>
        &constant_precomp,
    StreamContext &ctx,
    const std::vector<uint64_t> &h_P_in = std::vector<uint64_t>(),
    const std::vector<uint64_t> &h_mu_in = std::vector<uint64_t>(),
    int thread_id = 0, uint64_t q_val = 0) -> bool {
  (void)bit_len; std::vector<uint64_t> h_P = h_P_in;
  std::vector<uint64_t> h_mu = h_mu_in;

  mpz_t B_2d, mu;
  mpz_init(B_2d);
  mpz_init(mu);

  if (h_P.empty() || h_mu.empty()) {
    mpz_ui_pow_ui(B_2d, 2, 16 * 2 * d);
    mpz_fdiv_q(mu, B_2d, p);
    h_P.assign(N_val, 0);
    h_mu.assign(N_val, 0);
    gmp_to_limbs(p, h_P, N_val);
    gmp_to_limbs(mu, h_mu, N_val);
  }

  std::vector<uint64_t> h_x(N_val, 0);

  mpz_t x;
  mpz_init_set_ui(x, 2); // Base 2
  gmp_to_limbs(x, h_x, N_val);
  mpz_clear(x);

  ctx.init(N_val);
  ctx.d_P = h_P;
  ctx.d_P_freq = h_P;
  ctx.d_mu_freq = h_mu;
  ctx.d_X = h_x;

  thrust::device_vector<uint64_t> &d_P = ctx.d_P;
  thrust::device_vector<uint64_t> &d_P_freq = ctx.d_P_freq;
  thrust::device_vector<uint64_t> &d_mu_freq = ctx.d_mu_freq;
  thrust::device_vector<uint64_t> &d_X = ctx.d_X;
  thrust::device_vector<uint64_t> &d_T = ctx.d_T;
  thrust::device_vector<uint64_t> &d_Q = ctx.d_Q;
  thrust::device_vector<uint64_t> &d_Z1 = ctx.d_Z1;
  thrust::device_vector<uint64_t> &d_Z2 = ctx.d_Z2;

  cudaStream_t stream = ctx.stream;
  cudaGraph_t &graph = ctx.graph;
  cudaGraphExec_t &instance = ctx.instance;
  cudaEvent_t start = ctx.start;
  cudaEvent_t stop = ctx.stop;

  int threads = 256;
  int blocks = (N_val + threads - 1) / threads;

  launch_forward_ntt_fermat(N_val, thrust::raw_pointer_cast(d_P_freq.data()),
                            precomp_device_ptr, constant_precomp, stream);
  launch_forward_ntt_fermat(N_val, thrust::raw_pointer_cast(d_mu_freq.data()),
                            precomp_device_ptr, constant_precomp, stream);

  mpz_t p_minus_1;
  mpz_init(p_minus_1);
  mpz_sub_ui(p_minus_1, p, 1);
  size_t actual_bit_len = mpz_sizeinbase(p_minus_1, 2);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);

  uint64_t *raw_X = thrust::raw_pointer_cast(d_X.data());
  uint64_t *raw_T = thrust::raw_pointer_cast(d_T.data());
  uint64_t *raw_Z1 = thrust::raw_pointer_cast(d_Z1.data());
  uint64_t *raw_Q = thrust::raw_pointer_cast(d_Q.data());
  uint64_t *raw_Z2 = thrust::raw_pointer_cast(d_Z2.data());
  uint64_t *raw_mu_freq = thrust::raw_pointer_cast(d_mu_freq.data());
  uint64_t *raw_P_freq = thrust::raw_pointer_cast(d_P_freq.data());
  uint64_t *raw_P = thrust::raw_pointer_cast(d_P.data());
  uint64_t mod_val = modulus.get_modulus();

  // Step 1: T = X^2
  cudaMemcpyAsync(raw_T, raw_X, N_val * sizeof(uint64_t),
                  cudaMemcpyDeviceToDevice, stream);
  cudaEventRecord(ctx.ntt_start[0], stream);
  launch_forward_ntt_fermat(N_val, raw_T, precomp_device_ptr, constant_precomp,
                            stream);
  cudaEventRecord(ctx.ntt_end[0], stream);
  cudaEventRecord(ctx.pw_start[0], stream);
  pointwise_multiply_reverse_scale_kernel<<<blocks, threads, 0, stream>>>(
      raw_T, raw_T, raw_T, inv_n, mod_val, N_val);
  cudaEventRecord(ctx.pw_end[0], stream);
  cudaEventRecord(ctx.ntt_start[1], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_T, precomp_device_ptr,
                                  constant_precomp, stream);
  cudaEventRecord(ctx.ntt_end[1], stream);
  cudaEventRecord(ctx.carry_start[0], stream);
  launch_resolve_carries(raw_T, N_val, stream);
  cudaEventRecord(ctx.carry_end[0], stream);

  // Step 2: Q = floor(T / B^(d-1)) * mu
  shift_right_kernel<<<blocks, threads, 0, stream>>>(raw_Z1, raw_T, d - 1,
                                                     N_val);
  cudaEventRecord(ctx.ntt_start[2], stream);
  launch_forward_ntt_fermat(N_val, raw_Z1, precomp_device_ptr, constant_precomp,
                            stream);
  cudaEventRecord(ctx.ntt_end[2], stream);
  cudaEventRecord(ctx.pw_start[1], stream);
  pointwise_multiply_reverse_scale_kernel<<<blocks, threads, 0, stream>>>(
      raw_Z1, raw_Z1, raw_mu_freq, inv_n, mod_val, N_val);
  cudaEventRecord(ctx.pw_end[1], stream);
  cudaEventRecord(ctx.ntt_start[3], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_Z1, precomp_device_ptr,
                                  constant_precomp, stream);
  cudaEventRecord(ctx.ntt_end[3], stream);
  cudaEventRecord(ctx.carry_start[1], stream);
  launch_resolve_carries(raw_Z1, N_val, stream);
  cudaEventRecord(ctx.carry_end[1], stream);

  // Step 3: Q = Q / B^(d+1)
  shift_right_kernel<<<blocks, threads, 0, stream>>>(raw_Q, raw_Z1, d + 1,
                                                     N_val);

  // Step 4: Z2 = Q * P
  cudaMemcpyAsync(raw_Z2, raw_Q, N_val * sizeof(uint64_t),
                  cudaMemcpyDeviceToDevice, stream);
  cudaEventRecord(ctx.ntt_start[4], stream);
  launch_forward_ntt_fermat(N_val, raw_Z2, precomp_device_ptr, constant_precomp,
                            stream);
  cudaEventRecord(ctx.ntt_end[4], stream);
  cudaEventRecord(ctx.pw_start[2], stream);
  pointwise_multiply_reverse_scale_kernel<<<blocks, threads, 0, stream>>>(
      raw_Z2, raw_Z2, raw_P_freq, inv_n, mod_val, N_val);
  cudaEventRecord(ctx.pw_end[2], stream);
  cudaEventRecord(ctx.ntt_start[5], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_Z2, precomp_device_ptr,
                                  constant_precomp, stream);
  cudaEventRecord(ctx.ntt_end[5], stream);
  cudaEventRecord(ctx.carry_start[2], stream);
  launch_resolve_carries(raw_Z2, N_val, stream);
  cudaEventRecord(ctx.carry_end[2], stream);

  // Step 5: R = T - Z2
  cudaEventRecord(ctx.carry_start[3], stream);
  launch_sub_kernel(raw_X, raw_T, raw_Z2, N_val, stream);
  single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(
      raw_X, raw_P, d, N_val, false);
  cudaEventRecord(ctx.carry_end[3], stream);

  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaEventRecord(start, stream);
  (void)bit_len; (void)q_val;

  int squarings = 0;
  int multiplies = 0;

  int total_steps = actual_bit_len - 1;
  int twenty_percent = total_steps / 5;
  if (twenty_percent == 0) {
    twenty_percent = 1;
  }

  for (int i = actual_bit_len - 2; i >= 0; --i) {
    cudaGraphLaunch(instance, stream);
    squarings++;

    if (mpz_tstbit(p_minus_1, i)) {
      single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(
          raw_X, raw_P, d, N_val, true);
      multiplies++;
    }

    int one_percent = std::max<int>(1, total_steps / 100);
    if (squarings % one_percent == 0 || squarings == total_steps) {
      int pct = (squarings * 100) / total_steps;

      if (pct < 100) {
        std::lock_guard<std::mutex> lock(get_state_mutex());
        auto& states = get_global_thread_states();
        if ((size_t)thread_id < states.size()) {
          states[thread_id].pct = pct;
        }

        auto now = std::chrono::steady_clock::now();
        auto uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(now - get_program_start()).count();
        int u_h = uptime_sec / 3600;
        int u_m = (uptime_sec % 3600) / 60;
        int u_s = uptime_sec % 60;

        char uptime_buf[64];
        snprintf(uptime_buf, sizeof(uptime_buf), "\r[%02d:%02d:%02d]", u_h, u_m, u_s);
        std::cout << uptime_buf;

        for (size_t t = 0; t < states.size(); t++) {
          if (states[t].is_running) {
            auto cand_sec = std::chrono::duration_cast<std::chrono::seconds>(now - states[t].cand_start).count();
            int c_m = cand_sec / 60;
            int c_s = cand_sec % 60;
            char cand_buf[64];
            snprintf(cand_buf, sizeof(cand_buf), " [%lu %02d:%02d %02d%%]", states[t].q_val, c_m, c_s, states[t].pct);
            std::cout << cand_buf;
          }
        }
        std::cout << "          " << std::flush;
      }
    }
  }

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  thrust::host_vector<uint64_t> final_limbs = d_X;
  mpz_t final_val;
  mpz_init(final_val);
  limbs_to_gmp(std::vector<uint64_t>(final_limbs.begin(), final_limbs.end()),
               final_val, N_val);

  float total_sec = ms / 1000.0f;
  float avg_us = ms * 1000.0f / squarings;
  bool is_prime = false;
  std::string result_msg;
  if (mpz_cmp_ui(final_val, 1) == 0) {
    is_prime = true;
    result_msg = "*** FOUND PROBABLE PRIME! ***";
  } else {
    result_msg = "x != 1";
  }
  {
    std::lock_guard<std::mutex> lock(get_state_mutex());
    auto& states = get_global_thread_states();
    if ((size_t)thread_id < states.size()) {
      states[thread_id].is_running = false;
    }
    std::cout << "\r                                                                                                                                  \r";
    auto now = std::chrono::steady_clock::now();
    auto uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(now - get_program_start()).count();
    auto cand_sec = std::chrono::duration_cast<std::chrono::seconds>(now - states[thread_id].cand_start).count();
    int u_h = uptime_sec / 3600; int u_m = (uptime_sec % 3600) / 60; int u_s = uptime_sec % 60;
    int c_m = cand_sec / 60; int c_s = cand_sec % 60;
    char time_buf[64];
    snprintf(time_buf, sizeof(time_buf), "[%02d:%02d:%02d %02d:%02d] ", u_h, u_m, u_s, c_m, c_s);
    std::cout << time_buf << "Q " << q_val << " (" << mpz_sizeinbase(p, 2) << " bits) | SQ " << total_steps << " | 100% | " << result_msg << "\n";
  }
  mpz_clear(final_val);
  mpz_clear(p_minus_1);
  mpz_clear(mu);
  mpz_clear(B_2d);

  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  return is_prime;
}

struct Candidate {
  mpz_t p;
  size_t bit_len;
  size_t d;
  size_t N_val;
  uint64_t q_val;
  std::vector<uint64_t> h_P;
  std::vector<uint64_t> h_mu;

  Candidate() { mpz_init(p); }

  Candidate(const Candidate &other) {
    mpz_init_set(p, other.p);
    bit_len = other.bit_len;
    d = other.d;
    N_val = other.N_val;
    q_val = other.q_val;
    h_P = other.h_P;
    h_mu = other.h_mu;
  }

  auto operator=(const Candidate &other) -> Candidate & {
    if (this != &other) {
      mpz_set(p, other.p);
      bit_len = other.bit_len;
      d = other.d;
      N_val = other.N_val;
      q_val = other.q_val;
      h_P = other.h_P;
      h_mu = other.h_mu;
    }
    return *this;
  }

  ~Candidate() { mpz_clear(p); }
};


inline std::atomic<size_t>& get_candidate_index() {
  static std::atomic<size_t> idx{0};
  return idx;
}

auto prepare_next_candidate(mpz_t K, const std::vector<uint64_t> &sieve_primes) -> Candidate {
  Candidate c;
  size_t idx = get_candidate_index().fetch_add(1);
  if (idx >= sieve_primes.size()) {
    c.q_val = 0; // use 0 as a sentinel to mean we are done
    return c;
  }
  c.q_val = sieve_primes[idx];

  mpz_mul_ui(c.p, K, c.q_val);
  mpz_add_ui(c.p, c.p, 1);

  c.bit_len = mpz_sizeinbase(c.p, 2);
  c.d = (c.bit_len + 15) / 16;
  if (c.bit_len <= 32000) {
    c.N_val = 4096;
  } else {
    c.N_val = 65536;
  }

  mpz_t B_2d, mu;
  mpz_init(B_2d);
  mpz_init(mu);
  mpz_ui_pow_ui(B_2d, 2, 16 * 2 * c.d);
  mpz_fdiv_q(mu, B_2d, c.p);

  c.h_P.assign(c.N_val, 0);
  c.h_mu.assign(c.N_val, 0);
  gmp_to_limbs(c.p, c.h_P, c.N_val);
  gmp_to_limbs(mu, c.h_mu, c.N_val);

  mpz_clear(B_2d);
  mpz_clear(mu);
  return c;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
auto main(int argc, char **argv) -> int {
  get_program_start();
  std::cout << "Starting Fermat Harness" << '\n';
  // Set up NTT field
  const polyarith::Modulus modulus(UINT64_C(0x1fff'fff9'0000'0001), 3);

  // Allocate Precomputations on heap to avoid stack overflow
  auto precomp_device = thrust::uninitialized_allocate_unique<
      precomputation::Precomputation<MODULUS_BITS>>(
      thrust::device_allocator<precomputation::Precomputation<MODULUS_BITS>>());
  auto precomp_host =
      std::make_unique<precomputation::Precomputation<MODULUS_BITS>>(modulus);
  cudaMemcpy(thrust::raw_pointer_cast(precomp_device.get()), precomp_host.get(),
             sizeof(precomputation::Precomputation<MODULUS_BITS>),
             cudaMemcpyHostToDevice);
  const precomputation::ConstantPrecomputation<MODULUS_BITS> constant_precomp(
      modulus);

  mpz_t p;
  mpz_init(p);

  // Parse arguments
  std::string filename = "primes.txt";
  int phase = 2;
  size_t target_bits = 258000;
  int target_index = -1;

  std::string primes_file_name = "primes.txt";
  std::string sieve_file_name = "";
  bool is_crunch = false;

  // Default to crunch mode if no arguments are provided
  if (argc == 1) {
    is_crunch = true;
    primes_file_name = "primes.txt";
    sieve_file_name = "sieve_1e10.txt";
  }

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--file" && i + 1 < argc) {
      filename = argv[++i];
    }
    if (arg == "--phase" && i + 1 < argc) {
      phase = std::stoi(argv[++i]);
    }
    if (arg == "--target-bits" && i + 1 < argc) {
      target_bits = std::stoull(argv[++i]);
    }
    if (arg == "--index" && i + 1 < argc) {
      target_index = std::stoi(argv[++i]);
    }
    if (arg == "--primes" && i + 1 < argc) {
      primes_file_name = argv[++i];
      is_crunch = true;
    }
    if (arg == "--sieve" && i + 1 < argc) {
      sieve_file_name = argv[++i];
      is_crunch = true;
    }
  }

  if (is_crunch) {
    std::cout << "--- Crunch Mode ---" << '\n';
    std::vector<std::string> p_strs;
    std::ifstream pf(primes_file_name);
    std::string line;
    while (std::getline(pf, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      p_strs.push_back(line);
    }
    // sort by length
    std::sort(p_strs.begin(), p_strs.end(),
              [](const std::string &a, const std::string &b) -> bool {
                return a.length() < b.length();
              });

    mpz_t P_A, P_B, P_C, K;
    mpz_init_set_str(P_A, p_strs[p_strs.size() - 1].c_str(), 10);
    mpz_init_set_str(P_B, p_strs[p_strs.size() - 2].c_str(), 10);
    mpz_init_set_str(P_C, p_strs[p_strs.size() - 3].c_str(), 10);

    mpz_init(K);
    mpz_mul(K, P_A, P_B);
    mpz_mul(K, K, P_C);
    mpz_mul_ui(K, K, 2);

    std::vector<uint64_t> sieve_primes;
    std::ifstream sf(sieve_file_name);
    while (std::getline(sf, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      sieve_primes.push_back(std::stoull(line));
    }
    std::cout << "Loaded " << sieve_primes.size() << " sieve primes." << '\n';
    
    std::random_device rd;
    std::mt19937_64 shuffle_rng(rd());
    std::shuffle(sieve_primes.begin(), sieve_primes.end(), shuffle_rng);

    constexpr int BATCH_SIZE = 4;
    std::vector<StreamContext> stream_ctxs(BATCH_SIZE);

    std::mutex file_mutex;

    auto worker_thread = [&](int thread_id) {
      while (true) {
        Candidate cand = prepare_next_candidate(K, sieve_primes);
        if (cand.q_val == 0) {
            break; // No more candidates!
        }

        {
          std::lock_guard<std::mutex> lock(get_state_mutex());
          auto& states = get_global_thread_states();
          states[thread_id].q_val = cand.q_val;
          states[thread_id].pct = 0;
          states[thread_id].cand_start = std::chrono::steady_clock::now();
          states[thread_id].is_running = true;
        }

        uint64_t inv_n = modulus.invert(cand.N_val);
        bool passed = run_fermat_pipeline(
            cand.p, cand.bit_len, cand.d, cand.N_val, inv_n, modulus,
            thrust::raw_pointer_cast(precomp_device.get()), constant_precomp,
            stream_ctxs[thread_id], cand.h_P, cand.h_mu, thread_id, cand.q_val);

        if (passed) {
          std::lock_guard<std::mutex> flock(file_mutex);
          std::ofstream outf("found_primes.txt", std::ios::app);
          outf << "q=" << cand.q_val << " P=";
          char *p_str = mpz_get_str(nullptr, 10, cand.p);
          outf << p_str << '\n';
          free(p_str);
        }
      }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < BATCH_SIZE; ++i) {
      threads.emplace_back(worker_thread, i);
    }

    for (auto &t : threads) {
      t.join();
    }

    mpz_clear(P_A);
    mpz_clear(P_B);
    mpz_clear(P_C);
    mpz_clear(K);
    return 0;
  }

  if (phase == 2) {
    std::cout << "--- Phase 2: Full Fermat Primality ---" << '\n';
    std::ifstream primes_file(filename);
    if (!primes_file.is_open()) {
      std::cout << "Could not open " << filename << '\n';
      return 1;
    }

    std::string line;
    std::vector<std::string> candidates;
    while (std::getline(primes_file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      candidates.push_back(line);
    }

    int selected_idx = -1;
    if (target_index != -1) {
      selected_idx = target_index;
    } else {
      for (size_t i = 0; i < candidates.size(); i++) {
        mpz_init_set_str(p, candidates[i].c_str(), 10);
        size_t bit_len = mpz_sizeinbase(p, 2);
        mpz_clear(p);
        // Allow a small tolerance for target_bits match or take exact
        if (bit_len >= target_bits * 0.95 && bit_len <= target_bits * 1.05) {
          selected_idx = i;
          break;
        }
      }
    }

    if (selected_idx < 0 || (size_t)selected_idx >= candidates.size()) {
      std::cout << "Candidate not found!" << '\n';
      return 1;
    }

    line = candidates[selected_idx];
    mpz_init_set_str(p, line.c_str(), 10);

    size_t bit_len = mpz_sizeinbase(p, 2);
    size_t exact_digits = line.length();
    size_t d = (bit_len + 15) / 16;

    size_t N_val;
    if (bit_len <= 32000) {
      N_val = 4096;
    } else {
      N_val = 65536;
    }

    if (2 * d + 2 > N_val) {
      std::cout << "ERROR: N_val " << N_val
                << " is not sufficient for 2d+2=" << (2 * d + 2)
                << " limbs to prevent aliasing." << '\n';
      return 1;
    }

    uint64_t inv_n = modulus.invert(N_val);
    std::cout << "Testing prime index " << selected_idx << ":" << '\n';
    std::cout << "exact decimal digit count: " << exact_digits << '\n';
    std::cout << "bit count: " << bit_len << '\n';
    std::cout << "d=" << d
              << " limbs, max convolution length 2d+2=" << (2 * d + 2) << '\n';
    std::cout << "Chosen N_val: " << N_val
              << " (sufficient zero-padding guaranteed)" << '\n';

    StreamContext ctx;
    std::vector<uint64_t> empty_vec;
    run_fermat_pipeline(p, bit_len, d, N_val, inv_n, modulus,
                        thrust::raw_pointer_cast(precomp_device.get()),
                        constant_precomp, ctx, empty_vec, empty_vec, 0, 0);
  }
}
