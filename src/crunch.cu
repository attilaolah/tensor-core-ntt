#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
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

inline std::mutex &get_print_mutex() {
  static std::mutex print_mutex;
  return print_mutex;
}

class ProgressLogSchedule {
 public:
  ProgressLogSchedule() : start_(std::chrono::steady_clock::now()) {}

  auto claim_nonfinal_slot() -> bool {
    constexpr auto tick_duration = std::chrono::milliseconds(500);
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    const auto tick = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() /
        tick_duration.count());
    return claim_tick(tick);
  }

  // Any worker may claim the current tick.  The compare-exchange preserves
  // one human-status emitter per tick without making output depend on a
  // particular worker reaching a checkpoint at the right instant.
  auto claim_tick(uint64_t tick) -> bool {
    uint64_t previous_tick = last_emitted_tick_.load(std::memory_order_relaxed);
    while (previous_tick != tick) {
      if (last_emitted_tick_.compare_exchange_weak(
              previous_tick, tick, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
 }

 private:
  std::chrono::steady_clock::time_point start_;
  std::atomic<uint64_t> last_emitted_tick_{UINT64_MAX};
};

auto format_duration(std::chrono::seconds duration) -> std::string {
  const auto seconds = duration.count();
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%02lld:%02lld",
           static_cast<long long>(seconds / 60),
           static_cast<long long>(seconds % 60));
  return buffer;
}

auto format_uptime(std::chrono::seconds duration) -> std::string {
  const auto seconds = duration.count();
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%02lld:%02lld:%02lld",
           static_cast<long long>(seconds / 3600),
           static_cast<long long>((seconds / 60) % 60),
           static_cast<long long>(seconds % 60));
  return buffer;
}

void print_candidate_status(int thread_id, std::chrono::steady_clock::time_point
                                               candidate_start,
                            uint64_t q_val, size_t completed, size_t total,
                            const std::string &result, bool terminate_line) {
  const auto now = std::chrono::steady_clock::now();
  const auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
      now - get_program_start());
  const auto candidate_duration =
      std::chrono::duration_cast<std::chrono::seconds>(now - candidate_start);
  const double completion = total == 0
                                ? 0.0
                                : 100.0 * static_cast<double>(completed) /
                                      static_cast<double>(total);

  std::lock_guard<std::mutex> lock(get_print_mutex());
  std::cout << "[T" << std::setfill('0') << std::setw(2) << thread_id
            << std::setfill(' ') << " | " << format_uptime(uptime) << " | F "
            << format_duration(candidate_duration) << "] [C " << std::setw(5)
            << static_cast<unsigned long long>(completed) << " | N "
            << std::setw(5) << static_cast<unsigned long long>(total) << " | "
            << std::fixed << std::setprecision(2) << std::setw(6) << completion
            << "%] [Q " << std::setw(11) << q_val << "] " << result
            << (terminate_line ? '\n' : '\r')
            << std::defaultfloat << std::flush;
}

inline std::atomic<size_t> &get_completed_candidate_count() {
  static std::atomic<size_t> count{0};
  return count;
}

#include <deque>
#include <fstream>
#include <future>
#include <random>
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

__global__ void pointwise_multiply(uint64_t *out, const uint64_t *a,
                                   const uint64_t *b, uint64_t modulus_val,
                                   size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    unsigned __int128 p1 = static_cast<unsigned __int128>(a[idx]) * b[idx];
    out[idx] = p1 % modulus_val;
  }
}

// The forward tensor-core transform is a decimation-in-frequency transform,
// so its output is bit-reversed.  This is the matching decimation-in-time
// inverse: ascending stages consume that ordering and restore natural order.
// Stages are separate kernels because CUDA provides no grid-wide barrier.
__global__ void inverse_ntt_dit_stage(uint64_t *data,
                                      const uint64_t *inverse_powers,
                                      uint64_t modulus_val, size_t n,
                                      size_t half) {
  const size_t butterfly = blockIdx.x * blockDim.x + threadIdx.x;
  if (butterfly >= n / 2) {
    return;
  }
  const size_t j = butterfly % half;
  const size_t group = butterfly / half;
  const size_t first = group * (half * 2) + j;
  const size_t second = first + half;
  const size_t stride = n / (half * 2);
  const uint64_t twiddle = inverse_powers[j * stride];
  // Tensor-core forward stages may deliberately leave values unreduced.
  // DIT butterflies require canonical operands for their overflow-free add.
  const uint64_t x0 = data[first] % modulus_val;
  const uint64_t x1 = static_cast<unsigned __int128>(data[second]) * twiddle %
                      modulus_val;
  data[first] = x0 >= modulus_val - x1 ? x0 - (modulus_val - x1) : x0 + x1;
  data[second] = x0 >= x1 ? x0 - x1 : x0 + modulus_val - x1;
}

__global__ void inverse_ntt_scale(uint64_t *data, uint64_t inv_n,
                                  uint64_t modulus_val, size_t n) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = static_cast<unsigned __int128>(data[index]) * inv_n %
                  modulus_val;
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

__global__ void pointwise_multiply_kernel(uint64_t *out, const uint64_t *a,
                                          const uint64_t *b,
                                          uint64_t modulus_val, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<unsigned __int128>(a[idx]) * b[idx] % modulus_val;
  }
}

void launch_inverse_ntt_fermat_fused(
    size_t N_val, uint64_t *d_data,
    const uint64_t *inverse_powers, uint64_t inv_n, uint64_t modulus,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (N_val + threads - 1) / threads;
  for (size_t half = 1; half < N_val; half <<= 1) {
    inverse_ntt_dit_stage<<<blocks, threads, 0, stream>>>(
        d_data, inverse_powers, modulus, N_val, half);
  }
  inverse_ntt_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus,
                                                      N_val);
}

void launch_inverse_ntt_fermat(
    size_t N_val, uint64_t *d_data, uint64_t inv_n, uint64_t modulus,
    const uint64_t *inverse_powers,
    cudaStream_t stream) {
  launch_inverse_ntt_fermat_fused(N_val, d_data, inverse_powers, inv_n,
                                  modulus, stream);
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
  thrust::device_vector<uint64_t> d_inverse_powers;

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
      d_inverse_powers.resize(N_val / 2);
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
    int thread_id = 0, uint64_t q_val = 0, size_t candidate_total = 0,
    ProgressLogSchedule *progress_schedule = nullptr) -> bool {
  const auto candidate_start = std::chrono::steady_clock::now();
  (void)bit_len;
  std::vector<uint64_t> h_P = h_P_in;
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
  std::vector<uint64_t> h_inverse_powers(N_val / 2);
  uint64_t inverse_power = 1;
  const uint64_t inverse_root = modulus.get_root_inverse(N_val);
  for (size_t index = 0; index < h_inverse_powers.size(); ++index) {
    h_inverse_powers[index] = inverse_power;
    inverse_power = modulus.multiply(inverse_power, inverse_root);
  }
  ctx.d_inverse_powers = h_inverse_powers;
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
  const uint64_t *raw_inverse_powers =
      thrust::raw_pointer_cast(ctx.d_inverse_powers.data());
  uint64_t mod_val = modulus.get_modulus();

  // Step 1: T = X^2
  cudaMemcpyAsync(raw_T, raw_X, N_val * sizeof(uint64_t),
                  cudaMemcpyDeviceToDevice, stream);
  cudaEventRecord(ctx.ntt_start[0], stream);
  launch_forward_ntt_fermat(N_val, raw_T, precomp_device_ptr, constant_precomp,
                            stream);
  cudaEventRecord(ctx.ntt_end[0], stream);
  cudaEventRecord(ctx.pw_start[0], stream);
  pointwise_multiply_kernel<<<blocks, threads, 0, stream>>>(raw_T, raw_T,
                                                              raw_T, mod_val,
                                                              N_val);
  cudaEventRecord(ctx.pw_end[0], stream);
  cudaEventRecord(ctx.ntt_start[1], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_T, raw_inverse_powers, inv_n,
                                  mod_val, stream);
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
  pointwise_multiply_kernel<<<blocks, threads, 0, stream>>>(
      raw_Z1, raw_Z1, raw_mu_freq, mod_val, N_val);
  cudaEventRecord(ctx.pw_end[1], stream);
  cudaEventRecord(ctx.ntt_start[3], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_Z1, raw_inverse_powers, inv_n,
                                  mod_val, stream);
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
  pointwise_multiply_kernel<<<blocks, threads, 0, stream>>>(
      raw_Z2, raw_Z2, raw_P_freq, mod_val, N_val);
  cudaEventRecord(ctx.pw_end[2], stream);
  cudaEventRecord(ctx.ntt_start[5], stream);
  launch_inverse_ntt_fermat_fused(N_val, raw_Z2, raw_inverse_powers, inv_n,
                                  mod_val, stream);
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
  (void)bit_len;
  (void)q_val;

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

    // Poll every few graph launches rather than only at percentage boundaries.
    // This bounds scheduler latency for 33k+ bit exponents while avoiding a
    // clock read for every launch.
    constexpr int progress_poll_steps = 8;
    if (squarings % progress_poll_steps == 0 || squarings == total_steps) {
      int pct = (squarings * 100) / total_steps;

       if (pct < 100 &&
            (progress_schedule == nullptr ||
             progress_schedule->claim_nonfinal_slot())) {
         print_candidate_status(thread_id, candidate_start, q_val,
                                get_completed_candidate_count().load(),
                               candidate_total,
                                std::string(3 - std::to_string(pct).size(), ' ') +
                                    std::to_string(pct) + "%",
                                false);
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
  if (mpz_cmp_ui(final_val, 1) == 0) {
    is_prime = true;
  }
  const size_t completed = get_completed_candidate_count().fetch_add(1) + 1;
  print_candidate_status(thread_id, candidate_start, q_val, completed,
                          candidate_total, "100%", true);
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

inline std::atomic<size_t> &get_candidate_index() {
  static std::atomic<size_t> idx{0};
  return idx;
}

auto prepare_next_candidate(mpz_t K, const std::vector<uint64_t> &sieve_primes)
    -> Candidate {
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

auto is_decimal(const std::string &value) -> bool {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= '0' && character <= '9';
         });
}

auto read_ordered_lines(const std::string &path,
                        std::vector<std::string> &values, const char *kind,
                        bool allow_empty = false) -> bool {
  std::ifstream input(path);
  if (!input.is_open()) {
    std::cerr << "ERROR unable to open " << kind << " file: " << path << '\n';
    return false;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (!is_decimal(line) || (line.size() > 1 && line.front() == '0')) {
      std::cerr << "ERROR malformed " << kind << " entry\n";
      return false;
    }
    values.push_back(line);
  }
  if (input.bad() || (values.empty() && !allow_empty)) {
    std::cerr << "ERROR empty or unreadable " << kind << " file\n";
    return false;
  }
  return true;
}

auto run_ntt_differential(
    size_t N_val, const polyarith::Modulus &modulus,
    const precomputation::Precomputation<MODULUS_BITS> *precomp_device_ptr,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>
        &constant_precomp) -> bool {
  std::vector<uint64_t> input(N_val);
  for (size_t index = 0; index < N_val; ++index) {
    input[index] = (index * index * 17 + index * 13 + 5) % modulus.get_modulus();
  }
  std::vector<uint64_t> expected_forward(N_val);
  std::vector<uint64_t> expected_inverse(N_val);
  NttReference reference(N_val, modulus.get_modulus(), modulus.get_generator());
  reference.compute_forward(expected_forward.data(), input.data());
  reference.compute_inverse(expected_inverse.data(), expected_forward.data());

  thrust::device_vector<uint64_t> data(input);
  std::vector<uint64_t> inverse_powers(N_val / 2);
  uint64_t power = 1;
  const uint64_t inverse_root = modulus.get_root_inverse(N_val);
  for (size_t index = 0; index < inverse_powers.size(); ++index) {
    inverse_powers[index] = power;
    power = modulus.multiply(power, inverse_root);
  }
  thrust::device_vector<uint64_t> d_inverse_powers(inverse_powers);
  launch_forward_ntt_fermat(N_val, thrust::raw_pointer_cast(data.data()),
                            precomp_device_ptr, constant_precomp, nullptr);
  cudaDeviceSynchronize();
  thrust::host_vector<uint64_t> forward = data;
  for (size_t index = 0; index < N_val; ++index) {
    if (forward[index] % modulus.get_modulus() !=
        expected_forward[index] % modulus.get_modulus()) {
      std::cerr << "NTT differential forward mismatch N=" << N_val
                << " index=" << index << '\n';
      return false;
    }
  }
  launch_inverse_ntt_fermat(
      N_val, thrust::raw_pointer_cast(data.data()), modulus.invert(N_val),
      modulus.get_modulus(), thrust::raw_pointer_cast(d_inverse_powers.data()),
      nullptr);
  cudaDeviceSynchronize();
  thrust::host_vector<uint64_t> inverse = data;
  for (size_t index = 0; index < N_val; ++index) {
    if (inverse[index] % modulus.get_modulus() !=
        expected_inverse[index] % modulus.get_modulus()) {
      std::cerr << "NTT differential inverse mismatch N=" << N_val
                << " index=" << index << " expected=" << expected_inverse[index]
                << " actual=" << inverse[index] % modulus.get_modulus() << '\n';
      return false;
    }
  }
  std::cout << "NTT differential PASS N=" << N_val << '\n';
  return true;
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
  bool ordered_mode = false;
  size_t workers = 1;
  bool fermat_regression = false;
  std::string ordered_bases_file;
  std::string ordered_candidates_file;

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
    if (arg == "--ordered") {
      if (i + 2 >= argc) {
        std::cerr
            << "ERROR --ordered requires BASES_FILE and CANDIDATES_FILE\n";
        return 2;
      }
      ordered_mode = true;
      ordered_bases_file = argv[++i];
      ordered_candidates_file = argv[++i];
      is_crunch = true;
    }
    if (arg == "--workers") {
      if (i + 1 >= argc || !is_decimal(argv[i + 1])) {
        std::cerr << "ERROR --workers requires a positive integer\n";
        return 2;
      }
      try {
        workers = std::stoull(argv[++i]);
      } catch (const std::exception &) {
        std::cerr << "ERROR --workers requires a positive integer\n";
        return 2;
      }
      if (workers == 0) {
        std::cerr << "ERROR --workers requires a positive integer\n";
        return 2;
      }
    }
    if (arg == "--fermat-regression") {
      fermat_regression = true;
    }
  }

  if (fermat_regression) {
    if (!run_ntt_differential(4096, modulus,
                              thrust::raw_pointer_cast(precomp_device.get()),
                              constant_precomp) ||
        !run_ntt_differential(65536, modulus,
                              thrust::raw_pointer_cast(precomp_device.get()),
                              constant_precomp)) {
      return 1;
    }
    const std::vector<uint64_t> candidates = {3, 17, 65537, 41559263,
                                               9, 15, 21, 25};
    for (const uint64_t candidate_value : candidates) {
      mpz_set_ui(p, candidate_value);
      mpz_t oracle;
      mpz_t fermat_base;
      mpz_init(oracle);
      mpz_init_set_ui(fermat_base, 2);
      mpz_powm_ui(oracle, fermat_base, candidate_value - 1, p);
      // The GPU pipeline implements this Fermat congruence, not a primality
      // test.  Keep the GMP oracle at the same level so its return-value
      // convention cannot affect this regression.
      const bool expected_fermat = mpz_cmp_ui(oracle, 1) == 0;
      mpz_clear(oracle);
      mpz_clear(fermat_base);
      const size_t bit_len = mpz_sizeinbase(p, 2);
      const size_t d = (bit_len + 15) / 16;
      StreamContext context;
      const bool actual = run_fermat_pipeline(
          p, bit_len, d, 4096, modulus.invert(4096), modulus,
          thrust::raw_pointer_cast(precomp_device.get()), constant_precomp,
          context, {}, {}, 0, candidate_value, candidates.size());
      if (actual != expected_fermat) {
        std::cerr << "GPU Fermat mismatch p=" << candidate_value
                  << " expected=" << expected_fermat
                  << " actual=" << actual << '\n';
        mpz_clear(p);
        return 1;
      }
      std::cout << "GPU Fermat PASS p=" << candidate_value
                << " expected=" << expected_fermat
                << " actual=" << actual << '\n';
    }
    mpz_clear(p);
    return 0;
  }

  if (is_crunch) {
    if (ordered_mode) {
      std::vector<std::string> base_strings;
      std::vector<std::string> candidate_strings;
      if (!read_ordered_lines(ordered_bases_file, base_strings, "base") ||
          !read_ordered_lines(ordered_candidates_file, candidate_strings,
                              "candidate", true) ||
          base_strings.size() != 3) {
        std::cerr << "ERROR ordered mode requires exactly three base primes\n";
        return 2;
      }
      mpz_t K;
      mpz_init_set_ui(K, 2);
      for (const std::string &base_string : base_strings) {
        mpz_t base;
        mpz_init(base);
        if (mpz_set_str(base, base_string.c_str(), 10) != 0 ||
            mpz_cmp_ui(base, 1) <= 0) {
          std::cerr << "ERROR invalid base prime\n";
          mpz_clear(base);
          mpz_clear(K);
          return 2;
        }
        mpz_mul(K, K, base);
        mpz_clear(base);
      }
      std::cout << "WORKERS count=" << workers << '\n';
      std::cout << "PLAN count=" << candidate_strings.size() << '\n';
      // The plan itself is deterministic.  With multiple workers, TEST and
      // completion records may be emitted in a different order.
      std::vector<uint64_t> ordered_candidates;
      ordered_candidates.reserve(candidate_strings.size());
      for (size_t index = 0; index < candidate_strings.size(); ++index) {
        const std::string &q_string = candidate_strings[index];
        if (q_string.size() > 20) {
          std::cerr << "ERROR candidate q exceeds uint64\n";
          mpz_clear(K);
          return 2;
        }
        uint64_t q = 0;
        try {
          q = std::stoull(q_string);
        } catch (const std::exception &) {
          std::cerr << "ERROR invalid candidate q\n";
          mpz_clear(K);
          return 2;
        }
        if (q < 2) {
          std::cerr << "ERROR invalid candidate q\n";
          mpz_clear(K);
          return 2;
        }
        mpz_t candidate_p;
        mpz_init(candidate_p);
        mpz_mul_ui(candidate_p, K, q);
        mpz_add_ui(candidate_p, candidate_p, 1);
        const size_t candidate_bits = mpz_sizeinbase(candidate_p, 2);
        const size_t candidate_d = (candidate_bits + 15) / 16;
        const size_t candidate_n = candidate_bits <= 32000 ? 4096 : 65536;
        mpz_clear(candidate_p);
        if (2 * candidate_d + 2 > candidate_n) {
          std::cerr << "ERROR candidate exceeds supported transform size\n";
          mpz_clear(K);
          return 2;
        }
        ordered_candidates.push_back(q);
       }
       std::atomic<size_t> next_index{0};
        ProgressLogSchedule ordered_progress_schedule;
       auto ordered_worker = [&](size_t worker_id) {
        StreamContext context;
        while (true) {
          const size_t index = next_index.fetch_add(1);
          if (index >= ordered_candidates.size()) {
            return;
          }
          const uint64_t q = ordered_candidates[index];
          {
            std::lock_guard<std::mutex> lock(get_print_mutex());
            std::cout << "TEST index=" << index << " q=" << q << '\n'
                      << std::flush;
          }
          Candidate candidate;
          candidate.q_val = q;
          mpz_mul_ui(candidate.p, K, q);
          mpz_add_ui(candidate.p, candidate.p, 1);
          candidate.bit_len = mpz_sizeinbase(candidate.p, 2);
          candidate.d = (candidate.bit_len + 15) / 16;
          candidate.N_val = candidate.bit_len <= 32000 ? 4096 : 65536;
          const bool passed = run_fermat_pipeline(
              candidate.p, candidate.bit_len, candidate.d, candidate.N_val,
              modulus.invert(candidate.N_val), modulus,
               thrust::raw_pointer_cast(precomp_device.get()), constant_precomp,
               context, candidate.h_P, candidate.h_mu, static_cast<int>(worker_id),
               q, ordered_candidates.size(), &ordered_progress_schedule);
          if (passed) {
            char *p_string = mpz_get_str(nullptr, 10, candidate.p);
            std::lock_guard<std::mutex> lock(get_print_mutex());
            std::cout << "FOUND index=" << index << " q=" << q
                      << " p=" << p_string << '\n' << std::flush;
            free(p_string);
          }
        }
      };
      if (workers == 1) {
        ordered_worker(0);
      } else {
        std::vector<std::thread> ordered_threads;
        ordered_threads.reserve(workers);
        for (size_t worker_id = 0; worker_id < workers; ++worker_id) {
          ordered_threads.emplace_back(ordered_worker, worker_id);
        }
        for (auto &thread : ordered_threads) {
          thread.join();
        }
      }
      mpz_clear(K);
      std::lock_guard<std::mutex> lock(get_print_mutex());
      std::cout << "DONE\n" << std::flush;
      return 0;
    }
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

        uint64_t inv_n = modulus.invert(cand.N_val);
        bool passed = run_fermat_pipeline(
            cand.p, cand.bit_len, cand.d, cand.N_val, inv_n, modulus,
            thrust::raw_pointer_cast(precomp_device.get()), constant_precomp,
            stream_ctxs[thread_id], cand.h_P, cand.h_mu, thread_id, cand.q_val,
            sieve_primes.size());

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
                        constant_precomp, ctx, empty_vec, empty_vec, 0, 0, 1);
  }
}
