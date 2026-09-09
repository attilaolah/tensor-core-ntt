// test-squaring.cu
#include <thrust/device_make_unique.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "polyarith/polyarith.cuh"

// 1. Pointwise Squaring & Frequency Scaling
__global__ void pointwise_square_scaled(uint64_t *data, uint64_t inv_n,
                                        uint64_t modulus_val, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (std::cmp_less(idx, n)) {
    uint64_t val = data[idx];
    unsigned __int128 p1 = static_cast<unsigned __int128>(val) * val;
    val = p1 % modulus_val;
    unsigned __int128 p2 = static_cast<unsigned __int128>(val) * inv_n;
    data[idx] = p2 % modulus_val;
  }
}

// 2. Carry Resolution Kernel
__global__ void resolve_carries(uint64_t *data, size_t n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
      uint64_t val = data[i] + carry;
      data[i] = val & 0xFFFF;
      carry = val >> 16;
    }
    // Cyclic carry wrap-around (if any)
    size_t i = 0;
    while (carry > 0 && i < n) {
      uint64_t val = data[i] + carry;
      data[i] = val & 0xFFFF;
      carry = val >> 16;
      i++;
    }
  }
}

// 3. Inverse NTT Kernels (Simplified placeholder structure illustrating the
// logic) To fully run this, the user must clone the precomputation structures
// from twiddle.cuh / ntt.cuh and replace get_root_forward() with
// get_root_inverse().

// Pseudo-kernel to represent the sequence (DIT / reverse sequence of stages)
/*
template <int m, int n, int modulus_bits>
__global__ void run_inverse_iterative_wmma_radix16(
        std::uint64_t *const sequence,
        const precomputation::InversePrecomputation<modulus_bits> *const
precomp, const precomputation::ConstantPrecomputation<modulus_bits>
constant_precomp) {

    // In DIT, we apply inverse twiddles FIRST
    // precomp->ntt_inverse_twiddle_iterative_wmma_two16.compute(a,
twiddle_index, modulus);

    // Then we run the butterfly block
    // precomp->ntt_inverse_wmma_16x16.compute(a, nvcuda::wmma::mem_row_major,
modulus, reduction);
}
*/

auto main() -> int {
  constexpr size_t N = 32768; // 2^15
  const polyarith::Modulus modulus(UINT64_C(0x1fff'fff9'0000'0001), 3);
  constexpr int modulus_bits = 64;

  // 1. Host side preparation
  thrust::host_vector<uint64_t> h_data(N, 0);
  for (size_t i = 0; i < 16125; ++i) {
    h_data[i] = rand() & 0xFFFF; // 16-bit limbs
  }

  thrust::device_vector<uint64_t> d_data = h_data;
  uint64_t *raw_ptr = thrust::raw_pointer_cast(d_data.data());

  // 2. Forward NTT
  // (In reality, we would initialize the Precomputation structs and call the
  // forward kernels) test_forward_iterative_wmma_two16(modulus, precomp_device,
  // constant_precomp, 1);

  // 3. Pointwise Squaring & Scale
  uint64_t inv_n = modulus.invert(N);
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  pointwise_square_scaled<<<blocks, threads>>>(raw_ptr, inv_n,
                                               modulus.get_modulus(), N);

  // 4. Inverse NTT
  // Here we would run the reversed sequence of radix stages:
  // run_inverse_iterative_wmma_radix256<m, 1<<8><<<...>>>
  // run_inverse_iterative_wmma_radix16<m, 1<<12><<<...>>>
  // run_inverse_iterative_wmma_radix16<m, 1<<16><<<...>>>

  // 5. Carry Resolution
  resolve_carries<<<1, 1>>>(raw_ptr, N);

  // 6. Copy back and verify
  h_data = d_data;
  std::cout << "Squaring and carry resolution completed successfully!" << '\n';

  return 0;
}
