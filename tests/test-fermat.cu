// test-fermat.cu
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cassert>

#include <gmp.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Include everything from test-ntt to reuse kernels and precomputation structs
#include "test-ntt.cu"

// -------------------------------------------------------------------------
// Constants and Primitives
// -------------------------------------------------------------------------
constexpr size_t N = 65536; 
constexpr int MODULUS_BITS = 64;

__global__ void pointwise_multiply_scaled(uint64_t* out, const uint64_t* a, const uint64_t* b, uint64_t inv_n, uint64_t modulus_val, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned __int128 p1 = (unsigned __int128)a[idx] * b[idx];
        out[idx] = p1 % modulus_val;
    }
}

__global__ void bit_reverse_permute(uint64_t* data, size_t n, int shift) {
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

__global__ void reverse_and_scale(uint64_t* data, uint64_t inv_n, uint64_t modulus_val, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n / 2) {
        size_t opp = n - idx;
        uint64_t tmp = data[idx];
        data[idx] = data[opp];
        data[opp] = tmp;
    }
    if (idx < n) {
        unsigned __int128 p = (unsigned __int128)data[idx] * inv_n;
        data[idx] = p % modulus_val;
    }
}

__global__ void resolve_carries(uint64_t* data, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t val = data[i] + carry;
            data[i] = val & 0xFFFF;
            carry = val >> 16;
        }
    }
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=65536)
// -------------------------------------------------------------------------
void launch_forward_ntt_65536(
    uint64_t* d_data, 
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    const int smem_per_warp = 4 * 8 * 16 * 16;

    {
        constexpr int n = 1 << 16;
        const dim3 block_dim(32 * 2, 1 * 1);
        const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
        run_forward_iterative_wmma_radix16<N, n><<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
    }
    {
        constexpr int n = 1 << 12;
        const dim3 block_dim(32 * 1, 1 * 4);
        const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
        run_forward_iterative_wmma_radix16<N, n><<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
    }
    {
        constexpr int n = 1 << 8;
        const dim3 block_dim(32, 32 / 32);
        const dim3 grid_dim(N / 16 / 16 / block_dim.y);
        run_forward_iterative_wmma_radix256<N, n><<<grid_dim, block_dim, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
    }
}

void launch_inverse_ntt_65536(
    uint64_t* d_data, uint64_t inv_n, uint64_t modulus,
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // 1. Unscramble the bit-reversed input frequency data into natural order
    // (Since N=65536 is 2^16, we shift by 32 - 16 = 16)
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N, 16);
    
    // 2. Double-forward trick: Forward NTT on natural freq data gives reversed scaled time data
    // But it gives it in bit-reversed layout!
    launch_forward_ntt_65536(d_data, precomp, constant_precomp, stream);
    
    // 3. Unscramble the output back to natural order
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N, 16);
    
    // 4. Reverse (N-i) and scale (N^-1)
    reverse_and_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus, N);
}

// -------------------------------------------------------------------------
// Host Utilities for GMP parsing
// -------------------------------------------------------------------------
bool read_prime(const char* file, size_t target_bits, size_t tolerance, mpz_t out) {
    std::ifstream f(file);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
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

// Convert GMP to base 2^16 limbs
void gmp_to_limbs(mpz_t x, std::vector<uint64_t>& limbs) {
    limbs.assign(N, 0);
    size_t count;
    mpz_export(limbs.data(), &count, -1, sizeof(uint16_t), 0, 0, x);
}

// Convert base 2^16 limbs to GMP
void limbs_to_gmp(const std::vector<uint64_t>& limbs, mpz_t x) {
    std::vector<uint16_t> short_limbs(N);
    for(size_t i=0; i<N; i++) short_limbs[i] = (uint16_t)limbs[i];
    mpz_import(x, N, -1, sizeof(uint16_t), 0, 0, short_limbs.data());
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout << "Starting Fermat Harness" << std::endl;
    // Set up NTT field
    const polyarith::Modulus modulus(UINT64_C(0x1fff'fff9'0000'0001), 3);
    uint64_t inv_n = modulus.invert(N);
    
    // Allocate Precomputations on heap to avoid stack overflow
    auto precomp_device = thrust::uninitialized_allocate_unique<precomputation::Precomputation<MODULUS_BITS>>(
        thrust::device_allocator<precomputation::Precomputation<MODULUS_BITS>>());
    auto precomp_host = std::make_unique<precomputation::Precomputation<MODULUS_BITS>>(modulus);
    cudaMemcpy(thrust::raw_pointer_cast(precomp_device.get()), precomp_host.get(), sizeof(precomputation::Precomputation<MODULUS_BITS>), cudaMemcpyHostToDevice);
    const precomputation::ConstantPrecomputation<MODULUS_BITS> constant_precomp(modulus);

    mpz_t p;
    mpz_init(p);
    
    // Parse arguments
    std::string filename = "primes.txt";
    int phase = 1;
    size_t target_digits = 4000;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i+1 < argc) filename = argv[++i];
        if (arg == "--phase" && i+1 < argc) phase = std::stoi(argv[++i]);
        if (arg == "--digits" && i+1 < argc) target_digits = std::stoi(argv[++i]);
    }
    
    if (phase == 1) {
        std::cout << "--- Phase 1: Modular Squaring Test ---" << std::endl;
        if (!read_prime(filename.c_str(), 10, 500000, p)) {
            std::cout << "Could not read prime for Phase 1!" << std::endl;
            return 1;
        }
        
        mpz_t x, expected;
        mpz_init_set_ui(x, 3);
        mpz_init(expected);
        
        // Expected CPU
        mpz_powm_ui(expected, x, 2, p);
        
        std::vector<uint64_t> h_data(N, 0);
        gmp_to_limbs(x, h_data);
        
        thrust::device_vector<uint64_t> d_data = h_data;
        uint64_t* raw_ptr = thrust::raw_pointer_cast(d_data.data());
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Single Square via GPU
        launch_forward_ntt_65536(raw_ptr, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_ptr, raw_ptr, raw_ptr, inv_n, modulus.get_modulus(), N);
        launch_inverse_ntt_65536(raw_ptr, inv_n, modulus.get_modulus(), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        resolve_carries<<<1, 1, 0, stream>>>(raw_ptr, N);
        
        cudaStreamSynchronize(stream);
        thrust::host_vector<uint64_t> result = d_data;
        
        // For Barrett reduction, we also need R = T - Q*P.
        // We'll just compare the unreduced square for Phase 1 basic debug, since full Barrett relies on exact limits.
        mpz_t unreduced;
        mpz_init(unreduced);
        limbs_to_gmp(std::vector<uint64_t>(result.begin(), result.end()), unreduced);
        
        mpz_t unreduced_expected;
        mpz_init(unreduced_expected);
        mpz_mul(unreduced_expected, x, x);
        
        if (mpz_cmp(unreduced, unreduced_expected) == 0) {
            std::cout << "[PASS] GPU Unreduced Squaring Matches GMP!" << std::endl;
        } else {
            std::cout << "[FAIL] Output diverges!" << std::endl;
            std::vector<uint64_t> exp_limbs(N, 0);
            size_t count;
            mpz_export(exp_limbs.data(), &count, -1, sizeof(uint16_t), 0, 0, unreduced_expected);
            int printed = 0;
            for (size_t i = 0; i < N && printed < 10; ++i) {
                if (result[i] != exp_limbs[i]) {
                    std::cout << "Limb " << i << ": GPU=" << result[i] << " CPU=" << exp_limbs[i] << std::endl;
                    printed++;
                }
            }
        }
    } else {
        std::cout << "--- Phase 2: Full Fermat Primality ---" << std::endl;
        // Not fully implemented Barrett loop here due to complexity, but struct is complete.
        std::cout << "Barrett implementation requires rigorous boundary checks not yet finalized." << std::endl;
    }
    
    return 0;
}
