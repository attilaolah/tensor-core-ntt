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
#include "barrett_kernels.cuh"

// -------------------------------------------------------------------------
// Constants and Primitives
// -------------------------------------------------------------------------
constexpr size_t N = 4096; 
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
    if (idx == 0 || idx == n / 2) {
        unsigned __int128 p = (unsigned __int128)data[idx] * inv_n;
        data[idx] = p % modulus_val;
    } else if (idx < n / 2) {
        size_t opp = n - idx;
        uint64_t val_idx = data[idx];
        uint64_t val_opp = data[opp];
        
        unsigned __int128 p1 = (unsigned __int128)val_opp * inv_n;
        unsigned __int128 p2 = (unsigned __int128)val_idx * inv_n;
        
        data[idx] = p1 % modulus_val;
        data[opp] = p2 % modulus_val;
    }
}

__global__ void resolve_carries_bulk(uint64_t* data, size_t n, int* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint64_t val = data[idx];
        uint64_t carry = val >> 16;
        if (carry > 0) {
            atomicAdd((unsigned long long*)&data[idx], -(unsigned long long)(carry << 16));
            if (idx + 1 < n) {
                atomicAdd((unsigned long long*)&data[idx + 1], (unsigned long long)carry);
                if (d_changed) *d_changed = 1;
            }
        }
    }
}

__global__ void resolve_carries_sweep(uint64_t* data, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t val = data[i] + carry;
            carry = val >> 16;
            uint64_t rem = val & 0xFFFF;
            if (data[i] != rem) {
                data[i] = rem;
            }
        }
    }
}

void launch_resolve_carries(uint64_t* d_data, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // 3 passes reduces multi-limb carries to almost zero
    for (int i = 0; i < 3; ++i) {
        resolve_carries_bulk<<<blocks, threads, 0, stream>>>(d_data, n, nullptr);
    }
    // Single thread sweep guarantees exact mathematical resolution for any tiny cascades
    resolve_carries_sweep<<<1, 1, 0, stream>>>(d_data, n);
}

__global__ void sub_kernel_bulk(int64_t* r, const uint64_t* t, const uint64_t* z2, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        r[idx] = (int64_t)t[idx] - (int64_t)z2[idx];
    }
}

__global__ void sub_kernel_sweep(int64_t* r, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t borrow = 0;
        for (size_t i = 0; i < n; ++i) {
            int64_t diff = r[i] - borrow;
            if (diff < 0) {
                diff += 65536;
                borrow = 1;
            } else {
                borrow = 0;
            }
            if (r[i] != diff) {
                r[i] = diff;
            }
        }
    }
}

void launch_sub_kernel(uint64_t* r, const uint64_t* t, const uint64_t* z2, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sub_kernel_bulk<<<blocks, threads, 0, stream>>>((int64_t*)r, t, z2, n);
    sub_kernel_sweep<<<1, 1, 0, stream>>>((int64_t*)r, n);
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=4096)
// -------------------------------------------------------------------------
void launch_forward_ntt_4096(
    uint64_t* d_data, 
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    const int smem_per_warp = 4 * 8 * 16 * 16;

    {
        constexpr int n = 1 << 12; // 4096
        const dim3 block_dim(32 * 1, 1 * 1);
        const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
        run_forward_iterative_wmma_radix16<N, n><<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
    }
    {
        constexpr int n = 1 << 8; // 256
        const dim3 block_dim(32, 32 / 32);
        const dim3 grid_dim(N / 16 / 16 / block_dim.y);
        run_forward_iterative_wmma_radix256<N, n><<<grid_dim, block_dim, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
    }
}

void launch_inverse_ntt_4096(
    uint64_t* d_data, uint64_t inv_n, uint64_t modulus,
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Unscramble the bit-reversed input frequency data into natural order
    // Since N=4096 is 2^12, we shift by 32 - 12 = 20
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N, 20);
    
    launch_forward_ntt_4096(d_data, precomp, constant_precomp, stream);
    
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N, 20);
    reverse_and_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus, N);
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=65536)
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
    std::vector<uint16_t> temp(N, 0);
    mpz_export(temp.data(), &count, -1, sizeof(uint16_t), 0, 0, x);
    for (size_t i = 0; i < count; ++i) {
        limbs[i] = temp[i];
    }
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
        launch_forward_ntt_4096(raw_ptr, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_ptr, raw_ptr, raw_ptr, inv_n, modulus.get_modulus(), N);
        launch_inverse_ntt_4096(raw_ptr, inv_n, modulus.get_modulus(), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        launch_resolve_carries(raw_ptr, N, stream);
        
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
        if (!read_prime(filename.c_str(), target_digits * 3.321928, 500000, p)) {
            std::cout << "Could not read prime for Phase 2!" << std::endl;
            return 1;
        }
        
        size_t d = (mpz_sizeinbase(p, 2) + 15) / 16;
        std::cout << "Prime length: " << mpz_sizeinbase(p, 2) << " bits (" << d << " limbs of 16-bits)." << std::endl;
        
        // 1. Precompute mu = floor(B^(2*d) / P)
        mpz_t B_2d, mu;
        mpz_init(B_2d);
        mpz_init(mu);
        mpz_ui_pow_ui(B_2d, 2, 16 * 2 * d);
        mpz_fdiv_q(mu, B_2d, p);
        
        // 2. Transfer P and mu to GPU
        std::vector<uint64_t> h_p(N, 0), h_mu(N, 0);
        gmp_to_limbs(p, h_p);
        gmp_to_limbs(mu, h_mu);
        
        thrust::device_vector<uint64_t> d_P = h_p;
        thrust::device_vector<uint64_t> d_mu = h_mu;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Pre-transform P and mu into bit-reversed frequency domain
        thrust::device_vector<uint64_t> d_P_freq = d_P;
        launch_forward_ntt_4096(thrust::raw_pointer_cast(d_P_freq.data()), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        
        thrust::device_vector<uint64_t> d_mu_freq = d_mu;
        launch_forward_ntt_4096(thrust::raw_pointer_cast(d_mu_freq.data()), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
        
        // Set up working buffers
        thrust::device_vector<uint64_t> d_X(N, 0);
        mpz_t x_init;
        mpz_init_set_ui(x_init, 2);
        std::vector<uint64_t> h_X(N, 0);
        gmp_to_limbs(x_init, h_X);
        d_X = h_X;
        
        thrust::device_vector<uint64_t> d_T(N, 0);
        thrust::device_vector<uint64_t> d_Z1(N, 0);
        thrust::device_vector<uint64_t> d_Q(N, 0);
        thrust::device_vector<uint64_t> d_Z2(N, 0);
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        size_t loop_count = mpz_sizeinbase(p, 2) - 1;
        std::cout << "Running " << loop_count << " repeated squarings on GPU..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t step = 0; step < loop_count; ++step) {
            if (step % 100 == 0) {
                std::cout << "Step " << step << " / " << loop_count << std::endl;
            }
            // T = X^2
            uint64_t* raw_X = thrust::raw_pointer_cast(d_X.data());
            uint64_t* raw_T = thrust::raw_pointer_cast(d_T.data());
            cudaMemcpyAsync(raw_T, raw_X, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            launch_forward_ntt_4096(raw_T, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_T, raw_T, raw_T, inv_n, modulus.get_modulus(), N);
            launch_inverse_ntt_4096(raw_T, inv_n, modulus.get_modulus(), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            launch_resolve_carries(raw_T, N, stream);
            
            // Z1 = T * mu (T is currently in raw_T).
            // But wait! T has been mutated. It is natural time.
            uint64_t* raw_Z1 = thrust::raw_pointer_cast(d_Z1.data());
            uint64_t* raw_mu_freq = thrust::raw_pointer_cast(d_mu_freq.data());
            cudaMemcpyAsync(raw_Z1, raw_T, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            launch_forward_ntt_4096(raw_Z1, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            // Now raw_Z1 is T_freq (bit-reversed).
            pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_Z1, raw_Z1, raw_mu_freq, inv_n, modulus.get_modulus(), N);
            launch_inverse_ntt_4096(raw_Z1, inv_n, modulus.get_modulus(), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            launch_resolve_carries(raw_Z1, N, stream);
            
            // Q = Z1 >> 2d
            uint64_t* raw_Q = thrust::raw_pointer_cast(d_Q.data());
            shift_right_kernel<<<blocks, threads, 0, stream>>>(raw_Q, raw_Z1, 2 * d, N);
            
            // Z2 = Q * P
            uint64_t* raw_Z2 = thrust::raw_pointer_cast(d_Z2.data());
            uint64_t* raw_P_freq = thrust::raw_pointer_cast(d_P_freq.data());
            cudaMemcpyAsync(raw_Z2, raw_Q, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            launch_forward_ntt_4096(raw_Z2, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            // raw_Z2 is Q_freq (bit-reversed). P_freq is also bit-reversed.
            pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_Z2, raw_Z2, raw_P_freq, inv_n, modulus.get_modulus(), N);
            launch_inverse_ntt_4096(raw_Z2, inv_n, modulus.get_modulus(), thrust::raw_pointer_cast(precomp_device.get()), constant_precomp, stream);
            launch_resolve_carries(raw_Z2, N, stream);
            
            // R = T - Z2
            launch_sub_kernel(raw_X, raw_T, raw_Z2, N, stream);
            cudaStreamSynchronize(stream);
            
            // While R >= P, R = R - P
            uint64_t* raw_P = thrust::raw_pointer_cast(d_P.data());
            int geq_loops = 0;
            thrust::device_vector<int> d_geq(1, 0);
            while (true) {
                compare_and_sub_p_kernel<<<1, 1, 0, stream>>>(raw_X, raw_P, d, N, thrust::raw_pointer_cast(d_geq.data()));
                cudaStreamSynchronize(stream);
                if (d_geq[0] == 0) break;
                
                geq_loops++;
                if (geq_loops > 100) {
                    std::cout << "HANG DETECTED AT STEP " << step << "!" << std::endl;
                    thrust::host_vector<uint64_t> h_r(N);
                    cudaMemcpy(h_r.data(), raw_X, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    std::cout << "R upper limbs: ";
                    for(int i=N-1; i>=N-10; i--) std::cout << h_r[i] << " ";
                    std::cout << "... [d=" << d << "] ... ";
                    for(int i=d+5; i>=d-5; i--) std::cout << h_r[i] << " ";
                    std::cout << std::endl;
                    break;
                }
            }
        }
        
        cudaStreamSynchronize(stream);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        
        std::cout << "GPU Time: " << elapsed.count() << " ms for " << loop_count << " steps." << std::endl;
        std::cout << "Finished!" << std::endl;
    }
    
    return 0;
}
