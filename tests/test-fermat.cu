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

__global__ void mul_2_kernel(const uint64_t* in, uint64_t* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint64_t val = in[idx];
        uint64_t carry_in = (idx == 0) ? 0 : (in[idx-1] >> 15);
        out[idx] = ((val & 0x7FFF) << 1) | carry_in;
    }
}

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

__global__ void single_block_arbitrary_resolve_carries(uint64_t* data, size_t n) {
    __shared__ uint64_t smem[4097];
    int tid = threadIdx.x;
    
    uint64_t carry_in = 0;
    
    int num_chunks = (n + 4095) / 4096;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int base = chunk * 4096;
        
        for(int i=0; i<4; ++i) {
            int local_idx = tid + i*1024;
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
        while(changed) {
            changed = 0;
            for(int i=0; i<4; ++i) {
                int local_idx = tid + i*1024;
                if (base + local_idx >= n) continue;
                
                uint64_t val = smem[local_idx];
                uint64_t c = val >> 16;
                if (c > 0) {
                    changed = 1;
                    atomicAdd((unsigned long long*)&smem[local_idx], -(unsigned long long)(c << 16));
                    atomicAdd((unsigned long long*)&smem[local_idx + 1], (unsigned long long)c);
                }
            }
            changed = __syncthreads_or(changed);
        }
        
        carry_in = smem[4096];
        __syncthreads();
        
        for(int i=0; i<4; ++i) {
            int local_idx = tid + i*1024;
            int global_idx = base + local_idx;
            if (global_idx < n) {
                data[global_idx] = smem[local_idx];
            }
        }
        __syncthreads();
    }
}

void launch_resolve_carries(uint64_t* d_data, size_t n, cudaStream_t stream) {
    single_block_arbitrary_resolve_carries<<<1, 1024, 0, stream>>>(d_data, n);
}

__global__ void single_block_arbitrary_sub_kernel(uint64_t* r, const uint64_t* t, const uint64_t* z2, size_t n) {
    __shared__ uint64_t s_r[4097];
    int tid = threadIdx.x;
    
    int64_t borrow_in = 0;
    
    int num_chunks = (n + 4095) / 4096;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int base = chunk * 4096;
        
        for(int i=0; i<4; ++i) {
            int local_idx = tid + i*1024;
            int global_idx = base + local_idx;
            if (global_idx < n) {
                int64_t diff = (int64_t)t[global_idx] - (int64_t)z2[global_idx];
                s_r[local_idx] = (uint64_t)diff;
            } else {
                s_r[local_idx] = 0;
            }
        }
        if (tid == 0) {
            s_r[4096] = 0;
            int64_t diff = (int64_t)s_r[0] - borrow_in;
            s_r[0] = (uint64_t)diff;
        }
        __syncthreads();
        
        int changed = 1;
        while(changed) {
            changed = 0;
            for(int i=0; i<4; ++i) {
                int local_idx = tid + i*1024;
                if (base + local_idx >= n) continue;
                
                int64_t val = (int64_t)s_r[local_idx];
                if (val < 0) {
                    changed = 1;
                    atomicAdd((unsigned long long*)&s_r[local_idx], 65536ULL);
                    atomicAdd((unsigned long long*)&s_r[local_idx + 1], -1ULL);
                } else if (val >= 65536) {
                    changed = 1;
                    uint64_t carry = val >> 16;
                    atomicAdd((unsigned long long*)&s_r[local_idx], -(unsigned long long)(carry << 16));
                    atomicAdd((unsigned long long*)&s_r[local_idx + 1], carry);
                }
            }
            changed = __syncthreads_or(changed);
        }
        
        borrow_in = -(int64_t)s_r[4096];
        __syncthreads();
        
        for(int i=0; i<4; ++i) {
            int local_idx = tid + i*1024;
            int global_idx = base + local_idx;
            if (global_idx < n) {
                r[global_idx] = s_r[local_idx];
            }
        }
        __syncthreads();
    }
}

void launch_sub_kernel(uint64_t* r, const uint64_t* t, const uint64_t* z2, size_t n, cudaStream_t stream) {
    single_block_arbitrary_sub_kernel<<<1, 1024, 0, stream>>>(r, t, z2, n);
}

// -------------------------------------------------------------------------
// Wrapper for Forward/Inverse NTT on Tensor Cores (N=4096 or 65536)
// -------------------------------------------------------------------------
void launch_forward_ntt_fermat(
    size_t N_val,
    uint64_t* d_data, 
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    const int smem_per_warp = 4 * 8 * 16 * 16;

    if (N_val == 4096) {
        constexpr int N = 4096;
        constexpr int n = 1 << 12; // 4096
        const dim3 block_dim(32 * 1, 1 * 1);
        const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
        run_forward_iterative_wmma_radix16<N, n><<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
            
        constexpr int n2 = 1 << 8; // 256
        const dim3 block_dim2(32, 32 / 32);
        const dim3 grid_dim2(N / 16 / 16 / block_dim2.y);
        run_forward_iterative_wmma_radix256<N, n2><<<grid_dim2, block_dim2, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
    } else if (N_val == 65536) {
        constexpr int N = 65536;
        constexpr int n = 1 << 16;
        const dim3 block_dim(32 * 2, 1 * 1);
        const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16), N / n / block_dim.y);
        run_forward_iterative_wmma_radix16<N, n><<<grid_dim, block_dim, smem_per_warp *(block_dim.x * block_dim.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
            
        constexpr int n2 = 1 << 12;
        const dim3 block_dim2(32 * 1, 1 * 4);
        const dim3 grid_dim2(n2 / 16 / (block_dim2.x / 32 * 16), N / n2 / block_dim2.y);
        run_forward_iterative_wmma_radix16<N, n2><<<grid_dim2, block_dim2, smem_per_warp *(block_dim2.x * block_dim2.y / 32), stream>>>(
            d_data, precomp, constant_precomp);
            
        constexpr int n3 = 1 << 8;
        const dim3 block_dim3(32, 32 / 32);
        const dim3 grid_dim3(N / 16 / 16 / block_dim3.y);
        run_forward_iterative_wmma_radix256<N, n3><<<grid_dim3, block_dim3, smem_per_warp * 1, stream>>>(
            d_data, precomp, constant_precomp);
    }
}

void launch_inverse_ntt_fermat(
    size_t N_val,
    uint64_t* d_data, uint64_t inv_n, uint64_t modulus,
    const precomputation::Precomputation<MODULUS_BITS>* precomp,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp,
    cudaStream_t stream) 
{
    int threads = 256;
    int blocks = (N_val + threads - 1) / threads;
    
    int shift = (N_val == 65536) ? 16 : 20;
    
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);
    launch_forward_ntt_fermat(N_val, d_data, precomp, constant_precomp, stream);
    bit_reverse_permute<<<blocks, threads, 0, stream>>>(d_data, N_val, shift);
    
    reverse_and_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus, N_val);
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

// // Convert GMP to base 2^16 limbs
void gmp_to_limbs(mpz_t x, std::vector<uint64_t>& limbs, size_t N_val) {
    limbs.assign(N_val, 0);
    size_t count;
    std::vector<uint16_t> temp(N_val, 0);
    mpz_export(temp.data(), &count, -1, sizeof(uint16_t), 0, 0, x);
    for (size_t i = 0; i < count; ++i) {
        limbs[i] = temp[i];
    }
}

// Convert base 2^16 limbs to GMP
void limbs_to_gmp(const std::vector<uint64_t>& limbs, mpz_t x, size_t N_val) {
    std::vector<uint16_t> short_limbs(N_val);
    for(size_t i=0; i<N_val; i++) short_limbs[i] = (uint16_t)limbs[i];
    mpz_import(x, N_val, -1, sizeof(uint16_t), 0, 0, short_limbs.data());
}

void run_fermat_pipeline(
    mpz_t p, size_t bit_len, size_t d, size_t N_val, uint64_t inv_n,
    const polyarith::Modulus& modulus,
    const precomputation::Precomputation<MODULUS_BITS>* precomp_device_ptr,
    const precomputation::ConstantPrecomputation<MODULUS_BITS>& constant_precomp)
{
    // 1. Precompute mu = floor(B^(2*d) / P)
    mpz_t B_2d, mu;
    mpz_init(B_2d);
    mpz_init(mu);
    mpz_ui_pow_ui(B_2d, 2, 16 * 2 * d);
    mpz_fdiv_q(mu, B_2d, p);
    
    std::vector<uint64_t> h_P(N_val, 0);
    std::vector<uint64_t> h_mu(N_val, 0);
    std::vector<uint64_t> h_x(N_val, 0);
    
    gmp_to_limbs(p, h_P, N_val);
    gmp_to_limbs(mu, h_mu, N_val);
    
    mpz_t x;
    mpz_init_set_ui(x, 2); // Base 2
    gmp_to_limbs(x, h_x, N_val);
    mpz_clear(x);
    
    thrust::device_vector<uint64_t> d_P = h_P;
    thrust::device_vector<uint64_t> d_P_freq = h_P;
    thrust::device_vector<uint64_t> d_mu_freq = h_mu;
    thrust::device_vector<uint64_t> d_X = h_x;
    
    thrust::device_vector<uint64_t> d_T(N_val, 0);
    thrust::device_vector<uint64_t> d_Q(N_val, 0);
    thrust::device_vector<uint64_t> d_Z1(N_val, 0);
    thrust::device_vector<uint64_t> d_Z2(N_val, 0);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int threads = 256;
    int blocks = (N_val + threads - 1) / threads;
    
    launch_forward_ntt_fermat(N_val, thrust::raw_pointer_cast(d_P_freq.data()), precomp_device_ptr, constant_precomp, stream);
    launch_forward_ntt_fermat(N_val, thrust::raw_pointer_cast(d_mu_freq.data()), precomp_device_ptr, constant_precomp, stream);
    
    mpz_t p_minus_1;
    mpz_init(p_minus_1);
    mpz_sub_ui(p_minus_1, p, 1);
    size_t actual_bit_len = mpz_sizeinbase(p_minus_1, 2);
    
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    uint64_t* raw_X = thrust::raw_pointer_cast(d_X.data());
    uint64_t* raw_T = thrust::raw_pointer_cast(d_T.data());
    uint64_t* raw_Z1 = thrust::raw_pointer_cast(d_Z1.data());
    uint64_t* raw_Q = thrust::raw_pointer_cast(d_Q.data());
    uint64_t* raw_Z2 = thrust::raw_pointer_cast(d_Z2.data());
    uint64_t* raw_mu_freq = thrust::raw_pointer_cast(d_mu_freq.data());
    uint64_t* raw_P_freq = thrust::raw_pointer_cast(d_P_freq.data());
    uint64_t* raw_P = thrust::raw_pointer_cast(d_P.data());
    uint64_t mod_val = modulus.get_modulus();
    
    // Step 1: T = X^2
    cudaMemcpyAsync(raw_T, raw_X, N_val * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    launch_forward_ntt_fermat(N_val, raw_T, precomp_device_ptr, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_T, raw_T, raw_T, inv_n, mod_val, N_val);
    launch_inverse_ntt_fermat(N_val, raw_T, inv_n, mod_val, precomp_device_ptr, constant_precomp, stream);
    launch_resolve_carries(raw_T, N_val, stream);
    
    // Step 2: Q = floor(T / B^(d-1)) * mu
    shift_right_kernel<<<blocks, threads, 0, stream>>>(raw_Z1, raw_T, d - 1, N_val);
    launch_forward_ntt_fermat(N_val, raw_Z1, precomp_device_ptr, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_Z1, raw_Z1, raw_mu_freq, inv_n, mod_val, N_val);
    launch_inverse_ntt_fermat(N_val, raw_Z1, inv_n, mod_val, precomp_device_ptr, constant_precomp, stream);
    launch_resolve_carries(raw_Z1, N_val, stream);
    
    // Step 3: Q = Q / B^(d+1)
    shift_right_kernel<<<blocks, threads, 0, stream>>>(raw_Q, raw_Z1, d + 1, N_val);
    
    // Step 4: Z2 = Q * P
    cudaMemcpyAsync(raw_Z2, raw_Q, N_val * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    launch_forward_ntt_fermat(N_val, raw_Z2, precomp_device_ptr, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(raw_Z2, raw_Z2, raw_P_freq, inv_n, mod_val, N_val);
    launch_inverse_ntt_fermat(N_val, raw_Z2, inv_n, mod_val, precomp_device_ptr, constant_precomp, stream);
    launch_resolve_carries(raw_Z2, N_val, stream);
    
    // Step 5: R = T - Z2
    launch_sub_kernel(raw_X, raw_T, raw_Z2, N_val, stream);
    
    single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(raw_X, raw_P, d, N_val);
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    
    int squarings = 0;
    int multiplies = 0;
    
    int total_steps = actual_bit_len - 1;
    int twenty_percent = total_steps / 5;
    if (twenty_percent == 0) twenty_percent = 1;
    
    for (int i = actual_bit_len - 2; i >= 0; --i) {
        cudaGraphLaunch(instance, stream);
        squarings++;
        
        if (mpz_tstbit(p_minus_1, i)) {
            mul_2_kernel<<<blocks, threads, 0, stream>>>(raw_X, raw_T, N_val);
            cudaMemcpyAsync(raw_X, raw_T, N_val * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
            single_block_arbitrary_conditional_sub_p_kernel<<<1, 1024, 0, stream>>>(raw_X, raw_P, d, N_val);
            multiplies++;
        }
        
        if (squarings % twenty_percent == 0) {
            cudaStreamSynchronize(stream);
            std::cout << "Progress: " << (squarings * 100 / total_steps) << "% (" << squarings << "/" << total_steps << " squarings)" << std::endl;
        }
    }
    
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    thrust::host_vector<uint64_t> final_limbs = d_X;
    mpz_t final_val;
    mpz_init(final_val);
    limbs_to_gmp(std::vector<uint64_t>(final_limbs.begin(), final_limbs.end()), final_val, N_val);
    
    float total_sec = ms / 1000.0f;
    float avg_us = ms * 1000.0f / squarings;
    if (mpz_cmp_ui(final_val, 1) == 0) {
        size_t exact_digits = mpz_sizeinbase(p, 10);
        size_t exact_bits = mpz_sizeinbase(p, 2);
        std::cout << "[PASS] Prime: " << exact_digits << " digits (" << exact_bits << " bits) | Total Time: " << total_sec << " s | Avg per step: " << avg_us << " us" << std::endl;
    } else {
        std::cout << "  [FAIL] Final x != 1" << std::endl;
    }
    
    mpz_clear(final_val);
    mpz_clear(p_minus_1);
    mpz_clear(mu);
    mpz_clear(B_2d);
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout << "Starting Fermat Harness" << std::endl;
    // Set up NTT field
    const polyarith::Modulus modulus(UINT64_C(0x1fff'fff9'0000'0001), 3);
    
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
    int phase = 2;
    size_t target_bits = 258000;
    int target_index = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i+1 < argc) filename = argv[++i];
        if (arg == "--phase" && i+1 < argc) phase = std::stoi(argv[++i]);
        if (arg == "--target-bits" && i+1 < argc) target_bits = std::stoull(argv[++i]);
        if (arg == "--index" && i+1 < argc) target_index = std::stoi(argv[++i]);
    }
    
    if (phase == 2) {
        std::cout << "--- Phase 2: Full Fermat Primality ---" << std::endl;
        std::ifstream primes_file(filename);
        if (!primes_file.is_open()) {
            std::cout << "Could not open " << filename << std::endl;
            return 1;
        }
        
        std::string line;
        std::vector<std::string> candidates;
        while (std::getline(primes_file, line)) {
            if (line.empty() || line[0] == '#') continue;
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
        
        if (selected_idx < 0 || selected_idx >= candidates.size()) {
            std::cout << "Candidate not found!" << std::endl;
            return 1;
        }
        
        line = candidates[selected_idx];
        mpz_init_set_str(p, line.c_str(), 10);
        
        size_t bit_len = mpz_sizeinbase(p, 2);
        size_t exact_digits = line.length();
        size_t d = (bit_len + 15) / 16;
        
        size_t N_val;
        if (bit_len <= 32000) N_val = 4096;
        else N_val = 65536;
        
        if (2*d + 2 > N_val) {
            std::cout << "ERROR: N_val " << N_val << " is not sufficient for 2d+2=" << (2*d+2) << " limbs to prevent aliasing." << std::endl;
            return 1;
        }
        
        uint64_t inv_n = modulus.invert(N_val);
        std::cout << "Testing prime index " << selected_idx << ":" << std::endl;
        std::cout << "exact decimal digit count: " << exact_digits << std::endl;
        std::cout << "bit count: " << bit_len << std::endl;
        std::cout << "d=" << d << " limbs, max convolution length 2d+2=" << (2*d+2) << std::endl;
        std::cout << "Chosen N_val: " << N_val << " (sufficient zero-padding guaranteed)" << std::endl;
        
        run_fermat_pipeline(p, bit_len, d, N_val, inv_n, modulus, thrust::raw_pointer_cast(precomp_device.get()), constant_precomp);

    }
}
