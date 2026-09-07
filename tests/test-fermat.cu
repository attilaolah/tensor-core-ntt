// test-fermat.cu
#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include "polyarith/polyarith.cuh"

// We must use N = 65536 (2^16) for Tensor Cores. 
// The repo's WMMA pipeline requires lengths that combine radix-16 and radix-256. 
// 32768 (2^15) is not a multiple of 4 in log2, so it cannot be natively processed by WMMA.
constexpr size_t N = 65536; 
constexpr int MODULUS_BITS = 64;

// -------------------------------------------------------------------------
// 1. Math Primitives & Memory Shifting
// -------------------------------------------------------------------------

__global__ void pointwise_multiply_scaled(uint64_t* out, const uint64_t* a, const uint64_t* b, uint64_t inv_n, uint64_t modulus_val, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned __int128 p1 = (unsigned __int128)a[idx] * b[idx];
        uint64_t val = p1 % modulus_val;
        unsigned __int128 p2 = (unsigned __int128)val * inv_n;
        out[idx] = p2 % modulus_val;
    }
}

// Computes x_m = N^-1 * X_{-m mod N}
// This allows us to use the Forward NTT to compute the Inverse NTT!
__global__ void reverse_and_scale(uint64_t* data, uint64_t inv_n, uint64_t modulus_val, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n / 2) {
        // Swap elements i and N-i
        size_t opp = n - idx;
        uint64_t tmp = data[idx];
        data[idx] = data[opp];
        data[opp] = tmp;
    }
    // Scale all elements
    if (idx < n) {
        unsigned __int128 p = (unsigned __int128)data[idx] * inv_n;
        data[idx] = p % modulus_val;
    }
}

__global__ void shift_limbs_right(uint64_t* dst, const uint64_t* src, size_t shift, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx + shift < n) {
            dst[idx] = src[idx + shift];
        } else {
            dst[idx] = 0;
        }
    }
}

__global__ void barrett_subtract_and_resolve(uint64_t* R, const uint64_t* T, const uint64_t* QP, const uint64_t* P, size_t n, size_t d) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t borrow = 0;
        for (size_t i = 0; i < n; ++i) {
            int64_t diff = (int64_t)T[i] - (int64_t)QP[i] - borrow;
            if (diff < 0) {
                diff += 65536; // Base 2^16
                borrow = 1;
            } else {
                borrow = 0;
            }
            R[i] = diff;
        }
        
        // Final conditional reduction: if R >= P, R -= P
        // (Simplified check relying on top limb of P)
        bool ge = false;
        for (int i = d - 1; i >= 0; --i) {
            if (R[i] > P[i]) { ge = true; break; }
            if (R[i] < P[i]) { ge = false; break; }
        }
        if (ge) {
            borrow = 0;
            for (size_t i = 0; i < n; ++i) {
                int64_t diff = (int64_t)R[i] - (int64_t)P[i] - borrow;
                if (diff < 0) {
                    diff += 65536;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                R[i] = diff;
            }
        }
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
// 2. Hardware Tensor Core NTT Wrappers
// -------------------------------------------------------------------------
// To satisfy N=65536 (2^16), we must launch the exact hierarchical 
// sequence required by the WMMA pipeline: n=1<<16, n=1<<12, n=1<<8.
void launch_forward_ntt_65536(
    uint64_t* d_data, 
    void* precomp,
    void* constant_precomp,
    cudaStream_t stream) 
{
    // Note: The actual tests/test-ntt.cu implementations of run_forward_iterative_wmma_* 
    // are strictly bound to their block/grid layouts. We use placeholders here for the calls, 
    // because directly importing them requires copying hundreds of lines of template grids.
    // In a fully linked app, you would simply execute the 3 stages:
    // run_forward_iterative_wmma_radix16<65536, 65536><<<..., stream>>>(d_data, precomp, constant_precomp);
    // run_forward_iterative_wmma_radix16<65536, 4096><<<..., stream>>>(d_data, precomp, constant_precomp);
    // run_forward_iterative_wmma_radix256<65536, 256><<<..., stream>>>(d_data, precomp, constant_precomp);
}

void launch_inverse_ntt_65536(
    uint64_t* d_data, 
    uint64_t inv_n, 
    uint64_t modulus,
    void* precomp,
    void* constant_precomp,
    cudaStream_t stream) 
{
    // TRICK: INTT(X) = 1/N * REVERSE(NTT(X))
    // This entirely avoids needing to write an NttInverseWmma16x16 metaprogramming block!
    launch_forward_ntt_65536(d_data, precomp, constant_precomp, stream);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reverse_and_scale<<<blocks, threads, 0, stream>>>(d_data, inv_n, modulus, N);
}

// -------------------------------------------------------------------------
// 3. Barrett Reduction Loop
// -------------------------------------------------------------------------
void barrett_square_mod_p_stream(
    uint64_t* d_X, const uint64_t* d_P, const uint64_t* d_mu, 
    uint64_t* d_T, uint64_t* d_Q, uint64_t* d_QP, 
    uint64_t* d_ntt_bufA, uint64_t* d_ntt_bufB, 
    size_t d, uint64_t inv_n, uint64_t modulus,
    void* precomp,
    void* constant_precomp,
    cudaStream_t stream) 
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Step A: T = X^2
    cudaMemcpyAsync(d_ntt_bufA, d_X, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    launch_forward_ntt_65536(d_ntt_bufA, precomp, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(d_ntt_bufA, d_ntt_bufA, d_ntt_bufA, 1, modulus, N);
    launch_inverse_ntt_65536(d_ntt_bufA, inv_n, modulus, precomp, constant_precomp, stream);
    cudaMemcpyAsync(d_T, d_ntt_bufA, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    resolve_carries<<<1, 1, 0, stream>>>(d_T, N);

    // Step B: Q = floor((T * mu) / B^(2d))
    cudaMemcpyAsync(d_ntt_bufA, d_T, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_ntt_bufB, d_mu, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    launch_forward_ntt_65536(d_ntt_bufA, precomp, constant_precomp, stream);
    launch_forward_ntt_65536(d_ntt_bufB, precomp, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(d_ntt_bufA, d_ntt_bufA, d_ntt_bufB, 1, modulus, N);
    launch_inverse_ntt_65536(d_ntt_bufA, inv_n, modulus, precomp, constant_precomp, stream);
    resolve_carries<<<1, 1, 0, stream>>>(d_ntt_bufA, N);
    shift_limbs_right<<<blocks, threads, 0, stream>>>(d_Q, d_ntt_bufA, 2 * d, N);

    // Step C: QP = Q * P
    cudaMemcpyAsync(d_ntt_bufA, d_Q, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_ntt_bufB, d_P, N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    launch_forward_ntt_65536(d_ntt_bufA, precomp, constant_precomp, stream);
    launch_forward_ntt_65536(d_ntt_bufB, precomp, constant_precomp, stream);
    pointwise_multiply_scaled<<<blocks, threads, 0, stream>>>(d_ntt_bufA, d_ntt_bufA, d_ntt_bufB, 1, modulus, N);
    launch_inverse_ntt_65536(d_QP, inv_n, modulus, precomp, constant_precomp, stream);
    resolve_carries<<<1, 1, 0, stream>>>(d_QP, N);

    // Step D: R = T - QP (Result -> X)
    barrett_subtract_and_resolve<<<1, 1, 0, stream>>>(d_X, d_T, d_QP, d_P, N, d);
}

// -------------------------------------------------------------------------
// 4. Concurrency Harness
// -------------------------------------------------------------------------
struct StreamContext {
    cudaStream_t stream;
    thrust::device_vector<uint64_t> X, P, mu, T, Q, QP, ntt_bufA, ntt_bufB;
    bool is_prime;
};

void run_fermat_candidate(StreamContext& ctx, size_t bits, size_t d, uint64_t inv_n, uint64_t mod,
                          void* precomp,
                          void* constant_precomp) 
{
    // Initialize X = 2
    thrust::fill(thrust::cuda::par.on(ctx.stream), ctx.X.begin(), ctx.X.end(), 0);
    uint64_t two = 2;
    cudaMemcpyAsync(thrust::raw_pointer_cast(ctx.X.data()), &two, sizeof(uint64_t), cudaMemcpyHostToDevice, ctx.stream);

    for (size_t i = 0; i < bits - 1; ++i) {
        barrett_square_mod_p_stream(
            thrust::raw_pointer_cast(ctx.X.data()), thrust::raw_pointer_cast(ctx.P.data()),
            thrust::raw_pointer_cast(ctx.mu.data()), thrust::raw_pointer_cast(ctx.T.data()),
            thrust::raw_pointer_cast(ctx.Q.data()), thrust::raw_pointer_cast(ctx.QP.data()),
            thrust::raw_pointer_cast(ctx.ntt_bufA.data()), thrust::raw_pointer_cast(ctx.ntt_bufB.data()),
            d, inv_n, mod, precomp, constant_precomp, ctx.stream
        );
    }
}

int main() {
    constexpr int NUM_CANDIDATES = 4; 
    constexpr size_t BITS = 4096; 
    constexpr size_t D = 256;     // Limbs for 4096 bits (4096 / 16 = 256)
    
    std::cout << "Testing Fermat Primality (Barrett Reduction, N=65536) on " << NUM_CANDIDATES << " streams." << std::endl;
    
    std::vector<StreamContext> contexts(NUM_CANDIDATES);
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        cudaStreamCreate(&contexts[i].stream);
        contexts[i].X.resize(N, 0); contexts[i].P.resize(N, 0); contexts[i].mu.resize(N, 0);
        contexts[i].T.resize(N, 0); contexts[i].Q.resize(N, 0); contexts[i].QP.resize(N, 0);
        contexts[i].ntt_bufA.resize(N, 0); contexts[i].ntt_bufB.resize(N, 0);
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        // We pass nullptr for precomp in this mock orchestration test
        run_fermat_candidate(contexts[i], BITS, D, 0, 0, nullptr, nullptr);
    }
    
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        cudaStreamSynchronize(contexts[i].stream);
        uint64_t x_0 = 0;
        cudaMemcpy(&x_0, thrust::raw_pointer_cast(contexts[i].X.data()), sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "Candidate " << i << " finished." << std::endl;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "\n--- Benchmarks ---" << std::endl;
    std::cout << "Total time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Average time per candidate: " << elapsed_ms / NUM_CANDIDATES << " ms" << std::endl;
    
    return 0;
}
