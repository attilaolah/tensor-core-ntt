#pragma once
#include <cstdint>

__global__ void shift_right_kernel(uint64_t* out, const uint64_t* in, size_t shift, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx + shift < n) {
            out[idx] = in[idx + shift];
        } else {
            out[idx] = 0;
        }
    }
}

__global__ void sub_kernel_and_resolve(uint64_t* r, const uint64_t* t, const uint64_t* z2, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t borrow = 0;
        for (size_t i = 0; i < n; ++i) {
            int64_t diff = (int64_t)t[i] - (int64_t)z2[i] - borrow;
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

__global__ void parallel_conditional_sub_p_kernel(uint64_t* r, const uint64_t* p, size_t d, size_t n) {
    __shared__ uint64_t s_r[4096];
    
    int tid = threadIdx.x;
    for(int i=0; i<4; ++i) {
        s_r[tid + i*1024] = r[tid + i*1024];
    }
    __syncthreads();
    
    for (int iter = 0; iter < 2; iter++) {
        int local_highest_diff_idx = -1;
        int local_diff_sign = 0;
        
        for(int i=3; i>=0; --i) {
            int idx = tid + i*1024;
            uint64_t p_val = p[idx];
            if (s_r[idx] != p_val) {
                local_highest_diff_idx = idx;
                local_diff_sign = (s_r[idx] > p_val) ? 1 : -1;
                break;
            }
        }
        
        __shared__ int max_diff_idx;
        __shared__ int max_diff_sign;
        if (tid == 0) { max_diff_idx = -1; max_diff_sign = 0; }
        __syncthreads();
        
        if (local_highest_diff_idx != -1) {
            atomicMax(&max_diff_idx, local_highest_diff_idx);
        }
        __syncthreads();
        
        if (local_highest_diff_idx == max_diff_idx && local_highest_diff_idx != -1) {
            max_diff_sign = local_diff_sign;
        }
        __syncthreads();
        
        bool geq = (max_diff_idx == -1) || (max_diff_sign == 1);
        if (!geq) break;
        
        for(int i=0; i<4; ++i) {
            int idx = tid + i*1024;
            int64_t diff = (int64_t)s_r[idx] - (int64_t)p[idx];
            s_r[idx] = (uint64_t)diff;
        }
        __syncthreads();
        
        int changed = 1;
        while(changed) {
            changed = 0;
            for(int i=0; i<4; ++i) {
                int idx = tid + i*1024;
                int64_t val = (int64_t)s_r[idx];
                if (val < 0) {
                    changed = 1;
                    atomicAdd((unsigned long long*)&s_r[idx], 65536ULL);
                    if (idx + 1 < n) {
                        atomicAdd((unsigned long long*)&s_r[idx + 1], -1ULL);
                    }
                } else if (val >= 65536) {
                    changed = 1;
                    uint64_t carry = val >> 16;
                    atomicAdd((unsigned long long*)&s_r[idx], -(unsigned long long)(carry << 16));
                    if (idx + 1 < n) {
                        atomicAdd((unsigned long long*)&s_r[idx + 1], carry);
                    }
                }
            }
            changed = __syncthreads_or(changed);
        }
    }
    
    for(int i=0; i<4; ++i) {
        r[tid + i*1024] = s_r[tid + i*1024];
    }
}
