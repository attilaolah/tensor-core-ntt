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

__global__ void compare_and_sub_p_kernel(uint64_t* r, const uint64_t* p, size_t d, size_t n, int* d_geq) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bool geq = false;
        // Check if r >= p
        for (int i = n - 1; i >= 0; --i) {
            if (r[i] > p[i]) { geq = true; break; }
            if (r[i] < p[i]) { geq = false; break; }
        }
        
        if (geq) {
            *d_geq = 1;
            int64_t borrow = 0;
            for (size_t i = 0; i < n; ++i) {
                int64_t diff = (int64_t)r[i] - (int64_t)p[i] - borrow;
                if (diff < 0) {
                    diff += 65536;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                r[i] = diff;
            }
        } else {
            *d_geq = 0;
        }
    }
}
