template <int m, int n, int modulus_bits>
static __global__ void __launch_bounds__(1024) run_forward_iterative_radix16(
    std::uint64_t *const sequence,
    const precomputation::Precomputation<modulus_bits> *const precomp,
    const precomputation::ConstantPrecomputation<modulus_bits>
        constant_precomp) {
  const std::uint64_t modulus = constant_precomp.modulus;
  const typename decltype(precomp->ntt_forward_16x16)::reduction_type
      reduction = constant_precomp.friendly_reduction;

  const int twiddle_index = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
  const int subindex =
      n * (blockDim.y * blockIdx.y + threadIdx.y) + 16 * twiddle_index;

  polyarith::MatrixView<16, polyarith::UnreducedRatio<n / 16>, 16,
                        polyarith::UnreducedRatio<1>>
      block(&sequence[subindex]);

  __align__(16) std::uint64_t a[8];
  polyarith::cuda::load_matrix2_packed4cols_t(a, block);

  precomp->ntt_forward_16x16.compute_2t(a, modulus, reduction);

  if constexpr (n == (1 << 12)) {
    precomp->ntt_forward_twiddle_iterative_two12.compute(a, twiddle_index,
                                                         modulus);
  } else if constexpr (n == (1 << 16)) {
    precomp->ntt_forward_twiddle_iterative_two16.compute(a, twiddle_index,
                                                         modulus);
  } else if constexpr (n == (1 << 20)) {
    precomp->ntt_forward_twiddle_iterative_two20.compute(a, twiddle_index,
                                                         modulus);
  } else if constexpr (n == (1 << 24)) {
    precomp->ntt_forward_twiddle_iterative_two24.compute(a, twiddle_index,
                                                         modulus);
  } else if constexpr (n == (1 << 28)) {
    precomp->ntt_forward_twiddle_iterative_two28.compute(a, twiddle_index,
                                                         modulus);
  } else {
    assert(false);
  }

  polyarith::cuda::store_matrix2_packed2cols_n(block, a);
}

template <int m, int n, int modulus_bits>
static __global__ void __launch_bounds__(1024)
    run_forward_iterative_wmma_radix16(
        std::uint64_t *const sequence,
        const precomputation::Precomputation<modulus_bits> *const precomp,
        const precomputation::ConstantPrecomputation<modulus_bits>
            constant_precomp) {
  const std::uint64_t modulus = constant_precomp.modulus;
  const typename decltype(precomp->ntt_forward_wmma_16x16)::reduction_type
      reduction = constant_precomp.friendly_reduction;

  const int twiddle_index = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
  const int subindex =
      n * (blockDim.y * blockIdx.y + threadIdx.y) + 16 * twiddle_index;

  polyarith::MatrixView<16, polyarith::UnreducedRatio<n / 16>, 16,
                        polyarith::UnreducedRatio<1>>
      block(&sequence[subindex]);

  __align__(16) std::uint64_t a[8];
  polyarith::cuda::load_matrix_packed1col_t(a, block);

  precomp->ntt_forward_wmma_16x16.compute(a, nvcuda::wmma::mem_col_major,
                                          modulus, reduction);

  if constexpr (n == (1 << 12)) {
    precomp->ntt_forward_twiddle_iterative_wmma_two12.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 16)) {
    precomp->ntt_forward_twiddle_iterative_wmma_two16.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 20)) {
    precomp->ntt_forward_twiddle_iterative_wmma_two20.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 24)) {
    precomp->ntt_forward_twiddle_iterative_wmma_two24.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 28)) {
    precomp->ntt_forward_twiddle_iterative_wmma_two28.compute(a, twiddle_index,
                                                              modulus);
  } else {
    assert(false);
  }

  polyarith::cuda::store_matrix_packed1col_n(block, a);
}

template <int m, int n, int modulus_bits>
static __global__ void __launch_bounds__(1024) run_forward_iterative_radix256(
    std::uint64_t *const sequence,
    const precomputation::Precomputation<modulus_bits> *const precomp,
    const precomputation::ConstantPrecomputation<modulus_bits> constant_precomp)
  requires(n == (1 << 8))
{
  const std::uint64_t modulus = constant_precomp.modulus;
  const typename decltype(precomp->ntt_forward_16x16)::reduction_type
      reduction = constant_precomp.friendly_reduction;

  const int warpId = threadIdx.y;
  const int subindex = n * (blockDim.y * blockIdx.x + warpId);

  polyarith::MatrixView<16, polyarith::UnreducedRatio<n / 16>, 16,
                        polyarith::UnreducedRatio<1>>
      block(&sequence[subindex]);

  std::uint64_t a[8];
  polyarith::cuda::load_matrix2_packed4cols_t(a, block);

  precomp->ntt_forward_16x16.compute_2t(a, modulus, reduction);

  precomp->ntt_forward_twiddle_16x16.compute(a, modulus);

  polyarith::cuda::matrix_16x16_packed2cols_to_packed4cols(a);

  precomp->ntt_forward_16x16.compute_2n(a, modulus, reduction);

  polyarith::cuda::store_matrix2_packed2cols_n(block, a);
}

template <int m, int n, int modulus_bits>
static __global__ void
__launch_bounds__(1024) run_forward_iterative_wmma_radix256(
    std::uint64_t *const sequence,
    const precomputation::Precomputation<modulus_bits> *const precomp,
    const precomputation::ConstantPrecomputation<modulus_bits> constant_precomp)
  requires(n == (1 << 8))
{
  const std::uint64_t modulus = constant_precomp.modulus;
  const typename decltype(precomp->ntt_forward_wmma_16x16)::reduction_type
      reduction = constant_precomp.friendly_reduction;

  const int warpId = threadIdx.y;
  const int subindex = n * (blockDim.y * blockIdx.x + warpId);

  polyarith::MatrixView<16, polyarith::UnreducedRatio<n / 16>, 16,
                        polyarith::UnreducedRatio<1>>
      block(&sequence[subindex]);

  std::uint64_t a[8];
  polyarith::cuda::load_matrix_packed1col_t(a, block);

  precomp->ntt_forward_wmma_16x16.compute(a, nvcuda::wmma::mem_col_major,
                                          modulus, reduction);

  precomp->ntt_forward_twiddle_wmma_16x16.compute(a, modulus);

  precomp->ntt_forward_wmma_16x16.compute(a, nvcuda::wmma::mem_row_major,
                                          modulus, reduction);

  polyarith::cuda::store_matrix_packed1col_n(block, a);
}

template <int m, int n, int modulus_bits>
static __global__ void
__launch_bounds__(1024) run_forward_scalar_iterative_radix8(
    std::uint64_t *const sequence,
    const precomputation::Precomputation<modulus_bits> *const precomp,
    const precomputation::ConstantPrecomputation<modulus_bits> constant_precomp)
  requires(m >= (1 << 3) * 32 && n == (1 << 3))
{
  const std::uint64_t modulus = constant_precomp.modulus;

  const int laneId = threadIdx.x;
  const int warpId = threadIdx.y;
  const int subindex =
      n * (warpSize * (blockDim.y * blockIdx.x + warpId) + laneId);

  std::uint64_t a[8];
  for (int col = 0; col < 8; ++col) {
    a[col] = sequence[subindex + col];
  }

  precomp->ntt_forward_scalar_iterative_radix8_two3.compute(a, 0, modulus);

  for (int col = 0; col < 8; ++col) {
    sequence[subindex + col] = a[col];
  }
}

template <int m, int n, int modulus_bits>
static __global__ void __launch_bounds__(1024)
    run_forward_scalar_iterative_within_subsequence_radix8(
        std::uint64_t *const sequence,
        const precomputation::Precomputation<modulus_bits> *const precomp,
        const precomputation::ConstantPrecomputation<modulus_bits>
            constant_precomp) {
  const std::uint64_t modulus = constant_precomp.modulus;

  /*

   * x-axis and y-axis correspond to horizontal and vertical directions,
   respectively.
   */
  const int twiddle_index = blockDim.x * blockIdx.x + threadIdx.x;
  const int subindex =
      n * (blockDim.y * blockIdx.y + threadIdx.y) + twiddle_index;

  std::uint64_t a[8];
  for (int row = 0; row < 8; ++row) {
    a[row] = sequence[subindex + n / 8 * row];
  }

  if constexpr (n == (1 << 3)) {
    precomp->ntt_forward_scalar_iterative_radix8_two3.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 6)) {
    precomp->ntt_forward_scalar_iterative_radix8_two6.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 9)) {
    precomp->ntt_forward_scalar_iterative_radix8_two9.compute(a, twiddle_index,
                                                              modulus);
  } else if constexpr (n == (1 << 12)) {
    precomp->ntt_forward_scalar_iterative_radix8_two12.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 15)) {
    precomp->ntt_forward_scalar_iterative_radix8_two15.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 18)) {
    precomp->ntt_forward_scalar_iterative_radix8_two18.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 21)) {
    precomp->ntt_forward_scalar_iterative_radix8_two21.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 24)) {
    precomp->ntt_forward_scalar_iterative_radix8_two24.compute(a, twiddle_index,
                                                               modulus);
  } else {
    assert(false);
  }

  for (int row = 0; row < 8; ++row) {
    sequence[subindex + n / 8 * row] = a[row];
  }
}

template <int m, int n, int modulus_bits>
static __global__ void __launch_bounds__(65536 / 80)
    run_forward_scalar_iterative_within_subsequence_radix16(
        std::uint64_t *const sequence,
        const precomputation::Precomputation<modulus_bits> *const precomp,
        const precomputation::ConstantPrecomputation<modulus_bits>
            constant_precomp) {
  const std::uint64_t modulus = constant_precomp.modulus;

  const int twiddle_index = blockDim.x * blockIdx.x + threadIdx.x;
  const int subindex =
      n * (blockDim.y * blockIdx.y + threadIdx.y) + twiddle_index;

  std::uint64_t a[16];
  for (int row = 0; row < 16; ++row) {
    a[row] = sequence[subindex + n / 16 * row];
  }

  if constexpr (n == (1 << 4)) {
    precomp->ntt_forward_scalar_iterative_radix16_two4.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 8)) {
    precomp->ntt_forward_scalar_iterative_radix16_two8.compute(a, twiddle_index,
                                                               modulus);
  } else if constexpr (n == (1 << 12)) {
    precomp->ntt_forward_scalar_iterative_radix16_two12.compute(
        a, twiddle_index, modulus);
  } else if constexpr (n == (1 << 16)) {
    precomp->ntt_forward_scalar_iterative_radix16_two16.compute(
        a, twiddle_index, modulus);
  } else if constexpr (n == (1 << 20)) {
    precomp->ntt_forward_scalar_iterative_radix16_two20.compute(
        a, twiddle_index, modulus);
  } else if constexpr (n == (1 << 24)) {
    precomp->ntt_forward_scalar_iterative_radix16_two24.compute(
        a, twiddle_index, modulus);
  } else {
    assert(false);
  }

  for (int row = 0; row < 16; ++row) {
    sequence[subindex + n / 16 * row] = a[row];
  }
}

// MARK: Matrix, recursive

template <int modulus_bits>
static void test_forward_recursive_two4(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 4;

  thrust::host_vector<std::uint64_t> a(m * 16), b_correct(m * 16);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  for (int i = 0; i < 16; ++i) {
    ntt_ref.compute_forward(&b_correct[m * i], &a[m * i]);
  }

  thrust::device_vector<std::uint64_t> b(a);

  run_forward_two4_x16<<<1, 32>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_recursive_two8(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 8;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  run_forward_two8<<<1, 32>>>(thrust::raw_pointer_cast(b.data()),
                              precomp_device, constant_precomp);

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

// MARK: Matrix, recursive, WMMA

template <int modulus_bits>
static void test_forward_wmma_recursive_two4(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 4;

  thrust::host_vector<std::uint64_t> a(m * 16), b_correct(m * 16);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  for (int i = 0; i < 16; ++i) {
    ntt_ref.compute_forward(&b_correct[m * i], &a[m * i]);
  }

  thrust::device_vector<std::uint64_t> b(a);

  run_forward_wmma_two4_x16<<<1, 32, 4 * 8 * 16 * 16>>>(
      thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_wmma_recursive_two8(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 8;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  run_forward_wmma_two8<<<1, 32, 4 * 8 * 16 * 16>>>(
      thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

// MARK: Matrix, iterative

template <int modulus_bits>
static void test_forward_iterative_two8(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 8;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_two12(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 12;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_two16(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 16;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 2);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_two20(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 20;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 1, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 2);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 2);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_two24(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 24;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 24;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_two28(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 28;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 28;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 24;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_radix16<m, n><<<grid_dim, block_dim>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

// MARK: Iterative, WMMA

template <int modulus_bits>
static void test_forward_iterative_wmma_two8(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 8;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_wmma_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_wmma_two12(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 12;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      const dim3 grid_dim(m / 16 / 16 / block_dim.y);
      run_forward_iterative_wmma_radix256<m, n>
          <<<grid_dim, block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_wmma_two16(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 16;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_wmma_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_wmma_two20(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 20;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_wmma_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_wmma_two24(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 24;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 24;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_wmma_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

template <int modulus_bits>
static void test_forward_iterative_wmma_two28(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp,
    const int num_iters) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 28;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  if (num_iters == 1) {
    ntt_ref.compute_forward(b_correct.data(), a.data());
  }

  thrust::device_vector<std::uint64_t> b(a);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  const int smem_per_warp = 4 * 8 * 16 * 16;

  for (int i = 0; i < num_iters; ++i) {
    {
      constexpr int n = 1 << 28;
      const dim3 block_dim(32 * 2, 1 * 1);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 24;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 20;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 16;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 12;
      const dim3 block_dim(32 * 1, 1 * 4);
      const dim3 grid_dim(n / 16 / (block_dim.x / 32 * 16),
                          m / n / block_dim.y);
      run_forward_iterative_wmma_radix16<m, n>
          <<<grid_dim, block_dim,
             smem_per_warp *(block_dim.x * block_dim.y / 32)>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }

    {
      constexpr int n = 1 << 8;
      const dim3 block_dim(32, 32 / 32);
      run_forward_iterative_wmma_radix256<m, n>
          <<<dim3(m / 16 / 16 / block_dim.y), block_dim, smem_per_warp * 1>>>(
              thrust::raw_pointer_cast(b.data()), precomp_device,
              constant_precomp);
    }
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float duration;
  cudaEventElapsedTime(&duration, start, stop);
  std::clog << "Time: " << duration / 1000 << " seconds, "
            << duration / 1000 / num_iters << " second/iter" << std::endl;

  if (num_iters == 1) {
    const boost::icl::interval_set<int> mismatches =
        find_mismatches(b, b_correct, modulus.get_modulus());
    std::clog << "Mismatches: " << mismatches << std::endl;
  }
}

// MARK: Scalar, iterative, radix-8

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two3(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 3;
  constexpr int warps_per_block = 32;
  constexpr int blocks_per_grid = 123;
  constexpr int num_ntts = 32 * warps_per_block * blocks_per_grid;

  thrust::host_vector<std::uint64_t> a(m * num_ntts), b_correct(m * num_ntts);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  for (int i = 0; i < num_ntts; ++i) {
    ntt_ref.compute_forward(&b_correct[m * i], &a[m * i]);
  }

  thrust::device_vector<std::uint64_t> b(a);

  run_forward_scalar_iterative_radix8<m * num_ntts, m>
      <<<blocks_per_grid, dim3(32, warps_per_block)>>>(
          thrust::raw_pointer_cast(b.data()), precomp_device, constant_precomp);

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two9(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 9;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two12(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 12;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two15(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 15;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 15;
    const dim3 block_dim(32 * 2, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two18(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 18;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 18;
    const dim3 block_dim(32 * 2, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 15;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two21(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 21;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 21;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 18;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 15;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix8_two24(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 24;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 24;
    const dim3 block_dim(32 * 2, 1);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 21;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 18;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 15;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 9;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 6;
    const dim3 block_dim(8, 4 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 3;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 8 / block_dim.x, m / n / block_dim.y);
    // TODO: Adopt more optimized butterfly
    run_forward_scalar_iterative_within_subsequence_radix8<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

// MARK: Scalar, iterative, radix-16

template <int modulus_bits>
static void test_forward_scalar_iterative_radix16_two12(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 12;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 * 2, 1);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 8;
    const dim3 block_dim(16, 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 4;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix16_two16(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 16;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 16;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 8;
    const dim3 block_dim(16, 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 4;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix16_two20(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 20;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 20;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 16;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 8;
    const dim3 block_dim(16, 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 4;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

template <int modulus_bits>
static void test_forward_scalar_iterative_radix16_two24(
    const polyarith::Modulus &modulus,
    const precomputation::Precomputation<modulus_bits> *const precomp_device,
    const precomputation::ConstantPrecomputation<modulus_bits>
        &constant_precomp) {
  std::clog << "** " << __PRETTY_FUNCTION__ << std::endl;

  constexpr int m = 1 << 24;

  thrust::host_vector<std::uint64_t> a(m), b_correct(m);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<std::uint64_t> dist(0,
                                                    modulus.get_modulus() - 1);
  thrust::generate(a.begin(), a.end(), [&] { return dist(gen); });

  NttReference ntt_ref(m, modulus.get_modulus(), modulus.get_generator());
  ntt_ref.compute_forward(b_correct.data(), a.data());

  thrust::device_vector<std::uint64_t> b(a);

  {
    constexpr int n = 1 << 24;
    const dim3 block_dim(32, 1);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 20;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 16;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 12;
    const dim3 block_dim(32 / 2, 1 * 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 8;
    const dim3 block_dim(16, 2 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  {
    constexpr int n = 1 << 4;
    const dim3 block_dim(1, 32 * 2);
    const dim3 grid_dim(n / 16 / block_dim.x, m / n / block_dim.y);
    run_forward_scalar_iterative_within_subsequence_radix16<m, n>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(b.data()),
                                  precomp_device, constant_precomp);
  }

  const boost::icl::interval_set<int> mismatches =
      find_mismatches(b, b_correct, modulus.get_modulus());
  std::clog << "Mismatches: " << mismatches << std::endl;
}

// MARK: main

int main(const int argc, const char *const argv[]) {
  std::clog << "Built on " << __DATE__ << ' ' << __TIME__ << std::endl;

  if (argc != 1 + 1) {
    throw std::runtime_error("specify arguments");
  }
  int num_iters;
  std::istringstream(argv[1]) >> num_iters;

  /*
   * scalar_iterative does not support 62-bit moduli for now.
   */
  // const polyarith::Modulus modulus(UINT64_C(0x3fff'ffee'0000'0001), 3);
  const polyarith::Modulus modulus(UINT64_C(0x1fff'fff9'0000'0001), 3);
  // const polyarith::Modulus modulus(UINT64_C(0x0ffffc4900000001), 3);
  // const polyarith::Modulus modulus(UINT64_C(0x2b'0000'0001), 3);
  // const polyarith::Modulus modulus(UINT64_C(0xff'0000'0001), 13);
  constexpr int modulus_bits = 64;
  std::clog << "modulus = " << modulus.get_modulus() << std::endl;

  auto precomp_device = thrust::uninitialized_allocate_unique<
      precomputation::Precomputation<modulus_bits>>(
      thrust::device_allocator<precomputation::Precomputation<modulus_bits>>());
  const precomputation::Precomputation<modulus_bits> *const precomp_ptr =
      thrust::raw_pointer_cast(precomp_device.get());
  {
    const precomputation::Precomputation<modulus_bits> precomp(modulus);
    checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(precomp_device.get()),
                               &precomp, sizeof(precomp),
                               cudaMemcpyHostToDevice));
  }

  const precomputation::ConstantPrecomputation<modulus_bits> constant_precomp(
      modulus);

  test_forward_recursive_two4(modulus, precomp_ptr, constant_precomp);
  test_forward_recursive_two8(modulus, precomp_ptr, constant_precomp);

  test_forward_wmma_recursive_two4(modulus, precomp_ptr, constant_precomp);
  test_forward_wmma_recursive_two8(modulus, precomp_ptr, constant_precomp);

  test_forward_iterative_two8(modulus, precomp_ptr, constant_precomp,
                              (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_two12(modulus, precomp_ptr, constant_precomp,
                               (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_two16(modulus, precomp_ptr, constant_precomp,
                               (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_two20(modulus, precomp_ptr, constant_precomp,
                               (num_iters == 1) ? 1 : (num_iters << 8));
  test_forward_iterative_two24(modulus, precomp_ptr, constant_precomp,
                               (num_iters == 1) ? 1 : (num_iters << 4));
  test_forward_iterative_two28(modulus, precomp_ptr, constant_precomp,
                               (num_iters == 1) ? 1 : (num_iters << 0));

  test_forward_iterative_wmma_two8(modulus, precomp_ptr, constant_precomp,
                                   (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_wmma_two12(modulus, precomp_ptr, constant_precomp,
                                    (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_wmma_two16(modulus, precomp_ptr, constant_precomp,
                                    (num_iters == 1) ? 1 : (num_iters << 12));
  test_forward_iterative_wmma_two20(modulus, precomp_ptr, constant_precomp,
                                    (num_iters == 1) ? 1 : (num_iters << 8));
  test_forward_iterative_wmma_two24(modulus, precomp_ptr, constant_precomp,
                                    (num_iters == 1) ? 1 : (num_iters << 4));
  test_forward_iterative_wmma_two28(modulus, precomp_ptr, constant_precomp,
                                    (num_iters == 1) ? 1 : (num_iters << 0));

  test_forward_scalar_iterative_radix8_two3(modulus, precomp_ptr,
                                            constant_precomp);
  test_forward_scalar_iterative_radix8_two9(modulus, precomp_ptr,
                                            constant_precomp);
  test_forward_scalar_iterative_radix8_two12(modulus, precomp_ptr,
                                             constant_precomp);
  test_forward_scalar_iterative_radix8_two15(modulus, precomp_ptr,
                                             constant_precomp);
  test_forward_scalar_iterative_radix8_two18(modulus, precomp_ptr,
                                             constant_precomp);
  test_forward_scalar_iterative_radix8_two21(modulus, precomp_ptr,
                                             constant_precomp);
  test_forward_scalar_iterative_radix8_two24(modulus, precomp_ptr,
                                             constant_precomp);

  test_forward_scalar_iterative_radix16_two12(modulus, precomp_ptr,
                                              constant_precomp);
  test_forward_scalar_iterative_radix16_two16(modulus, precomp_ptr,
                                              constant_precomp);
  test_forward_scalar_iterative_radix16_two20(modulus, precomp_ptr,
                                              constant_precomp);
  test_forward_scalar_iterative_radix16_two24(modulus, precomp_ptr,
                                              constant_precomp);
}
