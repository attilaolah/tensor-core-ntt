namespace precomputation {

template <int modulus_bits_> class ConstantPrecomputation {
public:
  constexpr static int modulus_bits = modulus_bits_;

  using reduction_type =
      polyarith::modular::MontgomeryFriendlyReductionBy64<modulus_bits>;

  __align__(16) uint64_t modulus;
  __align__(16) reduction_type friendly_reduction;

  ConstantPrecomputation() = default;

  explicit ConstantPrecomputation(const polyarith::Modulus &modulus)

      : modulus(modulus.get_modulus()),
        friendly_reduction(modulus.get_modulus()) {}
};

/* Warning: Do not allocate this on stack. */

template <int modulus_bits_> class Precomputation {
public:
  constexpr static int modulus_bits = modulus_bits_;

  using reduction_type = ConstantPrecomputation<modulus_bits>::reduction_type;

  __align__(16) polyarith::cuda::NttForward16x16Coalesced<
      reduction_type> ntt_forward_16x16;
  __align__(32) polyarith::cuda::NttForwardWmma16x16<
      reduction_type> ntt_forward_wmma_16x16;

  __align__(16) polyarith::cuda::NttForwardTwiddle16x16Coalesced
      ntt_forward_twiddle_16x16;
  __align__(16) polyarith::cuda::NttForwardTwiddleWmma16x16
      ntt_forward_twiddle_wmma_16x16;

  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeCoalesced<
      (1 << 12)> ntt_forward_twiddle_iterative_two12;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeCoalesced<
      (1 << 16)> ntt_forward_twiddle_iterative_two16;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeCoalesced<
      (1 << 20)> ntt_forward_twiddle_iterative_two20;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeCoalesced<
      (1 << 24)> ntt_forward_twiddle_iterative_two24;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeCoalesced<
      (1 << 28)> ntt_forward_twiddle_iterative_two28;

  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeWmma<
      (1 << 12)> ntt_forward_twiddle_iterative_wmma_two12;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeWmma<
      (1 << 16)> ntt_forward_twiddle_iterative_wmma_two16;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeWmma<
      (1 << 20)> ntt_forward_twiddle_iterative_wmma_two20;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeWmma<
      (1 << 24)> ntt_forward_twiddle_iterative_wmma_two24;
  __align__(16) polyarith::cuda::NttForwardTwiddleIterativeWmma<
      (1 << 28)> ntt_forward_twiddle_iterative_wmma_two28;

  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 3), 8> ntt_forward_scalar_iterative_radix8_two3;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 6), 8> ntt_forward_scalar_iterative_radix8_two6;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 9), 8> ntt_forward_scalar_iterative_radix8_two9;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 12), 8> ntt_forward_scalar_iterative_radix8_two12;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 15), 8> ntt_forward_scalar_iterative_radix8_two15;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 18), 8> ntt_forward_scalar_iterative_radix8_two18;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 21), 8> ntt_forward_scalar_iterative_radix8_two21;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 24), 8> ntt_forward_scalar_iterative_radix8_two24;

  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 4), 16> ntt_forward_scalar_iterative_radix16_two4;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 8), 16> ntt_forward_scalar_iterative_radix16_two8;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 12), 16> ntt_forward_scalar_iterative_radix16_two12;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 16), 16> ntt_forward_scalar_iterative_radix16_two16;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 20), 16> ntt_forward_scalar_iterative_radix16_two20;
  __align__(16) polyarith::cuda::NttForwardScalarIterative<
      (1 << 24), 16> ntt_forward_scalar_iterative_radix16_two24;

  Precomputation() = default;

  explicit Precomputation(const polyarith::Modulus &modulus)
      : ntt_forward_16x16(modulus), ntt_forward_wmma_16x16(modulus),
        ntt_forward_twiddle_16x16(modulus),
        ntt_forward_twiddle_wmma_16x16(modulus),
        ntt_forward_twiddle_iterative_two12(modulus),
        ntt_forward_twiddle_iterative_two16(modulus),
        ntt_forward_twiddle_iterative_two20(modulus),
        ntt_forward_twiddle_iterative_two24(modulus),
        ntt_forward_twiddle_iterative_two28(modulus),
        ntt_forward_twiddle_iterative_wmma_two12(modulus),
        ntt_forward_twiddle_iterative_wmma_two16(modulus),
        ntt_forward_twiddle_iterative_wmma_two20(modulus),
        ntt_forward_twiddle_iterative_wmma_two24(modulus),
        ntt_forward_twiddle_iterative_wmma_two28(modulus),
        ntt_forward_scalar_iterative_radix8_two3(modulus),
        ntt_forward_scalar_iterative_radix8_two6(modulus),
        ntt_forward_scalar_iterative_radix8_two9(modulus),
        ntt_forward_scalar_iterative_radix8_two12(modulus),
        ntt_forward_scalar_iterative_radix8_two15(modulus),
        ntt_forward_scalar_iterative_radix8_two18(modulus),
        ntt_forward_scalar_iterative_radix8_two21(modulus),
        ntt_forward_scalar_iterative_radix8_two24(modulus),
        ntt_forward_scalar_iterative_radix16_two4(modulus),
        ntt_forward_scalar_iterative_radix16_two8(modulus),
        ntt_forward_scalar_iterative_radix16_two12(modulus),
        ntt_forward_scalar_iterative_radix16_two16(modulus),
        ntt_forward_scalar_iterative_radix16_two20(modulus),
        ntt_forward_scalar_iterative_radix16_two24(modulus) {}
};

} // namespace precomputation
