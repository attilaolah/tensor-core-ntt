# [`tensor-core-ntt`](https://github.com/Terminus-IMRC/tensor-core-ntt)

This repository provides an improved implementation of the [number theoretic
transform
(NTT)](https://en.wikipedia.org/wiki/Discrete_Fourier_transform_over_a_ring)
that leverages Tensor Cores on NVIDIA GPUs.

NTT is a generalization of the [fast Fourier transform
(FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform) over modular
integers, and it is used to perform convolutions and
polynomial/multiple-precision arithmetic exactly, in contrast to FFT, which uses
floating-point arithmetic and thus suffers from rounding errors.

Through careful analysis, we successfully removed the redundant operations in
the previous Tensor Core-based NTT implementations.
Along with efficient modular reduction algorithm, we achieved higher performance
than the previous work in most settings.

See the paper by Y. Sugizaki and D. Takahashi titled "Improved Implementation of
Number Theoretic Transform on NVIDIA GPU with Tensor Cores"
(doi:[10.1145/3773656.3773673](https://doi.org/10.1145/3773656.3773673)), which
was accepted to [SupercomputingAsia 2026 / International Conference on High
Performance Computing in Asia-Pacific Region 2026 (SCA/HPCAsia
2026)](https://www.sca-hpcasia2026.jp/), for more details.


## License and contribution

For license and copyright notices, see the SPDX file tags in each file.
Unless otherwise noted, files in this project are licensed under the Apache
License, Version 2.0 (SPDX short-form identifier: Apache-2.0) and copyrighted by
the contributors.

Everyone is encouraged to contribute to this project.
See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions.
