#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob

echo "Running clang-tidy..."
# Collect all relevant C/C++/CUDA files using globstar
FILES=(**/*.cu **/*.cuh **/*.cpp **/*.h **/*.c)

if [ ${#FILES[@]} -gt 0 ]; then
  clang-tidy -p . "${FILES[@]}" -- -Iinclude -x cuda --cuda-gpu-arch=sm_86 -std=c++20
fi

echo "Running nix build to check strict compilation (-Werror -Wall -Wextra)..."
nix --extra-experimental-features "nix-command flakes" build

echo "All linting and strict builds passed!"
