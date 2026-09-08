#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob

echo "Running clang-tidy --fix..."
FILES=(**/*.cu **/*.cuh **/*.cpp **/*.h **/*.c)

if [ ${#FILES[@]} -gt 0 ]; then
  clang-tidy -p . --fix-errors "${FILES[@]}" -- -Iinclude -x cuda --cuda-gpu-arch=sm_86 -std=c++20
fi
