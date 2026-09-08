#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
shopt -s globstar nullglob

echo "Formatting Nix files..."
nix --extra-experimental-features "nix-command flakes" fmt .

echo "Formatting C/C++/CUDA files..."
FILES=(**/*.cu **/*.cuh **/*.cpp **/*.h **/*.c)

if [ ${#FILES[@]} -gt 0 ]; then
  clang-format -i "${FILES[@]}"
fi

echo "Done formatting."
