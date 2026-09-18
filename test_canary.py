#!/usr/bin/env python3
"""Exercise the GPU Fermat pipeline with a randomly generated 1000-digit prime."""

import secrets
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BITS = 3320  # Every 3320-bit integer has exactly 1000 decimal digits.
MR_ROUNDS = 40


def is_probable_prime(value: int) -> bool:
    """Run Miller-Rabin rounds with independently sampled bases."""
    if value < 2:
        return False
    for small_prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % small_prime == 0:
            return value == small_prime

    exponent, twos = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        twos += 1
    for _ in range(MR_ROUNDS):
        base = secrets.randbelow(value - 3) + 2
        residue = pow(base, exponent, value)
        if residue in (1, value - 1):
            continue
        for _ in range(twos - 1):
            residue = pow(residue, 2, value)
            if residue == value - 1:
                break
        else:
            return False
    return True


def random_prime() -> int:
    """Generate a high-confidence 1000-digit prime using only the stdlib."""
    while True:
        candidate = secrets.randbits(BITS) | (1 << (BITS - 1)) | 1
        if is_probable_prime(candidate):
            assert len(str(candidate)) == 1000
            return candidate


def main() -> int:
    prime = random_prime()
    subprocess.run(["nix", "build"], cwd=ROOT, check=True)
    binary = ROOT / "result/bin/crunch_sm_86"
    with tempfile.NamedTemporaryFile("w", encoding="ascii") as input_file:
        input_file.write(f"{prime}\n")
        input_file.flush()
        result = subprocess.run(
            [
                str(binary),
                "--file",
                input_file.name,
                "--target-bits",
                str(prime.bit_length()),
                "--phase",
                "2",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    print(result.stdout, end="")
    assert result.returncode == 0, "GPU Fermat execution failed"
    assert "FERMAT PASS" in result.stdout, "GPU rejected a generated 1000-digit prime"
    print(f"CANARY PASS digits={len(str(prime))} bits={prime.bit_length()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
