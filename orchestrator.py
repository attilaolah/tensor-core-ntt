#!/usr/bin/env python3
"""Correctness-first CPU planner and GPU Fermat orchestrator.

The candidate plan is deliberately made on the CPU and passed verbatim to the
GPU executable.  GPU output is only a hint: a Pratt certificate is made on the
CPU before repository state is changed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from candidate_plan import MASK64, candidate_plan, read_plan, write_plan
DATA_DIR = ROOT.parent / "data"
TIP_FILE = ROOT.parent / "TIP"


def selected_primes(data_dir=DATA_DIR, max_digits=None):
    values = set()
    for path in data_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("P "):
                text = line[2:]
                if text.isdecimal() and (max_digits is None or len(text) <= max_digits):
                    values.add(int(text))
                break
    return sorted(values, reverse=True)[:3]


def certificate_path(prime, data_dir=DATA_DIR):
    return data_dir / hashlib.sha256(str(prime).encode("ascii")).hexdigest()[:16]


def factor_distinct(number):
    factors, divisor = [], 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            factors.append(divisor)
            while number % divisor == 0:
                number //= divisor
        divisor += 1 if divisor == 2 else 2
    return factors + ([number] if number > 1 else [])


def witness(prime, factors):
    for value in range(2, prime):
        if pow(value, prime - 1, prime) == 1 and all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors):
            return value
    return None


def certify_prime(prime, data_dir=DATA_DIR):
    if prime == 2:
        certificate_path(2, data_dir).write_text("V 1\nP 2\n", encoding="ascii")
        return
    factors = factor_distinct(prime - 1)
    proof = witness(prime, factors)
    if proof is None:
        raise ValueError("CPU Pratt verification failed")
    for factor in factors:
        certify_prime(factor, data_dir)
    counts = []
    for factor in factors:
        power, remaining = 0, prime - 1
        while remaining % factor == 0:
            power, remaining = power + 1, remaining // factor
        counts.append(f"F {factor}" + (f"^{power}" if power > 1 else ""))
    certificate_path(prime, data_dir).write_text("\n".join(["V 1", f"P {prime}", f"W {proof}", *counts]) + "\n", encoding="ascii")


def validate_protocol(output, bases, candidates):
    hits, done, tested = [], False, []
    for line in output.splitlines():
        if line == "DONE":
            done = True
        elif line.startswith("TEST "):
            fields = dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)
            if set(fields) != {"index", "q"}:
                raise ValueError("malformed TEST record")
            index, q = int(fields["index"]), int(fields["q"])
            if index != len(tested) or index >= len(candidates) or q != candidates[index]:
                raise ValueError("TEST does not match ordered candidate plan")
            tested.append(q)
        elif line.startswith("FOUND "):
            fields = dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)
            if set(fields) != {"index", "q", "p"}:
                raise ValueError("malformed FOUND record")
            index, q, prime = int(fields["index"]), int(fields["q"]), int(fields["p"])
            if not 0 <= index < len(candidates) or q != candidates[index]:
                raise ValueError("FOUND does not match ordered candidate plan")
            if prime != 2 * bases[0] * bases[1] * bases[2] * q + 1:
                raise ValueError("FOUND has incorrect constructed p")
            hits.append((q, prime))
    if not done:
        raise ValueError("GPU did not emit DONE")
    if tested != candidates:
        raise ValueError("GPU did not test the complete ordered candidate plan")
    return hits


def run(args):
    selected = selected_primes(max_digits=args.max_digits)
    if len(selected) != 3:
        raise ValueError("not enough eligible data certificates")
    digest = None
    if args.candidate_plan:
        plan = read_plan(Path(args.candidate_plan))
        if plan["max_digits"] != args.max_digits or plan["sieve_limit"] != args.sieve_limit or plan["seed"] != args.seed or tuple(selected) != plan["bases"]:
            raise ValueError("candidate plan metadata or bases do not match requested search")
        bases, candidates, digest = plan["bases"], plan["candidates"], plan["hash"]
    else:
        bases = selected
        candidates = candidate_plan(bases, args.sieve_limit, args.seed)
    if args.write_candidate_plan:
        digest = write_plan(Path(args.write_candidate_plan), bases, args.sieve_limit, args.seed, args.max_digits, candidates)
    print("[*] GPU Dynamic Fermat Search Orchestrator Started")
    print(f"[+] Found Base Primes: {len(str(bases[0]))} digits, {len(str(bases[1]))} digits, {len(str(bases[2]))} digits")
    print(f"[*] Built Sieve of Eratosthenes up to {args.sieve_limit}; {len(candidates)} candidates survive")
    print(f"[*] Seed: {args.seed}; plan hash: {digest or 'ephemeral'}")
    if args.write_candidate_plan:
        print(f"[*] Wrote candidate plan: {args.write_candidate_plan}")
    print("[*] Starting ordered GPU Fermat search...")
    binary = Path(args.gpu_binary) if args.gpu_binary else ROOT / "result/bin/crunch_sm_86"
    if not args.gpu_binary:
        subprocess.run(["nix", "build"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="gpu-plan-") as directory:
        directory = Path(directory)
        base_file, candidate_file = directory / "bases", directory / "candidates"
        base_file.write_text("\n".join(map(str, bases)) + "\n", encoding="ascii")
        candidate_file.write_text("\n".join(map(str, candidates)) + ("\n" if candidates else ""), encoding="ascii")
        result = subprocess.run([str(binary), "--ordered", str(base_file), str(candidate_file)], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"GPU program failed ({result.returncode}): {result.stderr.strip()}")
    hits = validate_protocol(result.stdout, bases, candidates)
    for q, prime in hits:
        # This independently proves primality (including q) before any TIP update.
        certify_prime(prime)
        TIP_FILE.write_text(certificate_path(prime).name + "\n", encoding="ascii")
        print(f"[+] CPU-certified GPU hit q={q}; updated TIP")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-digits", type=int)
    parser.add_argument("--sieve-limit", type=int, default=5_000_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-binary")
    parser.add_argument("--candidate-plan", help="consume a persisted plan before invoking crunch --ordered")
    parser.add_argument("--write-candidate-plan", help="persist the selected bases and ordered candidates")
    args = parser.parse_args()
    if args.max_digits is not None and args.max_digits <= 0:
        parser.error("--max-digits must be a positive integer")
    if args.sieve_limit <= 0:
        parser.error("--sieve-limit must be a positive integer")
    if not 0 <= args.seed <= MASK64:
        parser.error("--seed must be an unsigned 64-bit integer")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
