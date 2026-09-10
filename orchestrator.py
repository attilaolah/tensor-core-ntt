#!/usr/bin/env python3
"""Correctness-first CPU planner and GPU Fermat orchestrator.

The candidate plan is deliberately made on the CPU and passed verbatim to the
GPU executable.  GPU output is only a hint: a Pratt certificate is made on the
CPU before repository state is changed.
"""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import ModuleType

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
MASK64 = (1 << 64) - 1
candidate_plans: ModuleType | None = None
DATA_DIR = ROOT.parent / "data"
TIP_FILE = ROOT.parent / "TIP"


def candidate_plan(*args, **kwargs):
    """Compatibility proxy for planner-only users of this module."""
    return importlib.import_module("candidate_plan").candidate_plan(*args, **kwargs)


def plan_hash(*args, **kwargs):
    return importlib.import_module("candidate_plan").plan_hash(*args, **kwargs)


def read_plan(*args, **kwargs):
    return importlib.import_module("candidate_plan").read_plan(*args, **kwargs)


def write_plan(*args, **kwargs):
    return importlib.import_module("candidate_plan").write_plan(*args, **kwargs)


def _native_candidate_filter_path():
    """Return the configured helper, building the Nix output when necessary."""
    configured = os.environ.get("PRIMES_CANDIDATE_FILTER_LIB")
    if configured:
        library = Path(configured)
        if not library.is_file():
            raise OSError(f"PRIMES_CANDIDATE_FILTER_LIB does not name a shared library: {library}")
        return library
    try:
        output = subprocess.check_output(
            ["nix", "build", "--print-out-paths", ".#candidate-filter"],
            cwd=ROOT.parent, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OSError(
            "could not build Nix candidate-filter; install/configure Nix or set "
            "PRIMES_CANDIDATE_FILTER_LIB to libcandidate_filter"
        ) from error
    library_dir = Path(output) / "lib"
    library = next(
        (library_dir / f"libcandidate_filter{suffix}"
         for suffix in (".so", ".dylib", ".dll")
         if (library_dir / f"libcandidate_filter{suffix}").is_file()),
        None,
    )
    if library is None:
        raise OSError(f"Nix candidate-filter output has no shared library: {library_dir}")
    return library


def setup_candidate_filter():
    """Load the mandatory production filter before importing candidate_plan.

    Keeping this lazy lets protocol-only consumers import this module without a
    Nix build, while ``run`` invokes it before every plan read or generation.
    """
    global candidate_plans
    if candidate_plans is not None:
        return os.environ["PRIMES_CANDIDATE_FILTER_LIB"]
    library = _native_candidate_filter_path()
    os.environ["PRIMES_CANDIDATE_FILTER_LIB"] = str(library)
    # A planner-only caller may have imported candidate_plan before this GPU
    # run.  Reload so its import-time backend selection observes the native
    # library we configured above.
    candidate_plans = importlib.reload(importlib.import_module("candidate_plan"))
    if candidate_plans._CANDIDATE_FILTER is None:
        raise OSError(f"could not load native candidate filter: {library}")
    return str(library)


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


def certificate_documents(prime):
    """Build Pratt certificates in memory, so failed proofs leave no files."""
    if prime == 2:
        return {2: "V 1\nP 2\n"}
    factors = factor_distinct(prime - 1)
    proof = witness(prime, factors)
    if proof is None:
        raise ValueError("CPU Pratt verification failed")
    documents = {}
    for factor in factors:
        documents.update(certificate_documents(factor))
    counts = []
    for factor in factors:
        power, remaining = 0, prime - 1
        while remaining % factor == 0:
            power, remaining = power + 1, remaining // factor
        counts.append(f"F {factor}" + (f"^{power}" if power > 1 else ""))
    documents[prime] = "\n".join(["V 1", f"P {prime}", f"W {proof}", *counts]) + "\n"
    return documents


def certify_prime(prime, data_dir=DATA_DIR):
    for value, document in certificate_documents(prime).items():
        certificate_path(value, data_dir).write_text(document, encoding="ascii")


def certify_gpu_found(q, prime, bases, data_dir=DATA_DIR, tip_file=TIP_FILE):
    """Certify a GPU hit from its supplied, complete p - 1 factorization."""
    factors = [2, *bases, q]
    if len(bases) != 3 or prime - 1 != 2 * bases[0] * bases[1] * bases[2] * q:
        raise ValueError("GPU FOUND has incorrect known p - 1 factorization")
    if pow(2, prime - 1, prime) != 1:
        raise ValueError("GPU Fermat claim failed CPU verification")
    distinct = sorted(set(factors))
    proof = witness(prime, distinct)
    if proof is None:
        raise ValueError("CPU Pratt verification failed for GPU hit")

    # q is the sole unknown factor; never generically factor prime - 1.
    documents = certificate_documents(q)
    counts = []
    for factor in distinct:
        power = factors.count(factor)
        counts.append(f"F {factor}" + (f"^{power}" if power > 1 else ""))
    documents[prime] = "\n".join(["V 1", f"P {prime}", f"W {proof}", *counts]) + "\n"
    for value, document in documents.items():
        certificate_path(value, data_dir).write_text(document, encoding="ascii")
    tip_file.write_text(certificate_path(prime, data_dir).name + "\n", encoding="ascii")


class ProtocolValidator:
    """Validate ordered GPU records, invoking ``on_found`` before more work runs."""

    def __init__(self, bases, candidates, on_found=None):
        self.bases, self.candidates, self.on_found = bases, candidates, on_found
        self.hits, self.done, self.tested, self.planned = [], False, [], False
        self.certified = set()

    @staticmethod
    def _fields(line, record):
        parts = line.split()
        if not parts or parts[0] != record:
            raise ValueError(f"malformed {record} record")
        fields = {}
        for part in parts[1:]:
            if part.count("=") != 1:
                raise ValueError(f"malformed {record} record")
            key, value = part.split("=", 1)
            if not key or not value or key in fields:
                raise ValueError(f"malformed {record} record")
            fields[key] = value
        return fields

    def process(self, line):
        if self.done:
            raise ValueError("GPU emitted a record after DONE")
        if line.startswith("PLAN "):
            fields = self._fields(line, "PLAN")
            if fields != {"count": str(len(self.candidates))} or self.planned or self.tested:
                raise ValueError("PLAN does not match ordered candidate plan")
            self.planned = True
        elif line == "DONE":
            if not self.planned or self.tested != self.candidates:
                raise ValueError("GPU emitted DONE before the complete ordered candidate plan")
            self.done = True
        elif line.startswith("TEST "):
            if not self.planned:
                raise ValueError("GPU emitted TEST before PLAN")
            fields = self._fields(line, "TEST")
            if set(fields) != {"index", "q"}:
                raise ValueError("malformed TEST record")
            index, q = int(fields["index"]), int(fields["q"])
            if index != len(self.tested) or index >= len(self.candidates) or q != self.candidates[index]:
                raise ValueError("TEST does not match ordered candidate plan")
            self.tested.append(q)
        elif line.startswith("FOUND "):
            if not self.planned:
                raise ValueError("GPU emitted FOUND before PLAN")
            fields = self._fields(line, "FOUND")
            if set(fields) != {"index", "q", "p"}:
                raise ValueError("malformed FOUND record")
            index, q, prime = int(fields["index"]), int(fields["q"]), int(fields["p"])
            if not 0 <= index < len(self.candidates) or q != self.candidates[index] or index >= len(self.tested):
                raise ValueError("FOUND does not match ordered candidate plan")
            if prime != 2 * self.bases[0] * self.bases[1] * self.bases[2] * q + 1:
                raise ValueError("FOUND has incorrect constructed p")
            hit = (q, prime)
            self.hits.append(hit)
            if hit not in self.certified:
                if self.on_found is not None:
                    self.on_found(q, prime)
                self.certified.add(hit)
        elif line.startswith(("PLAN", "TEST", "FOUND", "DONE")):
            raise ValueError("malformed GPU protocol record")

    def finish(self):
        if not self.planned:
            raise ValueError("GPU did not emit PLAN")
        if not self.done:
            raise ValueError("GPU did not emit DONE")
        if self.tested != self.candidates:
            raise ValueError("GPU did not test the complete ordered candidate plan")
        return self.hits


def validate_protocol(output, bases, candidates):
    validator = ProtocolValidator(bases, candidates)
    for line in output.splitlines():
        if line.startswith(("PLAN", "TEST", "FOUND", "DONE")):
            validator.process(line)
    return validator.finish()


def _tee_stream(source, destination, captured):
    """Forward raw child bytes immediately while retaining them for validation."""
    reader = getattr(source, "read1", source.read)
    while chunk := reader(64 * 1024):
        captured.extend(chunk)
        destination.write(chunk)
        destination.flush()


def _stream_stdout(source, destination, captured, validator, failed, process):
    """Hide complete protocol lines while forwarding all non-protocol bytes."""
    reader, pending, passthrough = getattr(source, "read1", source.read), bytearray(), False
    prefixes = (b"PLAN", b"TEST", b"FOUND", b"DONE")
    try:
        while chunk := reader(64 * 1024):
            captured.extend(chunk)
            for byte in chunk:
                if passthrough:
                    destination.write(bytes((byte,)))
                    if byte in (ord("\r"), ord("\n")):
                        destination.flush()
                    if byte in (ord("\r"), ord("\n")):
                        passthrough = False
                    continue
                pending.append(byte)
                if byte == ord("\n"):
                    line = pending[:-1].decode("ascii")
                    validator.process(line)
                    pending.clear()
                elif not any(prefix.startswith(pending) or pending.startswith(prefix) for prefix in prefixes):
                    destination.write(pending)
                    if ord("\r") in pending:
                        destination.flush()
                    pending.clear()
                    passthrough = True
            destination.flush()
        if pending:
            destination.write(pending)
            destination.flush()
    except Exception as error:
        failed.append(error)
        try:
            process.terminate()
        except ProcessLookupError:
            pass


def run_streamed(command, bases=None, candidates=None, on_found=None):
    """Run a child with live, byte-for-byte output and collected protocol text."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    stdout, stderr = bytearray(), bytearray()
    stdout_sink = getattr(sys.stdout, "buffer", sys.stdout)
    stderr_sink = getattr(sys.stderr, "buffer", sys.stderr)
    validator = ProtocolValidator(bases, candidates, on_found) if bases is not None and candidates is not None else None
    failed = []
    stdout_target = _stream_stdout if validator is not None else _tee_stream
    stdout_args = (process.stdout, stdout_sink, stdout, validator, failed, process) if validator else (process.stdout, stdout_sink, stdout)
    threads = [threading.Thread(target=stdout_target, args=stdout_args), threading.Thread(target=_tee_stream, args=(process.stderr, stderr_sink, stderr))]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    process.stdout.close()
    process.stderr.close()
    if failed:
        raise failed[0]
    if validator is not None:
        validator.finish()
    return returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


def run(args):
    library = setup_candidate_filter()
    assert candidate_plans is not None
    plans = candidate_plans
    print(f"[*] Candidate filter backend: native ({library})", flush=True)
    selected = selected_primes(max_digits=args.max_digits)
    if len(selected) != 3:
        raise ValueError("not enough eligible data certificates")
    digest = None
    if args.candidate_plan:
        plan = plans.read_plan(Path(args.candidate_plan))
        if plan["max_digits"] != args.max_digits or plan["sieve_limit"] != args.sieve_limit or plan["seed"] != args.seed or tuple(selected) != plan["bases"]:
            raise ValueError("candidate plan metadata or bases do not match requested search")
        bases, candidates, digest = plan["bases"], plan["candidates"], plan["hash"]
    else:
        bases = selected
        candidates = plans.candidate_plan(bases, args.sieve_limit, args.seed)
    digest = digest or plans.plan_hash(bases, args.sieve_limit, args.seed, args.max_digits, candidates)
    if args.write_candidate_plan:
        digest = plans.write_plan(Path(args.write_candidate_plan), bases, args.sieve_limit, args.seed, args.max_digits, candidates)
    print("[*] GPU Dynamic Fermat Search Orchestrator Started")
    print(f"[+] Found Base Primes: {len(str(bases[0]))} digits, {len(str(bases[1]))} digits, {len(str(bases[2]))} digits")
    print(f"[*] Built Sieve of Eratosthenes up to {args.sieve_limit}; {len(candidates)} candidates survive")
    print(f"[*] Seed: {args.seed}; plan hash: {digest}; candidate count: {len(candidates)}", flush=True)
    if args.write_candidate_plan:
        print(f"[*] Wrote candidate plan: {args.write_candidate_plan}")
    print("[*] Starting ordered GPU Fermat search...", flush=True)
    binary = Path(args.gpu_binary) if args.gpu_binary else ROOT / "result/bin/crunch_sm_86"
    if not args.gpu_binary:
        subprocess.run(["nix", "build"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="gpu-plan-") as directory:
        directory = Path(directory)
        base_file, candidate_file = directory / "bases", directory / "candidates"
        base_file.write_text("\n".join(map(str, bases)) + "\n", encoding="ascii")
        candidate_file.write_text("\n".join(map(str, candidates)) + ("\n" if candidates else ""), encoding="ascii")
        def certify_found(q, prime):
            certify_gpu_found(q, prime, bases)
            path = certificate_path(prime)
            print(f"[+] CPU-certified GPU hit: {path} ({len(str(prime))} digits); updated TIP", flush=True)

        returncode, stdout, stderr = run_streamed(
            [str(binary), "--ordered", str(base_file), str(candidate_file)], bases, candidates, certify_found
        )
    if returncode:
        raise RuntimeError(f"GPU program failed ({returncode}): {stderr.strip()}")


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
