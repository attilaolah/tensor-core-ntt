import importlib.util
from pathlib import Path
import tempfile
import io
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from candidate_plan import read_plan, write_plan
SPEC = importlib.util.spec_from_file_location("orchestrator", ROOT / "orchestrator.py")
assert SPEC is not None and SPEC.loader is not None
orch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orch)


class OrchestratorTests(unittest.TestCase):
    def test_plan_is_seed_deterministic(self):
        bases = [3, 5, 7]
        self.assertEqual(orch.candidate_plan(bases, 50, 12), orch.candidate_plan(bases, 50, 12))
        self.assertNotEqual(orch.candidate_plan(bases, 50, 12), orch.candidate_plan(bases, 50, 13))

    def test_top_primes_apply_digit_filter_to_each_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for name, prime in enumerate([99991, 999, 97, 89]):
                (data / str(name)).write_text(f"V 1\nP {prime}\n", encoding="ascii")
            self.assertEqual(orch.selected_primes(data, 3), [999, 97, 89])

    def test_protocol_recomputes_ordered_candidate(self):
        bases, candidates = [3, 5, 7], [11, 13]
        prime = 2 * 3 * 5 * 7 * 13 + 1
        self.assertEqual(orch.validate_protocol(f"PLAN count=2\nTEST index=0 q=11\nTEST index=1 q=13\nFOUND index=1 q=13 p={prime}\nDONE\n", bases, candidates), [(13, prime)])
        with self.assertRaises(ValueError):
            orch.validate_protocol("PLAN count=2\nFOUND index=0 q=13 p=1\nDONE\n", bases, candidates)

    def test_tee_stream_preserves_carriage_returns(self):
        source, destination, captured = io.BytesIO(b"CUDA 10%\rCUDA 20%\r"), io.BytesIO(), bytearray()
        orch._tee_stream(source, destination, captured)
        self.assertEqual(destination.getvalue(), b"CUDA 10%\rCUDA 20%\r")
        self.assertEqual(bytes(captured), destination.getvalue())

    def test_streamed_child_output_is_teed_and_validated(self):
        output = "PLAN count=1\nCUDA 10%\\r\nTEST index=0 q=11\nDONE\n"
        stdout, stderr = io.BytesIO(), io.BytesIO()
        with mock.patch.object(orch.sys, "stdout", mock.Mock(buffer=stdout)), mock.patch.object(orch.sys, "stderr", mock.Mock(buffer=stderr)):
            returncode, protocol, errors = orch.run_streamed([sys.executable, "-c", f"import sys; sys.stdout.buffer.write({output.encode()!r})"])
        self.assertEqual(returncode, 0)
        self.assertEqual(errors, "")
        self.assertEqual(stdout.getvalue(), output.encode())
        self.assertEqual(orch.validate_protocol(protocol, [3, 5, 7], [11]), [])

    def test_persisted_plan_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan"
            digest = write_plan(path, [3, 5, 7], 50, 12, 2, orch.candidate_plan([3, 5, 7], 50, 12))
            self.assertEqual(read_plan(path)["hash"], digest)
            path.write_text(path.read_text(encoding="ascii") + "2\n", encoding="ascii")
            with self.assertRaises(ValueError):
                read_plan(path)

    def test_persisted_plan_rejects_rehashed_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan"
            # write_plan deliberately only serializes; readers must establish
            # that seed-backed metadata actually derives this sequence.
            write_plan(path, [3, 5, 7], 50, 12, 2, [2, 3, 5])
            with self.assertRaises(ValueError):
                read_plan(path)

    def test_gpu_found_writes_canonical_certificate_without_factoring_p_minus_one(self):
        # p = 2 * 3 * 5 * 7 * 11 + 1 = 2311.  The test rejects accidental use
        # of the generic p certificate path, which would factor p - 1.
        bases, q = [3, 5, 7], 11
        prime = 2 * 3 * 5 * 7 * q + 1
        with tempfile.TemporaryDirectory() as directory:
            data, tip = Path(directory) / "data", Path(directory) / "TIP"
            data.mkdir()
            original_factor_distinct = orch.factor_distinct
            with mock.patch.object(
                orch,
                "factor_distinct",
                side_effect=lambda value: self.fail("factored p - 1") if value == prime - 1 else original_factor_distinct(value),
            ):
                orch.certify_gpu_found(q, prime, bases, data, tip)
            self.assertEqual(
                orch.certificate_path(prime, data).read_text(encoding="ascii"),
                "V 1\nP 2311\nW 3\nF 2\nF 3\nF 5\nF 7\nF 11\n",
            )
            self.assertEqual(tip.read_text(encoding="ascii"), orch.certificate_path(prime, data).name + "\n")

    def test_invalid_gpu_found_does_not_write_data_or_tip(self):
        with tempfile.TemporaryDirectory() as directory:
            data, tip = Path(directory) / "data", Path(directory) / "TIP"
            data.mkdir()
            with self.assertRaises(ValueError):
                orch.certify_gpu_found(11, 2312, [3, 5, 7], data, tip)
            self.assertEqual(list(data.iterdir()), [])
            self.assertFalse(tip.exists())


if __name__ == "__main__":
    unittest.main()
