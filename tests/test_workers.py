import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "gpu_orchestrator", Path(__file__).parents[1] / "orchestrator.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load GPU orchestrator")
orchestrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orchestrator)


def test_protocol_accepts_out_of_order_worker_records():
    bases = (3, 5, 7)
    candidates = [11, 13]
    p = 2 * 3 * 5 * 7 * 13 + 1
    output = f"PLAN count=2\nTEST index=1 q=13\nTEST index=0 q=11\nFOUND index=1 q=13 p={p}\nDONE\n"
    assert orchestrator.validate_protocol(output, bases, candidates) == [(13, p)]


def test_protocol_rejects_duplicate_worker_test():
    validator = orchestrator.ProtocolValidator((3, 5, 7), [11])
    validator.process("PLAN count=1")
    validator.process("TEST index=0 q=11")
    try:
        validator.process("TEST index=0 q=11")
    except ValueError:
        return
    raise AssertionError("duplicate TEST was accepted")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_workers_must_be_positive(value):
    with pytest.raises(SystemExit):
        orchestrator.parse_args(["--workers", value])


def test_progress_scheduler_allows_any_worker_to_claim_each_tick():
    source = (Path(__file__).parents[1] / "src" / "crunch.cu").read_text()

    scheduler = source[source.index("class ProgressLogSchedule") : source.index("auto format_duration")]
    assert "claim_nonfinal_slot()" in scheduler
    assert "claim_tick(tick)" in scheduler
    assert "compare_exchange_weak" in scheduler
    assert "tick % worker_count" not in scheduler


def test_progress_scheduler_is_polled_more_often_than_percentage_updates():
    source = (Path(__file__).parents[1] / "src" / "crunch.cu").read_text()

    assert "constexpr int progress_poll_steps = 8;" in source
    assert "squarings % progress_poll_steps == 0" in source
