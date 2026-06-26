"""The eval harness runs offline and catches a regression."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_eval_passes_and_detects_regression():
    rows = run_eval.load_dataset()
    assert len(rows) >= 8
    healthy = run_eval.evaluate(rows, threshold=run_eval.LEX_THRESHOLD)
    degraded = run_eval.evaluate(rows, threshold=0.0)
    assert healthy["accuracy"] >= 0.8
    assert healthy["cites_source_pass"]
    assert degraded["accuracy"] < healthy["accuracy"], "regression must be detectable"


def test_eval_main_exits_zero():
    assert run_eval.main() == 0
