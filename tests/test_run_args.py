"""run_graph.parse_args — guards the --image parsing bug."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_graph", ROOT / "scripts" / "run_graph.py")
run_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_graph)
parse_args = run_graph.parse_args
DEFAULT_Q = run_graph.DEFAULT_Q


def test_image_only_uses_default_question():
    q, img = parse_args(["--image", "data/sample_dashboard.png"])
    assert q == DEFAULT_Q and img == "data/sample_dashboard.png"


def test_question_only():
    q, img = parse_args(["my question"])
    assert q == "my question" and img is None


def test_question_then_image():
    q, img = parse_args(["my question", "--image", "x.png"])
    assert q == "my question" and img == "x.png"


def test_image_then_question_order_independent():
    q, img = parse_args(["--image", "x.png", "my question"])
    assert q == "my question" and img == "x.png"
