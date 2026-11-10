"""Generated packages and probes never unpickle .pt files by default.

`torch.load(weights_only=False)` unpickles arbitrary objects, and unpickling
runs code stored in the file. The paradigm data templates used it for user
datasets in example_data/ and for their own _cache/ files, and the
scale-mismatch probe used it inside the driver process. A planted .pt would
then execute with the user's privileges.

The fix lives in the producers (templates, probe), not in run output:
  - tensor-only formats load with weights_only=True;
  - detection formats store images as byte records, which load safely,
    and the full-unpickle path is gated behind an explicit R2C_TRUST_PICKLE=1.
"""

from __future__ import annotations

import importlib.util
import pickle
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTION = (
    REPO_ROOT / "paradigms/supervised_ml/knowledge_distillation/detection"
)
DETECTION_TEMPLATE = DETECTION / "templates/method/data.py.template"
CROSS_MODAL_TEMPLATE = DETECTION / "cross_modal/templates/method/data.py.template"

_UNSAFE = re.compile(r"weights_only\s*=\s*False")
_GATE = 'if _env_flag("R2C_TRUST_PICKLE")'


def _producer_files() -> list[Path]:
    files = list((REPO_ROOT / "paradigms").rglob("*.template"))
    files += list((REPO_ROOT / "scripts").rglob("*.py"))
    return [f for f in files if f.is_file()]


def _ungated_unsafe_loads(text: str) -> list[int]:
    """Line numbers of weights_only=False calls not directly behind the gate."""
    lines = text.splitlines()
    bad = []
    for i, line in enumerate(lines):
        code = line.split("#", 1)[0]
        if not _UNSAFE.search(code):
            continue
        window = "\n".join(lines[max(0, i - 2):i])
        if _GATE not in window:
            bad.append(i + 1)
    return bad


def test_no_producer_unpickles_without_the_trust_gate():
    offenders = {
        str(f.relative_to(REPO_ROOT)): lines
        for f in _producer_files()
        if (lines := _ungated_unsafe_loads(f.read_text(encoding="utf-8")))
    }
    assert offenders == {}


def test_gate_check_flags_a_known_bad_load():
    bad = "def f(p):\n    return torch.load(p, weights_only=False)\n"
    good = (
        "    except pickle.UnpicklingError as e:\n"
        '        if _env_flag("R2C_TRUST_PICKLE"):\n'
        "            return torch.load(path, weights_only=False)\n"
    )
    assert _ungated_unsafe_loads(bad) == [2]
    assert _ungated_unsafe_loads(good) == []


# ---------------------------------------------------------------------------
# Runtime behaviour of the detection template (needs torch + Pillow)
# ---------------------------------------------------------------------------


def _load_template(path: Path, name: str):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class _Boom:
    """Pickles to a call that would run during unpickling."""

    def __reduce__(self):
        return (_record_side_effect, ("ran",))


_SIDE_EFFECTS: list[str] = []


def _record_side_effect(value):
    _SIDE_EFFECTS.append(value)
    return value


@pytest.fixture
def detection():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    return torch, _load_template(DETECTION_TEMPLATE, "r2c_det_data_template")


def test_synthetic_cache_round_trips_through_weights_only(detection, tmp_path):
    torch, mod = detection
    first = mod.load_data(tmp_path, train_size=3, n_test=2, seed=1)
    cache = tmp_path / "_cache" / "synthetic_detection.pt"
    assert cache.is_file()
    # The cache itself must load under the safe loader.
    raw = torch.load(cache, weights_only=True)
    assert isinstance(raw["images_train"][0], dict)

    second = mod.load_data(tmp_path, train_size=3, n_test=2, seed=1)
    assert len(second[0]) == 3 and len(second[2]) == 2
    for a, b in zip(first[0], second[0]):
        assert a.tobytes() == b.tobytes() and a.size == b.size
    assert torch.equal(first[1][0]["boxes"], second[1][0]["boxes"])


def test_user_pt_with_image_records_loads(detection, tmp_path):
    torch, mod = detection
    from PIL import Image

    img = Image.new("RGB", (8, 6), color=(10, 20, 30))
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}
    torch.save(
        {
            "images_train": [mod.image_to_record(img)],
            "targets_train": [target],
            "images_test": [mod.image_to_record(img)],
            "targets_test": [target],
        },
        tmp_path / "mine.pt",
    )
    images_train, _, images_test, _ = mod.load_data(tmp_path)
    assert images_train[0].size == (8, 6)
    assert images_test[0].tobytes() == img.tobytes()


def test_malicious_pt_is_refused_without_running_code(detection, tmp_path, monkeypatch):
    torch, mod = detection
    monkeypatch.delenv("R2C_TRUST_PICKLE", raising=False)
    _SIDE_EFFECTS.clear()
    torch.save({"images_train": [_Boom()], "targets_train": [],
                "images_test": [], "targets_test": []}, tmp_path / "evil.pt")

    with pytest.raises(RuntimeError, match="R2C_TRUST_PICKLE"):
        mod.load_data(tmp_path)
    assert _SIDE_EFFECTS == []


def test_trust_flag_allows_full_unpickling(detection, tmp_path, monkeypatch):
    torch, mod = detection
    _SIDE_EFFECTS.clear()
    torch.save({"images_train": [_Boom()], "targets_train": [],
                "images_test": [], "targets_test": []}, tmp_path / "legacy.pt")
    monkeypatch.setenv("R2C_TRUST_PICKLE", "1")
    images_train, *_ = mod.load_data(tmp_path)
    assert images_train == ["ran"] and _SIDE_EFFECTS == ["ran"]


def test_cross_modal_synthetic_cache_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    mod = _load_template(CROSS_MODAL_TEMPLATE, "r2c_xmodal_data_template")
    first = mod.load_data(tmp_path, train_size=3, n_test=2, seed=1)
    raw = torch.load(tmp_path / "_cache" / "synthetic_cross_modal.pt",
                     weights_only=True)
    assert isinstance(raw["student_inputs_train"][0], dict)
    second = mod.load_data(tmp_path, train_size=3, n_test=2, seed=1)
    assert second[0][0].tobytes() == first[0][0].tobytes()
    assert torch.equal(second[1][0], first[1][0])


def test_unpickling_error_is_what_torch_raises(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({"x": _Boom()}, tmp_path / "x.pt")
    with pytest.raises(pickle.UnpicklingError):
        torch.load(tmp_path / "x.pt", weights_only=True)
