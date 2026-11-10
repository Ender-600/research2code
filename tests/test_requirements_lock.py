"""requirements.lock pins every dependency with hashes and tracks requirements.txt.

CI installs with `pip install --require-hashes -r requirements.lock`, so a
compromised or re-uploaded package on the index fails the install instead
of running. requirements.txt stays the human-edited list of direct
dependencies; regenerate the lock after editing it:

    uv pip compile requirements.txt --generate-hashes --universal \
        --python-version 3.10 --no-header -o requirements.lock
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
LOCK = REPO_ROOT / "requirements.lock"

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def _direct_requirements() -> list[Requirement]:
    reqs = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            reqs.append(Requirement(line))
    return reqs


def _lock_entries(text: str) -> list[tuple[str, str, list[str]]]:
    """(name, version, hashes) for each pinned entry in a pip lock file."""
    entries: list[tuple[str, str, list[str]]] = []
    for line in text.splitlines():
        m = _PIN.match(line)
        if m:
            entries.append((canonicalize_name(m.group(1)), m.group(2), []))
        elif line.strip().startswith("--hash=") and entries:
            entries[-1][2].append(line.strip().rstrip("\\").strip())
    return entries


def _unhashed(text: str) -> list[str]:
    return [name for name, _, hashes in _lock_entries(text) if not hashes]


def test_every_lock_entry_has_a_hash():
    entries = _lock_entries(LOCK.read_text(encoding="utf-8"))
    assert len(entries) > 20
    assert _unhashed(LOCK.read_text(encoding="utf-8")) == []


def test_lock_covers_every_direct_requirement_within_its_range():
    entries = _lock_entries(LOCK.read_text(encoding="utf-8"))
    pinned: dict[str, list[str]] = {}
    for name, version, _ in entries:
        pinned.setdefault(name, []).append(version)
    problems = []
    for req in _direct_requirements():
        name = canonicalize_name(req.name)
        versions = pinned.get(name)
        if not versions:
            problems.append(f"{req.name}: missing from requirements.lock")
            continue
        outside = [v for v in versions if not req.specifier.contains(v, prereleases=True)]
        if outside:
            problems.append(f"{req.name}: lock pins {outside}, outside {req.specifier}")
    assert problems == [], "regenerate requirements.lock:\n" + "\n".join(problems)


def test_lock_check_flags_a_known_bad_entry():
    bad = "pyyaml==6.0.3\n    # via -r requirements.txt\n"
    good = "pyyaml==6.0.3 \\\n    --hash=sha256:abc\n"
    assert _unhashed(bad) == ["pyyaml"]
    assert _unhashed(good) == []


def test_ci_installs_from_the_lock_with_hashes():
    workflow = (REPO_ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "--require-hashes -r requirements.lock" in workflow
    assert "pip install -r requirements.txt" not in workflow
