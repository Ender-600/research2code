"""Every pipeline agent runs with the bash tool denied.

Pipeline agents produce files through the read/write tools; the driver, not
the agent, executes generated code (stage 2.d install + import, the smoke
gate, probes). An agent definition with no `permission` block inherits
opencode's default, which allows bash, so a prompt injection carried in the
paper text could run arbitrary shell commands with the user's privileges.
Three producer agents (method coder, architecture coder, notebook generator)
shipped that way; none of them needs a shell.

The interactive companion is the one sanctioned exception: it is the
user-facing primary agent and launches drivers on the user's request.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"

# Agents allowed to keep bash. Adding a name here needs a reason in review.
BASH_ALLOWED = {"r2c-companion"}


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path.name}: missing frontmatter"
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end]) or {}


AGENT_FILES = sorted(AGENTS_DIR.glob("*.md"))


def test_agent_files_found():
    assert len(AGENT_FILES) >= 10


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.stem)
def test_pipeline_agent_denies_bash(path: Path):
    if path.stem in BASH_ALLOWED:
        pytest.skip("sanctioned interactive agent")
    permission = _frontmatter(path).get("permission")
    assert isinstance(permission, dict), (
        f"{path.name} has no permission block; opencode's default allows bash"
    )
    assert permission.get("bash") == "deny", (
        f"{path.name} must set `bash: deny` (got {permission.get('bash')!r})"
    )


def test_known_bad_frontmatter_is_caught(tmp_path: Path):
    bad = tmp_path / "r2c-example.md"
    bad.write_text("---\ndescription: x\n---\n\nbody\n", encoding="utf-8")
    assert _frontmatter(bad).get("permission") is None
