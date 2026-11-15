"""
The pre-retirement check (tools/dependency_audit.py --retire) stops on a
file it cannot parse instead of skipping it.

retire_check skipped any file with a SyntaxError, so an importer inside
it was never listed and the check could answer "NONE - safe to retire"
wrongly. audit() already aborted on such a file; retire_check now does
the same, naming it.
"""
import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "dependency_audit.py"


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))    # its _repo_root import
    spec = importlib.util.spec_from_file_location("dependency_audit_t", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "target.py").write_text("def f():\n    return 1\n")
    (tmp_path / "user.py").write_text("from target import f\n")
    return mod


def test_importers_are_listed(audit):
    res = audit.retire_check("target")
    assert [h["importer"] for h in res["importers"]] == ["user.py"]


def test_package_imports_are_listed(audit, tmp_path):
    # the module is named by the LAST dotted part; matching the first part
    # missed every package import and answered "safe to retire"
    (tmp_path / "user.py").unlink()
    (tmp_path / "deep.py").write_text("from pkg.sub.target import f\n")
    (tmp_path / "mod.py").write_text("from pkg.sub import target\n")
    (tmp_path / "plain.py").write_text("import pkg.sub.target\n")
    (tmp_path / "other.py").write_text("from pkg.target_two import g\n")
    res = audit.retire_check("target")
    assert sorted(h["importer"] for h in res["importers"]) == [
        "deep.py", "mod.py", "plain.py"]


def test_an_unparsable_file_stops_the_check(audit, tmp_path):
    (tmp_path / "broken.py").write_text("from target import f\ndef (\n")
    with pytest.raises(SystemExit, match="broken.py does not parse"):
        audit.retire_check("target")
