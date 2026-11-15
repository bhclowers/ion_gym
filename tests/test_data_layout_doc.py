"""
docs/DATA_LAYOUT.md names only things that exist.

The guide once described ion_gym.paths (the module is ion_gym.io.paths),
a data-root order the code did not follow, and reference folders that
had been deleted (review NU-06). This checks that every `paths.<name>()`
the doc names is a function in ion_gym.io.paths, that every ION_GYM_*
variable it names is read by the package, and that the data-root order it
lists is the order paths._candidates() yields.
"""
import re
from pathlib import Path

from ion_gym.io import paths

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DATA_LAYOUT.md"


def test_named_functions_exist():
    names = set(re.findall(r"`paths\.(\w+)\(", DOC.read_text("utf-8")))
    assert names, "the doc names no paths functions"
    missing = sorted(n for n in names if not callable(getattr(paths, n,
                                                              None)))
    assert not missing, f"DATA_LAYOUT.md names missing functions: {missing}"


def test_named_environment_variables_are_read():
    doc_vars = set(re.findall(r"ION_GYM_[A-Z_]+", DOC.read_text("utf-8")))
    src = "\n".join(p.read_text("utf-8")
                    for p in (ROOT / "ion_gym").rglob("*.py"))
    unread = sorted(v for v in doc_vars if f'"{v}"' not in src)
    assert not unread, f"DATA_LAYOUT.md names unread variables: {unread}"


def test_the_doc_names_the_real_module():
    text = DOC.read_text("utf-8")
    assert "`ion_gym.io.paths`" in text
    assert "`ion_gym.paths`" not in text


def _doc_data_root_list():
    text = DOC.read_text("utf-8")
    start = text.index("**User data root**")
    block = text[start:text.index("\n\n", text.index("\n1. ", start))]
    return re.findall(r"^\d+\. (.+)$", block, re.M)


def test_data_root_order_matches_the_code(monkeypatch, tmp_path):
    override, env = tmp_path / "override", tmp_path / "env"
    override.mkdir()                     # an explicit root must exist
    env.mkdir()
    monkeypatch.setenv("ION_GYM_DATA", str(env))
    paths.set_data_root(override)
    try:
        got = list(paths._candidates())
    finally:
        paths.set_data_root(None)
    repo = paths._repo_root()
    want = [override.resolve(), env.resolve(), repo / "geometry", repo,
            paths.pkg_dir() / "data", paths.pkg_dir(), Path.cwd()]
    assert got == want
    items = _doc_data_root_list()
    # the doc lists the same candidates, in this order
    assert len(items) == len(want), items
    for item, key in zip(items, ("set_data_root", "ION_GYM_DATA",
                                 "<checkout>/geometry", "<checkout>",
                                 "ion_gym/data", "package folder",
                                 "working folder")):
        assert key in item, (key, item)


def test_an_explicit_root_that_does_not_exist_is_refused(monkeypatch,
                                                         tmp_path):
    """Review API-15: a missing explicit root used to be skipped, and the
    data root silently became another folder."""
    import pytest
    missing = tmp_path / "no_such_folder"
    with pytest.raises(FileNotFoundError, match="not a folder"):
        paths.set_data_root(missing)
    monkeypatch.setenv("ION_GYM_DATA", str(missing))
    with pytest.raises(FileNotFoundError, match="ION_GYM_DATA"):
        paths.data_root()
