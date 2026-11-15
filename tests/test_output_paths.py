"""
Per-machine output location: the checkout root in a clone, the current
working folder for an installed package (which has no repo, and whose
package folder is inside the Python installation).

The installed case is simulated by making the checkout lookup find no
checkout, then running from a temporary folder.
"""
from pathlib import Path

import pytest

from ion_gym.io import paths
from ion_gym.ui.telemetry import LoopStallWatchdog


@pytest.fixture
def installed(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_checkout_root", lambda: None)
    for var in ("ION_GYM_FIELDS", "ION_GYM_TRAJECTORIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_installed_package_writes_to_the_working_folder(installed):
    assert paths.outputs_dir() == installed / "outputs"
    assert (installed / "outputs").is_dir()
    assert paths.fields_dir() == installed / "fields"
    assert paths.trajectories_dir() == installed / "trajectories"


def test_installed_hang_dump_goes_to_the_working_folder(installed):
    dog = LoopStallWatchdog()
    assert Path(dog.dump_path) == (installed / "outputs" / "diagnostics"
                                   / "ion_gym_hang_dump.txt")


def test_installed_package_never_writes_under_the_package(installed):
    pkg = paths.pkg_dir().resolve()
    for p in (paths.outputs_dir(), paths.fields_dir(),
              paths.trajectories_dir()):
        assert not p.resolve().is_relative_to(pkg), p


def test_environment_overrides_still_win(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("ION_GYM_FIELDS", str(tmp_path / "f"))
    monkeypatch.setenv("ION_GYM_TRAJECTORIES", str(tmp_path / "t"))
    assert paths.fields_dir() == (tmp_path / "f").resolve()
    assert paths.trajectories_dir() == (tmp_path / "t").resolve()


def test_a_clone_still_writes_under_the_repo(monkeypatch, tmp_path):
    for var in ("ION_GYM_FIELDS", "ION_GYM_TRAJECTORIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)              # cwd must not matter in a clone
    root = paths.repo_root()
    assert paths.outputs_dir() == root / "outputs"
    assert paths.fields_dir() == root / "fields"
    assert paths.trajectories_dir() == root / "trajectories"
