"""
numba's on-disk kernel cache is keyed by the source of ion_gym.physics.

numba invalidates a cached kernel on the stamp of the file that defines it,
not on the files it calls: editing only collision3d.py once left every
r-z hard-sphere flight running the old collision physics, with no message,
until the .nbi/.nbc files were deleted by hand (review R-03). ion_gym.physics
now points numba's cache at <cache root>/numba/<key of the physics source>.

These run on a throwaway COPY of the package, in subprocesses, so the
repository's own files and caches are never touched. Slow (kernel
compiles): `pytest -m slow tests/test_kernel_cache_key.py`.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]

_CHILD = r"""
import json, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("always")
from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics.sim_build import build_run
import numba.core.config as cfg
rect = lambda x, y, w, h: ShapeSpec("rect", {"x_mm": x, "y_mm": y,
                                             "width_mm": w, "height_mm": h})
geo = GeometrySpec(width_mm=10.0, height_mm=6.0, depth_mm=0.0, mm_per_gu=0.5,
                   symmetry=SymmetrySpec(coords="rz"),
                   electrodes=[ElectrodeSpec(name="top", dc=5.0,
                                             shapes=[rect(0, 5, 10, 1)]),
                               ElectrodeSpec(name="bot", dc=0.0,
                                             shapes=[rect(0, 0, 10, 1)])])
src = SourceSpec(distribution="point", n_ions=1, x0_mm=2.0, y0_mm=2.0,
                 z0_mm=0.0, direction=[1.0, 0.0, 0.0], ke_lo=1.0, ke_hi=1.0,
                 mz_list=[200.0], tob_span_us=0.0, seed=3)
col = CollisionSpec(enabled=True, gas="N2", T_k=300.0, P_torr=5.0,
                    model="hs", sigma_model="fixed", sigma_m2=1.5e-18)
spec = SimSpec(geometry=geo, name="r03", source=src, collisions=col,
               integration=IntegrationSpec(dt_ns=2.0, t_max_us=2.0,
                                           rec_every=50))
_m, fly, cols, _b = build_run(spec)
tr = fly(0)[0]
print(json.dumps({"end": [float(v) for v in tr[-1]],
                  "cache": cfg.CACHE_DIR}))
"""


def _copy_package(tmp_path):
    dst = tmp_path / "tree"
    shutil.copytree(ROOT / "ion_gym", dst / "ion_gym",
                    ignore=shutil.ignore_patterns("__pycache__", "*.nbi",
                                                  "*.nbc"))
    return dst


def _run(tree, cache, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    env["ION_GYM_CACHE"] = str(cache)
    env.update(extra_env or {})
    out = subprocess.run([sys.executable, "-c", _CHILD, str(tree)],
                         capture_output=True, text=True, timeout=600,
                         env=env, cwd=str(tree))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


def test_an_edit_to_a_called_module_reaches_the_kernels(tmp_path):
    tree, cache = _copy_package(tmp_path), tmp_path / "cache"
    before, _e = _run(tree, cache)
    callee = tree / "ion_gym" / "physics" / "collision3d.py"
    original = callee.read_text(encoding="utf-8")
    old = "    return c_bar * ((s + 1.0 / (2.0 * s)) * 0.5 * math.sqrt(math.pi)"
    assert original.count(old) == 1
    # only the CALLEE changes: the mean relative speed scaled by 0.25
    callee.write_text(original.replace(old, old.replace(
        "return c_bar", "return 0.25 * c_bar")), encoding="utf-8")
    edited, _e = _run(tree, cache)
    assert edited["end"] != before["end"], "stale kernel: old physics flew"
    assert edited["cache"] != before["cache"]
    callee.write_text(original, encoding="utf-8")
    restored, _e = _run(tree, cache)
    assert restored["cache"] == before["cache"]     # the old key, reused
    assert restored["end"] == before["end"]


def test_the_kernel_cache_stays_out_of_the_field_inventory(tmp_path):
    tree, cache = _copy_package(tmp_path), tmp_path / "cache"
    got, _e = _run(tree, cache)
    assert Path(got["cache"]).parent == cache / ".numba"
    from ion_gym.io.fa_cache import entries
    assert not [e for e in entries(root=str(cache))
                if "numba" in e["key"]]


def test_a_user_numba_cache_dir_is_respected_and_warned(tmp_path):
    tree, cache = _copy_package(tmp_path), tmp_path / "cache"
    mine = tmp_path / "my_numba_cache"
    got, err = _run(tree, cache, {"NUMBA_CACHE_DIR": str(mine)})
    assert got["cache"] == str(mine)
    assert "NUMBA_CACHE_DIR is set" in err
