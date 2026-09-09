"""Tests for ion_gym.io.records — the name-addressed record layer.

Purpose-built: each test pins one property the
temperature bug taught us to guard. Fixtures are self-contained (K9).

What is pinned and why:
- zero-copy: the record must be a VIEW of the kernel buffer, not a copy —
  wrapping at the kernel boundary must cost nothing.
- schema refusal: a wrong-length or duplicated schema is a construction
  ERROR, never a silent misalignment (the bug class this layer exists
  to kill).
- assemble_record channel fill: every requested channel is populated with
  the physically correct value — the original bug was requested ke_x/
  ke_y/ke_z assembled as silent zeros.
- unfillable refusal: a requested channel the kernel cannot supply
  raises with a diagnostic instead of zero-filling.
"""
import numpy as np
import pytest

from ion_gym.io.records import TrajRecord, assemble_record, record

AMU = 1.66053906660e-27
E_CHG = 1.602176634e-19


def _rec():
    cols = ["t", "x", "y", "z", "vx", "vy", "vz"]
    raw = np.arange(35.0).reshape(5, 7)
    return raw, cols, TrajRecord(raw, cols)


def test_column_access_is_zero_copy_view():
    raw, cols, rec = _rec()
    v = rec["vx"]
    assert np.shares_memory(v, raw)
    raw[0, 4] = 123.0
    assert rec["vx"][0] == 123.0


def test_named_access_matches_positions():
    raw, cols, rec = _rec()
    for j, c in enumerate(cols):
        assert np.array_equal(rec[c], raw[:, j])


def test_unknown_channel_raises_with_available_names():
    _, _, rec = _rec()
    with pytest.raises(KeyError, match="available"):
        rec["nope"]


def test_schema_length_mismatch_refused():
    raw = np.zeros((4, 7))
    with pytest.raises(ValueError, match="schema mismatch"):
        TrajRecord(raw, ["t", "x", "y"])


def test_duplicate_schema_refused():
    raw = np.zeros((4, 3))
    with pytest.raises(ValueError, match="duplicate"):
        TrajRecord(raw, ["t", "x", "x"])


def test_row_decimate_and_len():
    raw, cols, rec = _rec()
    end = rec.row(-1)
    assert end["t"] == raw[-1, 0]
    assert len(rec) == 5
    assert rec.decimate(2).n_rows == 3


def test_setitem_writes_through_to_backing_array():
    raw, cols, rec = _rec()
    before = raw[:, 1].copy()
    rec["x"] += 100.0
    assert np.array_equal(raw[:, 1], before + 100.0)
    rec["y"] = np.zeros(raw.shape[0])
    assert np.array_equal(raw[:, 2], np.zeros(raw.shape[0]))


def test_setitem_unknown_channel_refused():
    _, _, rec = _rec()
    with pytest.raises(KeyError, match="available"):
        rec["nope"] = 1.0


def test_record_helper_passes_none_through():
    assert record(None, ["t"]) is None


def _kernel_output(n=50, seed=3, with_field=True):
    rng = np.random.default_rng(seed)
    o = dict(t_us=np.linspace(0.0, 1.0, n),
             x=rng.normal(0, 1, n),
             y=rng.normal(0, 1, n),
             z=rng.normal(0, 1, n),
             vx=rng.normal(0, 0.2, n),
             vy=rng.normal(0, 0.2, n),
             vz=rng.normal(0, 0.2, n),
             ncol=7)
    if with_field:
        o["ex"] = rng.normal(0, 5, n)
        o["ey"] = rng.normal(0, 5, n)
        o["ez"] = rng.normal(0, 5, n)
    return o


def test_assemble_fills_every_requested_channel():
    o = _kernel_output()
    mz = 120.0
    cols = ["t", "x", "y", "z", "vx", "vy", "vz",
            "speed", "ke_ev", "ke_x", "ke_y", "ke_z",
            "e_x", "e_field", "n_col"]
    tr = assemble_record(o, cols, mz_da=mz)
    rec = TrajRecord(tr, cols)
    ke_fac = 0.5 * mz * AMU * 1e6 / E_CHG
    assert np.allclose(rec["ke_x"], ke_fac * o["vx"] ** 2)
    assert np.allclose(rec["ke_ev"],
                       ke_fac * (o["vx"] ** 2 + o["vy"] ** 2
                                 + o["vz"] ** 2))
    assert np.allclose(rec["speed"],
                       np.sqrt(o["vx"] ** 2 + o["vy"] ** 2 + o["vz"] ** 2))
    assert np.allclose(rec["e_field"],
                       np.sqrt(o["ex"] ** 2 + o["ey"] ** 2 + o["ez"] ** 2))
    assert np.allclose(rec["n_col"], 7.0)
    # THE bug this layer exists to prevent: no requested channel is all-zero
    for c in ("ke_x", "ke_y", "ke_z", "speed", "ke_ev"):
        assert np.any(rec[c] != 0.0), f"{c} assembled as silent zeros"


def test_assemble_refuses_unfillable_channel():
    o = _kernel_output(with_field=False)     # SDS-like: no ex/ey/ez
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "e_field"]
    with pytest.raises(ValueError, match="cannot be derived"):
        assemble_record(o, cols, mz_da=120.0)


def test_assemble_ke_temperature_roundtrip():
    # velocities drawn at 300 K must yield ke channels whose mean gives
    # back ~300 K via T = 2<KE>/kB (the temperature chain's first link)
    KB = 1.380649e-23
    mz = 120.0
    m_kg = mz * AMU
    n = 60000
    rng = np.random.default_rng(11)
    sig = np.sqrt(KB * 300.0 / m_kg) / 1e3          # mm/us
    o = _kernel_output(n=n, with_field=False)
    o["vx"] = rng.normal(0, sig, n)
    o["vy"] = rng.normal(0, sig, n)
    o["vz"] = rng.normal(0, sig, n)
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "ke_x"]
    rec = TrajRecord(assemble_record(o, cols, mz_da=mz), cols)
    T = 2.0 * rec["ke_x"].mean() * E_CHG / KB
    assert abs(T - 300.0) < 6.0
