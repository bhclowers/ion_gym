"""
Gate: multidrive composer — compose_drive_channels
in build_stl3d, the spec->channel-pack step feeding the frozen-anchored
tracer3d multidrive kernel (test_multidrive3d Gate 1 covers the tracer).

Anchored on a REAL solved scene (native rasterize + solve, no external
STL assets), so the bases carry a real field gradient (a hand-built
degenerate geometry gives ~0 fields and anchors nothing).

  M1  single sin group -> exactly ONE sin channel; om = 2pi f; amp on the
      channel; DC-free channel offset (DC lives in EA).
  M2  MULTI-GROUP: an electrode in TWO groups (RF + AC) plus DC -> two
      channels of the right kinds AND the DC present in the static EA.
  M3  MULTI-FREQUENCY: two groups at different f both appear (the OLD
      _compose_AB refused this) — the whole point of the composer.
  M4  waveform kinds map: sin/cos/square/table -> 0/1/2/3; a table group
      carries its (t,v) into the pack.
  M5  REFUSALS (no silent mis-drive): unknown waveform; a member naming a
      group that isn't defined; a non-zero group with no members.

Run: python tests/test_multidrive_compose.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 RFGroupSpec, SourceSpec)
from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Box3D
from ion_gym.physics.solver3d import solve_bases
from ion_gym.physics.rasterize3d import rasterize
from ion_gym.physics import build_stl3d as B

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


class _D(dict):
    pass


def _solved_bases():
    """Real 3-rod scene -> unit bases (÷v_basis, as build_stl3d_run does),
    with the label grid stashed for edge-aware gradients."""
    rods = [(1, 3), (6, 3), (3.5, 5)]
    sc = GeomScene(
        grid=GridSpec(nx=90, ny=70, nz=61, mm_per_gu=0.1),
        electrodes=[Electrode(index=k, name=f"rod{k}", voltage=0.0,
                    shapes=[Shape(within=[Box3D(x1=x, y1=y, z1=0.5,
                                                x2=x + 1, y2=y + 1, z2=5.5)])])
                    for k, (x, y) in enumerate(rods, 1)],
        units="mm", name="anchor")
    lab = rasterize(sc)
    raw = solve_bases({i: (lab == i) for i in range(1, 4)},
                      v_basis=1e4, verbose=False)
    bases = _D({i: (raw[i] / 1e4).astype(np.float32) for i in raw})
    bases.ele = lab.astype(np.int16)
    return bases


def _spec(rf_groups, memberships):
    """memberships: list of (dc, [group names]) per rod 1..3. Phase lives
    on the GROUP (clean model), never on the electrode."""
    els = [ElectrodeSpec(name=f"rod{k}", dc=dc, rf_groups=list(gs))
           for k, (dc, gs) in enumerate(memberships, 1)]
    return SimSpec(
        geometry=GeometrySpec(width_mm=9, height_mm=7, depth_mm=6.1,
                              mm_per_gu=0.1, rf_groups=rf_groups,
                              electrodes=els),
        source=SourceSpec(seed=0), name="g")


def main():
    bases = _solved_bases()

    def m1():
        rf = [RFGroupSpec(name="RF", frequency_hz=1e6, amplitude_v=200.0,
                          phase_deg=0.0, waveform="sin")]
        # ONE rod driven, the others are the grounded return -> a real
        # field gradient (all rods in one group = uniform, no field)
        ch = B.compose_drive_channels(
            _spec(rf, [(0, ["RF"]), (0, []), (0, [])]), bases)
        assert list(ch["ch_kind"]) == [0], ch["ch_kind"]
        assert abs(ch["ch_om"][0] - 2 * np.pi * 1e6 * 1e-6) < 1e-9
        assert abs(ch["ch_amp"][0] - 200.0) < 1e-9
        assert abs(ch["ch_off"][0]) < 1e-12, "DC leaked into a channel"
        assert np.abs(ch["ExK"][0]).max() > 1e-3, "channel field is ~0"
    check("M1 single sin group -> one sin channel, om/amp right", m1)

    def m2():
        rf = [RFGroupSpec(name="RF", frequency_hz=1e6, amplitude_v=200.0,
                          waveform="sin"),
              RFGroupSpec(name="AC", frequency_hz=5e3, amplitude_v=10.0,
                          waveform="square")]
        ch = B.compose_drive_channels(
            _spec(rf, [(5.0, ["RF", "AC"]), (0, ["RF"]),
                       (0, ["RF"])]), bases)
        assert len(ch["ch_kind"]) == 2, ch["ch_kind"]
        assert set(ch["ch_kind"]) == {0, 2}, ch["ch_kind"]
        assert np.abs(ch["EAx"]).max() > 1e-3, "DC bias absent from EA"
    check("M2 electrode in 2 groups + DC -> 2 channels + static EA", m2)

    def m3():
        rf = [RFGroupSpec(name="RF", frequency_hz=1e6, amplitude_v=200.0,
                          waveform="sin"),
              RFGroupSpec(name="AC", frequency_hz=5e3, amplitude_v=10.0,
                          waveform="sin")]
        ch = B.compose_drive_channels(
            _spec(rf, [(0, ["RF", "AC"]), (0, ["RF"]),
                       (0, ["AC"])]), bases)
        oms = sorted(ch["ch_om"])
        assert len(oms) == 2 and oms[0] != oms[1], oms
        # the frequency the OLD path would have refused is present
        assert any(abs(o - 2 * np.pi * 5e3 * 1e-6) < 1e-9 for o in oms)
    check("M3 multi-frequency groups both compose (old path refused)", m3)

    def m4():
        rf = [RFGroupSpec(name="TAB", frequency_hz=1e3, amplitude_v=3.0,
                          waveform="table", table_t_us=[0.0, 100.0],
                          table_v=[0.0, 1.0])]
        ch = B.compose_drive_channels(
            _spec(rf, [(0, ["TAB"]), (0, []), (0, [])]),
            bases)
        assert list(ch["ch_kind"]) == [3], ch["ch_kind"]
        assert len(ch["tab_t"]) == 2 and ch["tab_v"][-1] == 1.0
    check("M4 table waveform -> kind 3 with (t,v) carried", m4)

    def m5():
        # unknown waveform
        try:
            B.compose_drive_channels(_spec(
                [RFGroupSpec(name="X", frequency_hz=1e6, amplitude_v=1.0,
                             waveform="triangle")],
                [(0, ["X"]), (0, []), (0, [])]), bases)
            raise AssertionError("unknown waveform not refused")
        except ValueError as e:
            assert "waveform" in str(e)
        # member names an undefined group
        try:
            B.compose_drive_channels(_spec(
                [RFGroupSpec(name="RF", frequency_hz=1e6, amplitude_v=1.0)],
                [(0, ["NOPE"]), (0, []), (0, [])]), bases)
            raise AssertionError("undefined group not refused")
        except ValueError as e:
            assert "not defined" in str(e) or "NOPE" in str(e)
        # defined non-zero group with no members
        try:
            B.compose_drive_channels(_spec(
                [RFGroupSpec(name="ORPHAN", frequency_hz=1e6,
                             amplitude_v=5.0)],
                [(0, []), (0, []), (0, [])]), bases)
            raise AssertionError("orphan group not refused")
        except ValueError as e:
            assert "no electrode" in str(e) or "nothing to drive" in str(e)
    check("M5 refusals: bad waveform / undefined group / orphan group", m5)

    def m6():
        # duty flows group -> pack; 0.5 default; out-of-range refused
        rf = [RFGroupSpec(name="TW", frequency_hz=6250.0, amplitude_v=2.5,
                          offset_v=2.5, waveform="square", duty=0.25),
              RFGroupSpec(name="RF", frequency_hz=1e6, amplitude_v=50.0,
                          waveform="sin")]
        ch = B.compose_drive_channels(
            _spec(rf, [(0, ["TW"]), (0, ["RF"]), (0, [])]), bases)
        {int(k): float(v) for k, v in
             zip(ch["ch_kind"], ch["ch_duty"])}
        assert abs(ch["ch_duty"][list(ch["ch_kind"]).index(2)] - 0.25)             < 1e-12, ch["ch_duty"]
        assert abs(ch["ch_duty"][list(ch["ch_kind"]).index(0)] - 0.5)             < 1e-12, "sin channel default duty"
        assert abs(ch["ch_off"][list(ch["ch_kind"]).index(2)] - 2.5)             < 1e-12, "offset_v not carried"
        try:
            bad = [RFGroupSpec(name="X", frequency_hz=1e3, amplitude_v=1.0,
                               waveform="square", duty=1.5)]
            B.compose_drive_channels(_spec(bad, [(0, ["X"]), (0, []),
                                                 (0, [])]), bases)
            raise AssertionError("duty 1.5 not refused")
        except ValueError as e:
            assert "duty" in str(e)
    check("M6 duty: group -> ch_duty (0.25), sin default 0.5, "
          "offset carried, out-of-range refused", m6)

    print("=" * 60)
    print(f"PASSED {6 - len(FAILED)}   FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
