"""
Gate: staged multi-region flight (staged_flight.py) — the non-coaxial
assembly model: fly an ion across SEPARATE
field regions with a pose handoff, so an orthogonal-acceleration stage
(pusher 90 deg to the beam) is expressible.

  S1  Pose round-trip: world->local->world is identity for translation +
      rotation; a 90-deg-about-z pose maps world +x velocity to local -y.
  S2  seam check MEASURES |E| and refuses a LIVE seam (field-dead
      doctrine: a handoff where both regions' fields act double-counts).
  S3  staged flight through a coaxial drift + a 90-deg-rotated region
      completes, hands off at the shared world point (continuous path),
      and carries velocity through the rotation correctly.
  S4  a region whose ion terminates (splat/timeout) before its exit plane
      STOPS the chain with a named fate, not a silent continue.

Field-free drift regions (analytic: straight lines) so the gate is fast
and exact -- it tests the HANDOFF math, not the tracer (which its own
gates cover).

Run: python tests/test_staged_flight.py
"""
import pytest
pytestmark = pytest.mark.slow  # staged flight >60 s

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.physics.staged_flight import (Region, Pose, ExitPlane,
                                           fly_staged, check_seam,
                                           _rot_matrix)

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


def _region(name, pose=None, exit=None, nx=60, ny=60, nz=60, h=0.5):
    z = np.zeros((nx, ny, nz), np.float32)
    # route: this pack is hand-built here rather than produced by a
    # builder, so nothing tags it. staged_flight._fly_region DISPATCHES
    # on this tag and refuses an untagged pack rather than guessing a
    # kernel -- correct, and the reason this fixture had to be updated:
    # a synthetic pack bypasses the builder that would have tagged it.
    # These are (nx, ny, nz) arrays with EAx/EAy/EAz, i.e. the 3-D pack.
    fields = dict(route="3d",
                  EAx=z, EAy=z.copy(), EAz=z.copy(),
                  ele=np.zeros((nx, ny, nz), bool), h_mm=h,
                  ExK=np.zeros((0, nx, ny, nz)), EyK=np.zeros((0, nx, ny, nz)),
                  EzK=np.zeros((0, nx, ny, nz)), ch_kind=np.zeros(0, np.int64),
                  ch_om=np.zeros(0), ch_ph=np.zeros(0), ch_amp=np.zeros(0),
                  ch_off=np.zeros(0), ch_duty=np.zeros(0), tab_t=np.zeros(0),
                  tab_v=np.zeros(0), tab_off=np.zeros(1, np.int64))
    return Region(name=name, fields=fields, pose=pose or Pose(), exit=exit,
                  dt_ns=2.0, t_max_us=40.0)


def main():
    def s1():
        p = Pose(offset_mm=[3, -2, 7], rot_deg=[0, 0, 90])
        pw, vw = p.local_to_world([1, 0, 0], [1, 0, 0])
        pl, vl = p.world_to_local(pw, vw)
        assert np.allclose(pl, [1, 0, 0], atol=1e-9), pl
        assert np.allclose(vl, [1, 0, 0], atol=1e-9), vl
        # 90 about z: world +x velocity -> local -y
        R = _rot_matrix([0, 0, 90])
        assert np.allclose(R.T @ np.array([1., 0, 0]), [0, -1, 0], atol=1e-9)
    check("S1 pose round-trip + 90deg velocity map", s1)

    def s2():
        a = _region("a", exit=ExitPlane("x", 25.0, +1))
        b = _region("b", pose=Pose(offset_mm=[25, 0, 0]))
        ok, emax, rep = check_seam(a, b, 300.0)
        assert ok and emax == 0.0, rep      # field-free -> dead
        # inject a live field on the exit plane -> must refuse
        a.fields["EAx"][50, :, :] = 9.0     # 9 V/mm at x=25mm (h=0.5)
        ok2, emax2, rep2 = check_seam(a, b, 300.0, tol_v_per_mm=1.0)
        assert not ok2 and emax2 >= 9.0, rep2
    check("S2 seam check refuses a live (field-alive) handoff", s2)

    def s3():
        a = _region("beamline", exit=ExitPlane("x", 25.0, +1))
        b = _region("pusher", pose=Pose(offset_mm=[25, 15, 15],
                                        rot_deg=[0, 0, 90]))
        res = fly_staged([a, b], 300.0, p0_world=[2, 15, 15],
                         v0_world_mm_us=[5, 0, 0])
        assert res["fate"] == "completed", res["fate"]
        assert [r["name"] for r in res["regions"]] == ["beamline", "pusher"]
        h = res["handoffs"][0]
        assert np.allclose(h["p_world"], [25, 15, 15], atol=0.5), h["p_world"]
        assert np.allclose(h["v_world"], [5, 0, 0], atol=0.2), h["v_world"]
        # continuous world path at the seam
        r1end = res["regions"][0]["x"][-1]
        r2start = res["regions"][1]["x"][0]
        assert abs(r1end - r2start) < 0.5, (r1end, r2start)
    check("S3 staged flight: coaxial drift -> 90deg pusher, continuous", s3)

    def s4():
        # exit plane the ion never reaches (wrong side) -> timeout, chain
        # stops with a named fate, not a silent continue
        a = _region("trap", exit=ExitPlane("x", 25.0, +1))
        b = _region("never", pose=Pose(offset_mm=[25, 0, 0]))
        # ion moving AWAY from the exit (-x): times out in region a
        res = fly_staged([a, b], 300.0, p0_world=[15, 15, 15],
                         v0_world_mm_us=[-5, 0, 0])
        assert "terminated in trap" in res["fate"], res["fate"]
        assert len(res["regions"]) == 1, "chain did not stop"
    check("S4 termination before seam stops the chain with a fate", s4)

    print("=" * 60)
    print(f"PASSED {4 - len(FAILED)}   FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
