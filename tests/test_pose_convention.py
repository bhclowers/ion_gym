"""
The stage-pose rotation convention, pinned (review C-13).

A stage's pose rot_deg = [rx, ry, rz] rotates about the FIXED world x,
then world y, then world z: R = Rz @ Ry @ Rx (extrinsic X-Y-Z). The
documentation once called it intrinsic X-Y-Z (R = Rx @ Ry @ Rz); the two
agree whenever only one angle is nonzero, and a census found no stored
pose with two or more (2 rotations, both single-axis), so the wording was
corrected to match the code rather than moving any stage. The flight
(staged_flight._rot_matrix) and the assembly view (viz_core) must stay on
the same convention.
"""
import itertools

import numpy as np
import pytest

from ion_gym.physics.staged_flight import Pose, _rot_matrix


def _Rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_the_review_case_is_extrinsic():
    # [90, 90, 0] on x-hat: extrinsic gives -z (intrinsic would give +y)
    v = _rot_matrix([90, 90, 0]) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0.0, 0.0, -1.0], atol=1e-12)


@pytest.mark.parametrize("rot", [(30, -45, 60), (90, 90, 0), (10, 0, -75),
                                 (0, 35, 120)])
def test_the_matrix_is_rz_ry_rx(rot):
    assert np.allclose(_rot_matrix(rot),
                       _Rz(rot[2]) @ _Ry(rot[1]) @ _Rx(rot[0]), atol=1e-12)


@pytest.mark.parametrize("axis", range(3))
def test_a_single_axis_pose_is_convention_free(axis):
    rot = [0.0, 0.0, 0.0]
    rot[axis] = 37.0
    intrinsic = _Rx(rot[0]) @ _Ry(rot[1]) @ _Rz(rot[2])
    assert np.allclose(_rot_matrix(rot), intrinsic, atol=1e-12)


def test_pose_round_trips():
    pose = Pose(offset_mm=[1.0, -2.0, 3.0], rot_deg=[30.0, -45.0, 60.0])
    p, v = np.array([0.3, 0.7, -1.1]), np.array([1.0, 0.5, 0.25])
    pw, vw = pose.local_to_world(p, v)
    pl, vl = pose.world_to_local(pw, vw)
    assert np.allclose(pl, p) and np.allclose(vl, v)


def test_the_assembly_view_uses_the_flight_convention():
    """viz_core.assembly_stage_draw_mode decides whether an r-z stage's
    posed axis lands on a world axis; that must agree with where the
    flight's matrix sends local x, for every grid of angles."""
    from ion_gym.io.paths import repo_root
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.viz.viz_core import assembly_stage_draw_mode
    spec = SimSpec.from_json(str(repo_root() / "examples/ion_funnel_rz.json"))
    for rot in itertools.product((0, 30, 90, -90, 180), repeat=3):
        a = _rot_matrix(rot)[:, 0]
        on_axis = abs(np.max(np.abs(a)) - 1.0) <= 1e-9
        want = "drawn" if on_axis else "placeholder"
        assert assembly_stage_draw_mode(spec, list(rot)) == want, rot
