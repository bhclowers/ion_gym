"""Gates for Shortley-Weller fractional boundaries (the pa-surf path):
curved-surface (staircase) potential error must COLLAPSE when solving with
exact-surface thetas, and the multigrid must carry theta faithfully."""
import numpy as np
from ion_gym.physics.scene3d import GeomScene, Electrode, Shape, Cylinder, GridSpec
from ion_gym.physics.solver3d import fractions_from_scene, solve3d
from ion_gym.physics.multigrid3d import solve3d_vmg


def _cyl_problem(n=176, a=20.0, b=80.0):
    c = (n - 1) / 2
    ii = np.arange(n)
    X, Y = np.meshgrid(ii, ii, indexing="ij")
    R = np.hypot(X - c, Y - c)
    fixed = ((R <= a) | (R >= b))[:, :, None]
    val = np.zeros((n, n, 1)); val[(R >= b)[:, :, None]] = 100.0
    sc = GeomScene(grid=GridSpec(nx=n, ny=n, nz=1, mm_per_gu=1.0), units="gu",
               electrodes=[
        Electrode(index=1, name="i", shapes=[Shape(
            within=[Cylinder(cx=c, cy=c, z=5, r=a, length=10)])]),
        Electrode(index=2, name="o", shapes=[Shape(
            within=[Cylinder(cx=c, cy=c, z=5, r=b + 40, length=10)],
            notin=[Cylinder(cx=c, cy=c, z=5, r=b, length=10)])])])
    Vex = 100 * np.log(np.maximum(R, 1e-9) / a) / np.log(b / a)
    m = (R > a) & (R < b)
    return fixed, val, sc, Vex, m, R - a


def test_sw_error_collapse():
    fixed, val, sc, Vex, m, d = _cyl_problem()
    Vi, _, _ = solve3d(fixed, val, tol=1e-5)
    th = fractions_from_scene(sc, fixed)
    assert (th < 0.999).sum() > 500, "too few fractional legs found"
    Vf, _, _ = solve3d(fixed, val, theta=th, tol=1e-5)
    surf = m & (d >= 0) & (d < 1)
    ei = np.abs(Vi[:, :, 0] - Vex)[surf].max()
    ef = np.abs(Vf[:, :, 0] - Vex)[surf].max()
    assert ei > 1.0, f"integer staircase error unexpectedly small: {ei}"
    assert ef < 0.1, f"fractional near-surface error did not collapse: {ef}"
    assert ef < ei / 20, f"improvement < 20x: {ei} -> {ef}"
    print(f"  SW collapse at curved surface: {ei:.2f} V -> {ef:.3f} V "
          f"({ei/ef:.0f}x)  OK")


def test_sw_multigrid_matches_sor():
    fixed, val, sc, *_ = _cyl_problem()
    th = fractions_from_scene(sc, fixed)
    Vs, _, _ = solve3d(fixed, val, theta=th, tol=1e-6)
    Vm, _, _ = solve3d_vmg(fixed, val, theta=th, tol=1e-5)
    dmax = np.abs(Vm - Vs).max()
    assert dmax < 5e-3, f"VMG-theta vs SOR-theta {dmax} V"
    print(f"  VMG carries theta: match {dmax:.1e} V  OK")


if __name__ == "__main__":
    test_sw_error_collapse(); test_sw_multigrid_matches_sor()
    print("all pa-surf gates passed")
