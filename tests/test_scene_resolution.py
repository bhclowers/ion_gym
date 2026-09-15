"""
test_scene_resolution.py -- gates for the GeomScene resolution knob and the
grid-bounds refusal.  Schema only: no solves, no fields.

THE BUGS THESE PIN
  1. GridSpec carried mm_per_gu AND nx/ny/nz -- two sources of truth for one
     quantity, with nothing enforcing agreement.  Changing the solve
     resolution of a loaded JSON meant hand-editing three ints and a float
     in step; get it wrong and the geometry silently rescales or the domain
     silently resizes.  Neither raised.
  2. GeomScene.check() never verified that the geometry FITS the grid.
     rasterize3d/voxelize index from arange(n) with the origin at 0, so
     out-of-box metal is not "elsewhere" -- it is NOT RASTERISED.  The solve
     then succeeds on an empty or clipped array.

Two unalike fixtures (a centred box pair; an off-origin ladder), so no fix
can be configuration-specific.

  S0  geometry invariance : at_resolution(h) changes the GRID, never the
                            geometry's own mm size
  S1  coverage            : the grid box always COVERS the geometry (the
                            floor()-vs-ceil() off-by-one that clipped metal)
  S2  refusal: out of box : metal outside [0, (n-1)*h] is REFUSED, not clipped
  S3  refusal: bad pitch  : mm_per_gu <= 0 is refused
  S4  round-trip          : to_json/from_json survives at_resolution
  S5  sizing adapter      : cost scales as h^-3 and warns on under-resolution
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ion_gym.physics.scene3d import GeomScene, GridSpec, Electrode, Shape, Box3D, Cylinder
from ion_gym.physics.sizing import propose_sizing_scene

FAILED = []


def gate(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    if not ok:
        FAILED.append(n)


# ---- FIXTURE A: two centred plates + a bored disc (negative coords) -----
def fixture_a():
    els = [
        Electrode(index=1, name="P1", voltage=100.0, shapes=[Shape(
            within=[Box3D(-5, -5, -2, 5, 5, -1)],
            notin=[Cylinder(cx=0, cy=0, z=0.0, r=1.5, length=4.0)])]),
        Electrode(index=2, name="P2", voltage=0.0, shapes=[Shape(
            within=[Box3D(-5, -5, 3, 5, 5, 4)])]),
    ]
    return GeomScene(grid=GridSpec(nx=2, ny=2, nz=2, mm_per_gu=1.0, mirror="xy"),
                 electrodes=els, units="mm", name="A_plates")


# ---- FIXTURE B: an off-origin, long-thin segmented ladder ---------------
def fixture_b():
    els = []
    for k in range(6):
        z0 = 20.0 + k * 4.2
        els.append(Electrode(index=k + 1, name=f"E{k}", voltage=float(k),
                             shapes=[Shape(within=[
                                 Box3D(2.0, 2.0, z0, 4.8, 3.6, z0 + 3.7)])]))
    return GeomScene(grid=GridSpec(nx=2, ny=2, nz=2, mm_per_gu=1.0),
                 electrodes=els, units="mm", name="B_ladder")


FIX = {"A_plates": fixture_a, "B_ladder": fixture_b}

print("\nS0  geometry invariance under at_resolution")
for nm, f in FIX.items():
    sc = f().at_resolution(1.0, margin_mm=1.0)     # declare the domain ONCE
    lo0, hi0 = sc.extent_mm()
    sizes = []
    for h in (0.5, 0.25, 0.1):
        s2 = sc.at_resolution(h)                   # preserves it
        lo, hi = s2.extent_mm()
        sizes.append(tuple(np.round(hi - lo, 9)))
    ok = all(s == sizes[0] for s in sizes)
    gate(f"S0.{nm}", ok,
         f"mm extent constant across h: {sizes[0]} (grid changes, geometry "
         f"does not)")

print("\nS1  the grid always COVERS the geometry")
for nm, f in FIX.items():
    sc = f().at_resolution(1.0, margin_mm=1.0)
    bad = []
    for h in (0.5, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1, 0.07):
        s2 = sc.at_resolution(h)
        lo, hi = s2.extent_mm()
        glo, ghi = s2.grid_extent_mm()
        if np.any(lo < glo - 1e-9) or np.any(hi > ghi + 1e-9):
            bad.append(h)
    gate(f"S1.{nm}", not bad, "covered at every pitch tested"
         if not bad else f"NOT covered at {bad}")

print("\nS2  refusal: geometry outside the grid box")
for nm, f in FIX.items():
    sc = f().at_resolution(0.25, margin_mm=1.0)
    # shrink the grid past the 1 mm margin -> metal falls outside the box
    bad = GeomScene(grid=GridSpec(nx=sc.grid.nx - 8, ny=sc.grid.ny,
                              nz=sc.grid.nz, mm_per_gu=sc.grid.mm_per_gu,
                              mirror=sc.grid.mirror),
                electrodes=sc.electrodes, units="mm")
    try:
        bad.check()
        gate(f"S2.{nm}", False, "clipped geometry ACCEPTED")
    except ValueError as e:
        gate(f"S2.{nm}", "does not fit its grid" in str(e),
             "clipped geometry refused with a diagnostic")

# and the negative-coordinate case: geometry before the origin
sc = fixture_a()
raw = GeomScene(grid=GridSpec(nx=40, ny=40, nz=40, mm_per_gu=0.25),
            electrodes=sc.electrodes, units="mm")
try:
    raw.check()
    gate("S2.origin", False, "geometry at negative coords ACCEPTED")
except ValueError as e:
    gate("S2.origin", "before the grid origin" in str(e),
         "centred (negative-coordinate) geometry refused -- rasterize3d "
         "indexes from arange(n), origin 0")

# --- declared overhang: the refusal is a DEFAULT, not a
# prohibition.  Full-physical-frame authoring (rods straddling a mirror
# fold; infinite constructs made finite-with-overhang) is legitimate and
# node-identical inside the stored region -- but it must be DECLARED
# (GridSpec overhang='allow'), and the relaxation must never reopen the
# hole it guards: an electrode rasterising EMPTY still refuses, always.
from ion_gym.physics.rasterize3d import rasterize

# S2.declared -- deliberately protruding geometry is accepted once declared,
# and only the in-box portion rasterises.  The fixture mirrors the real
# Rung-1 pattern: a rod straddling the mirror fold plane and a plate
# overrunning the high edge -- every electrode still owns in-box nodes.
# (fixture_a cannot serve here: its P1 lies ENTIRELY at z<0, so the
# unconditional empty-electrode guard refuses it -- correctly; see S2.empty.)
dec = GeomScene(grid=GridSpec(nx=10, ny=10, nz=12, mm_per_gu=1.0, mirror="xy",
                          overhang="allow"),
            electrodes=[
                Electrode(1, "fold_rod",
                          [Shape([Cylinder(cx=-2.0, cy=3.0, z=8.0, r=3.0,
                                           length=6.0)])]),
                Electrode(2, "edge_plate",
                          [Shape([Box3D(4, 4, 10, 13, 13, 11.5)])]),
            ], units="mm")
try:
    lab = rasterize(dec)
    n1, n2 = int((lab == 1).sum()), int((lab == 2).sum())
    gate("S2.declared", n1 > 0 and n2 > 0,
         f"declared overhang accepted; in-box nodes e1={n1} e2={n2}")
except ValueError as e:
    gate("S2.declared", False, f"declared overhang REFUSED: {e}")

# S2.badpolicy -- an unrecognised policy value is refused, never defaulted.
try:
    GeomScene(grid=GridSpec(nx=40, ny=40, nz=40, mm_per_gu=0.25,
                        overhang="alow"),
          electrodes=fixture_a().electrodes, units="mm").check()
    gate("S2.badpolicy", False, "typo'd overhang value ACCEPTED")
except ValueError as e:
    gate("S2.badpolicy", "bad overhang policy" in str(e),
         "unrecognised overhang value refused")

# S2.empty -- an electrode with ZERO stored-frame nodes is a dangling
# Dirichlet basis; refused in rasterize() regardless of the declaration.
ghost = GeomScene(grid=GridSpec(nx=10, ny=10, nz=10, mm_per_gu=1.0,
                            mirror="xy", overhang="allow"),
              electrodes=[Electrode(1, "ghost",
                                    [Shape([Box3D(-5, -5, 2, -2, -2, 4)])])],
              units="mm")
try:
    rasterize(ghost)
    gate("S2.empty", False, "zero-node electrode ACCEPTED (dangling basis)")
except ValueError as e:
    gate("S2.empty", "rasterises EMPTY" in str(e),
         "zero-node electrode refused even with overhang declared")

# S2.carry -- the declaration survives at_resolution and the JSON round
# trip.  A DROPPED declaration does not fail quietly: the new grid
# re-defaults to 'refuse' and at_resolution's own final check() raises on
# the protruding geometry -- so the failure mode here is an EXCEPTION, and
# a crash is not a red gate: catch it and say so.
try:
    car = dec.at_resolution(0.5)
    back = GeomScene.from_json(car.to_json())
    gate("S2.carry", car.grid.overhang == "allow"
         and back.grid.overhang == "allow",
         "overhang declaration carried through at_resolution + JSON")
except ValueError as e:
    gate("S2.carry", False,
         f"declaration DROPPED in transit (re-refused): {e}")

print("\nS3  refusal: bad pitch")
try:
    fixture_a().at_resolution(0.0, margin_mm=1.0)
    gate("S3.zero", False, "pitch 0 accepted")
except ValueError:
    gate("S3.zero", True, "pitch 0 refused")
try:
    fixture_a().at_resolution(-0.1, margin_mm=1.0)
    gate("S3.negative", False, "negative pitch accepted")
except ValueError:
    gate("S3.negative", True, "negative pitch refused")

print("\nS4  JSON round-trip")
for nm, f in FIX.items():
    s2 = f().at_resolution(0.2, margin_mm=1.0)
    back = GeomScene.from_json(s2.to_json())
    same_grid = (back.grid.nx, back.grid.ny, back.grid.nz,
                 back.grid.mm_per_gu) == (s2.grid.nx, s2.grid.ny,
                                          s2.grid.nz, s2.grid.mm_per_gu)
    lo1, hi1 = s2.extent_mm()
    lo2, hi2 = back.extent_mm()
    gate(f"S4.{nm}", same_grid and np.allclose(lo1, lo2)
         and np.allclose(hi1, hi2),
         "grid + geometry survive to_json/from_json; and the reloaded scene "
         "can be re-gridded again")
    # the whole point: a LOADED json can change resolution
    re = back.at_resolution(0.1)
    gate(f"S4.{nm}.regrid", re.grid.mm_per_gu == 0.1,
         f"loaded JSON re-gridded to 0.1 mm/gu -> "
         f"{re.grid.nx}x{re.grid.ny}x{re.grid.nz}")

print("\nS5  sizing adapter")
sc = fixture_b().at_resolution(0.2, margin_mm=1.0)
p1 = propose_sizing_scene(sc, pitch=0.2, min_feature_mm=0.5)
p2 = propose_sizing_scene(sc.at_resolution(0.1), pitch=0.1,
                          min_feature_mm=0.5)
ratio = p2.n_voxels / p1.n_voxels
gate("S5.scaling", 6.0 < ratio < 10.0,
     f"halving the pitch multiplies nodes by {ratio:.1f} (expect ~8)")
p3 = propose_sizing_scene(sc.at_resolution(0.4), pitch=0.4,
                          min_feature_mm=0.5)
gate("S5.warns", any("smeared" in w for w in p3.warnings),
     "under-resolved feature (0.5 mm at 0.4 mm/gu) is warned about")
gate("S5.fold", propose_sizing_scene(
    fixture_a().at_resolution(0.25, margin_mm=1.0), pitch=0.25).n_voxels_solved
    == propose_sizing_scene(fixture_a().at_resolution(0.25, margin_mm=1.0),
                            pitch=0.25).n_voxels // 4,
    "declared mirror xy is credited as a 4x reduction in the estimate")

print("\nS6  the DOMAIN is preserved across a pitch change")
# This is the invariant that failed on the Q3: re-gridding re-derived the box
# from the METAL, silently discarding the vacuum margin and pushing
# conductors onto the open boundary -- and a domain declared from the metal
# bbox while the metal sat offset inside a larger grid CLIPPED the far edge.
for nm, f in FIX.items():
    sc = f().at_resolution(0.5, margin_mm=2.0)
    g0 = sc.grid_extent_mm()
    m0 = sc.extent_mm()
    ok = True
    for h in (0.4, 0.25, 0.2, 0.1):
        s2 = sc.at_resolution(h)
        g1, m1 = s2.grid_extent_mm(), s2.extent_mm()
        # grid box may grow by <1 cell (ceil), never shrink; metal NEVER moves
        if not (np.allclose(m0[0], m1[0]) and np.allclose(m0[1], m1[1])):
            ok = False
        if np.any(g1[1] < g0[1] - 1e-9):
            ok = False
    gate(f"S6.{nm}", ok,
         f"metal stays at {np.round(m0[0], 2)}..{np.round(m0[1], 2)} mm and "
         f"the margin survives, at every pitch")

sc = fixture_a().at_resolution(0.5, margin_mm=2.0)
mlo, mhi = sc.extent_mm()
glo, ghi = sc.grid_extent_mm()
gate("S6.margin", bool(np.all(mlo - glo > 1.5) and np.all(ghi - mhi > 1.5)),
     f"declared 2 mm margin is real: metal {np.round(mlo,2)}..{np.round(mhi,2)} "
     f"inside grid {np.round(glo,2)}..{np.round(ghi,2)}")

# refusal: a scene with no declared domain cannot silently invent one
try:
    fixture_a().at_resolution(0.25)
    gate("S6.refuse_no_domain", False, "invented a domain silently")
except ValueError as e:
    gate("S6.refuse_no_domain", "no valid domain to preserve" in str(e),
         "an ungridded scene refuses to invent a domain")

print("\n" + "=" * 62)
print("FAILED: " + (", ".join(FAILED) if FAILED else "none"))
print("=" * 62)
sys.exit(1 if FAILED else 0)
