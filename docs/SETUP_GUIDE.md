# Setting up a simulation — the SimSpec schema

A simulation is one self-contained JSON deck (`ion_gym.io.sim_spec.SimSpec`).
A saved deck plus any STL files it references is a complete, portable run.
Lengths are mm, voltages V, times µs (the step `dt_ns` is ns), temperatures
K, pressures Torr, masses Da.

## The pieces

| You want to... | Where it lives in the deck |
|---|---|
| define geometry | `geometry.electrodes[].shapes` (inline `rect` / `ellipse` / `polygon` / `cutout`, mm) **or** `geometry.electrodes[].stl` (a file in `geometry.stl_dir`). `geometry.mm_per_gu` is the grid pitch (mm per grid unit); it is required. |
| choose 2-D, r-z or 3-D | `geometry.symmetry.coords` (`"rz"` or `"xyz"`) and `geometry.depth_mm` — see "Which route a deck flies" below |
| set DC voltages | per electrode: `dc` (V). Electrodes tuned together can share a `dc_group`. |
| set RF and travelling-wave drives | per electrode: `rf_groups`, the names of the drive groups it belongs to (an electrode may carry several). Each group in `geometry.rf_groups` carries `amplitude_v`, `frequency_hz`, `phase_deg` and `waveform` (`sin`, `square` or `table`). A rod pair is two groups 180° apart. Per-electrode `rf_amplitude` / `rf_frequency_hz` / `rf_phase_deg` are not part of the schema and are refused when non-zero. |
| specify the ions | `source`: `distribution` (`point`, `disc`, `line`, `grid`, `box`, `gaussian`, `file`), position (`x0_mm`, `y0_mm`, `z0_mm`), energy `ke_lo`..`ke_hi` (eV) along `direction` or a thermal `temperature_k`, `mz_list`, `charge`, `tob_span_us`, `n_ions`, `seed`. **`mz_list` holds ion masses in Da** (the name is historical); `charge` is the signed charge state, and m/z = mass / \|charge\|. `seed: null` draws a fresh seed per run and prints it, so any run can be repeated by declaring that number. |
| add a buffer gas | `collisions`: `enabled`, `gas`, `T_k`, `P_torr`, `model` (`"hs"` hard-sphere or `"sds"`), `flow_mm_us` (gas flow, mm/µs per axis). Hard-sphere cross sections come from a CCS fit at each ion's mass by default (`sigma_model: "ccs_fit"`, fitted in N2; another gas falls back, with a warning, to `sigma_m2`); state `sigma_model: "fixed"` with `sigma_m2` for one cross section for every ion. SDS mobilities can be declared per species in `sds_species` (see `docs/SDS_MODEL.md`). Leaving `collisions` out flies vacuum. `P_pa` is deprecated; write `P_torr`. |
| set the step and flight time | `integration`: `dt_ns`, `t_max_us`, `rec_every` (record every Nth step), `record_channels` |
| run it | `ion_gym.physics.sim_build.build_run(spec)` returns `(model, fly_fn, cols, births)`; `ion_gym.physics.ensemble_driver.fly_all(len(births), fly_fn, n_workers=...)` flies every ion and returns `[(trajectory, summary), ...]`. The workbench (`ion-gym dashboard`) does the same from a deck file. |
| save trajectories | pick `integration.record_channels`; each trajectory is an array whose columns are named by `cols`. |
| save the configuration | `spec.to_json("run.json")` and `SimSpec.from_json("run.json")` |

`spec.validate()` returns a list of problems (empty when the deck is
consistent); `build_run` refuses a deck that does not validate.

## Voltages are a re-weight, not a re-solve

Each electrode's field is solved once per geometry. Changing any `dc` or a
drive group's `amplitude_v` only changes how those solved fields are summed
(A + Σ s_k(t) B_k), so it is instant and exact. This is what lets the
workbench adjust voltages and redraw the field live, with or without flying
ions.

## Which route a deck flies

`ion_gym.physics.sim_build.build_route(spec)` answers this; `build_run` dispatches on the
same answer.

| `symmetry.coords` | `depth_mm` | electrodes | route | field |
|---|---|---|---|---|
| `"rz"` | — | shapes | r-z: x is the axis, y the radius | 2-D, axisymmetric |
| `"xyz"` | 0 | shapes | planar | 2-D (no E_z); ions and gas still move in z |
| `"xyz"` | 0 | STL | 2-D slice of the meshes | 2-D |
| `"xyz"` | > 0 | shapes, extruded (along z unless a shape's `extrude` says otherwise) | 3-D shapes | 3-D |
| `"xyz"` | > 0 | STL | 3-D meshes | 3-D |

### Where the metal surface sits

Electrodes are rasterized onto the grid: the nodes inside an electrode's
shapes (or mesh) are metal. The routes then decide whether an ion is in
metal in two ways, and the difference is a grid-scale offset of the wall:

- **planar and r-z** (and 2-D STL slices): an ion is in metal when its
  *nearest* node is metal. The surface lies half a grid cell (h/2, with
  h = `mm_per_gu`) outside the outermost metal nodes, and a plate one node
  thick still stops ions.
- **3-D routes**: an ion is in metal only inside a cell whose eight corners
  are all metal. The surface lies on the outermost metal nodes, and metal
  must be at least two nodes thick to stop an ion: an electrode none of
  whose metal can stop an ion is refused, and one with more than 5 % of its
  metal too thin is flown with a warning.

So the same geometry stands up to h/2 proud on each face on planar and
r-z: a slit of width w with its faces on nodes is open w − h there and w
on a 3-D route. The offset shrinks in proportion to h. For apertures and
slits only a few cells wide, refine `mm_per_gu` until the result stops
changing, and compare routes at the same pitch. `tests/test_wall_conventions.py`
pins both conventions. An electrode marked `is_grid` (a mesh grid) holds
its potential in the solve but never stops an ion, on every route.

`geometry.symmetry` may also be written as one of the strings
`"cylindrical"` (r-z), `"planar"` or `"none"` (both `xyz`); any other string
is refused. `symmetry.planes` declares mirror or translational planes per
axis; the builder verifies each against the geometry before using it to
shrink the solve.

## Views

`view.mode` (`"2d"` or `"3d"`) and `view.planes` (`"xy"`, `"xz"`, `"yz"`)
say how the workbench shows the deck. A 2-D route uses `xy`; a 3-D route can
show the three orthogonal slices or a capped 3-D trajectory bundle. View keys
are display only.

## Worked example

An r-z drift cell: two annular plates 20 mm apart, the entrance at 100 V,
20 ions of 100 Da, charge +1, in 1 Torr of N2. This block is executed by
`tests/test_setup_guide_example.py`, so it runs as written.

```python
from collections import Counter

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, ShapeSpec, SimSpec,
                                 SourceSpec, SymmetrySpec)
from ion_gym.physics.ensemble_driver import default_workers, fly_all
from ion_gym.physics.ion_envelope import fate_name
from ion_gym.physics.sim_build import build_route, build_run


def ring(name, x_mm, volts):
    """An annular plate in r-z: x is the axis, y the radius (mm)."""
    return ElectrodeSpec(name=name, dc=volts, shapes=[ShapeSpec(
        "rect", {"x_mm": x_mm, "y_mm": 1.0, "width_mm": 0.5,
                 "height_mm": 4.0})])


spec = SimSpec(
    name="drift cell",
    geometry=GeometrySpec(
        width_mm=22.0, height_mm=5.0, mm_per_gu=0.1,
        symmetry=SymmetrySpec(coords="rz"),
        electrodes=[ring("entrance", 0.5, 100.0), ring("exit", 21.0, 0.0)]),
    source=SourceSpec(distribution="point", n_ions=20, x0_mm=2.0, y0_mm=0.0,
                      direction=[1.0, 0.0, 0.0], ke_lo=1.0, ke_hi=1.0,
                      mz_list=[100.0], charge=1, tob_span_us=0.0, seed=1),
    collisions=CollisionSpec(enabled=True, gas="N2", T_k=300.0, P_torr=1.0),
    integration=IntegrationSpec(dt_ns=1.0, t_max_us=40.0, rec_every=100),
)
errors = spec.validate()
assert not errors, errors
print(build_route(spec))          # BuildRoute(builder='rz', ...)

model, fly, cols, births = build_run(spec)
results = fly_all(len(births), fly, n_workers=default_workers())
print(Counter(fate_name(summary["kind"]) for _traj, summary in results))

spec.to_json("drift_cell.json")
again = SimSpec.from_json("drift_cell.json")
assert again.to_dict() == spec.to_dict()
```

For complete decks, see `examples/` (for example
`SimSpec.from_json("examples/ion_funnel_rz.json")`).
