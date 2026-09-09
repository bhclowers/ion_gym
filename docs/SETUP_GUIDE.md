# Setting up a simulation — the SimSpec schema

A simulation is one self-contained JSON (`sim_spec.SimSpec`). A saved
spec + any referenced STL files is a complete, portable run. This answers
each setup question explicitly.

## The six pieces

| You want to...            | Where it lives in the spec |
|---------------------------|----------------------------|
| define geometry           | `geometry.electrodes[].shapes` (inline rect/ellipse/polygon/cutout, mm) **or** `geometry.electrodes[].stl` (a file in `geometry.stl_dir`) |
| set DC / RF voltages      | per electrode: `dc`, `rf_amplitude`, `rf_frequency_hz`, `rf_phase_deg` (ion_gym's exact field names) |
| specify ion source        | `source`: `distribution` (point/disc/line/grid/file), `ke_lo..ke_hi` or `temperature_k`, `mz_list`, `charge`, `direction`, `tob_span_us`, `n_ions` |
| set HS collisions        | `collisions`: `enabled`, `gas`, `T_k`, `P_pa`, `sigma_m2` (`enabled=false` = vacuum) |
| run / start               | `sim_build.build_run(spec)` -> `(model, fly_fn, cols, births)`, then `ensemble_driver.run(...)` (or the Panel app) |
| save trajectories/fields  | pick `integration.record_channels`; results carry them; export via a DataFrame. Fields: dump `model.A` / bases. |
| save the configuration    | `spec.to_json("run.json")` / `SimSpec.from_json("run.json")` |

## Voltages are a re-weight, not a re-solve
Electrode bases are solved once per geometry (the fast-adjust property,
banked since Rung 1). Changing any `dc`/`rf_amplitude` only changes how
the bases are summed into A + sum_k s_k(t) B_k — instant and exact. This
is what lets the app **adjust voltages and redraw the field live, with or
without flying ions**.

## Symmetry picks the solver / view
`geometry.symmetry`:
- `cylindrical` — 2-D r-z (funnel, IMS). **Wired and validated (F-1).**
- `planar` — 2-D cartesian slice (einzel, buncher). Pieces validated;
  builder wiring is the next rung.
- `none` — full 3-D (quad, reflectron). Composes solver3d (v16) + mesh
  legs (v18) + tracer3d; the milestone after the GB2605116B geometry lands.
The builder RUNS validated paths and cleanly REFUSES unwired ones (it
never fabricates an unvalidated field).

## Views
`view.mode` (`2d`|`3d`) and `view.planes` (`xy`/`xz`/`yz`). 2-D-symmetry
geometry uses `xy`; 3-D geometry can show all three orthogonal slices
(fast, quantitative) or a capped 3-D trajectory bundle. Volumetric field
rendering -> pyvista (see viz.viz_notes()).

## Minimal example
```python
from ion_gym.io.sim_spec import SimSpec, GeometrySpec, SourceSpec, CollisionSpec
spec = SimSpec(
    geometry=GeometrySpec(width_mm=16, height_mm=6.5, mm_per_gu=0.125,
                          symmetry="cylindrical", stl_dir="funnel_stls",
                          electrodes=[...]),           # inline or STL
    source=SourceSpec(distribution="disc", r_mm=4.0, ke_lo=0.1, ke_hi=1.9,
                      mz_list=[556.0], n_ions=100),
    collisions=CollisionSpec(enabled=True, gas="N2", T_k=273, P_pa=133.28),
)
spec.to_json("my_run.json")

from ion_gym.physics.sim_build import build_run
from ion_gym.physics.ensemble_driver import run
model, fly_fn, cols, births = build_run(spec)
res = run(len(births), fly_fn)
```
See `sim_spec.SimSpec.from_json("examples/ion_funnel_rz.json")` and `reflectron_vacuum_spec()` for
worked templates.
