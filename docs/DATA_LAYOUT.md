# Data layout and path resolution

Every path ion_gym reads or writes is resolved in one module,
`ion_gym.io.paths`; no module hardcodes an absolute path. Where things go
depends on whether ion_gym runs from a **source checkout** (a clone, with
`pyproject.toml` beside the `ion_gym/` package) or from an **installed
package** (`pip install`).

## What ion_gym reads

**Example decks** — `paths.examples_dir()`:
- checkout: `<checkout>/examples/`
- installed: the copy the wheel carries at `ion_gym/examples/`

`paths.default_deck()` is the deck the workbench opens when given none
(`ion_funnel_rz.json` in that folder). Shipped examples are read-only to the
workbench's save buttons: save an edited example under a new name.

**User data root** — `paths.data_root()`, the first of these that exists:
1. the folder passed to `paths.set_data_root(...)`;
2. the `ION_GYM_DATA` environment variable;
3. `<checkout>/geometry`;
4. `<checkout>`;
5. `ion_gym/data` inside the package;
6. the `ion_gym` package folder;
7. the current working folder.

For an installed package, items 3 and 4 refer to the package folder.
A folder you name explicitly (items 1 and 2) must exist: it is refused
rather than skipped in favour of the next candidate.
`paths.uploads_dir()` (`<data root>/uploads`) is where the workbench's
upload buttons install STL meshes (`stl/`) and decks with their meshes
(`decks/`). Set `ION_GYM_DATA`, or call `paths.set_data_root(...)` at the
top of a script or notebook, to keep that data elsewhere.

An STL deck finds its meshes in `geometry.stl_dir`, resolved relative to
the deck file, so a deck and its meshes travel together.

## What ion_gym writes

Per-machine output goes under an **output root**: the checkout root in a
clone, the current working folder for an installed package (never inside
the Python installation).

| Output | Location | Override |
|---|---|---|
| figures and reports | `<output root>/outputs/` (`paths.outputs_dir()`) | — |
| solved fields you save, and their label registry | `<output root>/fields/` (`paths.fields_dir()`) | `ION_GYM_FIELDS` |
| trajectories you save (`*.traj.npz`) | `<output root>/trajectories/` (`paths.trajectories_dir()`) | `ION_GYM_TRAJECTORIES` |

## Caches

`paths.cache_root()` is `ION_GYM_CACHE` if set, else `~/.ion_gym_cache`.
It holds:
- the solved per-electrode fields, keyed by geometry and by the code that
  solves them (`physics.solve_code`), so a deck with unchanged geometry
  loads its fields instead of solving again, and a change to a solver or a
  rasterizer re-solves each cached field once, on first use. Entries
  solved by older code stay on disk until cleared;
- the compiled numba kernels, under `.numba/<key>`, where the key is a hash
  of the physics source, so editing any physics module recompiles every
  kernel rather than leaving a stale one in use. A `NUMBA_CACHE_DIR` you
  set is used instead, with a warning that it is not keyed by the source.

Deleting the cache folder is always safe: fields are re-solved and kernels
recompiled on next use. The workbench's Cache tab lists and clears the
field cache.
