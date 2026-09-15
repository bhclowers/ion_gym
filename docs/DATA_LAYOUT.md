# Data layout & path resolution

All modules locate reference data through `ion_gym.paths` — no absolute
paths are hardcoded anywhere in the package.

## Resolution order (first that exists wins)
1. `paths.set_data_root("/path")` — programmatic, highest priority.
2. `ION_GYM_DATA` environment variable.
3. a `data/` directory beside the package.
4. the package directory itself (the shipped flat layout).
5. the current working directory.

## Expected folder names under the data root
```
quad/quad/            quad_monolithic field arrays        (native solver, fly)
quad_upload/quad/     STL-import round-trip reference
ims_hs/ims/          IMS HS example + ims-1.csv        (C-3)
funnel/ionfunnel/     funnel example + ionfunnel-1.csv  (F-1)
buncher/ octupole/ tof/ ref_220/einzel/                  (older gates)
uploads/              quad_stl_beamwin.npz, quad-stl-1.csv
```
Registered names live in `paths._REGISTRY`; add a line there to register
a new dataset rather than hardcoding a path in a module.

## Typical setups
- **Data beside code** (shipped zip): nothing to do.
- **Data in a separate Drive folder**: set `ION_GYM_DATA` once, or call
  `paths.set_data_root(...)` at the top of your notebook/script.
