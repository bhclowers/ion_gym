"""DC-potential contour primitives for the editor's 2-D views
for the editor's 2-D views.

DESIGN: contours are computed from the COMMITTED document itself, on
demand, so they can never be stale — there is no solved-field key to
mismatch because the field is built for exactly the geometry being
drawn.  The 2-D builders make this honest: the planar MRT stage builds
in ~1.3 s cold and is a disk-cache hit afterwards (build_run banks
bases by geometry key), and the r-z fixtures are faster still.  An
oversized grid refuses BY NAME before any solve is attempted.

AUTHORITIES REUSED (one authority per quantity):
  * potential composition: `pe_view.compute_component(model, None, 1,
    'dc')` — the same DC term the PE-surface tab shows.  mz=None means
    the RF pseudopotential is never added; the legend says so.
  * model construction: `sim_build.build_run` — the same builder and
    the same disk cache every other consumer uses.
  * contour extraction: contourpy (matplotlib's own engine), on a
    metal-masked array so lines terminate at electrode surfaces.

SCOPE: planar and rz only.  A 3-D field needs a slice-plane convention
(WHICH z?) before contours there mean anything — refused by name until
that convention exists.
"""
from __future__ import annotations

import json
import time
from typing import List

import numpy as np

from ion_gym.edit.session import EditSession

# grid guard: refuse before solving anything bigger than this
NODE_BUDGET = 2_000_000
N_LEVELS_DEFAULT = 15
# decimation cap across all polylines (payload + draw cost)
MAX_TOTAL_POINTS = 150_000
# a DC range smaller than this is a flat field, not a contourable one
FLAT_RANGE_EV = 1e-9


def contour_primitives(session: EditSession,
                       n_levels: int = N_LEVELS_DEFAULT) -> dict:
    """Contour polylines of the DC potential for the session's COMMITTED
    document.  Returns either
      {"levels", "polylines" (list of {"level", "pts"}), "unit",
       "note", "n_points", "build_s"}
    or {"refused": <named reason>} — a refusal is a displayable outcome
    for the legend, not an exception; genuine malfunctions still raise.
    """
    pol = session.policy
    if pol.out_of_plane == "per_shape_extrude":
        return {"refused":
                "contours on a 3-D field need a slice-plane ruling "
                "(which z?) — not yet chartered; 2-D routes only"}

    g = session.spec.geometry
    h = float(g.mm_per_gu)
    n_nodes = (round(float(g.width_mm) / h) + 1) \
        * (round(float(g.height_mm) / h) + 1)
    if n_nodes > NODE_BUDGET:
        return {"refused":
                f"grid of ~{n_nodes:,} nodes exceeds the interactive "
                f"contour budget ({NODE_BUDGET:,}); view fields in "
                f"sim_app instead"}

    try:
        # lazy: pe_view pulls plotly; the editor stays importable without
        # it and the failure is named here instead of at module import
        from ion_gym.physics.sim_build import build_run
        from ion_gym.viz.pe_view import compute_component
        import contourpy
    except ImportError as e:
        return {"refused": f"contour dependencies unavailable: {e}"}

    # the COMMITTED document — what the viewer draws is what gets solved
    from ion_gym.io.sim_spec import SimSpec
    spec = SimSpec.from_dict(json.loads(session.to_document_bytes()))
    t0 = time.time()
    try:
        model, *_ = build_run(spec)
        x, y, PE, ele = compute_component(model, None, 1, "dc")
    except Exception as e:
        return {"refused": f"field build failed: "
                           f"{type(e).__name__}: {e}"}
    build_s = time.time() - t0

    # PE arrives (nx, ny); contourpy wants z[ny, nx] with x (nx,), y (ny,)
    Z = np.ma.array(PE.T, mask=(ele.T > 0))
    vals = Z.compressed()
    if vals.size == 0:
        return {"refused": "field is all metal — nothing to contour"}
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < FLAT_RANGE_EV:
        return {"refused":
                f"DC field is flat ({vmin:.3g} eV everywhere) — set "
                f"electrode dc values to see contours"}
    levels = list(np.linspace(vmin, vmax, n_levels + 2)[1:-1])

    # r-z models return the MIRRORED full plane (y in [-r_max, r_max],
    # their own display convention).  The editor's r-x view draws the
    # STORED half only, so contours are clipped to r >= 0 here — the
    # omitted half is identical by construction, and the note says so.
    clip_r = (pol.out_of_plane == "revolved")

    gen = contourpy.contour_generator(
        x=np.asarray(x, float), y=np.asarray(y, float), z=Z,
        line_type=contourpy.LineType.Separate)
    polylines: List[dict] = []
    total = 0
    for lv in levels:
        for arr in gen.lines(float(lv)):
            if arr is None or len(arr) < 2:
                continue
            arr = np.asarray(arr, float)
            runs = [arr]
            if clip_r:
                keep = arr[:, 1] >= -1e-9
                runs = []
                start = None
                for i, k in enumerate(keep):
                    if k and start is None:
                        start = i
                    elif not k and start is not None:
                        runs.append(arr[start:i]); start = None
                if start is not None:
                    runs.append(arr[start:])
            for run in runs:
                if len(run) < 2:
                    continue
                total += len(run)
                polylines.append({"level": float(lv), "pts": run})
    if total > MAX_TOTAL_POINTS:
        stride = int(np.ceil(total / MAX_TOTAL_POINTS))
        total = 0
        for pl in polylines:
            pts = pl["pts"]
            keep = np.unique(np.r_[np.arange(0, len(pts), stride),
                                   len(pts) - 1])
            pl["pts"] = pts[keep]
            total += len(pl["pts"])
    for pl in polylines:
        pl["pts"] = [[float(a), float(b)] for a, b in pl["pts"]]

    return {
        "levels": [float(v) for v in levels],
        "vmin": vmin, "vmax": vmax,
        "polylines": polylines,
        "n_points": int(total),
        "build_s": round(build_s, 3),
        "extent": [float(np.min(x)), float(np.max(x)),
                   (0.0 if clip_r else float(np.min(y))),
                   float(np.max(y))],
        "unit": "q\u00b7\u03c6_DC (eV at q=+1; numerically \u03c6 in V)",
        "note": ("DC potential only — RF drives are NOT included; "
                 "computed from the committed document (never stale)"
                 + ("; reflected r<0 half omitted (symmetric by "
                    "construction)" if clip_r else "")),
    }
