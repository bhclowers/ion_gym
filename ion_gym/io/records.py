"""Name-addressed record access over dense trajectory arrays.

The flight kernels (numba @njit) fill dense ``(n_records, n_cols)`` float
buffers — that layout is a hard performance boundary and never changes.
Everything DOWNSTREAM of the kernel addresses channels by NAME through
:class:`TrajRecord`, a zero-copy wrapper pairing the raw matrix with its
column schema. ``rec["vx"]`` returns a numpy VIEW of the column (strided,
no copy — the dict and the matrix are the same memory), so wrapping costs
nothing and vectorized math on channels is unchanged.

Why this exists: positional access (``traj[:, 4]`` or
call-site ``cols.index("vx")``) silently breaks when the recorded column
set changes — a real temperature-assessment bug was exactly this
class of failure. A named accessor is self-describing, survives
reordering, and travels between functions carrying its own schema.

Provenance: schema order is owned by ``sim_spec.column_names()``
(BASE_CHANNELS + OPTIONAL_CHANNELS order); this module only enforces it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TrajRecord", "record"]


class TrajRecord:
    """Zero-copy, name-addressed view over a dense trajectory matrix.

    Parameters
    ----------
    raw : (n, ncols) ndarray
        The kernel-produced record buffer (rows = record steps).
    columns : sequence of str
        Channel names in column order — the single source of truth for
        this array's layout (normally ``spec.column_names()``).

    Access
    ------
    - ``rec["vx"]``        -> 1-D column VIEW (no copy)
    - ``rec.get("ke_x")``  -> column view or None if absent
    - ``"vx" in rec``, ``rec.keys()``, ``rec.items()``
    - ``rec.row(k)``       -> {name: scalar} for one record step
    - ``rec.decimate(d)``  -> TrajRecord over ``raw[::d]`` (view)
    - ``rec.where(mask)``  -> TrajRecord over ``raw[mask]`` (copy, numpy)
    - ``rec.raw``          -> the dense matrix (kernel/serialisation use)
    - ``np.asarray(rec)``  -> the dense matrix (via __array__)
    - ``rec.as_dataframe()`` -> pandas DataFrame with named columns
    """

    __slots__ = ("_raw", "_cols", "_ix")

    def __init__(self, raw, columns):
        raw = np.asarray(raw)
        if raw.ndim != 2:
            raise ValueError(
                "TrajRecord needs a 2-D (n, ncols) array, got ndim="
                f"{raw.ndim} shape={raw.shape}")
        cols = tuple(str(c) for c in columns)
        if raw.shape[1] != len(cols):
            raise ValueError(
                f"column schema mismatch: array has {raw.shape[1]} columns "
                f"but {len(cols)} names were given ({cols}). The schema "
                "must come from the SAME source that built the array "
                "(spec.column_names()).")
        if len(set(cols)) != len(cols):
            raise ValueError(f"duplicate column names in schema: {cols}")
        self._raw = raw
        self._cols = cols
        self._ix = {c: i for i, c in enumerate(cols)}

    # ------------------------------------------------------------ mapping
    def __getitem__(self, name):
        try:
            return self._raw[:, self._ix[name]]
        except KeyError:
            raise KeyError(
                f"no channel {name!r} in this record; available: "
                f"{self._cols}") from None

    def __setitem__(self, name, value):
        # write a whole column BY NAME (rec["x"] = arr, or the augmented
        # rec["x"] += dx which Python lowers to get+set). Writes through to
        # the backing array — the record is a live view, not a copy.
        try:
            self._raw[:, self._ix[name]] = value
        except KeyError:
            raise KeyError(
                f"no channel {name!r} in this record; available: "
                f"{self._cols}") from None

    def get(self, name, default=None):
        i = self._ix.get(name)
        return default if i is None else self._raw[:, i]

    def __contains__(self, name):
        return name in self._ix

    def keys(self):
        return self._cols

    def items(self):
        return ((c, self._raw[:, i]) for c, i in self._ix.items())

    # ------------------------------------------------------------- shape
    @property
    def raw(self):
        return self._raw

    @property
    def columns(self):
        return self._cols

    @property
    def n_rows(self):
        return self._raw.shape[0]

    def __len__(self):
        # length = number of RECORD STEPS (rows), matching len(traj)
        return self._raw.shape[0]

    def __array__(self, dtype=None):
        return (self._raw if dtype is None
                else self._raw.astype(dtype, copy=False))

    # --------------------------------------------------------------- ops
    def row(self, k):
        """One record step as {name: scalar} (e.g. rec.row(-1) endpoint)."""
        r = self._raw[k]
        return {c: r[i] for c, i in self._ix.items()}

    def decimate(self, every):
        """Every Nth record step, as a view-backed TrajRecord."""
        step = max(int(every), 1)
        return TrajRecord(self._raw[::step], self._cols)

    def where(self, mask):
        """Boolean-mask the record steps (numpy fancy indexing copies)."""
        return TrajRecord(self._raw[np.asarray(mask)], self._cols)

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._raw, columns=list(self._cols))

    def __repr__(self):
        return (f"TrajRecord({self._raw.shape[0]} steps, "
                f"channels={self._cols})")


def record(traj, columns):
    """Convenience wrapper: ``record(r.traj, cols)["vx"]``. Returns None
    for a None trajectory (fates-only runs), so call sites can guard with
    a single truthiness check instead of two."""
    return None if traj is None else TrajRecord(traj, columns)


# constants for derived-channel physics: the ONE authority (L-532; these
# were retyped here)
from ion_gym.physics.constants import KG_AMU as _AMU_KG   # noqa: E402
from ion_gym.physics.constants import E_CHG as _E_CHG     # noqa: E402

_AXES = "xyz"


def radial_values(name, rec, frame):
    """A radial channel (sim_spec.RADIAL_CHANNELS) DERIVED from a record
    about `frame` (an OpticAxisSpec: SimSpec.radial_frame()), or None when
    the record lacks the columns it needs (positions for radius; e_x,
    e_y, e_z for the field channels). THE one derivation, used after
    flight by every route and by the app's plots (L-539: each route and
    the app had its own axis -- x, z through the origin, or the domain
    centre).

      radius   = distance from the line (axis through through_mm)
      e_axial  = e_<axis> (signed, along +axis)
      e_radial = |E perpendicular to the axis| (>= 0)"""
    if frame is None:
        raise ValueError(f"radial channel {name!r} needs an optic axis "
                         f"(SimSpec.radial_frame() returned None)")
    a = _AXES.index(frame.axis)
    p, q = _AXES[(a + 1) % 3], _AXES[(a + 2) % 3]        # cyclic in-plane
    cp, cq = (float(v) for v in frame.through_mm)
    if name == "radius":
        if p not in rec or q not in rec:
            return None
        return np.sqrt((rec[p] - cp) ** 2 + (rec[q] - cq) ** 2)
    if name in ("e_axial", "e_radial"):
        need = (f"e_{frame.axis}", f"e_{p}", f"e_{q}")
        if not all(c in rec for c in need):
            return None
        if name == "e_axial":
            return np.array(rec[need[0]], dtype=float)
        return np.sqrt(rec[need[1]] ** 2 + rec[need[2]] ** 2)
    raise ValueError(f"radial_values: {name!r} is not a radial channel "
                     f"(known: radius, e_axial, e_radial)")


def apply_radial_frame(traj, columns, frame):
    """Overwrite, in place, every radial channel present in `columns` with
    its value derived about `frame`. Refuses a radial column it cannot
    derive (its inputs are not recorded) rather than leaving the
    kernel's own-axis value in it."""
    rec = TrajRecord(traj, columns)
    for c in ("radius", "e_axial", "e_radial"):
        if c in rec:
            v = radial_values(c, rec, frame)
            if v is None:
                raise ValueError(
                    f"radial channel {c!r} cannot be derived: the record "
                    f"lacks its inputs (columns {list(columns)})")
            rec[c] = v
    return traj


def assemble_record(o, columns, mz_da, frame=None):
    """Build the dense trajectory record from a kernel output dict,
    filling EVERY requested channel and REFUSING any it cannot derive
    (a requested channel silently left at zero is a wrong
    answer — the temperature bug above was exactly this: ke_x/ke_y/
    ke_z present in the schema but assembled as zeros).

    Parameters
    ----------
    o : dict
        Kernel per-ion output. Required keys: ``t_us, x, y, z, vx, vy,
        vz`` (per-step arrays). Optional: ``ex, ey, ez`` (per-step field,
        enables the e_* channels) and ``ncol`` (enables n_col). Callers
        with axis remaps (e.g. slim3d) pass an already-remapped dict.
    columns : sequence of str
        The record schema — normally ``spec.column_names()``.
    mz_da : float
        This ion's m/z in Da (for the KE channels).
    frame : OpticAxisSpec or None
        The axis the radial channels are measured about
        (SimSpec.radial_frame()); required when one is requested.

    Returns the dense (n, ncols) float64 array. Every column is written;
    an unfillable requested channel raises with a diagnostic naming what
    the kernel provided.
    """
    n = len(o["t_us"])
    tr = np.zeros((n, len(columns)))
    rec = TrajRecord(tr, columns)
    vx, vy, vz = o["vx"], o["vy"], o["vz"]
    base = {"t": o["t_us"], "x": o["x"], "y": o["y"], "z": o["z"],
            "vx": vx, "vy": vy, "vz": vz}
    m_kg = float(mz_da) * _AMU_KG
    # 0.5 m v^2 / e with v in mm/us (*1e3 -> m/s)
    _ke = 0.5 * m_kg * 1e6 / _E_CHG
    have_e = all(k in o for k in ("ex", "ey", "ez"))
    for c in columns:
        if c in base:
            rec[c][:] = base[c]
        elif c == "speed":
            rec[c][:] = np.sqrt(vx * vx + vy * vy + vz * vz)
        elif c == "ke_ev":
            rec[c][:] = _ke * (vx * vx + vy * vy + vz * vz)
        elif c == "ke_x":
            rec[c][:] = _ke * vx * vx
        elif c == "ke_y":
            rec[c][:] = _ke * vy * vy
        elif c == "ke_z":
            rec[c][:] = _ke * vz * vz
        elif c == "radius":
            # about the declared/route axis (L-539; this assumed z
            # through the origin while e_axial below assumed x)
            rec[c][:] = radial_values("radius", base, frame)
        elif c == "n_col":
            # No default: a missing count written as 0 reads as a
            # collisionless flight. Under SDS the provider passes
            # ncol = nan (sds.SDS_NO_COLLISION_COUNT, every route).
            if "ncol" not in o:
                raise ValueError(
                    "record channel 'n_col' was requested but the kernel "
                    "output carries no 'ncol' (pass nan for a model with "
                    f"no discrete collisions, e.g. SDS); provided: "
                    f"{sorted(o)}")
            rec[c][:] = float(o["ncol"])
        elif c in ("e_x", "e_y", "e_z", "e_field", "e_axial",
                   "e_radial") and have_e:
            ex, ey, ez = o["ex"], o["ey"], o["ez"]
            if c == "e_x":
                rec[c][:] = ex
            elif c == "e_y":
                rec[c][:] = ey
            elif c == "e_z":
                rec[c][:] = ez
            elif c == "e_field":
                rec[c][:] = np.sqrt(ex * ex + ey * ey + ez * ez)
            elif c in ("e_axial", "e_radial"):
                if frame is None:
                    radial_values(c, rec, None)       # refuses by name
                a = _AXES.index(frame.axis)
                comp = {"x": ex, "y": ey, "z": ez}
                p, q = _AXES[(a + 1) % 3], _AXES[(a + 2) % 3]
                rec[c][:] = (comp[frame.axis] if c == "e_axial" else
                             np.sqrt(comp[p] ** 2 + comp[q] ** 2))
        else:
            raise ValueError(
                f"assemble_record: requested channel {c!r} cannot be "
                f"derived from this kernel's outputs (has: "
                f"{sorted(o.keys())}). Remove it from record_channels for "
                "this path, or extend the kernel — a silent zero column "
                "is not an option (H8).")
    return tr
