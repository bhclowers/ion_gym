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


# constants for derived-channel physics (duplicated nowhere else in this
# module's callers; the tracer kernels carry their own numba-local copies)
_AMU_KG = 1.66053906660e-27
_E_CHG = 1.602176634e-19


def assemble_record(o, columns, mz_da):
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
            rec[c][:] = np.sqrt(o["x"] ** 2 + o["y"] ** 2)
        elif c == "n_col":
            rec[c][:] = float(o.get("ncol", 0))
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
            elif c == "e_axial":
                rec[c][:] = ex          # transport axis is x
            elif c == "e_radial":
                rec[c][:] = np.sqrt(ey * ey + ez * ez)
        else:
            raise ValueError(
                f"assemble_record: requested channel {c!r} cannot be "
                f"derived from this kernel's outputs (has: "
                f"{sorted(o.keys())}). Remove it from record_channels for "
                "this path, or extend the kernel — a silent zero column "
                "is not an option (H8).")
    return tr
