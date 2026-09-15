"""
ion_gym.sim_config
------------------
A self-describing, versioned simulation configuration: one JSON object
that captures everything needed to reproduce a run — physics parameters,
the ion source, the RF drive, AND which per-step trajectory channels to
record. Load it, edit it (by hand or via the Panel app), save it; a saved
config is a complete, human-readable simulation script.

Design:
  * SimConfig is a plain dataclass with typed fields and per-field
    metadata (label, unit, kind, range) carried in FIELD_SPECS, so the
    UI can build itself from the schema rather than duplicating it (the
    "self-describing" part — add a field here and the app grows a
    control for it).
  * record_channels selects which quantities each ion's trajectory
    stores. The base channels (t,x,y,z,vx,vy,vz) are always present;
    optional channels (speed, ke_ev, e_field, e_axial, e_radial, radius,
    n_col) are computed in the kernel and appended, so downstream code
    reads named columns, not magic indices.
  * to_json/from_json round-trip losslessly; validate() catches
    out-of-range or unknown values before a run starts.

Versioned: SCHEMA_VERSION bumps if the field set changes; from_json
tolerates older configs by filling defaults for new fields.
"""

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import List

SCHEMA_VERSION = 1

# The optional, per-step recordable channels beyond the base kinematics.
# name -> (label, unit). Base channels t,x,y,z,vx,vy,vz are implicit and
# always stored first, in that order.
# SINGLE OWNER. This module carried its own
# OPTIONAL_CHANNELS dict that had DIVERGED from the one in sim_spec -- two
# definitions of the same vocabulary, guaranteed to drift, and the drift was
# already real. Its only importer (the funnel ionbench) was retired the same
# day. Delegate to the owner: sim_spec declares the channels, everyone reads
# them from there.
from ion_gym.io.sim_spec import OPTIONAL_CHANNELS   # noqa: F401  (re-export)
BASE_CHANNELS = ["t", "x", "y", "z", "vx", "vy", "vz"]


# Per-field UI/validation metadata: (label, unit, kind, lo, hi, step).
# kind in {int, float, choice, bool, multichoice}. For choice/multichoice
# the (lo) slot carries the option list.
FIELD_SPECS = {
    "n_ions":   ("ions", "", "int", 1, 2000, 10),
    "disc_mm":  ("source disc radius", "mm", "float", 0.0, 20.0, 0.5),
    "src_x0":   ("source x0", "mm", "float", -200.0, 200.0, 0.5),
    "src_y0":   ("source y0", "mm", "float", -200.0, 200.0, 0.5),
    "src_z0":   ("source z0", "mm", "float", -200.0, 200.0, 0.5),
    "src_dist": ("source distribution", "", "choice",
                 ["disc", "point"], None, None),
    "ke_lo":    ("birth KE min", "eV", "float", 0.0, 50.0, 0.05),
    "ke_hi":    ("birth KE max", "eV", "float", 0.0, 50.0, 0.05),
    "mz":       ("m/z", "Da", "float", 1.0, 1e5, 1.0),
    "gas":      ("buffer gas", "", "choice",
                 ["He", "H2", "N2", "air", "Ar", "CO2", "Kr", "Xe"],
                 None, None),
    "T_k":      ("temperature", "K", "float", 4.0, 1000.0, 1.0),
    "P_pa":     ("pressure", "Pa", "float", 0.0, 5000.0, 1.0),
    "sigma_m2": ("collision sigma", "m^2", "float", 1e-20, 1e-16, None),
    "rf_V":     ("RF amplitude", "V", "float", 0.0, 500.0, 5.0),
    "rf_freq_mhz": ("RF frequency", "MHz", "float", 0.05, 5.0, 0.05),
    "dc_first":   ("DC ring 1", "V", "float", -500.0, 500.0, 1.0),
    "dc_gradient": ("DC gradient", "V/ring", "float", -20.0, 20.0, 0.1),
    "dt_ns":    ("time step", "ns", "float", 0.1, 20.0, 0.1),
    "t_max_us": ("max flight", "us", "float", 10.0, 5000.0, 10.0),
    "rec_every": ("record stride (steps)", "", "int", 1, 2000, 10),
    "use_csv_births": ("use reference exact births", "", "bool",
                       None, None, None),
    "seed0":    ("base seed", "", "int", 0, 10_000_000, 1),
}


@dataclass
class SimConfig:
    # --- ion source
    n_ions: int = 100
    disc_mm: float = 4.0
    src_x0: float = 1.0
    src_y0: float = 0.0
    src_z0: float = 0.0
    src_dist: str = "disc"          # point | disc (planar-ready for einzel)
    ke_lo: float = 0.1
    ke_hi: float = 1.9
    mz: float = 556.0
    use_csv_births: bool = False
    # --- buffer gas
    gas: str = "N2"
    T_k: float = 273.0
    P_pa: float = 133.28
    sigma_m2: float = 2.27e-18
    # --- RF drive
    rf_V: float = 50.0
    rf_freq_mhz: float = 0.5
    dc_first: float = 100.0
    dc_gradient: float = -2.1333   # V/ring; default reproduces 100->68 over 15
    # --- integration / recording
    dt_ns: float = 1.0
    t_max_us: float = 400.0
    rec_every: int = 80
    record_channels: List[str] = field(
        default_factory=lambda: ["speed", "ke_ev", "e_field"])
    seed0: int = 60601
    # --- provenance
    name: str = "funnel run"
    notes: str = ""

    # ---------------------------------------------------------------
    def column_names(self):
        """Ordered channel names for a recorded trajectory: base + the
        selected optional channels (validated order)."""
        opt = [ch for ch in self.record_channels
               if ch in OPTIONAL_CHANNELS]
        return BASE_CHANNELS + opt

    def channel_flags(self):
        """Boolean tuple over OPTIONAL_CHANNELS order — the kernel takes
        these to decide which extra columns to fill (numba-friendly)."""
        return tuple(ch in self.record_channels
                     for ch in OPTIONAL_CHANNELS)

    def validate(self):
        errs = []
        for name, spec in FIELD_SPECS.items():
            kind = spec[2]
            v = getattr(self, name)
            if kind in ("int", "float"):
                lo, hi = spec[3], spec[4]
                if lo is not None and v < lo:
                    errs.append(f"{name}={v} < {lo}")
                if hi is not None and v > hi:
                    errs.append(f"{name}={v} > {hi}")
            elif kind == "choice" and v not in spec[3]:
                errs.append(f"{name}={v!r} not in {spec[3]}")
        if self.ke_hi < self.ke_lo:
            errs.append("ke_hi < ke_lo")
        for ch in self.record_channels:
            if ch not in OPTIONAL_CHANNELS:
                errs.append(f"unknown record channel {ch!r}")
        return errs

    # ---------------------------------------------------------------
    def to_json(self, path=None, indent=2):
        d = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        s = json.dumps(d, indent=indent)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_json(cls, path_or_str):
        s = str(path_or_str)
        # a JSON document starts with '{'; anything else we try as a path.
        # (Path.exists() raises OSError on very long strings, so never
        # stat a candidate that is obviously JSON.)
        if s.lstrip().startswith("{"):
            raw = s
        else:
            p = Path(s)
            raw = p.read_text() if p.exists() else s
        d = json.loads(raw)
        d.pop("schema_version", None)
        known = {f.name for f in fields(cls)}
        # forward-compatible: ignore unknown keys, default missing ones
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)

    def copy_with(self, **changes):
        d = asdict(self)
        d.update(changes)
        return SimConfig(**d)


def default_config():
    return SimConfig()
