"""
ion_gym.beam_interface
----------------------
The BEAM_INTERFACE ledger -- the contract between the beam
producer (extraction stack / transfer column) and the consumer (mirror /
analyzer), realised as a measurement at a named port plane
rather than a document convention.

A Ledger is built from a plane_gate.Recording plus the field it was
recorded in. Every row carries a KIND:
  MEASURED  computed from the recorded per-ray states at the plane;
  DECLARED  carried by declaration because this plane/model cannot
            measure it (e.g. Kv in the width plane, thermal turnaround
            for a deterministic bundle) -- honest, labelled, bounded;
  DERIVED   computed from MEASURED rows (waist reconstruction, emittance,
            correlations);
  BUDGET    consumer-side confrontation rows (mirror-acceptance
            predictions via the E2 law and a live mirror fan).

The ledger header carries the full plane definition (station, normal,
window, direction) so recordings are self-describing; the normal field is
written now (axis-aligned +s) so the P1 tilted-plane generalisation needs
no schema migration.

Ledgers serialise to JSON (round-trip gated in validate_beam_E3).
"""
import json
import time
from dataclasses import dataclass, asdict

import numpy as np

MEASURED, DECLARED, DERIVED, BUDGET = ("MEASURED", "DECLARED", "DERIVED",
                                       "BUDGET")


@dataclass
class Row:
    name: str
    value: float
    unit: str
    kind: str
    note: str = ""


class Ledger:
    def __init__(self, name, plane_def, mz, pot_ref, meta=None):
        self.name = name
        self.plane_def = dict(plane_def)
        if "normal" not in self.plane_def:
            self.plane_def["normal"] = [1.0, 0.0]     # +s, P1-ready
        self.mz = float(mz)
        self.pot_ref = float(pot_ref)
        self.meta = dict(meta or {})
        self.meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M"))
        self.rows = []

    def add(self, name, value, unit, kind, note=""):
        self.rows.append(Row(name, float(value), unit, kind, note))

    def get(self, name):
        for r in self.rows:
            if r.name == name:
                return r.value
        raise KeyError(name)

    # ---------------------------------------------------------- serialise
    def to_json(self, path=None):
        d = dict(name=self.name, plane_def=self.plane_def, mz=self.mz,
                 pot_ref=self.pot_ref, meta=self.meta,
                 rows=[asdict(r) for r in self.rows])
        s = json.dumps(d, indent=1, sort_keys=True)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        led = cls(d["name"], d["plane_def"], d["mz"], d["pot_ref"],
                  d["meta"])
        for r in d["rows"]:
            led.rows.append(Row(**r))
        return led

    def __str__(self):
        w = max(len(r.name) for r in self.rows) + 2
        out = [f"BEAM_INTERFACE ledger '{self.name}'  (m/z {self.mz:g}, "
               f"pot_ref {self.pot_ref:g} V)",
               f"  plane: {self.plane_def}"]
        for r in self.rows:
            out.append(f"  {r.name:<{w}}{r.value:>12.5g} {r.unit:<10}"
                       f"[{r.kind}] {r.note}")
        return "\n".join(out)


def ledger_from_recording(rec, field, mz, name, pot_ref=None,
                          declared=None, n_launched=None):
    """Build the ledger from a port-plane Recording flown in `field`.
    pot_ref: potential (field gauge) defining the drift reference for
    energy rows; default = potential on the plane axis (valid when the
    port sits in field-free drift -- assert-checked by the E3 gates,
    not assumed here). declared: list of (name, value, unit, note) rows
    this plane cannot measure."""
    a = rec.arrays()
    p = rec.plane
    if a["ray"].size < 2:
        raise ValueError("ledger needs >= 2 crossings")
    if pot_ref is None:
        pot_ref = float(field.pot(np.array([p.s0]), np.array([0.0]))[0])
    led = Ledger(name,
                 dict(station_s_mm=p.s0, role=p.role,
                      direction=p.direction, u_lo=p.u_lo, u_hi=p.u_hi,
                      plane_name=p.name),
                 mz, pot_ref)

    t, u, vs, vu = a["t"], a["u"], a["vs"], a["vu"]
    slope = vu / vs
    pot = field.pot(a["s"], u)
    K = mz * (vs**2 + vu**2) / (2 * 96.485) + pot - pot_ref
    led.add("n_crossings", t.size, "", MEASURED)
    if n_launched is not None:
        led.add("transmission_at_port", t.size / n_launched, "", MEASURED,
                "in-stack losses upstream of this port are real beam")

    def stats(name, x, unit, note=""):
        led.add(f"{name}_mean", x.mean(), unit, MEASURED, note)
        led.add(f"{name}_pp", np.ptp(x), unit, MEASURED)
        led.add(f"{name}_rms", x.std(), unit, MEASURED)

    stats("t", t, "us")
    stats("u", u, "mm")
    stats("slope", slope * 1e3, "mrad")
    stats("K", K, "eV", "drift-referenced")
    led.add("dK_over_K_pp", np.ptp(K) / K.mean(), "", DERIVED)

    # phase-space ellipse (centred second moments) and waist
    du, da = u - u.mean(), slope - slope.mean()
    m_uu, m_aa, m_ua = (du * du).mean(), (da * da).mean(), (du * da).mean()
    eps = np.sqrt(max(m_uu * m_aa - m_ua**2, 0.0))
    led.add("emittance_rms", eps * 1e3, "mm.mrad", DERIVED,
            "sqrt(<u2><a2>-<ua>2)")
    if m_aa > 0:
        z_w = -m_ua / m_aa
        w = np.sqrt(max(m_uu - m_ua**2 / m_aa, 0.0))
        led.add("waist_position", z_w, "mm", DERIVED,
                "downstream of the plane (+)")
        led.add("waist_size_rms", w, "mm", DERIVED)

    def corr(x, y):
        sx, sy = x.std(), y.std()
        return 0.0 if sx == 0 or sy == 0 else \
            ((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)

    led.add("corr_u_slope", corr(u, slope), "", DERIVED,
            "phase-space tilt; -1 = converging fan")
    led.add("corr_u_K", corr(u, K), "", DERIVED,
            "the previously unquantified deck item")
    if u.std() > 0:
        led.add("dK_du", ((u - u.mean()) * (K - K.mean())).mean()
                / (u.std()**2 + 1e-300), "eV/mm", DERIVED)
    led.add("dT_dK", (((K - K.mean()) * (t - t.mean())).mean()
                      / (K.std()**2 + 1e-300)) * 1e3, "ns/eV", DERIVED,
            "first-order t-K tilt the mirror drift can trade against")

    for d in (declared or []):
        led.add(d[0], d[1], d[2], DECLARED, d[3] if len(d) > 3 else "")
    return led


def confront_mirror(led, rec, leg_mm, mirror_fan, rail_mm=8.0,
                    c2_ns_mm2=7.06, floor_ns=1.5):
    """Consumer-side BUDGET rows: propagate the recorded rays ballistically
    over `leg_mm` to the mirror, apply the E2 acceptance law, and combine
    with a LIVE mirror energy fan.

    mirror_fan: callable dk_half -> ppdT_ns at the operational focus
    (validate_mirror_E2.axis_fan + focus; live numbers, not hardcoded).
    Appends rows to `led` and returns the predicted Rp."""
    a = rec.arrays()
    slope = a["vu"] / a["vs"]
    u_mir = a["u"] + slope * leg_mm
    alive = (np.abs(a["u"]) <= rail_mm) & (np.abs(u_mir) <= rail_mm)
    led.add("mirror_leg", leg_mm, "mm", BUDGET)
    led.add("transmission_to_mirror", alive.mean(), "", BUDGET,
            "rail-clip survival of the raw (uncolumned) fan")
    u_pp = np.ptp(u_mir[alive]) if alive.any() else np.nan
    led.add("u_mirror_pp", u_pp, "mm", BUDGET, "E2 law input")
    dk_half = led.get("dK_over_K_pp") / 2.0
    pp_K = mirror_fan(dk_half)
    pp_u = c2_ns_mm2 * (u_pp / 2.0)**2 * 2.0 if np.isfinite(u_pp) else \
        np.nan
    T0 = led.meta.get("T0_us", 21.53)
    pp = np.hypot(pp_K, pp_u)
    rp = T0 * 1e3 / (2.0 * np.hypot(pp, floor_ns))
    led.add("mirror_ppdT_energy", pp_K, "ns", BUDGET,
            f"live fan at +/-{dk_half:.2%}")
    led.add("mirror_ppdT_uplane", pp_u, "ns", BUDGET,
            f"c2 = {c2_ns_mm2} ns/mm^2 (small-u; worse above 1 mm)")
    led.add("Rp_predicted", rp, "", BUDGET,
            "quadrature + 1.5 ns floor; survivor-only")
    return rp
