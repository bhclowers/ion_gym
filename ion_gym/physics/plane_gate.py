"""
ion_gym.plane_gate
------------------
TASK P0: the Plane object -- terminate / record / port semantics for the
checked width-plane flyer, and the record->replay (emit) machinery that
hands packets between benches.

WHY ONE OBJECT: terminate, record
and port planes share the same geometry (an s = const station with an
optional u window and a declared crossing direction) and the same crossing
arithmetic -- the frozen Gate-C linear in-step interpolation that E1's
exit/domain events already use. Role is DATA, not code. The legacy exit
plane and the mirror-entrance RETURNED relabel become special cases of
this object rather than parallel machinery.

ROLES
  terminate : first-event-wins participant. Ranked AFTER the frozen s_end
              exit (bit-identity) and BEFORE electrode hits at equal chord
              fraction -- a detector face laid on metal reports the
              detector, not the metal. Fate = PLANE with the plane's name
              as the label. A finite u window is an APERTURE: rays outside
              the window pass through untouched.
  record    : transparent. Every qualifying crossing appends the
              interpolated state (ray, t, s0, u, vs, vu) to the plane's
              Recording -- multiple crossings per ray are kept (mirror
              entrance in/out), and crossings later than the ray's terminal
              event on the same step do not exist and are not logged.
  port      : a record plane whose Recording is meant to be EMITTED into
              the next bench (emit() / replay()). Mechanically identical
              to record; the role name documents intent and lets a bench
              assembler wire seams by role.

BEAM-INTERFACE HOOK: the beam-interface ledger is a Recording at a
named port/record
plane -- ledger rows are derived columns of these arrays (see derive_K).

CONVENTIONS
  * Crossing = strict passage of s0 between step endpoints (landing
    exactly ON the plane counts, per the frozen exit convention); a ray
    STARTING on the plane and moving away is not a crossing, so
    consecutive steps never double-count.
  * direction: +1 records only +s crossings, -1 only -s, 0 both.
  * u at the crossing must lie in [u_lo, u_hi] (closed) to qualify.
  * Interpolation of (t, u, vs, vu) at the crossing fraction is the same
    linear-in-step arithmetic as the frozen exit record(); s is s0 exact.
"""
from dataclasses import dataclass
import numpy as np



@dataclass(frozen=True)
class Plane:
    s0: float
    role: str = "record"            # "terminate" | "record" | "port"
    name: str = "plane"
    direction: int = 0              # +1 / -1 / 0 (both)
    u_lo: float = -np.inf
    u_hi: float = np.inf

    def __post_init__(self):
        if self.role not in ("terminate", "record", "port"):
            raise ValueError(f"unknown plane role {self.role!r}")
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1")


class Recording:
    """Append-only crossing ledger for one plane (grow-by-chunks lists,
    finalized to arrays on first read)."""

    def __init__(self, plane):
        self.plane = plane
        self._buf = []
        self._arr = None

    def _append(self, ray, t, u, vs, vu):
        if ray.size:
            self._buf.append((ray, t, u, vs, vu))
            self._arr = None

    def arrays(self):
        """dict of arrays: ray, t, s, u, vs, vu (crossing order)."""
        if self._arr is None:
            if self._buf:
                ray, t, u, vs, vu = (np.concatenate(c) for c in
                                     zip(*self._buf))
            else:
                ray = np.zeros(0, int)
                t = u = vs = vu = np.zeros(0)
            self._arr = dict(ray=ray, t=t,
                             s=np.full(ray.size, self.plane.s0),
                             u=u, vs=vs, vu=vu)
        return self._arr

    def __len__(self):
        return self.arrays()["ray"].size


def crossing_fraction(planes, s, u, s_n, u_n, live):
    """Chord fractions to each plane. Returns (n_planes, n_rays) array of
    f in [0, 1] (inf = no qualifying crossing). Frozen-exit arithmetic:
    f = (s0 - s) / (s_n - s), evaluated only for strict passage."""
    n = s.size
    F = np.full((len(planes), n), np.inf)
    ds = s_n - s
    for ip, p in enumerate(planes):
        fwd = (s < p.s0) & (s_n >= p.s0)
        bwd = (s > p.s0) & (s_n <= p.s0)
        if p.direction == 1:
            hit = fwd
        elif p.direction == -1:
            hit = bwd
        else:
            hit = fwd | bwd
        hit &= live
        if not hit.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(hit, (p.s0 - s) / ds, np.inf)
        u_c = u + f * (u_n - u)
        ok = hit & (u_c >= p.u_lo) & (u_c <= p.u_hi)
        F[ip] = np.where(ok, f, np.inf)
    return F


# ------------------------------------------------------------ emit/replay
def emit(recording, sel=None):
    """Recording -> launch kwargs for fly_checked at the plane station.
    sel: optional boolean/index selection over crossings. Returns
    (kwargs, t0): initial state dict and the per-ray time offsets that a
    replayed flight's times must be added to (total = t0 + t_downstream).
    """
    a = recording.arrays()
    if sel is None:
        sel = slice(None)
    kw = dict(s0=a["s"][sel].copy(), u0=a["u"][sel].copy(),
              vs0=a["vs"][sel].copy(), vu0=a["vu"][sel].copy())
    return kw, a["t"][sel].copy()


def replay(field, geom, recording, sel=None, **fly_kw):
    """Emit a recording into a fresh fly_checked over (field, geom).
    Returns the fly result with res['t'] upgraded to TOTAL time
    (recording time + downstream time) and res['t_leg'] the downstream
    leg alone. Restarting the integrator at an interpolated state resets
    the velocity-Verlet phase: agreement with a through-flight is
    O(dt^2) local truncation, gated (not assumed) in validate_plane_P0.
    """
    from ion_gym.physics.flyer_plane import fly_checked
    kw, t0 = emit(recording, sel)
    res = fly_checked(field, geom, **kw, **fly_kw)
    res["t_leg"] = res["t"].copy()
    res["t"] = res["t"] + t0
    return res


def derive_K(recording, field, mz):
    """Ledger helper: total energy (eV, gauge of `field`) per crossing
    from the recorded kinematics -- K = m(vs^2+vu^2)/2q + phi(s, u)."""
    a = recording.arrays()
    return mz * (a["vs"]**2 + a["vu"]**2) / (2 * 96.485) \
        + field.pot(a["s"], a["u"])
