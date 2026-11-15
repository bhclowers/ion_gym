"""
ion_gym.ion_envelope — THE route-independent per-ion envelope.

Every fly wrapper (planar, r-z, the 3-D family) needs the same
scaffolding around its physics kernel: which mass this ion flies, which
seed reproduces it, and what its summary must contain. Previously
each wrapper reimplemented that envelope by hand — the structural cause
of earlier parity escapes (mass desync, missing summary m/z). This
module is the one implementation; a wrapper that bypasses it is a
PARITY_RULE.md violation.

Deliberately NOT here:
- Birth refusal PREDICATES: each route refuses with its OWN kernel's
  impact rule (nearest-node in 2-D, all-eight-corner in 3-D, four-corner
  in tw2d) — the refusal boundary must equal that route's impact
  boundary, so the predicate cannot be shared. The *requirement* to
  guard is contract-level, enforced by test_route_conformance R2.
- tw2d: not spec-routed; mass/seed are direct arguments by design
  (enumerated in PARITY_RULE.md).
"""
from collections import namedtuple

PerIon = namedtuple("PerIon", ["mz", "seed"])


def run_seed_base(spec, verbose: bool = True) -> int:
    """Resolve the run-level seed policy:
    source.seed = <int>  -> FIXED / CRN: rerun of ion i is
                            bit-identical; A/B compares on the same
                            collision draws.
    source.seed = null   -> RANDOM: fresh entropy each run; the drawn
                            base is PRINTED so a lucky run can be
                            reproduced by pinning it.
    Absent behaves as 0 (legacy fixed).  Callers resolve ONCE per run
    and pass the base into per_ion.

    RANDOM goes through resolve_run_seed, the same single draw the births
    use. This once drew a second, independent base, so a 3-D run printed
    two seeds "to reproduce" it -- births from one, collisions from the
    other -- and pinning either one reproduced neither."""
    raw = getattr(spec.source, "seed", 0)
    if raw is None:
        return resolve_run_seed(spec)
    base = int(raw)
    if verbose:
        print(f"[seed] policy FIXED (CRN): source.seed = {base}; "
              f"reruns are bit-identical -- set source.seed: null "
              f"for fresh entropy per run")
    return base

# THE per-ion summary contract. test_route_conformance imports this —
# the gate tests the declared contract, not a private copy of it.
REQUIRED_SUMMARY_KEYS = ("kind", "tof", "mz", "x_end", "y_end", "z_end")


def resolve_run_seed(spec):
    """THE run seed, resolved once per build (without this, a
    random/seeded checkbox reads as broken — same impact points).
    source.seed set -> that number. source.seed None (random mode) ->
    draw fresh OS entropy ONCE, cache it on the spec so births and every
    per-ion kernel seed share the same base within a run, and PRINT it —
    an unrecorded random seed makes the run unreproducible, which
    violates the certified-numbers rule. (The old path was `seed or 0`:
    None coerced to 0, so 'random' silently meant 'always 0'.)
    sim_build.build_run clears the cache at the start of every build, so
    each build of a random-mode spec is a fresh run (a second build of the
    same spec object once repeated the first)."""
    s = spec.source
    if s.seed is not None:
        return int(s.seed)
    drawn = getattr(s, "_drawn_seed", None)
    if drawn is None:
        import secrets
        drawn = secrets.randbits(31)
        s._drawn_seed = drawn
        print(f"[seed] random mode: drew {drawn} for births and collisions "
              f"— set source.seed: {drawn} to reproduce this run exactly",
              flush=True)
    return int(drawn)


def per_ion(spec, i, seed_base=None):
    """The two per-ion scalars every route derives identically:
    mz from sim_build.mz_of (the single mass authority — contiguous
    blocks, i // n_ions, matching generate_births), and the
    reproducibility seed spec.source.seed + i (the expression every
    wrapper used verbatim before; centralised so it can never
    diverge per route again)."""
    from ion_gym.physics.sim_build import mz_of
    return PerIon(mz=float(mz_of(spec, i)),
                  seed=(int(seed_base) if seed_base is not None
                        else resolve_run_seed(spec)) + int(i))


# A summary that carries NO fate (damaged or legacy result) reads as this
# named code -- never as a real fate ("keep the unknown");
# consumers had guessed 2 (timeout) or -1.
UNKNOWN_FATE = -1

FATE_NAME = {0: "impact / exit", 1: "boundary exit", 2: "timeout",
             3: "bounding plane", 4: "transporter max_passes",
             5: "station impact plane", 6: "station detect",
             UNKNOWN_FATE: "unknown (no fate recorded)"}

FATE_COLOR = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e", 3: "#9467bd",
              4: "#8c564b", 5: "#e377c2", 6: "#17becf"}


def fate_name(kind):
    """Label for a fate code, for display. THE one table.

    It previously lived in four places — sim_app._FATE_NAME,
    stats.FATE_NAME, viz_core.FATE_NAMES and tracer3d._KIND_NAME — and
    the copies drifted: three of them stopped at 3, so the station fates
    the kernels have been emitting (5 impact_plane, 6 detect) had no
    label and no colour anywhere in the UI. An absorbed ion therefore
    looked like nothing had happened, which is part of why a working
    splat still read as "does nothing". Fates are constructed here by
    make_summary, so the naming lives here too and every consumer
    imports it.
    """
    return FATE_NAME.get(int(kind), f"unknown fate {int(kind)}")


def fate_color(kind):
    """Colour for a fate code, from the same single table."""
    return FATE_COLOR.get(int(kind), "#7f7f7f")


# --------------------------------------------------- summary accessors
# THE one way consumers read a per-ion summary. Each consumer used
# to fill an absent field with its own plausible value -- m/z 1.0, fate 2
# or -1, an end position of 0.0 -- so damaged or legacy results were shown
# and exported as if real. Absence is now NAMED: the fate reads
# UNKNOWN_FATE (labelled by fate_name), and a numeric field reads None,
# which the caller must report rather than replace.
def summary_fate(summary):
    """The fate code, or UNKNOWN_FATE when the summary carries none."""
    k = (summary or {}).get("kind")
    if k is None:
        return UNKNOWN_FATE
    try:
        return int(k)
    except (TypeError, ValueError):
        return UNKNOWN_FATE


def summary_value(summary, key):
    """A numeric summary field (mz, tof, x_end, ...) as float, or None when
    it is absent or not a finite number. Never a substitute value."""
    import math
    v = (summary or {}).get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def make_summary(*, kind, tof, mz, x_end, y_end, z_end, **extras):
    """Construct a per-ion summary that SATISFIES THE CONTRACT by
    construction: the six required keys are keyword-only arguments, so
    omitting one is a TypeError at the call site (the strongest refusal
    — the code does not run), and a None value refuses here with the
    key named (absence never presented as a value). Route-specific
    extras (r_end, n_col, ke_end, v*_end, ...) pass through untouched.
    """
    req = dict(kind=kind, tof=tof, mz=mz,
               x_end=x_end, y_end=y_end, z_end=z_end)
    for k, v in req.items():
        if v is None:
            raise ValueError(
                f"make_summary: required key '{k}' is None — a route "
                f"must supply a real value or refuse upstream, never "
                f"pass absence through")
    s = dict(kind=int(kind), tof=float(tof), mz=float(mz),
             x_end=float(x_end), y_end=float(y_end), z_end=float(z_end))
    s.update(extras)
    return s


def birth_splat(err, *, col_names, birth, mz, label):
    """THE born-in-metal outcome, identical on every route:
    the ion SPLATS AT BIRTH (fate 0, metal impact, at its birth point and
    birth time), the user is WARNED, and the other ions of the run fly on.
    A birth OUTSIDE the solved domain (birth_domain.BirthOutsideDomain, a
    BirthInMetal) is recorded the same way as fate 1, left the domain,
    with born_outside_domain=True.

    Every route's birth check raises raster2d.BirthInMetal; each per-ion
    flight entry (planar / r-z / 3-D fly_fn, the staged packet) catches
    that one class and returns this instead of aborting the ensemble.

    err   : the BirthInMetal raised (its message is the diagnostic)
    birth : the ion's birth row (x, y, z, vx, vy, vz, tob) in the frame the
            route RECORDS in
    Returns (traj, summary): a one-row record at birth -- base channels
    filled, every optional channel NaN (nothing was flown) -- and a fate-0
    summary carrying born_in_metal=True and the diagnostic. n_col is None:
    the ion never flew, so there is no count."""
    import numpy as np
    from ion_gym.physics.birth_domain import BirthOutsideDomain
    x, y, z, vx, vy, vz, tob = (float(v) for v in birth[:7])
    outside = isinstance(err, BirthOutsideDomain)
    print(f"[flight] WARNING: {label} was born "
          + ("outside the solved domain and leaves it at birth"
             if outside else "inside metal and splats at birth")
          + f" -- {err}", flush=True)
    traj = np.full((1, len(col_names)), np.nan)
    for name, v in zip(("t", "x", "y", "z", "vx", "vy", "vz"),
                       (tob, x, y, z, vx, vy, vz)):
        if name in col_names:
            traj[0, list(col_names).index(name)] = v
    # outside the domain: fate 1 (left the domain), which is what the
    # kernel records at the first step -- now with the reason attached
    flag = (dict(born_outside_domain=True) if outside
            else dict(born_in_metal=True))
    summary = make_summary(kind=1 if outside else 0, tof=tob, mz=mz,
                           x_end=x, y_end=y, z_end=z, n_col=None,
                           note=str(err), **flag)
    return traj, summary


# ------------------------------------------------------------ ion labels
# THE one formatter for naming an ion species on any surface. Every route stores the ion's MASS in Da (source.mz_list,
# the per-ion summary key "mz"; the names are historical) with the signed
# charge in source.charge. A label therefore shows either the mass or
# m/z = mass / |charge|, by the caller's DISPLAY MODE, and always the
# charge, so a mass is never presented as an m/z. Rules: the mode is
# a display setting (workbench selector, session only), the default is
# "mass", and charge is written in the mass-spectrometry form "2+" / "1−".
ION_LABEL_MODES = ("mass", "m/z")
# Numbers print to 10 significant figures, the precision of
# viz_core.mz_key: the label is what a filter option shows, so two masses
# the filter tells apart must never print alike.
DEFAULT_ION_LABEL = "mass"
_MINUS = "−"            # typographic minus for negative charge states


def charge_str(charge) -> str:
    """'2+', '1+', '1−' for a signed integer charge. Refuses 0 and
    non-integers: a neutral or fractional 'ion' is a deck error."""
    z = int(charge)
    if z != charge or z == 0:
        raise ValueError(f"charge_str: charge must be a nonzero integer, "
                         f"got {charge!r}")
    return f"{abs(z)}{'+' if z > 0 else _MINUS}"


def mz_value(mass_da, charge) -> float:
    """m/z of an ion of mass `mass_da` (Da) and signed charge `charge`."""
    z = int(charge)
    if z != charge or z == 0:
        raise ValueError(f"mz_value: charge must be a nonzero integer, "
                         f"got {charge!r}")
    return float(mass_da) / abs(z)


def ion_label(mass_da, charge, mode=DEFAULT_ION_LABEL) -> str:
    """Species label in the display mode: 'mass' -> '492.5 Da, 2+',
    'm/z' -> 'm/z 246.3 (2+)'. An unknown mode refuses by name.
    charge None (a record that carries no charge state) is allowed only
    in 'mass' mode and prints the mass alone; m/z cannot be formed
    without the charge, so that combination refuses."""
    if charge is None:
        if mode == "mass":
            return f"{float(mass_da):.10g} Da"
        raise ValueError("ion_label: m/z needs the ion's charge; this "
                         "record carries none")
    if mode == "mass":
        return f"{float(mass_da):.10g} Da, {charge_str(charge)}"
    if mode == "m/z":
        return f"m/z {mz_value(mass_da, charge):.10g} ({charge_str(charge)})"
    raise ValueError(f"ion_label: mode {mode!r}; one of {ION_LABEL_MODES}")


def ion_label_both(mass_da, charge) -> str:
    """Both quantities, for readouts whose physics depends on m/z
    (Mathieu q, pseudopotential, secular frequency, PE surface):
    'm/z 246.3 (492.5 Da, 2+)'. Independent of the display mode."""
    return (f"m/z {mz_value(mass_da, charge):.10g} "
            f"({float(mass_da):.10g} Da, {charge_str(charge)})")


def ion_mode_noun(mode=DEFAULT_ION_LABEL) -> str:
    """The quantity name a mode shows, for axis titles and filter names:
    'mass (Da)' or 'm/z'."""
    if mode == "mass":
        return "mass (Da)"
    if mode == "m/z":
        return "m/z"
    raise ValueError(f"ion_mode_noun: mode {mode!r}; one of "
                     f"{ION_LABEL_MODES}")
