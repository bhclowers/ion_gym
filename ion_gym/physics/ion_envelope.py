"""
ion_gym.ion_envelope — THE route-independent per-ion envelope.

Every fly wrapper (planar, r-z, the 3-D family) needs the same
scaffolding around its physics kernel: which mass this ion flies, which
seed reproduces it, and what its summary must contain. Through v281
each wrapper reimplemented that envelope by hand — the structural cause
of the 2026-08 parity escapes (mass desync, missing summary m/z). This
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
    and pass the base into per_ion."""
    raw = getattr(spec.source, "seed", 0)
    if raw is None:
        import numpy as _np
        base = int(_np.random.SeedSequence().entropy % (2 ** 31))
        if verbose:
            print(f"[seed] policy RANDOM: entropy base {base} drawn "
                  f"for this run -- set source.seed: {base} to "
                  f"reproduce it exactly")
        return base
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
    None coerced to 0, so 'random' silently meant 'always 0'.)"""
    s = spec.source
    if s.seed is not None:
        return int(s.seed)
    drawn = getattr(s, "_drawn_seed", None)
    if drawn is None:
        import secrets
        drawn = secrets.randbits(31)
        s._drawn_seed = drawn
        print(f"[seed] random mode: drew {drawn} — set source.seed={drawn} "
              "to reproduce this run", flush=True)
    return int(drawn)


def per_ion(spec, i, seed_base=None):
    """The two per-ion scalars every route derives identically:
    mz from sim_build.mz_of (the single mass authority — contiguous
    blocks, i // n_ions, matching generate_births), and the
    reproducibility seed spec.source.seed + i (the expression every
    wrapper used verbatim through v281; centralised so it can never
    diverge per route again)."""
    from ion_gym.physics.sim_build import mz_of
    return PerIon(mz=float(mz_of(spec, i)),
                  seed=(int(seed_base) if seed_base is not None
                        else resolve_run_seed(spec)) + int(i))


FATE_NAME = {0: "impact / exit", 1: "boundary exit", 2: "timeout",
             3: "bounding plane", 4: "transporter max_passes",
             5: "station impact plane", 6: "station detect"}

FATE_COLOR = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e", 3: "#9467bd",
              4: "#8c564b", 5: "#e377c2", 6: "#17becf"}


def fate_name(kind):
    """Label for a fate code, for display. THE one table (2026-09-12).

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
