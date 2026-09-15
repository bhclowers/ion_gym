"""test_provenance.py -- gate P: origin is recorded, and never invented.

Charter P1-P5 (v144).  A number is worth what its origin is worth.  A formula
from a paper, a geometry read off a figure, a constant tuned by hand, and a
value an external tool happened to print are four different kinds of claim; once they are
all just floats in a module they are indistinguishable, and that is how a fitted
fudge factor gets cited as physics.

This gate does NOT check that citations are correct -- no gate can.  It checks
the two things a gate CAN check:
  1. the physics-bearing modules carry a PROVENANCE block at all;
  2. 'unknown' is spelled, not omitted.  P2: inventing a plausible citation is
     the worst outcome available, worse than silence, because a fabricated
     provenance survives review.

The list SHRINKS by adding provenance, never by deleting a name from it.
"""
import _bootstrap  # noqa: F401
import pathlib
import sys
from pathlib import Path

# Modules that assert a NUMBER about the physical world.
# SHIPPED physics modules only (the
# public tree may not reference internals). Internal physics modules
# register their own duties beside themselves,
# merged below when that tree is present; on a public checkout the
# shipped list is complete and the merge is a stated no-op.
NEEDS_PROVENANCE = [
    # q3r_geom.py REMOVED 2026-09-08: the Q3 study moved to
    # internal/projects/q3/ with its gate. This list is SHIPPED
    # modules only, so a module that no longer ships must not be
    # required here -- it failed as "module missing", which reads
    # as a lost provenance duty rather than a deliberate move. Its
    # duty travels with it, in the internal registration below.
    "w1_wiley.py", "sds.py",
    # Two example modules were REMOVED with
    # ion_gym/examples/: geometry-building Python left the package, and the
    # decks they built now ship as JSON. A list of files that must carry
    # provenance cannot name files the package no longer ships.
    # The five recovered legacy modules, provenance backfilled at
    # reconciliation (the list grows by adding provenance, never shrinks)
    "tracer_tdep.py", "tracer_gas.py",
    # quad_scene_build.py removed from the package (zero live callers).
]
FIELDS = ("origin", "derived", "extracted", "authored", "verified", "note")
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _merged_list():
    lst = list(NEEDS_PROVENANCE)
    extra = Path(__file__).resolve().parents[1] / "internal" / \
        "tests_oaTOF" / "provenance_internal.py"
    if extra.exists():
        ns = {}
        exec(extra.read_text(), ns)
        lst += list(ns["EXTRA"])
        print(f"  (+ {len(ns['EXTRA'])} internal modules from the dev tree)")
    return lst


def main():
    bad, ok = [], 0
    for f in _merged_list():
        try:
            p = _bootstrap.locate(f)
        except FileNotFoundError:
            bad.append((f, "module missing")); continue
        src = p.read_text()
        if "# PROVENANCE" not in src:
            bad.append((f, "no PROVENANCE block")); continue
        block = src.split("# PROVENANCE", 1)[1].split("\n\n", 1)[0]
        keys = [k for k in FIELDS if f"#   {k}" in block]
        if not keys:
            bad.append((f, "PROVENANCE block has no recognised field")); continue
        if "origin" not in keys:
            bad.append((f, "no origin line -- say 'unknown' if it is unknown"))
            continue
        note = "  (origin: unknown -- STATED, per P2)" if "unknown" in block.split(
            "origin", 1)[1].split("\n")[0] else ""
        print(f"  {f:28s} {', '.join(keys)}{note}")
        ok += 1

    print()
    if bad:
        print("PROVENANCE: FAIL")
        for f, why in bad:
            print(f"   {f}: {why}")
        return 1
    print(f"PROVENANCE: PASS ({ok} physics modules carry an origin; "
          f"'unknown' is spelled where it is unknown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
