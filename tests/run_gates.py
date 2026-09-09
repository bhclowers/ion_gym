"""run_gates.py -- run the SCRIPT-STYLE gates, the ones pytest cannot collect.

Why this exists
---------------
Some gates in this folder are pytest modules (`def test_...`).  Others are
standalone scripts that run on import and sys.exit() with a pass/fail count.
conftest.py detects the second kind STRUCTURALLY and adds them to
`collect_ignore`, because pytest importing a script that calls sys.exit kills
the whole run with INTERNALERROR before a single test executes.

That much was right.  But `collect_ignore` only stops pytest from RUNNING them
-- and the runner conftest pointed at (this file) was never written.  So after
the flat -> tests/ move, four gates were silently not executed:

    test_ui_smoke        (30 checks -- the gate that catches UI crashes)
    test_polygon_raster
    test_scene_resolution
    test_ele_labels

`pytest tests/ -q` went green while a third of the gate surface was dark.  A
gate that silently does not run is worse than no gate: no gate is a known hole,
a dark gate is a false all-clear.  Hence: DISCOVERED, not listed.  A hardcoded
list would rot the first time someone adds a script -- and would have rotted
exactly the way this did.

Exit code is nonzero if ANY script gate fails, so CI cannot go green on a
dark gate.
"""
import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def script_gates() -> list[Path]:
    """Every gate script: a test_*.py with no `def test_` in it, PLUS every
    validate_*.py.  Same structural rule conftest uses, so the two can never
    disagree about which files are scripts.

    validate_* is where the PHYSICS is certified -- the gbstack gates, the
    immersion gates, the mirror gates, compact stack, the co-optimisation.  The
    first version of this runner globbed test_* only, so 32 of the 38
    validate_* gates sat broken and unnoticed behind a green `pytest tests/`.
    A runner that does not run the gates that certify the numbers is a runner
    that certifies nothing."""
    out = []
    for p in sorted(list(HERE.glob("test_*.py")) + list(HERE.glob("validate_*.py"))):
        tree = ast.parse(p.read_text())
        has_pytest_fn = any(
            isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            for n in ast.walk(tree))
        if not has_pytest_fn:
            out.append(p)
    return out


def main() -> int:
    import argparse
    from tiers import tier_of, TIERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2, 3],
                    help="run tiers 1..N (0 = ALL; the release gate is 0)")
    args = ap.parse_args()

    gates = script_gates()
    if not gates:
        print("no script gates found -- has the structural rule drifted from "
              "conftest? refusing to report success on an empty run")
        return 1

    # REFUSE an unclassified gate before running ANYTHING.  A gate discovered on
    # disk but absent from tiers.py is a gate nobody decided about; running it
    # (or not) would both be guesses.
    try:
        plan = [(p, tier_of(p.stem)) for p in gates]
    except KeyError as e:
        print(f"REFUSING TO RUN: {e.args[0]}")
        return 1

    want = [3, 2, 1] if args.tier == 0 else list(range(1, args.tier + 1))
    run = [p for p, t in plan if t in want]
    skip = [(p, t) for p, t in plan if t not in want]

    # NO-SILENCE BINDS THE RUNNER.  The suite once went green while 32 validators were dark: a
    # tier that is not run and not VISIBLE is indistinguishable from one that
    # passed.  So every skip is NAMED.  A skipped gate that announces itself is
    # a schedule; a skipped gate that is silent is a lie.
    lbl = "ALL TIERS (release)" if args.tier == 0 else \
          f"tiers 1..{args.tier} ({', '.join(TIERS[t] for t in want)})"
    print(f"script gates: {len(run)} of {len(gates)} -- {lbl}\n")
    if skip:
        print(f"NOT RUN ({len(skip)} gates, BY DESIGN -- this is a schedule, "
              f"not a pass):")
        for t in sorted({t for _, t in skip}):
            names = sorted(p.stem for p, tt in skip if tt == t)
            print(f"    tier {t} {TIERS[t]}: {', '.join(names)}")
        print("    -> `python3 tests/run_gates.py` (no --tier) runs "
              "everything. NOTHING IS CERTIFIED UNTIL IT DOES.\n")

    failed = []
    for p in run:
        print(f"--- {p.name} " + "-" * (58 - len(p.name)))
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT,
                           capture_output=True, text=True)
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-3:]
        for ln in tail:
            print("   ", ln)
        if r.returncode != 0:
            failed.append(p.name)
            print(f"    !! FAILED (exit {r.returncode})")
            if r.stderr.strip():
                print("   ", r.stderr.strip().splitlines()[-1])
        print()

    print("=" * 66)
    if failed:
        print(f"SCRIPT GATES: {len(run) - len(failed)} passed, "
              f"{len(failed)} FAILED -> {', '.join(failed)}")
        return 1
    if skip:
        print(f"SCRIPT GATES: all {len(run)} RUN gates passed -- "
              f"{len(skip)} NOT RUN (see above). NOT A RELEASE CERTIFICATION.")
        return 0
    print(f"SCRIPT GATES: all {len(gates)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
