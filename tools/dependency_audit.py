"""tools/dependency_audit.py — code-placement / dependency-direction audit.

Enforces the K1–K9 placement rules (CODE PLACEMENT + DRY GUARD RAILS), mechanically,
so the einzel-retirement regression (a reusable helper trapped in a test file,
imported by two live gates) cannot recur unnoticed. Pure AST analysis: reads
source, writes nothing, runs nothing it audits. Zero compute beyond parsing.

What it reports, by rule:

  K2 (SEVERITY 1)  a CORE module importing from a test/validate module.
                   Core is load-bearing on the test suite — the hard
                   violation. FAILS the audit (exit 1).
  K2 (SEVERITY 2)  a test/validate module importing ANY symbol from another
                   test/validate module — fixtures and *_spec included
                   (K9, GOVERNING). A shared test symbol is a shared
                   dependency; moving/retiring its host breaks importers at
                   collection. FAILS.
  K4/K7 (REVIEW)   the same public symbol defined in 2+ live modules. NOT an
                   automatic failure: most are coincidental (K7) or ubiquitous
                   entry points. Reported for a human MOVE/MERGE/LEAVE call;
                   does not affect exit code unless --strict-dupes.
  K5 (INFO)        for a named --retire target, every live importer of every
                   symbol it defines (the grep-for-the-symbol check).

Run:
  python tools/dependency_audit.py                 # audit, human report
  python tools/dependency_audit.py --json          # machine-readable
  python tools/dependency_audit.py --retire NAME    # K5 pre-retirement check
  python tools/dependency_audit.py --strict-dupes   # dup symbols also fail

Exit code: 0 clean, 1 if any SEVERITY-1/2 violation (CI/pre-commit gate).

# PROVENANCE
#   origin    the K1–K9 placement rules, authored from the einzel
#             regression + the dry-principle skill review
#   authored  the acceptance signature: an import whose
#             SOURCE is less stable / not the conceptual owner of the IMPORTER
"""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

# Repo root = this file's parent's parent (tools/ lives at the repo root).
# Derived, never hardcoded (no per-machine paths).
from _repo_root import repo_root       # marker, not position
REPO = repo_root()

# Entry-point / dunder names that are legitimately defined in many modules;
# a shared name here is not a DRY fork. Extend deliberately, with reason.
UBIQUITOUS = {"main", "build", "run", "load", "setup", "gate", "check",
              "report", "store", "spec", "bundle", "evaluate", "parse"}


def _is_test(rel: Path) -> bool:
    """A module is 'test-side' if it lives under tests/ or is named
    test_*/validate_*. These are the LEAST stable modules; nothing more
    stable may depend on them (K2)."""
    return (rel.parts and rel.parts[0] == "tests") or \
           rel.name.startswith(("test_", "validate_"))


def _is_retired(rel: Path) -> bool:
    return "retired" in rel.parts


CORE_DIRS = frozenset({"physics", "io", "viz", "ui", "cad"})


# Register of tests DECLARED to own a project -- i.e. they ARE that project's
# own gates, so importing it is legitimate. A test belongs to exactly ONE
# project and reaches only into it; a core-concern test reaches into none. Any
# test importing a project it is not registered for is a project-isolation
# violation (this is what step-2 severed the six core gates for). New project
# gates are added here CONSCIOUSLY -- that deliberate act is the ratchet.
# The oa-TOF instrument (immersion/mirror/gbstack/instrument) is ONE project,
# oaTOF, so a gate spanning several of its subsystems still owns just "oaTOF".
TEST_PROJECT_OWNERS = {
    # -- q3 --
    # The cadquery_emit gate's C4 case is a
    # DECLARED regression pin against the real q3 caller ("the
    # abstraction did not break the real caller") — the q3 import is the
    # point, not a leak. Registry catches up to the design.
    "test_cadquery_emit": "q3",
    # -- oaTOF (the oa-TOF instrument and its subsystems) --
    "test_W1": "oaTOF", "test_W2": "oaTOF", "test_W3": "oaTOF",
    "test_W3d": "oaTOF", "test_bench_M_PA0": "oaTOF",
    "test_bench_M_PA1": "oaTOF", "test_bench_M_PA1_certified": "oaTOF",
    "test_bench_M_PA2": "oaTOF", "test_bench_M_PA3": "oaTOF",
    "test_bench_crop": "oaTOF", "test_bench_is_grid": "oaTOF",
    "test_export_frame_agnostic": "oaTOF", "test_instrument_diagram": "oaTOF",
    "test_mirror_fidelity": "oaTOF", "test_spec_loader": "oaTOF",
    "validate_beam_E3": "oaTOF", "validate_bench_certified": "oaTOF",
    "validate_gbstack_gateA": "oaTOF", "validate_gbstack_gateB": "oaTOF",
    "validate_gbstack_gateC": "oaTOF", "validate_gbstack_gateD": "oaTOF",
    "validate_compact_gateC": "oaTOF", "validate_coopt_v100": "oaTOF",
    "validate_hit2d": "oaTOF", "validate_immersion_gateACD": "oaTOF",
    "validate_immersion_gateB": "oaTOF", "validate_mirror_E2": "oaTOF",
    "validate_mirror_Mnew": "oaTOF", "validate_mock_detector": "oaTOF",
    "validate_plane_P0": "oaTOF", "validate_viz_M_VIZ": "oaTOF",
    # -- q3 (parked leaf project) --
    "validate_q3_2d": "q3",
}


def _layer(rel: Path) -> str:
    """Package-aware layer of a repo-relative .py path.

    FOUNDATION -> UPPER is forbidden (a governing rule): core and
    examples reach into core only; projects reach into core (+ sibling
    subsystems, tolerated) but q3 is a LEAF no sibling may import; tests and
    studies are scaffolding that may import anything (except test<-test, K9).
    """
    if _is_retired(rel):
        return "retired"
    if _is_test(rel):
        return "test"
    parts = rel.parts
    if parts and parts[0] == "ion_gym" and len(parts) >= 2:
        if parts[1] in CORE_DIRS:
            return "core"
        if parts[1] == "projects":
            return "project"
        if parts[1] == "examples":
            return "example"
        return "core"                        # ion_gym/__init__.py, version, ...
    if parts and parts[0] in ("studies", "tools"):
        return "aux"
    if len(parts) == 1:
        return "core"                        # legacy root module (none post-move)
    return "other"


def _project_of(rel: Path):
    """Project name for an ion_gym/projects/<name>/... path, else None."""
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "ion_gym" and parts[1] == "projects":
        return parts[2]
    return None


def _target_layer(module: str, test_stems):
    """(layer, project|None) of an import TARGET. Package imports carry their
    full dotted path (ion_gym.projects.q3.q3_cell), so the target layer is read
    from the PATH, not from the first component (which is always 'ion_gym')."""
    comps = module.split(".")
    if comps[0] == "ion_gym" and len(comps) >= 2:
        if comps[1] in CORE_DIRS:
            return ("core", None)
        if comps[1] == "projects":
            return ("project", comps[2] if len(comps) >= 3 else None)
        if comps[1] == "examples":
            return ("example", None)
        return ("core", None)
    if module in test_stems:
        return ("test", None)
    return (None, None)                      # stdlib / external / non-test


def _iter_py():
    for p in REPO.rglob("*.py"):
        rel = p.relative_to(REPO)
        if any(x in rel.parts for x in
               ("__pycache__", ".venv", "node_modules", "tools")):
            continue
        yield p, rel


def _defines(tree):
    """Top-level public symbols a module defines."""
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))
            and not n.name.startswith("_")}


def _all_defines(tree):
    """Every top-level def/class name (incl. private), for K5 symbol grep."""
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))}


def audit():
    files = list(_iter_py())
    trees = {}
    for p, rel in files:
        try:
            trees[rel] = ast.parse(p.read_text())
        except SyntaxError as e:
            # A file that will not parse is a real problem — surface it,
            # never skip silently.
            raise SystemExit(f"AUDIT ABORTED: {rel} does not parse: {e}")

    test_stems = {rel.stem for rel in trees if _is_test(rel) and
                  not _is_retired(rel)}

    sev1, sev2, dupes, symbol_home = [], [], [], defaultdict(list)
    layering, xcode, testiso = [], [], []    # layout-aware edges (all gated)

    for rel, tree in trees.items():
        if _is_retired(rel):
            continue
        ilayer = _layer(rel)
        iproj = _project_of(rel)
        # record symbol homes for K4/K7
        for s in _defines(tree):
            symbol_home[s].append(rel.stem)
        # scan imports (K2 + layering), carrying the FULL module path
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods = [(n.module, [a.name for a in n.names])]
            elif isinstance(n, ast.Import):
                # plain `import test_x [as y]` is the same dependency —
                # the form test_W2 used to slip past the first audit
                mods = [(a.name, ["<module>"]) for a in n.names]
            for module, names in mods:
                if module.split(".")[0] in ("tiers", "_bootstrap"):
                    # tiers = runner config; _bootstrap = sys.path infra --
                    # importing them couples no helper code
                    continue
                tlayer, tproj = _target_layer(module, test_stems)
                if tlayer is None:
                    continue
                rec = {"importer": str(rel), "source": module, "names": names}
                # --- K2: importing a TEST module ---
                if tlayer == "test" and module != rel.stem:
                    if ilayer == "core":
                        sev1.append(rec)
                    elif ilayer == "test":
                        # K9 (GOVERNING): NO test imports ANY
                        # symbol from another test — fixtures included. Every
                        # cross-test import is a violation, no name filter.
                        sev2.append(rec)
                # --- LAYERING RULE: a FOUNDATION layer must never
                #     reach UP into projects/examples. core imports core only;
                #     examples import core only. ---
                if (ilayer == "core" and tlayer in ("project", "example")) or \
                   (ilayer == "example" and tlayer == "project"):
                    layering.append(rec)
                # --- CROSS-PROJECT CODE: a project's code
                #     may be imported only by that project (its own code and its
                #     own tests). ANY project importing ANOTHER project's code is
                #     forbidden. This generalises the old q3-leaf rule to every
                #     project and retires the 'tolerated composition' review: the
                #     oa-TOF subsystems are ONE project (oaTOF), so intra-oaTOF
                #     edges are same-project and never fire here. ---
                if ilayer == "project" and tlayer == "project" \
                        and iproj and tproj and iproj != tproj:
                    xcode.append(rec)
                # --- TEST PROJECT-ISOLATION: a test may import
                #     ONLY the project it is registered to own. A test importing a
                #     project it does not own -- or that is not a declared project
                #     gate at all (a core-concern test reaching in) -- is a
                #     violation. This closes the gap that let the six step-2 core
                #     gates reach into projects for fixtures. ---
                if ilayer == "test" and tlayer == "project" and tproj:
                    owner = TEST_PROJECT_OWNERS.get(rel.stem)
                    if owner != tproj:
                        testiso.append(dict(rec, owns=owner or "<undeclared>",
                                            imports=tproj))

    for s, homes in symbol_home.items():
        uniq = sorted(set(homes))
        if len(uniq) > 1 and s not in UBIQUITOUS:
            dupes.append({"symbol": s, "modules": uniq})

    return {"severity1_core_from_test": sev1,
            "severity2_test_helper_from_test": sev2,
            "severity1_layering_foundation_into_upper": layering,
            "severity1_cross_project_code": sorted(
                xcode, key=lambda r: (r["importer"], r["source"])),
            "severity1_test_project_isolation": sorted(
                testiso, key=lambda r: (r["importer"], r["source"])),
            "review_duplicate_symbols": sorted(dupes,
                                               key=lambda d: d["symbol"])}


def retire_check(name):
    """K5: for a file about to be retired/moved, list every live importer of
    every symbol it defines — the grep-the-symbol check, not grep-the-name."""
    target = None
    for p, rel in _iter_py():
        if rel.stem == name:
            target = (p, rel)
            break
    if target is None:
        # also look in retired/ (already moved) to still report importers
        for p in REPO.rglob(f"{name}.py"):
            target = (p, p.relative_to(REPO))
            break
    if target is None:
        raise SystemExit(f"no module named {name!r} found")
    p, rel = target
    syms = _all_defines(ast.parse(p.read_text()))
    hits = []
    for q, qrel in _iter_py():
        if qrel.stem == name:
            continue
        try:
            t = ast.parse(q.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module:
                if n.module.split(".")[0] == name:
                    used = [a.name for a in n.names]
                    hits.append({"importer": str(qrel), "symbols": used})
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] == name:
                        hits.append({"importer": str(qrel),
                                     "symbols": ["<module>"]})
    return {"target": str(rel), "defines": sorted(syms), "importers": hits}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--retire", metavar="MODULE",
                    help="K5 pre-retirement importer check for MODULE")
    ap.add_argument("--strict-dupes", action="store_true",
                    help="duplicate symbols also fail the audit")
    args = ap.parse_args()

    if args.retire:
        res = retire_check(args.retire)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"K5 pre-retirement check: {res['target']}")
            print(f"  defines: {', '.join(res['defines']) or '(none)'}")
            if not res["importers"]:
                print("  live importers: NONE — safe to retire/move.")
            else:
                print(f"  live importers ({len(res['importers'])}) — repoint "
                      f"these FIRST:")
                for h in res["importers"]:
                    print(f"    {h['importer']}: {h['symbols']}")
        return 0

    r = audit()
    s1, s2, dup = (r["severity1_core_from_test"],
                   r["severity2_test_helper_from_test"],
                   r["review_duplicate_symbols"])
    lay = r["severity1_layering_foundation_into_upper"]
    xco = r["severity1_cross_project_code"]
    tiso = r["severity1_test_project_isolation"]
    hard = (bool(s1) or bool(s2) or bool(lay) or bool(xco) or bool(tiso))
    if args.json:
        # JSON mode is EXCLUSIVE: stdout is machine-readable ONLY, no human
        # summary mixed in (a consumer json.loads()es stdout). Exit code still
        # signals pass/fail.
        print(json.dumps(r, indent=2))
        return 1 if (hard or (args.strict_dupes and dup)) else 0
    if True:
        print("=== SEVERITY 1 — CORE imports from TEST (K2 hard) ===")
        for x in s1:
            print(f"  {x['importer']} <- {x['source']}: {x['names']}")
        print(f"  {len(s1)} site(s)")
        print("\n=== SEVERITY 2 — test helper imported from test (K2) ===")
        for x in s2:
            print(f"  {x['importer']} <- {x['source']}: {x['names']}")
        print(f"  {len(s2)} site(s)")
        print("\n=== SEVERITY 1 — LAYERING: core/example reaching UP into "
              "projects/examples (the PI's rule) ===")
        for x in lay:
            print(f"  {x['importer']} <- {x['source']}: {x['names']}")
        print(f"  {len(lay)} site(s)")
        print("\n=== SEVERITY 1 — CROSS-PROJECT code: a project imports "
              "another project's code (forbidden; oaTOF subsystems are one "
              "project) ===")
        for x in xco:
            print(f"  {x['importer']} <- {x['source']}: {x['names']}")
        print(f"  {len(xco)} site(s)")
        print("\n=== SEVERITY 1 — TEST project-isolation: a test imports a "
              "project it does not own (see TEST_PROJECT_OWNERS) ===")
        for x in tiso:
            print(f"  {x['importer']} owns {x.get('owns')}, imports "
                  f"{x.get('imports')} ({x['source']})")
        print(f"  {len(tiso)} site(s)")
        print("\n=== REVIEW — duplicate public symbols (K4/K7 — human "
              "MOVE/MERGE/LEAVE) ===")
        for x in dup:
            print(f"  {x['symbol']}: {x['modules']}")
        print(f"  {len(dup)} candidate(s) — most are coincidental (K7); "
              f"not an auto-fail")

    fail = hard or (args.strict_dupes and bool(dup))
    if fail:
        print(f"\nAUDIT: FAIL — {len(s1)} sev-1 core<-test, {len(s2)} sev-2 "
              f"test<-test, {len(lay)} layering, {len(xco)} cross-project code, "
              f"{len(tiso)} test-isolation"
              + (f", {len(dup)} dupes (strict)" if args.strict_dupes else ""))
        return 1
    print("\nAUDIT: PASS — no core<-test, test<-test, layering, cross-project "
          "code, or test-isolation violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
