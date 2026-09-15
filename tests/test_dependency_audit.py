"""test_dependency_audit.py — the K1–K9 placement audit is a GATE.

Wraps tools/dependency_audit.py so the runner enforces code placement every
release: no CORE module imports from a test file (K2 sev-1), no test imports a
working helper from another test (K2 sev-2). Duplicate symbols are REVIEW, not
failure (K7 — most are coincidental), so they do not fail here.

This gate exists because the audit is only protection if it RUNS. A tool that
must be remembered is a tool that gets forgotten (the einzel regression: the
handoff already warned 'grep before retiring' and it still happened).

NOTE: under governing K9 (no fixture exemption) the suite had
4 sev-1 + 23 sev-2 KNOWN violations, pending the
cleanup. Until that lands this gate holds the LINE (fails on any INCREASE),
naming the debt rather than hiding it. When the cleanup completes, the
baseline drops to zero and any regression re-reddens it.
"""
import _bootstrap  # noqa: F401
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "tools" / "dependency_audit.py"

# Known-violation baseline, under GOVERNING K9 (no *_spec
# fixture exemption — every cross-test import counts). The cleanup drives
# these to (0, 0); a NEW violation pushes a count above baseline and fails.
BASELINE_SEV1 = 0   # lowered from 4: placement refactor
BASELINE_SEV2 = 0   # lowered from 23 — any new
                    # cross-test or core<-test import is now a hard FAIL
BASELINE_LAYER = 0  # core/example must not reach UP into
                    # projects/examples (the governing layering rule)
BASELINE_XCODE = 0  # a project's code is imported only by that
                    # project (generalises the q3-leaf rule to every project)
BASELINE_TISO = 0   # a test imports ONLY the project it owns
                    # (TEST_PROJECT_OWNERS); a core-concern test owns none


def main():
    # Charter: a missing dependency audit script is a clear failure, not a skip.
    # Do NOT swallow the error; report it loudly so we know the gate is wedged.
    if not AUDIT.exists():
        print(f"AUDIT GATE: FAIL — {AUDIT} does not exist. "
              f"The dependency audit script is missing. Create it or skip "
              f"this gate.")
        return 1
    out = subprocess.run([sys.executable, str(AUDIT), "--json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"AUDIT GATE: FAIL — subprocess failed (code {out.returncode})")
        print(f"stderr: {out.stderr}")
        return 1
    try:
        r = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        print(f"AUDIT GATE: FAIL — invalid JSON from audit script: {e}")
        print(f"stdout: {out.stdout[:200]}")
        return 1
    s1 = len(r["severity1_core_from_test"])
    s2 = len(r["severity2_test_helper_from_test"])
    lay = len(r["severity1_layering_foundation_into_upper"])
    xco = len(r["severity1_cross_project_code"])
    tiso = len(r["severity1_test_project_isolation"])
    dup = len(r["review_duplicate_symbols"])
    print(f"K2 sev-1 (core<-test): {s1} (baseline {BASELINE_SEV1})")
    print(f"K2 sev-2 (test helper<-test): {s2} (baseline {BASELINE_SEV2})")
    print(f"Layering (core/example -> projects/examples): {lay} "
          f"(baseline {BASELINE_LAYER})")
    print(f"Cross-project code (project <- other project): {xco} "
          f"(baseline {BASELINE_XCODE})")
    print(f"Test project-isolation (test <- non-owned project): {tiso} "
          f"(baseline {BASELINE_TISO})")
    print(f"K4/K7 duplicate symbols (REVIEW, not gated): {dup}")
    over = (s1 > BASELINE_SEV1 or s2 > BASELINE_SEV2 or
            lay > BASELINE_LAYER or xco > BASELINE_XCODE or
            tiso > BASELINE_TISO)
    under = (s1 < BASELINE_SEV1 or s2 < BASELINE_SEV2 or
             lay < BASELINE_LAYER or xco < BASELINE_XCODE or
             tiso < BASELINE_TISO)
    if over:
        print("AUDIT GATE: FAIL — new placement/layering/isolation violation "
              f"(sev1 {s1}/{BASELINE_SEV1}, sev2 {s2}/{BASELINE_SEV2}, "
              f"layer {lay}/{BASELINE_LAYER}, xcode {xco}/{BASELINE_XCODE}, "
              f"tiso {tiso}/{BASELINE_TISO})")
        return 1
    if under:
        print("AUDIT GATE: PASS — and IMPROVED; lower the relevant baseline in "
              "this file to lock the gain.")
        return 0
    print("AUDIT GATE: PASS — at known baseline, no new violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
