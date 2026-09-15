"""
LINT GATE (adopted with ruff).

Policy: the SEVERE real-bug classes gate at ZERO — F821 (undefined
name: the n_ion_traces-crash class), F811 (redefinition shadowing,
outside the three known-benign files declared in ruff.toml), F601
(duplicate dict keys: the FIELD_SPECS class), and E9 (syntax) — while
the MECHANICAL tier (F401 unused imports, F841 unused vars, F541
empty f-strings) is REPORTED as info but does not fail until its
cleanup pass lands, after which this gate tightens to the
full F set.

Refusals are loud: a missing ruff binary is a FAIL naming the fix,
never a skip (a gate that silently doesn't run is worse than no
gate).

Run: python tests/test_lint.py
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = ["ion_gym", "tests", "tools"]
# TIGHTENED: the mechanical tier (F401/F841/F541, 173
# findings) was cleared by the cleanup pass — the gate now
# holds the FULL pyflakes set at zero.
SEVERE = "F,E9"
MECHANICAL = ""


def main():
    if shutil.which("ruff") is None:
        print("LINT GATE: FAIL — ruff is not installed "
              "(pip install ruff); a lint gate that cannot run is a "
              "failure, not a skip.")
        return 1

    sev = subprocess.run(
        ["ruff", "check", *TARGETS, "--select", SEVERE, "--quiet"],
        cwd=ROOT, capture_output=True, text=True)
    if sev.returncode != 0:
        # Charter: surface what failed, don't swallow. If ruff reported
        # findings, show them; if it errored, show stderr.
        output = sev.stdout.strip() or sev.stderr.strip()
        if output:
            print(output)
        else:
            print(f"(ruff exited {sev.returncode}, no output)")
        print("=" * 60)
        print(f"LINT GATE: FAIL — ruff check found issues [{SEVERE}]")
        print("=" * 60)
        return 1
    print("=" * 60)
    print(f"LINT GATE: PASS — zero severe findings [{SEVERE}]")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
