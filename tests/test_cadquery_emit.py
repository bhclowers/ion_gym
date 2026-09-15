"""
Gate: ion_gym.cad.cadquery_emit — the reusable CadQuery script emitter
abstracted out of a project-specific geometry module.

  C1  clean_num strips binary float noise (5.449999999999999 -> 5.45).
  C2  render() emits a SYNTACTICALLY VALID python script (ast.parse) with
      the numbering doctrine, export_stls, and the STEP save line.
  C3  duplicate electrode numbers are REFUSED (a number is the basis
      index; two conductors at one number is a silent-merge defect).
  C4  the Q3 device still emits a valid full script through the emitter
      (regression: the abstraction did not break the real caller).

Run: python tests/test_cadquery_emit.py
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.cad.cadquery_emit import CadEmitter, clean_num, fmt_params

FAILED = []


SKIPPED = []


def check(name, fn):
    try:
        # A case may return the string "skipped" when its SUBJECT is
        # absent (not when it is inconvenient). Counted separately and
        # printed, so the tally can never read a skip as a pass -- the
        # case itself prints the reason.
        if fn() == "skipped":
            SKIPPED.append(name)
            return
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


def main():
    def c1():
        assert clean_num(5.449999999999999) == "5.45"
        assert clean_num(100.0) == "100"
        assert clean_num(3.84) == "3.84"
        blk = fmt_params([("R0", 3.84, "field radius"), ("N", 36, "")])
        assert "R0" in blk and "3.84" in blk and "9999" not in blk
    check("C1 clean_num / fmt_params strip float noise", c1)

    def c2():
        em = CadEmitter(title="Dev", params=[("R0", 3.84, "r")])
        em.add_electrode(1, "A", 'cq.Workplane("XY").circle(R0).extrude(10)',
                         "gold")
        s = em.render()
        ast.parse(s)                       # valid python
        assert "ONE FILE PER ELECTRODE" in s
        assert "def export_stls" in s
        assert "assembly.step" in s
        assert "9999999" not in s
    check("C2 render emits valid script + doctrine + export + step", c2)

    def c3():
        em = CadEmitter(title="Dev")
        em.add_electrode(1, "A", "cq.Workplane()")
        try:
            em.add_electrode(1, "B", "cq.Workplane()")
            raise AssertionError("duplicate number not refused")
        except ValueError as e:
            assert "already used" in str(e)
        # empty assembly refused
        try:
            CadEmitter(title="empty").render()
            raise AssertionError("empty assembly not refused")
        except ValueError as e:
            assert "no electrodes" in str(e)
    check("C3 duplicate number / empty assembly refused", c3)

    def c4():
        # The Q3 device is an INTERNAL study (moved out of the shipped
        # package). On a dev checkout the projects __path__ bridge makes
        # it importable and this case runs; on a public checkout it is
        # absent, and an absent internal subject is a NAMED SKIP, never a
        # failure and never silent -- C1-C3 above certify the emitter
        # itself, which is the part that ships.
        try:
            from ion_gym.projects.q3.q3r_geom import Q3RGeom
        except ModuleNotFoundError:
            print("  [SKIP] C4 Q3 device emits valid script -- the Q3 "
                  "study is internal and is not present in this checkout; "
                  "the emitter itself is certified by C1-C3.")
            return "skipped"
        g = Q3RGeom()
        s = g.cadquery_numbered()
        ast.parse(s)
        n_el = len(g.electrode_table())
        assert s.count("assy.add(_solid") == n_el, "electrode count drift"
        # the second method too
        ast.parse(g.cadquery())
    check("C4 Q3 device emits valid script through the emitter", c4)

    print("=" * 60)
    print(f"PASSED {4 - len(FAILED) - len(SKIPPED)}   "
          f"FAILED {len(FAILED)}   SKIPPED {len(SKIPPED)}"
          + (f" -> {', '.join(SKIPPED)}" if SKIPPED else ""))
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
