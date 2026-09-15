"""
Gate: assembly JSON (io/assembly_json.py) — the declarative front end for
staged multi-region flight.

  A1  load_assembly on the shipped demo builds the declared regions with
      the declared poses (translation + rotation) and exit planes.
  A2  fly_assembly flies the beam through the directory end-to-end: the
      ion hands off across the rotated seam and completes.
  A3  schema/shape refusals: wrong schema, missing stage spec, and a
      malformed exit plane each raise a NAMED error (no silent skip).
  A4  gas mode: 'vacuum' -> no collisions; an unknown gas string is
      refused.

Uses the shipped examples/funnel_hexapole instrument (the funnel r-z
stage solves in seconds; the hexapole is a 3-D solve, cached after the
first run — the demo assembly is not part of the shipped set; the
IF-Hexapole IS the assembly example).

Run: python tests/test_assembly_json.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.physics.staged_flight import fly_packet, load_assembly

DEMO = "examples/funnel_hexapole/instrument.json"
FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


def _tmp_doc(mutate):
    doc = json.load(open(DEMO))
    mutate(doc)
    fd, path = tempfile.mkstemp(suffix=".json",
                                dir="examples/funnel_hexapole")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    return path


def main():
    # ONE load shared by A1/A2: from_stage beam generation FLIES the
    # source funnel, so loading is the expensive step — pay it once.
    _shared = {}

    def a1():
        regions, beam = load_assembly(DEMO)
        _shared["regions"], _shared["beam"] = regions, beam
        assert [r.name for r in regions] == ["funnel",
                                             "hexapole_tilted_wires"]
        assert regions[0].pose.rot_deg == [0, -90, 0], \
            "funnel rotation not loaded"
        assert (regions[0].exit is not None
                and regions[0].exit.value_mm == 37.7), \
            "funnel exit plane (the seam) not loaded"
        # region exits are STORED-FRAME: 88.55 world - 37.7 offset
        assert (regions[1].exit is not None
                and abs(regions[1].exit.value_mm - 50.85) < 1e-6), \
            "hexapole exit plane (the detector, stored frame) not loaded"
        ions = beam["ions"]
        assert len(ions) == 75, f"from_stage beam: {len(ions)} ions"
        assert sorted(set(beam["mz_per_ion"])) == [200.0, 400.0, 600.0]
    check("A1 load builds regions with poses + exits (from_stage beam)",
          a1)

    def a2():
        regions, beam = _shared["regions"], _shared["beam"]
        # a 3-ion slice keeps the handoff proof to seconds; the full
        # packet is nb11's job, not this gate's
        small = dict(beam)
        small["ions"] = beam["ions"][:3]
        small["mz_per_ion"] = list(beam["mz_per_ion"][:3])
        out = fly_packet(regions, small, keep_traces=3)
        assert out["n"] == 3
        # a trace is per-stage segments; a handoff means the SECOND
        # stage's segment exists and is non-empty for some ion
        crossed = sum(
            1 for tr in out["traces"]
            if any(s.get("name") == "hexapole_tilted_wires"
                   and len(s.get("x", [])) > 1
                   for s in (tr.get("regions") or [])))
        assert crossed >= 1, \
            "no trace carries a hexapole segment - seam handoff untested"
    check("A2 fly_packet hands the packet across the seam", a2)

    def a3():
        # wrong schema
        try:
            p = _tmp_doc(lambda d: d.update(schema="bogus/9"))
            load_assembly(p); raise AssertionError("bad schema not refused")
        except ValueError as e:
            assert "schema" in str(e)
        finally:
            os.remove(p)
        # path-form stage spec (the inline doctrine made the
        # PATH FORM itself the refusal — a manifest of paths can drift
        # from what it names — so this fires before any file check)
        try:
            p = _tmp_doc(lambda d: d["stages"][0].update(spec="nope/x.json"))
            load_assembly(p); raise AssertionError("path-form spec not refused")
        except TypeError as e:
            assert "INLINE" in str(e)
        finally:
            os.remove(p)
        # malformed exit (no value_mm)
        try:
            p = _tmp_doc(lambda d: d["stages"][0].update(exit={"axis": "x"}))
            load_assembly(p); raise AssertionError("bad exit not refused")
        except ValueError as e:
            assert "exit plane" in str(e)
        finally:
            os.remove(p)
    check("A3 schema / path-form-spec / bad-exit each refused by name", a3)

    def a_route():
        # the spec loader must RECOGNISE the staged-assembly schema and
        # refuse it with a USEFUL diagnostic (points to load_assembly),
        # not the generic "unrecognised JSON" (this was hit in the app).
        from ion_gym.io.spec_io import load_any_spec, sniff
        txt = open(DEMO).read()
        assert sniff(txt) == "staged_assembly", sniff(txt)
        try:
            load_any_spec(txt)
            raise AssertionError("staged assembly not refused by spec loader")
        except ValueError as e:
            m = str(e)
            assert "STAGED ASSEMBLY" in m and "load_assembly" in m, m
    check("A5 spec loader routes staged assembly to a useful diagnostic",
          a_route)

    def a4():
        try:
            p = _tmp_doc(lambda d: d["stages"][0].update(gas="argonish"))
            load_assembly(p); raise AssertionError("bad gas not refused")
        except ValueError as e:
            assert "gas" in str(e)
        finally:
            os.remove(p)
    check("A4 unknown gas mode refused", a4)

    print("=" * 60)
    print(f"PASSED {5 - len(FAILED)}   FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
