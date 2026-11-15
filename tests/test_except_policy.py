"""test_except_policy.py -- gate E: no exception handler may HIDE a special case.

Broad handlers must not hide special use cases.

The failure this gates against is not that exceptions are caught; it is that a
BROAD catch silently substitutes a DIFFERENT BEHAVIOUR for the one asked for,
so the code takes a hidden branch on one configuration and nobody is told:

  * build_planar._solve: `except Exception` around the multigrid import ALSO
    caught multigrid THROWING, and silently recomputed the field with a
    DIFFERENT SOLVER (SOR).  A certified number then depends on whether an
    exception happened to fire.
  * build_planar._cache_key: `except Exception` meant a key that RAISED
    silently produced a DIFFERENT key -- the very thing its own docstring warns
    about ("two notions of the same geometry is one cache too many").
  * build_planar._will_solve: `except Exception: return False` answered "is this
    cached?" with a confident NO whenever anything went wrong, and the UI
    believed it.
  * pe_view.compute_component: `except TypeError` used as a capability probe,
    which reported a THROWING SOLVER as a missing geometry (see D1).

POLICY.  In the modules that decide PHYSICS -- what gets solved, by which
solver, with which cache key -- a handler must be NARROW (a named, expected
exception) and must not swallow.  `except Exception` there is refused.

This is an ALLOWLIST that must SHRINK.  It is not a licence: every entry is a
known handler that has been read and judged, and the count may never grow
without someone reading the new one.
"""
import _bootstrap  # noqa: F401
import pathlib
import sys

# Modules where a hidden branch changes a NUMBER, not a pixel.
PHYSICS = ["build_planar.py", "build_rz.py", "build_stl3d.py", "solver3d.py",
           "multigrid3d.py", "basis_cache.py", "sim_build.py",
           # ADDED in a cleanup pass: stats
           # decides the NUMBERS on the run card (per-ion m/z, TOF stats);
           # build_stl decides WHICH SOLVER runs and whether a solve
           # happens at all.  Both carried unlisted broad handlers (the
           # multigrid silent-substitution archetype among them); both are now at budget 0
           # and covered so they cannot regrow one silently.
           "stats.py", "build_stl.py",
           # DISPLAY path.  A hidden branch here cannot corrupt a NUMBER, but it
           # can make one INVISIBLE -- and an absent electrode is a wrong figure
           # just as surely as a wrong voltage is (absence is a different
           # answer).  Two of these were hiding exactly that: pe_view's electrode
           # overlay was wrapped in `except Exception: pass` AND cast its int16
           # labels to a boolean mask, so every electrode drew identically; and
           # sim_app swallowed a VizError from apply_zoom_policy, so the zoom
           # policy silently never applied.
           "pe_view.py", "viz_core.py", "sim_app.py"]

# Known-and-judged broad handlers, by (file, kind).  MUST ONLY SHRINK.
# A RATCHET, not a licence.  Every entry is a handler that has been READ and
# judged; the number may only shrink.
ALLOW = {
    # AUDITED (every handler read): pe_view's
    # three remaining broad handlers are UI boundaries that ALL REPORT
    # to the status line (:858 compute boundary; :885/:1298 slider-
    # range updates, formerly silent passes). Four others were
    # NARROWED (scipy ImportError; 2x slice-index TypeError/ValueError;
    # stride AttributeError/TypeError).
    # RE-AUDITED, full evaluation. All FOUR
    # read: :498 captures into st["err"]/st["tb"] and is re-raised on
    # the document thread (a worker thread cannot surface its own
    # error); :1082, :1109 and :1594 each assign an informative message
    # to self.status.object naming the exception type and text. None is
    # silent; none substitutes behaviour. Budget raised 3 -> 4 to the
    # READ census, not to the observed count.
    ("pe_view.py", "Exception"): 4,
    ("viz_core.py", "Exception"): 0,    # loop-eater removed: collects+reports
    # sim_app: an earlier audit read ALL its broad handlers one at a time.  Nine
    # were narrowed or root-caused away (five real defects: a degrade-to-
    # geometry-only that hid draw errors; a hardcoded-xy redraw on clear-runs
    # -- plane-capability leak, 4th copy; the DC-ladder gradient measured
    # across the RADIUS on r-z imports and silently ABSENT on native builds;
    # a TypeError capability sniff on potential_image; silently dropped mz
    # tokens).  The 11 that remain ALL REPORT to the status line: they are the
    # app's top-level refuse-with-diagnostic boundaries -- thread workers and
    # widget callbacks with nothing above them to catch, where an uncaught
    # raise dies silently in the event loop, which is WORSE than a broad catch
    # that reports.  None is silent; none substitutes behaviour.
    # RE-AUDITED (a budget of 11 had grown to 39 as
    # feature work added handlers unaudited). Every broad handler was
    # read and classified by the governing criterion (reports or raises):
    # 12 were narrowed or converted (scene-box/mirror probes, curdoc
    # probes, optional-cache clears, keystroke summary -> NARROWED;
    # spec-signature trio + solve-necessity check -> broad but now
    # PRINT their substitution; per-ion mz colour fallback -> narrowed
    # to the refusal class and COUNTED on the status line). ALL
    # remaining broad handlers verified reporting (AST-checked: zero
    # silent bodies) — top-level UI/thread boundaries with nothing
    # above them to catch. Budget = the audited census; may only
    # shrink.
    # RE-AUDITED, full evaluation. Every broad
    # handler in this file was read individually against the governing
    # criterion -- "reports or raises" -- rather than the weaker test
    # the previous budget was frozen with ("not `except: pass`"). Those
    # are not the same bar, which is why this pass was worth doing.
    #
    # RESULT: 38 of the 39 handlers across both files report or forward.
    #   * 2 RE-RAISE.
    #   * 2 FORWARD an exception a caller surfaces: pe_view:498 and
    #     sim_app:3614 stash exc + traceback into st["err"]/st["tb"] for
    #     the document thread; sim_app:7924 stashes st["error"], which
    #     :7940 renders as "**assembly flight failed:** ...".
    #   * The rest assign self._err_status(label, e) -- which formats the
    #     type, the message AND a five-frame traceback -- or append a
    #     named failure to the status line (":6908/:6951 -> '/flight NOT
    #     banked: <type>: <msg>'"), or print with a labelled prefix.
    #
    # ONE GENUINE VIOLATION FOUND AND FIXED, not allowlisted:
    # _on_ion_count_change was `except Exception:` writing "" to
    # w_ion_total -- a silent substitution in which an unparseable m/z
    # list looked exactly like an unfilled field. NARROWED to
    # (ValueError, TypeError, AttributeError) and made to report, which
    # is why this census is 34 and not 35.
    #
    # A NOTE FOR THE NEXT AUDITOR, so this is not re-derived: an
    # automated classifier is not sufficient here. A scan that looks for
    # print/log calls misses BOTH the status-surface assignments and the
    # capture-and-forward pattern, and flags ~25 false violations; a scan
    # that additionally looks for literal strings misses
    # self._err_status(...) because the assigned value is a Call. The
    # handlers have to be read.
    #
    # Later the count had drifted to 37 on main
    # (the gate was red). Four were narrowed, each read: the view-slice
    # range (2-D models named, the rest raises to the redraw boundary),
    # the scene build (VizError/ValueError, reason on the status line),
    # NPZ pre-generation (ValueError/TypeError/MemoryError, reported), and
    # the assembly callback stop (the shared _stop_periodic). Ratcheted
    # to the new count, as this allowlist requires.
    #
    # Then +2, both read at write time. The
    # thermal-assess off-loop worker is the capture-and-forward pattern
    # already cited above (st["err"]/st["tb"] stashed on the worker, the
    # 150 ms poll renders type+message in the Analysis table and prints
    # the traceback -- a worker thread cannot surface its own error).
    # _update_sigma_honored is a param-watcher boundary with nothing
    # above it to catch; it renders "**σ readout failed:** <type>:
    # <msg>" IN the pane it would otherwise have filled, so a failed
    # readout can never look like a valid σ. Neither is silent; neither
    # substitutes behaviour.
    #
    # Batch 9 (every handler in the file read): 35 -> 31. Three
    # signature helpers were narrowed to the serialisation errors they
    # meet; the census also counts tuple/alias forms now. All 31 remaining
    # bind and report the exception (checked by the gate, not by comment).
    # 31 -> 32: SimApp._do_reset, the body of
    # Reset, runs on the next tick with nothing above it to catch; it
    # clears the spinner and opens the error dialog with the exception
    # (_report_exception), so a failed Reset is reported, not logged
    # only to the server console.
    ("sim_app.py", "Exception"): 32,
}

# SILENT handlers (broad or narrow) whose body is only pass / continue /
# a constant / `return <constant>` and that never touch the exception:
# each is pinned by (file, enclosing function, caught types) -> (how many
# such handlers that function may hold, the reason none is a swallow).
# Identity, not a file-wide count, so one cannot be swapped for another;
# an entry that no longer matches fails the gate (the list only shrinks).
# Each was read and judged (batch 9).
SILENT_ALLOW = {
    ("pe_view.py", "PeSurfaceTab._stride_for",
     ("AttributeError", "TypeError")):
        (1, "model not built yet -> full resolution; a pe_surface failure "
            "on a ready model raises to the compute boundary"),
    ("pe_view.py", "_run_off_doc._poll", ("RuntimeError", "ValueError")):
        (1, "stopping an already-stopped periodic callback is benign; the "
            "result is delivered right after"),
    ("sim_app.py", "SimApp._ensure_watchdog_pulse", ("RuntimeError",)):
        (1, "no session loop to schedule on yet; the handle stays None and "
            "every arming site retries until one lands"),
    ("sim_app.py", "SimApp._sniff_assembly", ("TypeError", "ValueError")):
        (1, "unreadable text is not an assembly; the normal loader then "
            "gives the real diagnostic"),
    ("sim_app.py", "SimApp._on_json_edited",
     ("AttributeError", "KeyError", "TypeError", "ValueError")):
        (1, "per-keystroke pitch sync of mid-edit (usually invalid) text; "
            "the summary shows the box does not parse and Apply reports"),
    ("sim_app.py", "SimApp._base_figure", ("VizError",)):
        (1, "nothing drawn to measure: the declared extent stands"),
    ("sim_app.py", "SimApp._domain_range",
     ("AttributeError", "KeyError", "ValueError")):
        (1, "a pre-build Scene falls through to the documented spec-extent "
            "rule just below"),
    ("sim_app.py", "SimApp._solve_then.poll",
     ("AttributeError", "RuntimeError")):
        (3, "clearing the spinner of a pane torn down mid-solve; the "
            "solve's outcome is reported by the status lines around each"),
}

ROOT = pathlib.Path(__file__).resolve().parent.parent
from tools.except_policy import scan_source  # noqa: E402 -- the scanner


def main():
    bad, ok = [], 0
    seen_silent = set()
    for f in PHYSICS:
        # a module the gate cannot find is a FAILURE: a rename used to drop
        # it out of the gate silently
        try:
            p = _bootstrap.locate(f)
        except FileNotFoundError as e:
            bad.append(f"{f}: not found ({e}); update PHYSICS")
            continue
        broad, problems, silent = scan_source(p.read_text(), f)
        budget = sum(v for (fn, _), v in ALLOW.items() if fn == f)
        over = len(broad) > budget
        print(f"  {f:20s} broad handlers: {len(broad):2d}  allowed: "
              f"{budget}{'   <-- OVER' if over else ''}")
        for ln, k in broad:
            print(f"        line {ln}: except {', '.join(k)}")
        if over:
            bad.append(f"{f}: {len(broad)} broad handlers, budget {budget}")
        bad += [f"{f}: {msg}" for msg in problems]
        per_key = {}
        for ln, key in silent:
            seen_silent.add(key)
            per_key.setdefault(key, []).append(ln)
        for key, lines in per_key.items():
            allowed = SILENT_ALLOW.get(key, (0, ""))[0]
            if len(lines) > allowed:
                bad.append(f"{f}: line(s) {lines}: silent handler(s) in "
                           f"{key[1]} (except {', '.join(key[2])}), "
                           f"{len(lines)} > {allowed} pinned -- report it, "
                           f"narrow its job, or pin it in SILENT_ALLOW with "
                           f"the reason")
        if not (over or problems):
            ok += 1
    for key in sorted(set(SILENT_ALLOW) - seen_silent):
        bad.append(f"SILENT_ALLOW entry {key} matches no handler; remove it")

    print()
    if bad:
        for b in bad:
            print(f"  !! {b}")
        print("EXCEPT POLICY: FAIL -- a handler in the physics/display path "
              "can hide a failure or substitute a different answer.")
        return 1
    print(f"EXCEPT POLICY: PASS ({ok} modules clean; broad budget "
          f"{sum(ALLOW.values())}, {len(SILENT_ALLOW)} pinned silent "
          f"handlers; both may only shrink)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
