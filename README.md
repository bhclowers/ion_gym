# ion_gym

A Python-native toolkit for ion-optics simulation:
multigrid field solvers, RF/DC/traveling-wave drives, hard-sphere (HS)
and statistical-diffusion (SDS) collision models, declarative JSON
geometry (2-D planar, r-z cylindrical, and full-3-D extruded/STL
routes), and an interactive Panel/Plotly workbench.

Developed in the Clowers Research Group, Washington State University,
as the design and validation environment for systems used to manipulate gas-phase ions.

![A simulated ion trajectory through a 3-D electrode geometry, drawn in
the workbench's flight view. The path is coloured by time of
flight (purple at birth through to yellow), and the electrodes are shown
semi-transparent so the trajectory stays visible inside the
structure.](docs/images/example_trajectory_3d.png)

<!-- CAPTION IS INCOMPLETE — it states only what is visible in the
     render. Per the standing rule that a figure carries its operating
     point, this needs the deck name and the tune (drive amplitudes and
     frequencies, pressure, m/z) before it is a result rather than a
     picture. -->

## Overview

- **What is ion_gym?** A Python-native toolkit that takes a declarative
  JSON geometry to flown ions in one scriptable path: solve the fields,
  cache per-electrode bases (a voltage change is a re-weight, not a
  re-solve), drive them with RF/DC/traveling waves, and fly ions in
  vacuum or buffer gas — with the same machinery behind the interactive
  workbench and the teaching notebooks.
- **What can it do today?** 2-D planar, r-z, and full-3-D
  (extruded/STL) routes; HS and SDS collision models for buffer-gas
  work; a Panel/Plotly workbench with a geometry editor; a shipped
  example gallery spanning einzel lenses, quadrupoles, ion funnels,
  hexapoles, SLIM boards, drift cells, Paul and Kingdon traps, an
  Orbitrap-class geometry, ELIT, and reflectron TOFs; and a numbered
  notebook series written for 1st/2nd-year graduate students.
- **What can it *not* do yet?** No space charge (ions fly
  independently), no ion–neutral reaction chemistry, no magnetic
  fields, no GPU backend (CPU with NumPy/numba), and the field solver
  is finite-difference electrostatics on a regular grid — not BEM/FMM.
  If those are load-bearing for your problem, see
  [docs/RELATED_SOFTWARE.md](docs/RELATED_SOFTWARE.md) for tools that
  cover them.
- **Who is it for?** Researchers designing and validating ion-optics
  instruments, and instructors teaching ion optics and mass
  spectrometry at the undergraduate and graduate levels.
- **Expectations.** ion_gym favors explicit, reproducible workflows
  over breadth: stated refusals instead of silent fallbacks, figures
  that carry their operating points, and validation against analytic
  references. Where its scope overlaps specialized tools, validate
  against them rather than treating any single package as ground truth.

## Quick start

**Before you install, two things about where it lives.**

*Do not put the checkout inside OneDrive, Dropbox, iCloud Drive, or any
other synced folder* (on Windows that includes `Desktop` and `Documents`
when Known Folder Move is on, which is the default). ion_gym writes
solved-field caches, saved runs, `__pycache__` and notebook checkpoints
into the tree as it works. A sync client will lock, copy and upload every
one of them while the app is using them, which shows up as the app
freezing, as saves that fail with a permission error, and as caches that
are silently half-written. Use a plain local path such as
`C:\dev\ion_gym` or `~/dev/ion_gym`.

*Install into a named environment, not conda's `base`.* `conda create -n
ion_gym python=3.11 && conda activate ion_gym` first. Installing into
`base` works until the first dependency conflict with something else you
have, and then it is hard to undo.

    pip install -e ".[all]"     # everything, incl. notebooks + editor
    # (equivalent: pip install -r requirements.txt; plain `pip install -e .`
    #  is the core toolkit only — the notebooks then lack matplotlib/cma)

Then start the dashboard. **These are three ways of doing the same
thing — run ONE of them, not all three:**

    ion-gym dashboard                                  # (a) console script, after pip install -e .
    python -m ion_gym dashboard                        # (b) same thing, module form — use if the
                                                       #     console script is not on your PATH
    ion-gym dashboard --port 5007 path/to/spec.json    # (c) same as (a), with a port and a deck

All three serve the same three pages on one server: the app at `/`, the
geometry editor at `/editor`, and the 3-D flight viewer at `/flight`.

or open **`notebooks/sim_workbench.ipynb`** — the interactive workbench
notebook: dependency check, example gallery, and a launch cell that is
safe to *Run All*. From plain Python:

    from ion_gym.ui.sim_app import SimApp
    from ion_gym.ui.serve import serve_dashboard
    server = serve_dashboard(app=SimApp(), port=5006, show=True)

Use `serve_dashboard`, not `SimApp(...).panel().show()`: the latter
starts a server carrying only `/`, so the app's own **Edit geometry**
and **Flight view** buttons link to pages that do not exist and answer
**404**. Pass `threaded=True` from a notebook so the call returns
instead of blocking the cell.

Every example in `examples/` is a declarative JSON spec you can load,
edit, solve, and fly from the app — integration time steps ship
pre-tuned against the RF-period and collision-sampling bounds the Gas
tab's Δt advisor displays.

## Troubleshooting

If the dashboard appears to hang or stop responding in the browser,
refresh the page — nine times out of ten that fixes it. The simulation
state lives in the server process, not the page, so a refresh reattaches
to the running app without losing your work.

Some operations genuinely hold the interface while they run and say so in
the Status tab: rendering a large geometry, and rebuilding the control
column after a deck load or a Reset. Those are expected, and the hang
watchdog labels them as such in its log rather than reporting them as
faults. A stall the app did not advertise is written to
`outputs/diagnostics/ion_gym_hang_dump.txt` with the stack that caused it;
send that file if you hit one.

If the app freezes at unpredictable moments and the checkout is in a
synced folder, move it out before looking any further — see the note at
the top of Quick start.

## Documentation

`docs/` holds the user manual plus the reference set: GLOSSARY,
MODULES (architecture map), DATA_LAYOUT, SETUP_GUIDE, and
RELATED_SOFTWARE (where ion_gym sits among related simulation
packages, with references). The
numbered validation-notebook series covers field
accuracy vs. analytic potentials, Mathieu stability, TOF timing vs.
Wiley–McLaren, collisional thermalization, and transport/Einstein
closure.

## Physics provenance

The SDS collision model implements the published algorithm of
Appelhans & Dahl, *Int. J. Mass Spectrom.* 244 (2005) 1–14
(doi:10.1016/j.ijms.2005.03.010). Its data tables
(`ion_gym/physics/sds_jump_icdf.dat`, `sds_mobility.dat`) are ion_gym
artifacts regenerated by Monte Carlo from the published procedure
(the regeneration recipe, seeds, and validation are documented with
the tables); no external simulator files are included.

## Citation and copyright

Copyright (C) 2026 Brian Clowers / Washington State University.

## Acknowledgment

Support for this work was provided in part by NIGMS R35GM161833.

## License

GPL v3 — see `LICENSE`.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for details.
