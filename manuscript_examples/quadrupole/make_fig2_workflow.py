"""Figure 2 of the ion_gym framework manuscript: the workflow from deck to
recorded flights, drawn from the solver products of the Results quadrupole.

The deck is quad_lib.quad_spec, the rod quadrupole the stability notebooks
fly (r0 = 3 mm, 1 MHz). The RF amplitude puts m/z 500 at q near 0.70, close
to the a = 0 operating region of a mass filter, with q read from the solved
field by quad_lib.calibrate_v_per_q and printed at run time. Two ions of
different mass are flown through the same drive. Because q scales as
1/(m/z), the lighter ion falls outside the first stability zone (q > 0.908),
so panel F shows two recorded fates from one deck.

Run:  python manuscript_examples/quadrupole/make_fig2_workflow.py
Writes outputs/manuscript/fig2_workflow.{png,pdf} under the repo root.
"""
from __future__ import annotations

import quad_lib as QL

from ion_gym.io.paths import outputs_dir
from ion_gym.physics.build_planar import build_planar_model
from ion_gym.viz.viz_core import (OKABE_ITO, apply_house_style,
                                  save_figure, set_style)
from ion_gym.viz.workflow_figure import workflow_figure

# ---- operating point ------------------------------------------------------
V_RF = 330.0           # V, zero-to-peak RF amplitude (q ~ 0.70 at m/z 500)
F_RF_MHZ = 1.0         # MHz
MZ_FLOWN = (500.0, 350.0)  # Th; same drive, stable then unstable mass
T_MAX_US = 40.0        # us, the flight time limit written into the deck
V_CAL = 30.0           # V, amplitude of the curvature calibration
DECK_NAME = "rod quadrupole"
RF_NAMES = ("RFA", "RFB")  # x-pair and y-pair drive groups

# ---- what panels B and D show --------------------------------------------
BASIS_ELECTRODES = ("xp", "yp")   # one rod of each RF pair
PHASES_DEG = (90.0,)             # RF phase of the summed-potential map

OUT_STEM = "fig2_workflow"


def build_figure():
    """Figure 2 for the operating point above (returns the Figure)."""
    apply_house_style()
    set_style(groups=list(OKABE_ITO))
    v_per_q, _q_cal = QL.calibrate_v_per_q(f_rf_mhz=F_RF_MHZ, v_cal=V_CAL,
                                           mz=MZ_FLOWN[0])
    for mz in MZ_FLOWN:
        q = V_RF / v_per_q * MZ_FLOWN[0] / mz
        print(f"m/z {mz:g}: q = {q:.4f} at {V_RF:g} V, {F_RF_MHZ:g} MHz "
              f"(solved-field calibration)")
    specs = [QL.quad_spec(V_RF, f_rf_mhz=F_RF_MHZ, mz=mz, t_max=T_MAX_US,
                          name=DECK_NAME, rf_names=RF_NAMES)
             for mz in MZ_FLOWN]
    # the masses differ, the geometry does not: one model serves all panels
    model = build_planar_model(specs[0])
    flights = []
    for sp in specs:
        _m, tr, _ci, status = QL.fly_one(sp)
        flights.append((tr, status, f"m/z {sp.source.mz_list[0]:g}"))
    return workflow_figure(specs[0], basis_electrodes=BASIS_ELECTRODES,
                           phases_deg=PHASES_DEG, flights=flights,
                           model=model)


if __name__ == "__main__":
    fig = build_figure()
    for ext in ("png", "pdf"):
        p = save_figure(fig, outputs_dir("manuscript_examples", "quadrupole", f"{OUT_STEM}.{ext}"))
        print(f"wrote {p}")
