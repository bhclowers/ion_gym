"""Deck GENERATOR for the RF drift cell.

RELOCATED OUT OF THE PACKAGE: every UI example is a JSON deck.
Geometry-building Python
inside ion_gym was the violation; a generator that WRITES a deck,
living outside the package beside its study notebooks, is not.
The shipped deck is the JSON this produced; parameter sweeps that
vary geometry (ring count, bore) call this generator directly.

Original module docstring follows.

example_rf_drift_cell.py -- rf-confining drift cell (Allen & Bush 2016).

Stacked-ring drift cell with a linear DC ladder and 180-degree
alternating RF on every ring: the geometry of the "rf-confining drift
cell" of S. J. Allen & M. F. Bush, J. Am. Soc. Mass Spectrom. 27,
2054-2063 (2016), used here as the G1 referee for the question "why does
arrival time grow with RF amplitude?"

Every physics quantity is an argument with a documented default; the
defaults are the paper's stated operating point, not design choices of
ours.  Nothing here is fitted.
"""
# PROVENANCE
#   origin   : Allen & Bush 2016 (Methods): 168 gold-coated electrodes,
#              0.5 mm thick, 1.5 mm centre-to-centre, i.d. 5 or 7 mm,
#              constant DC gradient via a divider network, RF 2.8 MHz on
#              every electrode with adjacent electrodes 180 deg apart,
#              Vpp 15..200 V, 3.5 Torr He.  Their trajectory model used
#              one-third of the length (8.4 cm).  Ion: GRGDS2+, 490 Da.
#   caveat   : the paper does not print the GRGDS2+ He collision cross
#              section it used (cites [38]); SIGMA_M2 below is ref 38's own
#              published value.  The ring OUTER radius
#              is not stated in the paper either; R_OUT_MM is our choice
#              (annulus wide enough that the outer edge is field-free).
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SourceSpec, CollisionSpec,
                                 IntegrationSpec, ViewSpec, SymmetrySpec,
                                 RFGroupSpec, DCGroupSpec, BoundsSpec)

# --- paper geometry (mm) --------------------------------------------------
PITCH_MM = 1.5           # centre-to-centre ring spacing
THICK_MM = 0.5           # ring thickness
R_OUT_MM = 6.0           # ring outer radius (OUR choice; not in the paper)
MM_GU = 0.125            # solve pitch: ring = 4 gu, pitch = 12 gu

# --- paper drive ------------------------------------------------------------
RF_FREQ_HZ = 2.8e6
P_TORR = 3.5
GAS = "He"
T_K = 298.0

# --- ion (GRGDS [M+2H]2+) ----------------------------------------------------
# Ion MASS (not m/z; the spec convention is mass in Da plus a signed
# charge). 492.5 Da = the protonated [M+2H]2+ ion (m/z 246.3 in Allen,
# Giles, Gilbert, Bush, Analyst 141 (2016) 884, ref 38 of Allen & Bush
# 2016); the neutral peptide is 490.5 Da.
ION_MASS_DA = 492.5
ION_CHARGE = 2
# He collision cross section for GRGDS2+, used as the HS1 hard-sphere
# sigma: 141 A^2 = 1.41e-18 m^2, measured in this same RF-confining cell
# (ref 38, ESI Tables S3/S4). The DC drift-tube value is 145.6 A^2
# (Kemper et al., Int. J. Mass Spectrom. 287 (2009) 46), 3% larger.
# replacing the 1.55e-18 placeholder.
SIGMA_M2 = 1.41e-18


def rf_drift_cell_spec(*, bore_id_mm=5.0, n_rings=56, rf_vpp=200.0,
                       e_over_p=0.507, rf_freq_hz=RF_FREQ_HZ,
                       p_torr=P_TORR, T_k=T_K, gas=GAS,
                       ion_mass_da=ION_MASS_DA, ion_charge=ION_CHARGE,
                       sigma_m2=SIGMA_M2, n_ions=100, dt_ns=0.25,
                       t_max_us=None, rec_every=None, seed=60601,
                       lead_mm=2.0, source_r_mm=0.3, collisions=True,
                       source_ring=3, exit_ring=None):
    """Allen & Bush rf-confining drift cell as a SimSpec.

    bore_id_mm : ring inner diameter (paper: 5.0 or 7.0)
    n_rings    : rings in the model (paper model: 56 = 8.4 cm; full: 168)
    rf_vpp     : PEAK-TO-PEAK rf amplitude (paper: 15..200 V). RFGroupSpec
                 takes zero-to-peak, so amplitude_v = rf_vpp / 2.
    e_over_p   : drift field in V cm^-1 Torr^-1 (paper: 0.35 .. 7.0)
    dt_ns      : 0.25 ns = the gas-transport timing plateau (doctrine)
    t_max_us   : default = 3 x the Mason-Schamp drift time of THIS ion
                 (same cross section the kernel flies) over the ladder
                 (generous; ions leave through the x_max bound plane at
                 the last ring, fate 3). Refused in vacuum: a vacuum
                 flight has no drift time to size from, so pass t_max_us.
    rec_every  : default sized for ~2000 rows per ion.
    source_ring: 1-based ring index; ions are born half a pitch
                 downstream of this ring's centre. Default 3: start three rings in, clear of the entry
                 fringe; the former hardcoded birth sat half a pitch
                 past ring 1).
    exit_ring  : 1-based ring index whose centre plane is the x_max
                 bound (ions crossing it end with fate 3). Default
                 n_rings - 3: end three rings before
                 the end (previously n_rings - 4). The truncated
                 stack's LAST ring is an RF fringe -- an axial
                 pseudopotential barrier that stalls low-E ions (seen
                 in the pilot: ions parked at the ring-19
                 fringe of a 20-ring model with the ladder's last
                 0.27 V unable to lift them out). The paper's device
                 has downstream optics we do not model, so the interior
                 is the measurement region and the fringe is excluded.
    Source: thermal disc on axis, radius source_r_mm, born half a
    pitch downstream of ring source_ring's centre, births spread over
    one RF period (phase sampling).
    """
    r_in = 0.5 * bore_id_mm
    x0 = lead_mm
    if exit_ring is None:
        exit_ring = n_rings - 3
    if not (2 <= exit_ring <= n_rings):
        raise ValueError(f"exit_ring {exit_ring} outside 2..{n_rings}")
    if not (1 <= source_ring < exit_ring):
        raise ValueError(
            f"source_ring {source_ring} must be >= 1 and upstream of "
            f"exit_ring {exit_ring}: ions born on or past the exit bound "
            f"never fly")
    L_ladder = (n_rings - 1) * PITCH_MM                    # ring-1 to ring-N
    E_v_per_mm = e_over_p * p_torr / 10.0                  # V/cm/Torr -> V/mm
    v_drop = E_v_per_mm * L_ladder
    rf_groups = [RFGroupSpec("RFA", frequency_hz=rf_freq_hz,
                             amplitude_v=0.5 * rf_vpp, phase_deg=0.0),
                 RFGroupSpec("RFB", frequency_hz=rf_freq_hz,
                             amplitude_v=0.5 * rf_vpp, phase_deg=180.0)]
    els = []
    for k in range(n_rings):
        xc = x0 + k * PITCH_MM
        els.append(ElectrodeSpec(
            name=f"ring_{k+1:03d}", dc=0.0,
            dc_group="ladder", dc_index=k + 1,
            rf_groups=["RFA" if k % 2 == 0 else "RFB"],
            shapes=[ShapeSpec("rect", {"x_mm": xc - 0.5 * THICK_MM,
                                       "y_mm": r_in,
                                       "width_mm": THICK_MM,
                                       "height_mm": R_OUT_MM - r_in})]))
    width = x0 + (n_rings - 1) * PITCH_MM + lead_mm
    height = R_OUT_MM + 0.5
    collisions_spec = CollisionSpec(enabled=collisions, gas=gas, T_k=T_k,
                                    P_torr=p_torr, sigma_m2=sigma_m2,
                                    model="hs1", sigma_model="fixed")
    # -- default flight duration: 3 x the Mason-Schamp drift time of THIS
    # ion over the ladder, from the same cross section the kernel flies
    # (replaces a former K0 = 7 cm^2/V/s guess -- a guessed constant that
    # happened to sit near the placeholder sigma's mobility)
    if t_max_us is None:
        if not collisions:
            raise ValueError(
                "t_max_us has no default in vacuum: the default is sized "
                "from the Mason-Schamp drift time, which needs gas. Pass "
                "t_max_us explicitly when collisions=False.")
        from ion_gym.physics.ccs import low_field_mobility_m2Vs
        # the ion's MASS in Da (review C-11: the function takes mass for
        # every charge; passing m/z gave a lighter reduced mass)
        K_m2Vs = low_field_mobility_m2Vs(collisions_spec, ion_mass_da,
                                         ion_charge)
        # K [m^2/Vs] x E [V/mm] -> mm/us (the m and mm factors cancel:
        # K*E[V/m] is m/s, and m/s = 1e-3 mm/us undoes the 1e3 of V/mm)
        v_d_mm_us = K_m2Vs * E_v_per_mm
        t_max_us = 3.0 * (L_ladder + 2 * lead_mm) / v_d_mm_us
    if rec_every is None:
        rec_every = max(int(t_max_us / (dt_ns * 1e-3) / 2000), 1)
    spec = SimSpec(
        name=f"rf-confining drift cell (Allen & Bush 2016) id {bore_id_mm:g} mm",
        notes=(f"Allen & Bush 2016 geometry: {n_rings} rings x 0.5 mm, pitch "
               f"1.5 mm, i.d. {bore_id_mm:g} mm, r_out {R_OUT_MM:g} mm (ours); "
               f"RF {rf_vpp:g} Vpp @ {rf_freq_hz/1e6:g} MHz alternating; "
               f"E/P {e_over_p:g} V/cm/Torr -> {E_v_per_mm:.5f} V/mm, ladder "
               f"{v_drop:.3f} V over {L_ladder:g} mm; {p_torr:g} Torr {gas} "
               f"{T_k:g} K; ion {ion_mass_da:g} Da z={ion_charge:+d}, sigma "
               f"{sigma_m2:.3g} m^2."),
        geometry=GeometrySpec(
            width_mm=width, height_mm=height, mm_per_gu=MM_GU,
            symmetry=SymmetrySpec(coords="rz"), electrodes=els,
            rf_groups=rf_groups,
            dc_groups=[DCGroupSpec("ladder", v_in=v_drop, v_out=0.0,
                                   interp="linear")]),
        source=SourceSpec(distribution="disc", n_ions=n_ions,
                          x0_mm=x0 + (source_ring - 0.5) * PITCH_MM,
                          y0_mm=0.0,
                          r_mm=source_r_mm, axis="x",
                          direction=[1.0, 0.0, 0.0], ke_lo=0.0, ke_hi=0.0,
                          temperature_k=T_k, mz_list=[ion_mass_da],
                          charge=ion_charge,
                          tob_span_us=1e6 / rf_freq_hz, seed=seed),
        collisions=collisions_spec,
        integration=IntegrationSpec(dt_ns=dt_ns, t_max_us=t_max_us,
                                    rec_every=rec_every,
                                    record_channels=["radius", "n_col",
                                                     "path_mm",
                                                     "e_axial_tint",
                                                     "ke_tint"]),
        bounds=BoundsSpec(x_max_on=True,
                          x_max=x0 + (exit_ring - 1) * PITCH_MM),
        view=ViewSpec(mode="2d", planes=["xy"]))
    return spec
