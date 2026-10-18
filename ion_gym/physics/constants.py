"""Physical constants — the single authority for the package.

CODATA 2018 / SI 2019 (PI ruling 2026-10-13, R1: "Move CODATA 2018
everywhere"). Before this module the package carried three vintages at
once: the 1998 set in sds.py, the "source's value" kB/amu in
collision3d.py that eight physics modules imported, and CODATA 2018 in
beam_spec/traj_stats/tracer3d. Masses and energies therefore differed in
the 6th digit depending on which route computed them.

Every value here is either exact by the 2019 SI definitions, a CODATA
2018 recommended value, or derived from those in this file — never
retyped. Import from here; do not define a second copy.

Dependency-free on purpose: io/ and viz/ import it without pulling in
numba or any physics module.
"""

# --- exact by definition (SI 2019) -------------------------------------
E_CHG = 1.602176634e-19        # C, elementary charge
K_B = 1.380649e-23             # J/K, Boltzmann constant
N_A = 6.02214076e23            # 1/mol, Avogadro constant

# --- CODATA 2018 recommended -------------------------------------------
KG_AMU = 1.66053906660e-27     # kg per Da (atomic mass constant)

# --- conventional reference conditions (definitions) -------------------
T_STP_K = 273.15               # K, standard temperature
P_STP_PA = 101325.0            # Pa, standard atmosphere
ATM_TORR = 760.0               # Torr per standard atmosphere, exact
TORR_PA = P_STP_PA / ATM_TORR  # Pa per Torr, exact

# --- derived (computed here, never retyped elsewhere) ------------------
K_B_EV = K_B / E_CHG                         # eV/K
MOL_VOLUME_STP = N_A * K_B * T_STP_K / P_STP_PA   # m^3/mol, ideal gas at STP
# Charge-to-mass acceleration in the plane flyers' units:
# a [mm/us^2] = ACC * q * E [V/mm] / m [Da]. e/amu is in C/kg, i.e.
# (m/s^2)/(V/m); V/mm -> V/m is 1e3, m/s^2 -> mm/us^2 is 1e3/1e12.
ACC = E_CHG / KG_AMU * 1e-6                  # mm/us^2 per (V/mm / Da)
