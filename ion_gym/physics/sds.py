"""
ion_gym.sds
-----------
Faithful port of the Statistical Diffusion Simulation (SDS) collision
model (Appelhans & Dahl, Int. J. Mass Spectrom. 244 (2005) 1-14), translated
from the published SDS formulation used in the tetramer worked example.

Two effects per time step:
  * Stokes'-law viscous mobility: acceleration is damped toward the local
    mobility-limited drift velocity (accel_adjust / apply_stokes_damping).
  * ICDF random-walk diffusion: a jump of magnitude drawn from the
    inverse-CDF statistics, scaled by the local MFP (other_actions /
    apply_diffusion).

Data files (ion_gym-native regenerations, in the package dir):
  sds_jump_icdf.dat -- 5 x 1002 diffusion ICDFs (mass ratios 10^0..10^4)
  sds_mobility.dat  -- mass(u), diameter(nm), Ko(1e-4 m^2/V/s) table

Units match that source: mm, us, amu, V, K, Torr, nm (diameters).
"""

JUMP_ICDF_FILE = "sds_jump_icdf.dat"
MOBILITY_FILE = "sds_mobility.dat"

# PROVENANCE
#   origin   : Appelhans & Dahl, Int. J. Mass Spectrom. 244 (2005) 1-14 --
#              the Statistical Diffusion Simulation (SDS) collision model.
#   extracted: faithful port, not a re-derivation.  Deviations from the paper
#              are bugs, not improvements.
import math
import numpy as np
from numba import njit

# ---- physical constants (source values, verbatim) ----
ELEMENTARY_CHARGE = 1.602176462e-19
K_BOLTZMANN = 1.3806503e-23
N_AVOGADRO = 6.02214199e23
MOL_VOLUME = 22.413996e-3
AMU_TO_KG = 1.66053873e-27
PI = math.pi
STP_TEMP = 273.15
M_AIR = 28.94515
D_AIR = 0.366

N_DIST_COLLISIONS = 100000
N_DIST = 5
N_DIST_POINTS = 1002


# ------------------------------------------------------------- data loaders
def _read_numbers(path):
    """Read comma/space separated numbers, ignoring ';' comments."""
    out = []
    for line in open(path):
        line = line.split(";")[0]
        for tok in line.replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def load_diffusion_statistics(path):
    raw = _read_numbers(path)
    assert len(raw) == N_DIST_POINTS * N_DIST, (
        f"{JUMP_ICDF_FILE} has {len(raw)} numbers, expected "
        f"{N_DIST_POINTS * N_DIST}")
    return np.array(raw, np.float64).reshape(N_DIST, N_DIST_POINTS)


def load_massdata(path):
    raw = _read_numbers(path)
    rows = []
    for i in range(0, len(raw) - 2, 3):
        m, d, ko = raw[i], raw[i + 1], raw[i + 2]
        if m == 0:
            break
        rows.append((abs(m), abs(d), abs(ko)))
    return rows


# ------------------------------------------------------------- mass -> params
def get_air_to_gas(d_ion, mass_ion, d_gas, mass_gas):
    rm_air = mass_ion * M_AIR / (mass_ion + M_AIR)
    rm_gas = mass_ion * mass_gas / (mass_ion + mass_gas)
    return ((d_ion + D_AIR) / (d_ion + d_gas))**2 * math.sqrt(rm_air / rm_gas)


def estimate_d_ion(mass_ion, ko, d_gas, mass_gas):
    d_ion = 0.120415405 * mass_ion**(1/3)
    if ko:
        Koair = ko / get_air_to_gas(d_ion, mass_ion, d_gas, mass_gas)
        logkm = math.log10(Koair * 1.0e5)
        d_ion = 10**(3.0367 - 0.8504*logkm + 0.1137*logkm**2 - 0.0135*logkm**3)
    return d_ion


def estimate_ko(mass_ion, d_ion, d_gas, mass_gas):
    logdm = math.log10(d_ion)
    Koair = 1.0e-5 * 10**(4.9137 - 1.4491*logdm - 0.2772*logdm**2
                          + 0.0717*logdm**3)
    return Koair * get_air_to_gas(d_ion, mass_ion, d_gas, mass_gas)


def _Vo_MFPo(mass_ion, d_ion, d_gas, mass_gas):
    Vk = math.sqrt(8*K_BOLTZMANN*STP_TEMP/PI/AMU_TO_KG) * 1.0e-3   # mm/us
    Vio = Vk * math.sqrt(1/mass_ion)
    Vgo = Vk * math.sqrt(1/mass_gas)
    No = N_AVOGADRO / MOL_VOLUME / 1.0e9
    Fio = 1.0e-12 * No * PI * ((math.sqrt(2)-1/4)*((d_gas+d_ion)/2)**2*Vio
                               + (1/4)*d_ion**2*Vgo)
    Lio = Vio / Fio
    return Vio, Lio                                                # vo, mfpo


def ion_params(mass_ion, charge, gas_mass, gas_diam, T_k, P_torr, massdata):
    """Full per-ion SDS parameters at local T,P. Returns dict with damping
    (1/us), mfp_mm, V_mm_us, log_mr_ratio, ko, d_ion — mirrors update_ion +
    update_ions_local in that source."""
    d_ion = ko = 0.0
    for (m, d, k) in massdata:
        if m == mass_ion:
            d_ion, ko = d, k
            break
    # complete_records logic for this mass
    if d_ion == 0 and ko == 0:
        d_ion = estimate_d_ion(mass_ion, None, gas_diam, gas_mass)
        ko = estimate_ko(mass_ion, d_ion, gas_diam, gas_mass)
    elif d_ion == 0:
        d_ion = estimate_d_ion(mass_ion, ko, gas_diam, gas_mass)
    elif ko == 0:
        ko = estimate_ko(mass_ion, d_ion, gas_diam, gas_mass)
    vo, mfpo = _Vo_MFPo(mass_ion, d_ion, gas_diam, gas_mass)

    emu = ELEMENTARY_CHARGE / AMU_TO_KG
    damping_STP = emu * 0.01 / ko * (abs(charge) / mass_ion)     # 1/us
    t_ratio = T_k / STP_TEMP
    pt_ratio = t_ratio * (760.0 / P_torr)
    return dict(
        damping=damping_STP / pt_ratio,
        mfp_mm=mfpo * pt_ratio,
        V_mm_us=vo * math.sqrt(t_ratio),
        log_mr_ratio=math.log10(mass_ion / gas_mass),
        ko=ko, d_ion=d_ion, vo=vo, mfpo=mfpo)


# ------------------------------------------------------------- njit kernels
@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _diff_dist_steps(stats, log_mr_ratio):
    """Random jump distance (in normalized 'steps') from the ICDF table, for
    this mass ratio. Faithful port of diff_dist_steps."""
    n = np.random.random() * (N_DIST_POINTS - 2)
    if log_mr_ratio <= 1.0:
        iicdf = 0
    elif log_mr_ratio <= 2.0:
        iicdf = 1
    elif log_mr_ratio <= 3.0:
        iicdf = 2
    else:
        iicdf = 3
    ilow = int(math.floor(n))
    ihigh = ilow + 1
    weight = n - math.floor(n)
    d1 = (stats[iicdf, ihigh] - stats[iicdf, ilow]) * weight + stats[iicdf, ilow]
    d2 = (stats[iicdf+1, ihigh] - stats[iicdf+1, ilow]) * weight + stats[iicdf+1, ilow]
    d1 = math.log10(d1); d2 = math.log10(d2)
    w2 = log_mr_ratio - iicdf                       # (iicdf+1)-1 = iicdf
    return 10.0**((d2 - d1) * w2 + d1)


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _sphere_rand(r):
    """Uniform random vector of length r (Marsaglia)."""
    while True:
        xp = 2.0*np.random.random() - 1.0
        yp = 2.0*np.random.random() - 1.0
        S = xp*xp + yp*yp
        if S <= 1.0:
            break
    z = (2.0*S - 1.0) * r
    f = 2.0*r*math.sqrt(1.0 - S)
    return xp*f, yp*f, z
