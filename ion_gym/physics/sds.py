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

# ---------------------------------------------------------------- regime
# Threshold on gamma/Omega. A NAMED DEFAULT, not a derived number: it is
# the order of magnitude at which the mobility limit is reached within an
# RF cycle. The exact value wants measuring -- sweep gamma/Omega over a
# couple of decades at fixed geometry, run HS and SDS at each point, and
# take the ratio where SDS departs from HS beyond tolerance. HS is the
# reference because it resolves collisions and assumes no terminal
# velocity. Until that campaign runs, this is a judgement, and the
# warning says so by printing the ratio rather than a verdict alone.
SDS_MIN_GAMMA_OVER_OMEGA = 1.0
SDS_ASSUMED_MZ_WHEN_UNSET = 300.0


def rf_regime_note(spec):
    """Warn when SDS is used with RF outside the regime it is valid in.

    Returns a multi-line warning string, or None when the operating
    point is fine or the check does not apply.

    THE RATIO IS gamma/Omega, NOT COLLISIONS PER CYCLE (corrected
    2026-09-15). gamma is the MOMENTUM-TRANSFER relaxation rate
    q/(m*K) that ion_params() already computes from the shipped mobility
    table -- the same number the integrator damps with. Omega = 2*pi*f.
    An earlier version of this guard counted kinetic-theory COLLISIONS
    per cycle and was wrong by two orders of magnitude on the PI's deck
    (1.35 against the true 0.015), because a collision is not a
    relaxation: a heavy ion needs roughly its mass ratio in light-gas
    collisions before it loses directed velocity. Mobility accounts for
    that; a collision count does not. Using the model's own damping also
    removes the assumed cross-section and assumed relative speed
    entirely -- nothing here is estimated except the mass when the deck
    does not declare one.

    WHY IT MATTERS. SDS damps acceleration toward the LOCAL
    MOBILITY-LIMITED DRIFT VELOCITY each step and then adds an ICDF
    diffusive jump; both halves assume that relaxation COMPLETES within
    a step, i.e. gamma >> Omega. RF confinement is the opposite
    condition: the pseudopotential is built from MICROMOTION, the driven
    oscillation out of phase with the field gradient, which survives
    only when gamma << Omega. The two cannot both hold. Where RF
    confinement is physical, SDS cannot represent it, and the ions go
    unconfined and time out while the same deck under hard-sphere
    transits normally.

    MEASURED on the PI's bent flatapole (0.05 Torr N2, 273 K, 2 MHz):
    gamma/Omega = 0.027 at m/z 100, 0.015 at 300, 0.008 at 622 -- deep
    in the RF-confined regime, and getting worse with mass. Reaching
    gamma/Omega = 1 at m/z 300 and 2 MHz would need about 6.4 Torr.

    THIS WARNS, IT DOES NOT REFUSE (PI ruling). The boundary is soft,
    the ratio degrades gradually rather than failing at a threshold, and
    working near it is legitimate. See Allen and Bush, Anal. Chem. 88
    (2016) -- RF confinement in ion mobility, apparent mobilities and
    effective temperatures.
    """
    import math
    import pathlib as _pl
    coll = getattr(spec, "collisions", None)
    if coll is None or not getattr(coll, "enabled", False):
        return None
    if str(getattr(coll, "model", "")).lower() != "sds":
        return None
    rf = [g for g in (spec.geometry.rf_groups or [])
          if float(getattr(g, "amplitude_v", 0.0) or 0.0) != 0.0]
    if not rf:
        return None                       # no RF: nothing to erase
    f_hz = max(float(getattr(g, "frequency_hz", 0.0) or 0.0) for g in rf)
    p_torr = float(getattr(coll, "P_torr", 0.0) or 0.0)
    t_k = float(getattr(coll, "T_k", 0.0) or 0.0)
    if f_hz <= 0 or p_torr <= 0 or t_k <= 0:
        return None                       # not enough to judge; say nothing

    # SourceSpec declares masses as mz_list; "mz" is not a field on it,
    # so a getattr default would silently fall through to the assumed
    # mass on EVERY deck and never report a real number.
    mz = getattr(spec.source, "mz_list", None)
    if isinstance(mz, (list, tuple)) and mz:
        masses, mz_note = sorted(float(x) for x in mz), ""
    elif isinstance(mz, (int, float)) and mz:
        masses, mz_note = [float(mz)], ""
    else:
        masses = [SDS_ASSUMED_MZ_WHEN_UNSET]
        mz_note = (f" (source.mz_list is unset, so m/z {masses[0]:g} was "
                   f"ASSUMED -- declare it and this is exact)")

    gas_mass = float(getattr(coll, "gas_mass_amu", 0.0) or 28.0)
    gas_diam = float(getattr(coll, "gas_diam_nm", 0.0) or 0.366)
    charge = abs(int(getattr(spec.source, "charge", 1) or 1))
    md = load_massdata(str(_pl.Path(__file__).parent / "sds_mobility.dat"))
    omega = 2.0 * math.pi * f_hz

    rows = []
    for m in masses:
        pr = ion_params(m, charge, gas_mass, gas_diam, t_k, p_torr, md)
        gamma = pr["damping"] * 1.0e6                     # 1/us -> 1/s
        rows.append((m, gamma / omega, pr["mfp_mm"], pr["ko"]))
    worst = min(r[1] for r in rows)
    if worst >= SDS_MIN_GAMMA_OVER_OMEGA:
        return None

    # pressure that would reach the threshold for the worst mass
    m_worst = [r[0] for r in rows if r[1] == worst][0]
    need_p = None
    _p = p_torr
    for _ in range(40):
        _p *= 2.0
        _g = ion_params(m_worst, charge, gas_mass, gas_diam, t_k, _p,
                        md)["damping"] * 1.0e6
        if _g / omega >= SDS_MIN_GAMMA_OVER_OMEGA:
            need_p = _p
            break

    lines = [
        f"[sds] WARNING: SDS with RF at gamma/Omega = {worst:.4f} -- "
        f"below the {SDS_MIN_GAMMA_OVER_OMEGA:g} this model needs."
        f"{mz_note}",
        f"[sds]   gamma is the momentum-transfer relaxation rate "
        f"q/(m*K) from the shipped mobility table; Omega = 2*pi*"
        f"{f_hz / 1e6:.3f} MHz, at {p_torr:.4g} Torr / {t_k:.0f} K:",
    ]
    for m, ratio, mfp, ko in rows:
        lines.append(f"[sds]     m/z {m:<8g} Ko {ko:.4g}  "
                     f"gamma/Omega {ratio:.4f}  mfp {mfp:.3f} mm")
    lines += [
        "[sds]   SDS assumes relaxation completes within a step "
        "(gamma >> Omega). RF confinement needs the opposite "
        "(gamma << Omega): the pseudopotential is made of MICROMOTION, "
        "which the damping erases. Expect ions to go UNCONFINED and "
        "time out while the same deck under model='hs' transits.",
        (f"[sds]   To reach gamma/Omega = "
         f"{SDS_MIN_GAMMA_OVER_OMEGA:g} at m/z {m_worst:g}: pressure "
         f"about {need_p:.2f} Torr at this frequency"
         if need_p else
         f"[sds]   No reachable pressure below 2^40 x {p_torr:g} Torr "
         f"puts m/z {m_worst:g} in regime at this frequency"),
        "[sds]   Otherwise use model='hs', which resolves collisions "
        "and keeps the micromotion.",
    ]
    return "\n".join(lines)
