"""
ion_gym.physics.sds
-------------------
Statistical Diffusion Simulation (SDS): ion transport through a gas at
pressures where individual collisions are too many to fly. Written for
ion_gym from the published method (Appelhans & Dahl, Int. J. Mass Spectrom.
244 (2005) 1-14, Sections 2.1-2.5) and first-order kinetic theory. It is
not a translation of any program's source.

Each time step the flight kernels do two things with the per-ion numbers
this module supplies (transport_params):
  * VISCOUS DRIFT. The velocity relaxes toward the mobility-limited drift
    velocity K*E at the rate gamma = q / (m K) (the kernels integrate that
    relaxation exactly over the step).
  * DIFFUSION. The ion jumps a random distance in a random direction. The
    distance is drawn from tabulated statistics of hard-sphere random walks
    (sds_jump_icdf.dat, our own Monte Carlo of the paper's Sec. 2.5
    procedure), so the SHAPE of the spread follows the ion/gas mass ratio.

WHAT THIS MODULE ADDS TO THE PAPER, stated because they are choices:
  * CHARGE. The paper's mobility formula is written for a singly charged
    ion. Here the cross section is taken from the ion's MASS (the singly
    charged relations) and the mobility scales with charge:
        K_z = abs(z) * K1(m)      (Mason-Schamp at fixed cross section).
    The drift direction follows the sign of z. This first-order model
    neglects the growth of the cross section with z from the ion-induced
    dipole (so it OVERESTIMATES K for multiply charged ions, most for
    small, compact ions in polarisable gases) and any charge-driven change
    of shape. With K_z = abs(z) K1 the relaxation rate is e / (m K1),
    independent of z.
  * DIFFUSION SCALE. The jump size is tied to the Einstein relation,
        D = K1 * k T / e      (= K_z k T / (abs(z) e), independent of z),
    so diffusion always agrees with the mobility in use: the jump scale is
    set so that the mean squared 3-D jump per step is 6 D dt exactly, for
    the tabulated jump statistics at this mass ratio. (The paper instead
    scales jumps by a hard-sphere mean free path from the ion diameter,
    which can disagree with a mobility given independently.)
  * NO MOBILITY TABLE is shipped. A deck may declare K0 and/or the ion
    diameter per ion mass (collisions.sds_species), each K0 tagged with
    the gas it was measured in; a K0 from another gas is converted,
    loudly, by the Mason-Schamp hard-sphere ratio. Undeclared masses come
    from the paper's estimators (Sec. 2.1-2.2): diameter from mass, the
    air K0-diameter fits, and the conversion from air to the deck's gas.

Units at the kernel interface: mm, us, amu, V, K, Torr, nm.
"""

JUMP_ICDF_FILE = "sds_jump_icdf.dat"

# PROVENANCE
#   origin   : Appelhans & Dahl, Int. J. Mass Spectrom. 244 (2005) 1-14,
#              Sec. 2.1-2.5: v = K E; K0 at 273.15 K and 760 Torr; the ion
#              diameter from mass (constant density, from the CRC Handbook
#              of Chemistry and Physics, 83rd ed.); the air K0 <-> diameter
#              cubic fits; the air-to-gas Mason-Schamp conversion with
#              Omega = pi ((d_ion + d_gas) / 2)^2; jump statistics by mass
#              ratio, interpolated log-log, scaled as sqrt(collisions).
#   derived  : the charge rule K_z = abs(z) K1(m) and the Einstein-tied
#              jump scale (first-order kinetic theory; stated above).
#   note     : the air mass is the paper's 28.954515 amu (dry air is
#              28.965), kept because the air fits were made with it.
import math
import pathlib
import numpy as np
from numba import njit

from ion_gym.physics.constants import (E_CHG, K_B, KG_AMU, T_STP_K,
                                       ATM_TORR)

# ---- the paper's air reference and estimators (Sec. 2.1-2.2) ----
AIR_MASS_AMU = 28.954515
AIR_DIAMETER_NM = 0.366
# d_ion(nm) = 0.120415405 * mass(amu)^(1/3): constant density, scaled to a
# 130 amu ion of 0.61 nm.
MASS_TO_DIAMETER_NM = 0.120415405
# log10 d(nm) as a cubic in log10 K9, with K9 = 1e5 * K0 (K0 in cm^2/(V s))
_LOGD_OF_LOGK9 = (3.0367, -0.8504, 0.1137, -0.0135)
# log10 (1e5 * K0) as a cubic in log10 d(nm)
_LOGK9_OF_LOGD = (4.9137, -1.4491, -0.2772, 0.0717)

# ---- jump statistics table layout (sds_jump_icdf.dat) ----
N_DIST_COLLISIONS = 100000      # collisions each tabulated walk ran
N_DIST = 5                      # mass ratios 10^0 .. 10^4
N_DIST_POINTS = 1002            # min, 0.1 % .. 99.9 %, max
# points of the midpoint rule that takes the mean squared jump of the
# kernels' own draw (u uniform in [0, 1))
_MEAN_SQ_POINTS = 200000


# ------------------------------------------------------------ jump table
def _read_numbers(path):
    """Numbers in a comma- or space-separated file, ';' starting a comment.
    A token that is not a number REFUSES, naming file, line and token: the
    table is read positionally, so a skipped token would shift every later
    value."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            for tok in line.split(";")[0].replace(",", " ").split():
                try:
                    out.append(float(tok))
                except ValueError:
                    raise ValueError(
                        f"{path}:{ln}: {tok!r} is not a number. The SDS table "
                        f"is read positionally, so skipping it would shift "
                        f"every later value; fix or remove the token.") from None
    return out


def load_diffusion_statistics(path):
    """(N_DIST, N_DIST_POINTS) jump statistics, in mean free paths after
    N_DIST_COLLISIONS collisions, one row per mass ratio 10^0 .. 10^4."""
    raw = _read_numbers(path)
    if len(raw) != N_DIST_POINTS * N_DIST:
        raise ValueError(f"{path}: {len(raw)} numbers, expected "
                         f"{N_DIST_POINTS * N_DIST} ({N_DIST} rows of "
                         f"{N_DIST_POINTS})")
    return np.array(raw, np.float64).reshape(N_DIST, N_DIST_POINTS)


_STATS = None


def jump_statistics():
    """The shipped jump statistics, read once per process. Every SDS route
    (planar, r-z, 3-D, staged regions) reads them through here."""
    global _STATS
    if _STATS is None:
        _STATS = load_diffusion_statistics(
            str(pathlib.Path(__file__).with_name(JUMP_ICDF_FILE)))
    return _STATS


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def _icdf_steps_at(stats, log_mr, u):
    """Jump distance (mean free paths, after N_DIST_COLLISIONS collisions)
    at probability u in [0, 1) for mass ratio 10^log_mr: linear in u within
    each tabulated row, then log-log between the two rows that bracket the
    mass ratio (extrapolated outside 10^0 .. 10^4)."""
    x = u * (N_DIST_POINTS - 2)
    i = int(x)
    w = x - i
    k = int(math.floor(log_mr))
    if k < 0:
        k = 0
    elif k > N_DIST - 2:
        k = N_DIST - 2
    t = log_mr - k
    lo = stats[k, i] + w * (stats[k, i + 1] - stats[k, i])
    hi = stats[k + 1, i] + w * (stats[k + 1, i + 1] - stats[k + 1, i])
    llo = math.log10(lo)
    return 10.0 ** (llo + t * (math.log10(hi) - llo))


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def icdf_jump_steps(stats, log_mr):
    """One random jump distance from the statistics (kernel helper)."""
    return _icdf_steps_at(stats, log_mr, np.random.random())


@njit(cache=True, fastmath=False, inline="always", nogil=True)
def isotropic_vector(r):
    """A vector of length r in a uniformly random direction: the cosine of
    the polar angle is uniform on [-1, 1] and the azimuth on [0, 2 pi)
    (no rejection loop)."""
    c = 2.0 * np.random.random() - 1.0
    ph = 2.0 * math.pi * np.random.random()
    s = r * math.sqrt(max(0.0, 1.0 - c * c))
    return s * math.cos(ph), s * math.sin(ph), r * c


@njit(cache=True, nogil=True)
def _mean_sq_steps(stats, log_mr, n):
    acc = 0.0
    for k in range(n):
        d = _icdf_steps_at(stats, log_mr, (k + 0.5) / n)
        acc += d * d
    return acc / n


_MEAN_SQ = {}


def mean_sq_jump_steps(log_mr):
    """<d^2> of icdf_jump_steps at this mass ratio (the kernels' own draw,
    integrated by the midpoint rule), cached per mass ratio."""
    key = round(float(log_mr), 12)
    if key not in _MEAN_SQ:
        _MEAN_SQ[key] = float(_mean_sq_steps(jump_statistics(), key,
                                             _MEAN_SQ_POINTS))
    return _MEAN_SQ[key]


# ------------------------------------------------------------ estimators
def _cubic(c, x):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def diameter_from_mass_nm(mass_amu):
    """Ion diameter (nm) from mass (paper Sec. 2.1, constant density)."""
    return MASS_TO_DIAMETER_NM * float(mass_amu) ** (1.0 / 3.0)


def air_K0_from_diameter(d_nm):
    """Reduced mobility in AIR, cm^2/(V s), from the ion diameter (paper's
    log-log cubic)."""
    return 1.0e-5 * 10.0 ** _cubic(_LOGK9_OF_LOGD, math.log10(d_nm))


def diameter_from_air_K0(K0_air):
    """Ion diameter (nm) from a reduced mobility in AIR (paper's cubic)."""
    return 10.0 ** _cubic(_LOGD_OF_LOGK9, math.log10(1.0e5 * K0_air))


def gas_over_air(mass_amu, d_nm, gas_mass_amu, gas_d_nm):
    """K0(gas) / K0(air) at fixed ion size: the Mason-Schamp ratio
    sqrt(mu_air / mu_gas) * Omega_air / Omega_gas with hard-sphere
    Omega = pi ((d_ion + d_neutral) / 2)^2 (paper Sec. 2.2)."""
    m = float(mass_amu)
    mu_air = m * AIR_MASS_AMU / (m + AIR_MASS_AMU)
    mu_gas = m * gas_mass_amu / (m + gas_mass_amu)
    return (math.sqrt(mu_air / mu_gas)
            * ((d_nm + AIR_DIAMETER_NM) / (d_nm + gas_d_nm)) ** 2)


def estimated_K0(mass_amu, gas_mass_amu, gas_d_nm):
    """(K0 in the deck's gas, cm^2/(V s), singly charged; diameter nm) from
    the ion's mass alone (paper Sec. 2.1-2.2)."""
    d = diameter_from_mass_nm(mass_amu)
    return (air_K0_from_diameter(d)
            * gas_over_air(mass_amu, d, gas_mass_amu, gas_d_nm)), d


# ------------------------------------------------- declared species values
SPECIES_KEYS = ("mass_amu", "gas", "charge", "K0_cm2_per_Vs", "d_nm",
                "gas_d_nm")
# two declared masses closer than this are the same ion
_SAME_MASS_DA = 1e-6


def _positive(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v > 0)


def species_errors(entries, deck_gas):
    """Refusals for collisions.sds_species entries, as messages naming the
    entry. Each entry gives a mass and at least one of K0 or the ion
    diameter; a K0 names the gas and the charge state it was measured at,
    and a gas other than the deck's needs that gas's diameter for the
    conversion."""
    from ion_gym.physics.collision3d import GASES
    errs, seen = [], []
    if not isinstance(entries, list):
        return [f"collisions.sds_species must be a list of entries, got "
                f"{type(entries).__name__}"]
    for i, e in enumerate(entries):
        at = f"collisions.sds_species[{i}]"
        if not isinstance(e, dict):
            errs.append(f"{at} must be an object, got {type(e).__name__}")
            continue
        extra = sorted(set(e) - set(SPECIES_KEYS))
        if extra:
            errs.append(f"{at}: unknown key(s) {extra}; an entry takes "
                        f"{list(SPECIES_KEYS)}")
        m = e.get("mass_amu")
        if not _positive(m):
            errs.append(f"{at}: mass_amu = {m!r} must be a finite number > 0")
        elif any(abs(m - s) <= _SAME_MASS_DA for s in seen):
            errs.append(f"{at}: mass {m:g} amu is declared twice")
        else:
            seen.append(m)
        k0, d = e.get("K0_cm2_per_Vs"), e.get("d_nm")
        if k0 is None and d is None:
            errs.append(f"{at}: states neither K0_cm2_per_Vs nor d_nm; give "
                        f"at least one (or remove the entry to estimate the "
                        f"ion from its mass)")
        for key in ("K0_cm2_per_Vs", "d_nm", "gas_d_nm"):
            if e.get(key) is not None and not _positive(e[key]):
                errs.append(f"{at}: {key} = {e[key]!r} must be a finite "
                            f"number > 0")
        if k0 is not None:
            z = e.get("charge")
            if (not isinstance(z, int) or isinstance(z, bool)) or z == 0:
                errs.append(
                    f"{at}: charge = {z!r}. A measured K0 belongs to the "
                    f"charge state it was measured at: state it as a nonzero "
                    f"integer 'charge' (an ion of that charge flies the value "
                    f"as given).")
            gas = e.get("gas")
            if gas is None:
                errs.append(f"{at}: a K0 is specific to the gas it was "
                            f"measured in; state that gas as 'gas'")
            elif gas not in GASES:
                errs.append(f"{at}: gas {gas!r} is not one ion_gym knows "
                            f"({sorted(GASES)})")
            elif gas != deck_gas and e.get("gas_d_nm") is None:
                errs.append(
                    f"{at}: K0 was measured in {gas!r} but the deck's gas is "
                    f"{deck_gas!r}. Converting it needs the diameter of "
                    f"{gas!r}: state it as 'gas_d_nm', or give a K0 "
                    f"measured in {deck_gas!r}.")
    return errs


def species_entry(entries, mass_amu):
    """The collisions.sds_species entry for this ion mass, or None."""
    for e in entries or ():
        if abs(float(e["mass_amu"]) - float(mass_amu)) <= _SAME_MASS_DA:
            return e
    return None


def resolve_K0(mass_amu, gas_name, gas_mass_amu, gas_d_nm, entry):
    """(K0 in the deck's gas, cm^2/(V s), PER UNIT CHARGE; ion diameter nm;
    how it was obtained) for one ion. The order is the paper's (Sec. 2.2):
      no entry           -> diameter from mass, K0 from the air fit, to gas;
      diameter only      -> K0 from the air fit at that diameter, to gas;
      K0 (and diameter)  -> used as given when measured in the deck's gas;
                            converted at the ion's diameter otherwise
                            (a missing diameter is found from the K0 via
                            the air fit, starting from the mass estimate).
    A declared K0 measured at charge z_m is referred to unit charge as
    K0 / abs(z_m) (Mason-Schamp at fixed cross section), so an ion of
    charge z_m flies abs(z_m) * K0 / abs(z_m) = the measured value exactly.
    """
    from ion_gym.physics.collision3d import GASES
    m = float(mass_amu)
    if entry is None:
        K0, d = estimated_K0(m, gas_mass_amu, gas_d_nm)
        return K0, d, "estimated from mass"
    k0e, de = entry.get("K0_cm2_per_Vs"), entry.get("d_nm")
    if k0e is None:
        d = float(de)
        return (air_K0_from_diameter(d)
                * gas_over_air(m, d, gas_mass_amu, gas_d_nm), d,
                f"from the declared diameter {d:g} nm")
    egas = entry["gas"]
    zm = abs(int(entry["charge"]))
    k1e = float(k0e) / zm                      # per unit charge
    at = f"{egas} at charge {int(entry['charge']):+d}"
    same = egas == gas_name
    e_mass = gas_mass_amu if same else GASES[egas]
    e_d = gas_d_nm if same else float(entry["gas_d_nm"])
    if de is None:
        d0 = diameter_from_mass_nm(m)
        d = diameter_from_air_K0(k1e / gas_over_air(m, d0, e_mass, e_d))
    else:
        d = float(de)
    if same:
        return k1e, d, f"declared {float(k0e):g} in {at}"
    ratio = (gas_over_air(m, d, gas_mass_amu, gas_d_nm)
             / gas_over_air(m, d, e_mass, e_d))
    return (k1e * ratio, d,
            f"declared {float(k0e):g} in {at}, CONVERTED to {gas_name} "
            f"(x {ratio:.4g})")


# ------------------------------------------------------- per-ion numbers
def transport_params(mass_amu, charge, gas_mass_amu, gas_d_nm, T_k, P_torr,
                     *, gas_name=None, species=None):
    """The per-ion SDS numbers at the local T and P. `species` is this
    ion's collisions.sds_species entry (or None to estimate from mass);
    `gas_name` is the deck's gas, which a declared K0 is compared with.

    damping       1/us   relaxation rate gamma = abs(z) e / (m K_z)
                         (= e / (m K1), independent of z)
    mfp_mm        mm     the jump-scale length the kernels use: chosen so
                         that the mean squared 3-D jump per step is 6 D dt
                         (it is NOT a physical mean free path)
    V_mm_us       mm/us  ion mean thermal speed sqrt(8 k T / (pi m))
    log_mr_ratio         log10(ion mass / gas mass): the jump statistics row
    K0                   reduced mobility in the deck's gas, cm^2/(V s),
                         per unit charge (a declared K0 divided by the
                         abs(charge) it was measured at)
    K0_source            how K0 was obtained (resolve_K0)
    K_mm2_per_V_us       the ion's mobility at T, P and its charge, abs(z) K1
    D_mm2_per_us         diffusion coefficient K1 k T / e
    d_ion_nm             the ion diameter used
    """
    m = float(mass_amu)
    if not (m > 0.0):
        raise ValueError(f"SDS: ion mass must be positive, got {mass_amu!r}")
    if not (P_torr > 0.0 and T_k > 0.0):
        raise ValueError(f"SDS: T and P must be positive, got T = {T_k!r} K, "
                         f"P = {P_torr!r} Torr")
    if species is not None and gas_name is None:
        raise ValueError("SDS: a declared species value needs the deck's "
                         "gas name to compare its measurement gas with")
    K0, d, how = resolve_K0(m, gas_name, gas_mass_amu, gas_d_nm, species)
    # K0 is defined at 273.15 K and 760 Torr: K = K0 (760 / P) (T / 273.15)
    K1 = K0 * 1.0e-4 * (ATM_TORR / P_torr) * (T_k / T_STP_K)   # m^2/(V s)
    m_kg = m * KG_AMU
    D = K1 * K_B * T_k / E_CHG             # m^2/s, numerically mm^2/us
    V = math.sqrt(8.0 * K_B * T_k / (math.pi * m_kg)) * 1.0e-3
    log_mr = math.log10(m / float(gas_mass_amu))
    msq = mean_sq_jump_steps(log_mr)
    return dict(
        damping=E_CHG / (m_kg * K1) * 1.0e-6,
        mfp_mm=6.0 * D * N_DIST_COLLISIONS / (V * msq),
        V_mm_us=V,
        log_mr_ratio=log_mr,
        K0=K0,
        K0_source=how,
        K_mm2_per_V_us=abs(int(charge)) * K1,    # m^2/(V s) == mm^2/(V us)
        D_mm2_per_us=D,
        d_ion_nm=d)


# ------------------------------------------------- one door for every route
def sds_active(collisions):
    """True when this collisions argument (None, dict or CollisionSpec)
    asks for gas ON under the SDS model. An unknown model name REFUSES by
    name: the routes only know 'hs' and 'sds', and guessing which one a
    misspelled model meant is how an SDS deck once flew hard-sphere."""
    from ion_gym.physics.collision3d import resolve_collisions
    col = resolve_collisions(collisions)
    if col is None:
        return False
    model = str(col.model).lower()
    if model not in ("hs", "sds"):
        raise ValueError(
            f"collisions.model {col.model!r} is not a known model; the "
            f"flight routes implement 'hs' (hard-sphere) and 'sds'.")
    return model == "sds"


def sds_ion_params(collisions, mass_amu, charge):
    """transport_params for this ion on the deck's gas (by name), its
    diameter, T and P, the ion's own mass and charge. ONE call shared by
    every route that flies SDS.

    Arguments (units in brackets; [-] is dimensionless):
      collisions  [-]   CollisionSpec or dict: gas (name), gas_diam_nm
                        [nm], T_k [K], P_torr [Torr] -- Torr here, where
                        the hard-sphere kernels take P_pa [Pa]
      mass_amu    [Da]  ion mass
      charge      [e]   signed charge state"""
    from ion_gym.physics.collision3d import gas_mass, resolve_collisions
    col = resolve_collisions(collisions)
    if col is None:
        raise ValueError("SDS parameters need gas: collisions is None or "
                         "disabled.")
    return transport_params(float(mass_amu), int(charge), gas_mass(col.gas),
                            float(col.gas_diam_nm), float(col.T_k),
                            float(col.P_torr), gas_name=col.gas,
                            species=species_entry(col.sds_species, mass_amu))


def species_notes(spec):
    """Advisories for the SDS mobilities a spec flies: the K0 each declared
    mass flies with and where it came from (displayed equals solver input),
    a WARNING naming every K0 converted from another gas, and a note when
    species values are declared but the model is not SDS. [] otherwise."""
    from ion_gym.physics.collision3d import resolve_collisions
    col = resolve_collisions(getattr(spec, "collisions", None))
    if col is None:
        return []
    if str(col.model).lower() != "sds":
        if col.sds_species:
            return [f"collisions: sds_species declares "
                    f"{len(col.sds_species)} value(s), but model is "
                    f"{col.model!r}, which does not read them."]
        return []
    rows, converted = [], []
    for m in sorted(float(x) for x in (spec.source.mz_list or [])):
        p = sds_ion_params(col, m, int(spec.source.charge))
        rows.append(f"m {m:g}: K0 {p['K0']:.4g} ({p['K0_source']})")
        if "CONVERTED" in p["K0_source"]:
            converted.append(f"m {m:g} ({p['K0_source']})")
    out = [f"collisions: SDS reduced mobilities in {col.gas} (cm^2/(V s), "
           f"per unit charge; each ion flies abs(z) x K): " + "; ".join(rows)]
    if converted:
        out.append(f"WARNING: SDS mobilities measured in another gas were "
                   f"converted to {col.gas} by the Mason-Schamp hard-sphere "
                   f"ratio at the ion's diameter: " + "; ".join(converted)
                   + ". State a K0 measured in the deck's gas to fly it "
                   "as given.")
    return out


# THE COLLISION-COUNT RULE under SDS, identical on every route: the per-ion
# summary carries n_col = None, a recorded n_col column is NaN, and this
# advisory says why. Not a refusal: switching a deck's model must never
# block a flight because a recording column was requested.
SDS_NO_COLLISION_COUNT = (
    "collisions: the n_col (collision count) channel is NaN and the "
    "per-ion collision count is empty, because model='sds' does not "
    "simulate individual ion-gas collisions. SDS (Appelhans and Dahl "
    "2005) treats the gas as a continuum: every time step it (1) relaxes "
    "the ion's velocity toward its mobility-limited drift velocity K*E "
    "and (2) moves the ion by a random diffusion jump drawn from "
    "statistics of hard-sphere random walks. The collisions are averaged "
    "into those two terms instead of being sampled one at a time, so no "
    "collision event exists to count. Use model='hs' (hard-sphere), which "
    "samples every collision, when a count is needed.")


def blank_collision_channel(traj, col_names):
    """Apply the rule to a recorded trajectory: the n_col column, if
    recorded, becomes NaN (in place). Returns traj."""
    if "n_col" in col_names:
        traj[:, list(col_names).index("n_col")] = np.nan
    return traj


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
# The note's "pressure that would reach the threshold" search doubles the
# deck pressure up to this many times (2**40 x P), then says no reachable
# pressure was found rather than guessing.
SDS_NEED_P_MAX_DOUBLINGS = 40
# dt guidance: the relaxation time 1/damping must be resolved; the app
# advises dt <= (1/damping) / SDS_DT_PER_DAMPING_TIME.
SDS_DT_PER_DAMPING_TIME = 10.0


def rf_regime_note(spec):
    """Warn when SDS is used with RF outside the regime it is valid in.

    Returns a multi-line warning string, or None when the operating
    point is fine or the check does not apply.

    THE RATIO IS gamma/Omega, NOT COLLISIONS PER CYCLE. gamma is the
    MOMENTUM-TRANSFER relaxation rate q/(m*K) from transport_params --
    the same number the integrator damps with. Omega = 2*pi*f. A collision
    is not a relaxation: a heavy ion needs roughly its mass ratio in
    light-gas collisions before it loses directed velocity. Mobility
    accounts for that; a collision count does not.

    WHY IT MATTERS. SDS relaxes the velocity toward the LOCAL
    MOBILITY-LIMITED DRIFT VELOCITY each step and then adds a diffusive
    jump; both halves assume that relaxation COMPLETES within a step, i.e.
    gamma >> Omega. RF confinement is the opposite condition: the
    pseudopotential is built from MICROMOTION, the driven oscillation out
    of phase with the field gradient, which survives only when
    gamma << Omega. The two cannot both hold. Where RF confinement is
    physical, SDS cannot represent it, and the ions go unconfined and time
    out while the same deck under hard-sphere transits normally.

    THIS WARNS, IT DOES NOT REFUSE. The boundary is soft, the ratio
    degrades gradually rather than failing at a threshold, and working near
    it is legitimate. See Allen and Bush, Anal. Chem. 88 (2016) -- RF
    confinement in ion mobility, apparent mobilities and effective
    temperatures.
    """
    from ion_gym.physics.collision3d import gas_mass as _gas_mass
    from ion_gym.physics.collision3d import resolve_collisions
    # the one gas authority: the deck's gas, temperature and pressure
    coll = resolve_collisions(getattr(spec, "collisions", None))
    if coll is None:
        return None
    if str(coll.model).lower() != "sds":
        return None
    rf = [g for g in (spec.geometry.rf_groups or [])
          if float(getattr(g, "amplitude_v", 0.0) or 0.0) != 0.0]
    if not rf:
        return None                       # no RF: nothing to erase
    f_hz = max(float(getattr(g, "frequency_hz", 0.0) or 0.0) for g in rf)
    p_torr = float(coll.P_torr)
    t_k = float(coll.T_k)
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
        mz_note = (f" (source.mz_list is unset, so a mass of {masses[0]:g} Da "
                   f"was ASSUMED -- declare it and this is exact)")

    gas_mass = _gas_mass(coll.gas)
    gas_diam = float(coll.gas_diam_nm)
    charge = int(spec.source.charge)
    omega = 2.0 * math.pi * f_hz

    rows = []
    for m in masses:
        pr = transport_params(m, charge, gas_mass, gas_diam, t_k, p_torr,
                              gas_name=coll.gas,
                              species=species_entry(coll.sds_species, m))
        gamma = pr["damping"] * 1.0e6                     # 1/us -> 1/s
        rows.append((m, gamma / omega, pr["D_mm2_per_us"], pr["K0"]))
    worst = min(r[1] for r in rows)
    if worst >= SDS_MIN_GAMMA_OVER_OMEGA:
        return None

    # pressure that would reach the threshold for the worst mass
    m_worst = [r[0] for r in rows if r[1] == worst][0]
    need_p = None
    _p = p_torr
    for _ in range(SDS_NEED_P_MAX_DOUBLINGS):
        _p *= 2.0
        _g = transport_params(m_worst, charge, gas_mass, gas_diam, t_k, _p,
                              gas_name=coll.gas,
                              species=species_entry(coll.sds_species,
                                                    m_worst))["damping"] * 1.0e6
        if _g / omega >= SDS_MIN_GAMMA_OVER_OMEGA:
            need_p = _p
            break

    lines = [
        f"[sds] WARNING: SDS with RF at gamma/Omega = {worst:.4f} -- "
        f"below the {SDS_MIN_GAMMA_OVER_OMEGA:g} this model needs."
        f"{mz_note}",
        f"[sds]   gamma is the momentum-transfer relaxation rate "
        f"q/(m*K) of the SDS mobility; Omega = 2*pi*"
        f"{f_hz / 1e6:.3f} MHz, at {p_torr:.4g} Torr / {t_k:.0f} K:",
    ]
    for m, ratio, D, K0 in rows:
        lines.append(f"[sds]     mass {m:<8g} Da  K0 {K0:.4g}  "
                     f"gamma/Omega {ratio:.4f}  D {D:.4g} mm^2/us")
    lines += [
        "[sds]   SDS assumes relaxation completes within a step "
        "(gamma >> Omega). RF confinement needs the opposite "
        "(gamma << Omega): the pseudopotential is made of MICROMOTION, "
        "which the damping erases. Expect ions to go UNCONFINED and "
        "time out while the same deck under model='hs' transits.",
        (f"[sds]   To reach gamma/Omega = "
         f"{SDS_MIN_GAMMA_OVER_OMEGA:g} at mass {m_worst:g} Da: pressure "
         f"about {need_p:.2f} Torr at this frequency"
         if need_p else
         f"[sds]   No reachable pressure below "
         f"2^{SDS_NEED_P_MAX_DOUBLINGS} x {p_torr:g} Torr "
         f"puts mass {m_worst:g} Da in regime at this frequency"),
        "[sds]   Otherwise use model='hs', which resolves collisions "
        "and keeps the micromotion.",
    ]
    return "\n".join(lines)
