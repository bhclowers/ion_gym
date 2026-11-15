"""Per-ion hard-sphere collision cross sections: THE authority.

Every hard-sphere kernel takes one scalar `sigma_m2` per ion. Before
this module every route read it from `collisions.sigma_m2`, a single value
for every ion in the run, so m/z 118 and m/z 5000 collided with the same
size. SIMION's collision_hs1.lua has the same design (`_sigma_m2` is one
adjustable for the run), and its default of 2.27e-18 m^2 is, by its own
comment, He with a roughly 200 amu ion. ion_gym inherited both.

`ion_sigma_m2(collisions, mz, charge)` is now the one place a kernel's
cross section comes from. `collisions.sigma_model` selects:

  "fixed"    `collisions.sigma_m2` for every ion. A deck states this
             when it means one cross section for every ion.
  "ccs_fit"  THE DEFAULT (SIGMA_MODEL_DEFAULT): a
             deck that names no model flies this. A power law
             CCS = A * (m/z) ** B fitted to drift-tube CCS values in N2 (Picache et al., Chem. Sci. 2019, 10, 983-993,
             Unified CCS Compendium, singly charged ions). The fitted
             subset is `collisions.ccs_fit_class` ("all" by default, or a
             polarity or a chemical super class). CCS in A^2 is used as the
             hard-sphere cross section directly: for a hard sphere the
             momentum-transfer cross section equals the geometric one.

Stated limits of "ccs_fit":
  * N2 only. The compendium has no drift-gas column and is N2 data, so
    the fit is never applied to another gas. For another gas the
    hard-sphere flight FALLS BACK to the static cross section
    collisions.sigma_m2 for every ion, LOUDLY: a console warning once per
    case and a WARNING advisory (fallback_note) name the gas, the sigma
    flown and how to state it on purpose. (It was once refused.) Saved
    decks write sigma_model, so a default and a chosen "ccs_fit" cannot be
    told apart; the fallback applies to both.
  * The fit is of singly charged ions. For any charge the cross section is
    taken from the fit at the ion's MASS (mz_list holds mass; for z = 1
    mass and m/z agree), the same rule the SDS model uses, and the drift
    then scales with charge through the force. The growth of the cross
    section with charge (ion-induced dipole) and any charge-driven change
    of shape are neglected, so a multiply charged ion's mobility is an
    upper estimate. charge_note says so in the advisories. (abs(z) != 1
    once fell back to the static sigma_m2, and before that was refused.)
  * Outside the fitted m/z range the power law is EXTRAPOLATED with a loud
    warning, never refused (an ion beyond m/z 2834, the top of the "all"
    fit, still flies).
  * The values were measured near room temperature and are used unchanged
    at the deck's T_k. This is a reasonable estimate, not a
    temperature-corrected cross section.
  * The fit has a residual scatter (9.9 % for "all"), so any one ion's
    true CCS can differ from the line by that much. The scatter is carried
    in the table and reported with the per-ion values.

The coefficient table ships beside this module (ccs_fit_n2_z1.json). The
compendium itself is not redistributed. The table is regenerated from it
by a development-tree script that does not ship.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ion_gym.advisories import advise

SIGMA_MODELS = ("fixed", "ccs_fit")
# THE default sigma model. CollisionSpec's field default is set from this
# constant, as is every fallback in this module, so the two cannot
# disagree.
SIGMA_MODEL_DEFAULT = "ccs_fit"
# The fixed cross section ion_gym used before the CCS fit became the
# default: SIMION's collision_hs1.lua value (He with a ~200 amu ion). It is
# still CollisionSpec.sigma_m2's default, so a deck whose sigma_m2 differs
# from it stated its own value (see stated_sigma_note).
LEGACY_FIXED_SIGMA_M2 = 2.27e-18
CCS_FIT_GAS = "N2"
A2_TO_M2 = 1.0e-20
_TABLE_FILE = "ccs_fit_n2_z1.json"


class CCSModelRefused(ValueError):
    """A cross-section request the selected model cannot honour."""


@lru_cache(maxsize=1)
def load_fit_table():
    """The shipped coefficient table, keyed by fit label."""
    path = Path(__file__).with_name(_TABLE_FILE)
    table = json.loads(path.read_text(encoding="utf-8"))
    table["by_label"] = {row["label"]: row for row in table["fits"]}
    return table


def fit_classes():
    """Labels a deck may name as collisions.ccs_fit_class."""
    return tuple(load_fit_table()["by_label"])


def _fit_row(fit_class):
    rows = load_fit_table()["by_label"]
    if fit_class not in rows:
        raise CCSModelRefused(
            f"collisions.ccs_fit_class {fit_class!r} is not in the CCS fit "
            f"table. Known classes: {sorted(rows)}")
    return rows[fit_class]


def ccs_fit_A2(mz, fit_class="all"):
    """(CCS in A^2, extrapolated flag, fit row) for one m/z."""
    row = _fit_row(fit_class)
    mz = float(mz)
    if mz <= 0.0:
        raise CCSModelRefused(f"m/z must be positive, got {mz}")
    ccs = row["A_A2"] * mz ** row["B"]
    extrapolated = not (row["mz_min"] <= mz <= row["mz_max"])
    return ccs, extrapolated, row


def fit_domain_reason(collisions, charge):
    """Why the CCS fit does not apply to this gas, or None when it does.
    The fit is N2 data. Charge is not a limit: the fit is evaluated at the
    ion's mass for every charge (charge_note states the approximation)."""
    gas = str(collisions.gas)
    if gas != CCS_FIT_GAS:
        return (f"the CCS fit is drift-tube data in {CCS_FIT_GAS}, and the "
                f"deck's gas is {gas!r}")
    return None


def fallback_text(reason, sigma_m2):
    """The one wording of the static-sigma fallback (console and advisory)."""
    legacy = (" This is the legacy default (SIMION's He value for a ~200 amu "
              "ion), not a value the deck chose."
              if float(sigma_m2) == LEGACY_FIXED_SIGMA_M2 else "")
    return (f"collisions.sigma_model 'ccs_fit' does not apply: {reason}. "
            f"The hard-sphere flight falls back to the static cross section "
            f"collisions.sigma_m2 = {float(sigma_m2):.4g} m^2 for every "
            f"ion.{legacy} State sigma_model 'fixed' (with a sigma_m2 for "
            f"this gas and charge) to fly a static cross section on "
            f"purpose.")


# Both warnings are IonGymAdvisory: the warning filter governs repeats
# (by default, each distinct text once), and a caller can collect,
# silence or escalate them. The text carries the case (reason and sigma;
# m/z and fit class), so a different case is stated anew.
def _warn_fallback(reason, sigma_m2):
    advise(f"[CCS] {fallback_text(reason, sigma_m2)}", stacklevel=2)


def _warn_extrapolation(mz, fit_class, row, ccs):
    advise(f"[CCS] m/z {mz:g} lies outside the fitted range "
           f"{row['mz_min']:.0f} to {row['mz_max']:.0f} of CCS fit class "
           f"{fit_class!r} (n = {row['n']}). The power law is EXTRAPOLATED "
           f"to CCS = {ccs:.1f} A^2. Fit scatter inside the range is "
           f"{100 * row['resid_sd_frac']:.1f} %, and nothing bounds the "
           f"error outside it.", stacklevel=2)


def ion_sigma_m2(collisions, mz, charge):
    """THE hard-sphere cross section (m^2) for one ion.

    collisions : CollisionSpec (or None for vacuum, returns 0.0, which
                 the kernels never read because collisions are off).
    mz         : the ion's MASS in Da (source.mz_list). The fit, made of
                 singly charged ions, is evaluated at this mass for every
                 charge (see the module notes and charge_note).
    charge     : the ion's signed charge state.
    """
    if collisions is None or not getattr(collisions, "enabled", True):
        # vacuum: the kernels read no cross section (P = 0), and 0.0 is
        # what gas_kernel_scalars handed them before this module existed,
        # so vacuum flights stay bit-identical
        return 0.0
    model = str(getattr(collisions, "sigma_model", SIGMA_MODEL_DEFAULT))
    if model == "fixed":
        return float(collisions.sigma_m2)
    elif model == "ccs_fit":
        reason = fit_domain_reason(collisions, charge)
        if reason is not None:
            # warned only where a hard-sphere sigma is flown: the planar
            # and r-z builders ask for it under SDS too, which never reads
            # it (SDS in He once refused there, at flight, and not on 3-D)
            if _hs_sigma_in_play(collisions):
                _warn_fallback(reason, collisions.sigma_m2)
            return float(collisions.sigma_m2)
        fit_class = str(collisions.ccs_fit_class)
        ccs, extrapolated, row = ccs_fit_A2(mz, fit_class)
        if extrapolated:
            _warn_extrapolation(mz, fit_class, row, ccs)
        return ccs * A2_TO_M2
    else:
        raise CCSModelRefused(
            f"collisions.sigma_model {model!r} is unknown. Known models: "
            f"{SIGMA_MODELS}")


def _hs_sigma_in_play(collisions):
    """True when a hard-sphere cross section is read by the flight: gas on
    and a collision model other than SDS. SDS reads gas_diam_nm, never a
    sigma, so the sigma model's checks and notes do not apply to it (the
    CCS default is scoped to the HS model). An unknown
    collisions.model is not judged here: validate() refuses it by name.
    """
    if collisions is None or not getattr(collisions, "enabled", False):
        return False
    return str(getattr(collisions, "model", "hs") or "").lower() != "sds"


def validate_sigma_model(collisions, charge):
    """Refusals for a deck that cannot mean what it asks (validate()).
    An unknown sigma_model name is refused whatever the collision model.
    The fit class is checked only when a hard-sphere sigma is in play
    (_hs_sigma_in_play): an SDS or vacuum deck is not refused over a
    cross section its flight never reads. A gas or charge outside the
    fit is not a refusal: the flight falls back (fallback_note)."""
    out = []
    model = str(getattr(collisions, "sigma_model", SIGMA_MODEL_DEFAULT))
    if model not in SIGMA_MODELS:
        out.append(f"collisions.sigma_model {model!r} is unknown. Known "
                   f"models are {SIGMA_MODELS}")
        return out
    if not _hs_sigma_in_play(collisions):
        return out
    if model == "fixed":
        # nothing model-specific to check: sigma_m2 is validated with
        # the other required collision fields
        return out
    # Outside the fit's gas and charge the flight falls back to sigma_m2
    # (fallback_note says so); a misspelled fit class is still a refusal.
    try:
        _fit_row(str(collisions.ccs_fit_class))
    except CCSModelRefused as e:
        out.append(str(e))
    return out


def _fallback_reason(spec):
    """fit_domain_reason for a spec that flies the CCS fit under HS, else
    None (no hard-sphere sigma in play, or a 'fixed' model)."""
    col = spec.collisions
    if not _hs_sigma_in_play(col):
        return None
    if str(getattr(col, "sigma_model", SIGMA_MODEL_DEFAULT)) != "ccs_fit":
        return None
    return fit_domain_reason(col, spec.source.charge)


def fallback_note(spec):
    """A WARNING advisory when the CCS fit does not apply and the flight
    falls back to the static sigma_m2, or None."""
    reason = _fallback_reason(spec)
    if reason is None:
        return None
    return "WARNING: " + fallback_text(reason, spec.collisions.sigma_m2)


def charge_note(spec):
    """The approximation a multiply charged ion flies under the CCS fit,
    as an advisory, or None (abs(z) = 1, no fit in play, or a fallback)."""
    z = int(spec.source.charge)
    if abs(z) == 1 or _fit_rows(spec) is None:
        return None
    return (f"collisions: the CCS fit is of singly charged ions. At charge "
            f"{z:+d} each ion's cross section is the fit at its MASS, and "
            f"its drift scales with charge through the force. The growth "
            f"of the cross section with charge (ion-induced dipole) and any "
            f"charge-driven change of shape are neglected, so the mobility "
            f"is an upper estimate. State sigma_model 'fixed' with a "
            f"cross section measured for this charge state to fly that "
            f"instead.")


def _fit_rows(spec):
    """(fit_class, fit row, [(mz, CCS in A^2, extrapolated), ...]) for
    every declared m/z when the spec flies a hard-sphere sigma from the
    CCS fit, else None. The one builder sigma_note and extrapolation_note
    share."""
    col = spec.collisions
    if not _hs_sigma_in_play(col):
        return None
    if str(getattr(col, "sigma_model", SIGMA_MODEL_DEFAULT)) != "ccs_fit":
        return None
    if _fallback_reason(spec) is not None:
        return None          # the fit is not flown; fallback_note says so
    fit_class = str(col.ccs_fit_class)
    row = _fit_row(fit_class)
    vals = []
    for mz in spec.source.mz_list:
        ccs, extrapolated, _row = ccs_fit_A2(mz, fit_class)
        vals.append((float(mz), ccs, extrapolated))
    return fit_class, row, vals


def sigma_note(spec):
    """Advisory text listing the cross section each declared m/z flies
    with under 'ccs_fit', so the displayed value is the solver input.
    None when no hard-sphere sigma is in play or the model is 'fixed'."""
    fit = _fit_rows(spec)
    if fit is None:
        return None
    fit_class, row, vals = fit
    parts = [f"m/z {mz:g}: {ccs:.1f} A^2" + (" EXTRAPOLATED" if ex else "")
             for mz, ccs, ex in vals]
    return (f"collisions: hard-sphere cross sections from the {CCS_FIT_GAS} "
            f"CCS fit, class {fit_class!r} (CCS = {row['A_A2']:.3f} * "
            f"(m/z)^{row['B']:.4f}, fitted m/z {row['mz_min']:.0f} to "
            f"{row['mz_max']:.0f}, scatter "
            f"{100 * row['resid_sd_frac']:.1f} %, room-temperature data "
            f"used at {float(spec.collisions.T_k):g} K): "
            + ", ".join(parts))


def extrapolation_note(spec):
    """A WARNING advisory naming every declared m/z the CCS fit
    extrapolates to, or None. It is separate from sigma_note (which lists
    every value) so the warning is not buried in that list. Advisories
    reach the console and the app's advisory line.
    Extrapolation warns and never refuses.
    """
    fit = _fit_rows(spec)
    if fit is None:
        return None
    fit_class, row, vals = fit
    out = [f"m/z {mz:g} ({ccs:.1f} A^2)" for mz, ccs, ex in vals if ex]
    if not out:
        return None
    return (f"WARNING: the CCS fit (class {fit_class!r}) is EXTRAPOLATED "
            f"for {', '.join(out)}, outside the fitted m/z range "
            f"{row['mz_min']:.0f} to {row['mz_max']:.0f}. Nothing bounds "
            f"the error there.")


def stated_sigma_note(spec):
    """A WARNING advisory when a deck states its own sigma_m2 but flies
    the CCS fit, or None (warn, not refuse).

    A value equal to the legacy default (LEGACY_FIXED_SIGMA_M2) cannot
    be told apart from one nobody chose (decks saved before saves left
    it out carry it), and is not flagged. Any other value was stated on
    purpose, and under the CCS-fit default it is not what the flight
    uses, so the advisory names it and says how to fly it instead.
    """
    col = spec.collisions
    if not _hs_sigma_in_play(col):
        return None
    if str(getattr(col, "sigma_model", SIGMA_MODEL_DEFAULT)) != "ccs_fit":
        return None
    if _fallback_reason(spec) is not None:
        return None          # the stated sigma IS flown (fallback_note)
    sig = float(col.sigma_m2)
    if sig == LEGACY_FIXED_SIGMA_M2:
        return None
    return (f"WARNING: collisions.sigma_m2 = {sig:.4g} m^2 is stated but "
            f"unused. The deck flies the per-ion CCS fit "
            f"(sigma_model 'ccs_fit'). State sigma_model 'fixed' to fly "
            f"{sig:.4g} m^2 for every ion.")


def low_field_mobility_m2Vs(collisions, mass_amu, charge):
    """Low-field ion mobility K (m^2 / (V s)) from the Mason-Schamp
    relation, using the SAME cross section the hard-sphere kernel flies:

        K = (3 / 16) (q / N) sqrt(2 pi / (mu k T)) / Omega

    q = |z| e, N = P / (k T) the gas number density, mu the ion-gas reduced
    mass and Omega = ion_sigma_m2(collisions, mass_amu, charge). mass_amu
    is the ion's MASS in Da for every charge (what mz_list holds and every
    route flies), so K scales with |z| through q alone. It is the
    reference a flown drift time is checked against (a hard-sphere ion at
    sigma = Omega should reproduce it in the low-field limit), and the
    basis for sizing a flight-time limit. First-order Mason-Schamp only:
    no high-field or temperature correction.
    """
    import math
    from ion_gym.physics.collision3d import KB, KG_AMU, E_CHG, gas_mass
    if collisions is None or not getattr(collisions, "enabled", True):
        raise CCSModelRefused(
            "a mobility needs gas: collisions is None or disabled")
    T = float(collisions.T_k)
    n_gas = float(collisions.P_pa) / (KB * T)
    # the mass as given: it used to be multiplied by abs(z) (an m/z
    # reading), so the reference mobility of a multiply charged ion used a
    # heavier reduced mass than the ion the kernels fly (review C-11)
    m_ion = float(mass_amu) * KG_AMU
    m_gas = gas_mass(collisions.gas) * KG_AMU
    mu = m_ion * m_gas / (m_ion + m_gas)
    q = abs(int(charge)) * E_CHG
    sigma = ion_sigma_m2(collisions, mass_amu, charge)
    return (3.0 / 16.0) * (q / n_gas) * math.sqrt(
        2.0 * math.pi / (mu * KB * T)) / sigma
