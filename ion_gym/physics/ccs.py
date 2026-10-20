"""Per-ion hard-sphere collision cross sections: THE authority.

Every hard-sphere kernel takes one scalar `sigma_m2` per ion. Until
2026-10-17 every route read it from `collisions.sigma_m2`, a single value
for every ion in the run, so m/z 118 and m/z 5000 collided with the same
size. SIMION's collision_hs1.lua has the same design (`_sigma_m2` is one
adjustable for the run), and its default of 2.27e-18 m^2 is, by its own
comment, He with a roughly 200 amu ion. ion_gym inherited both.

`ion_sigma_m2(collisions, mz, charge)` is now the one place a kernel's
cross section comes from. `collisions.sigma_model` selects:

  "fixed"    `collisions.sigma_m2` for every ion. The default, so every
             existing deck flies and saves exactly as before.
  "ccs_fit"  a power law CCS = A * (m/z) ** B fitted to drift-tube CCS
             values in N2 (Picache et al., Chem. Sci. 2019, 10, 983-993,
             Unified CCS Compendium, singly charged ions). The fitted
             subset is `collisions.ccs_fit_class` ("all" by default, or a
             polarity or a chemical super class). CCS in A^2 is used as the
             hard-sphere cross section directly: for a hard sphere the
             momentum-transfer cross section equals the geometric one.

Stated limits of "ccs_fit" (PI rulings 2026-10-17):
  * N2 only. The compendium has no drift-gas column and is N2 data, so a
    fit requested for another gas is refused rather than misapplied.
  * Singly charged ions only, for the same reason (the fit is per ion at
    |z| = 1). Other charge states are refused.
  * Outside the fitted m/z range the power law is EXTRAPOLATED with a loud
    warning, never refused (Brian: "extrapolation beyond 2834 should occur
    but with a warning, not a refusal").
  * The values were measured near room temperature and are used unchanged
    at the deck's T_k. This is a reasonable estimate, not a
    temperature-corrected cross section.
  * The fit has a residual scatter (9.9 % for "all"), so any one ion's
    true CCS can differ from the line by that much. The scatter is carried
    in the table and reported with the per-ion values.

The coefficient table ships beside this module (ccs_fit_n2_z1.json). The
compendium itself is not redistributed. The table is regenerated from it
by internal/studies/funnel_opt/fit_ccs_compendium.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SIGMA_MODELS = ("fixed", "ccs_fit")
CCS_FIT_GAS = "N2"
CCS_FIT_CHARGE_ABS = 1
A2_TO_M2 = 1.0e-20
_TABLE_FILE = "ccs_fit_n2_z1.json"

# m/z values already warned about as extrapolated, per fit class. A
# repeat of the identical warning carries no information. The first
# print says so, so nothing is silently withheld.
_EXTRAPOLATION_WARNED = set()


class CCSModelRefused(ValueError):
    """A cross-section request the selected model cannot honour."""


@lru_cache(maxsize=1)
def load_fit_table():
    """The shipped coefficient table, keyed by fit label."""
    path = Path(__file__).with_name(_TABLE_FILE)
    table = json.loads(path.read_text())
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


def _check_fit_applies(collisions, charge):
    gas = str(collisions.gas)
    if gas != CCS_FIT_GAS:
        raise CCSModelRefused(
            f"collisions.sigma_model='ccs_fit' is fitted to drift-tube CCS "
            f"in {CCS_FIT_GAS}, but the deck's gas is {gas!r}. Use "
            f"sigma_model='fixed' with a sigma_m2 for that gas.")
    if abs(int(charge)) != CCS_FIT_CHARGE_ABS:
        raise CCSModelRefused(
            f"collisions.sigma_model='ccs_fit' is fitted to singly charged "
            f"ions, but the ion's charge is {int(charge)}. Use "
            f"sigma_model='fixed' with a sigma_m2 for that charge state.")


def _warn_extrapolation(mz, fit_class, row, ccs):
    key = (round(float(mz), 6), fit_class)
    if key in _EXTRAPOLATION_WARNED:
        return
    _EXTRAPOLATION_WARNED.add(key)
    print(f"[CCS] WARNING: m/z {mz:g} lies outside the fitted range "
          f"{row['mz_min']:.0f} to {row['mz_max']:.0f} of CCS fit class "
          f"{fit_class!r} (n = {row['n']}). The power law is EXTRAPOLATED "
          f"to CCS = {ccs:.1f} A^2. Fit scatter inside the range is "
          f"{100 * row['resid_sd_frac']:.1f} %, and nothing bounds the "
          f"error outside it. [printed once per m/z per session]",
          flush=True)


def ion_sigma_m2(collisions, mz, charge):
    """THE hard-sphere cross section (m^2) for one ion.

    collisions : CollisionSpec (or None for vacuum, returns 0.0, which
                 the kernels never read because collisions are off).
    mz         : the ion's m/z as every route already derives it.
    charge     : the ion's signed charge state.
    """
    if collisions is None or not getattr(collisions, "enabled", True):
        # vacuum: the kernels read no cross section (P = 0), and 0.0 is
        # what gas_kernel_scalars handed them before this module existed,
        # so vacuum flights stay bit-identical
        return 0.0
    model = str(getattr(collisions, "sigma_model", "fixed"))
    if model == "fixed":
        return float(collisions.sigma_m2)
    elif model == "ccs_fit":
        _check_fit_applies(collisions, charge)
        fit_class = str(collisions.ccs_fit_class)
        ccs, extrapolated, row = ccs_fit_A2(mz, fit_class)
        if extrapolated:
            _warn_extrapolation(mz, fit_class, row, ccs)
        return ccs * A2_TO_M2
    else:
        raise CCSModelRefused(
            f"collisions.sigma_model {model!r} is unknown. Known models: "
            f"{SIGMA_MODELS}")


def validate_sigma_model(collisions, charge):
    """Refusals for a deck that cannot mean what it asks (validate())."""
    out = []
    model = str(getattr(collisions, "sigma_model", "fixed"))
    if model not in SIGMA_MODELS:
        out.append(f"collisions.sigma_model {model!r} is unknown; known "
                   f"models are {SIGMA_MODELS}")
        return out
    if model == "fixed":
        # nothing model-specific to check: sigma_m2 is validated with
        # the other required collision fields
        return out
    try:
        _check_fit_applies(collisions, charge)
        _fit_row(str(collisions.ccs_fit_class))
    except CCSModelRefused as e:
        out.append(str(e))
    return out


def sigma_note(spec):
    """Advisory text listing the cross section each declared m/z flies
    with under 'ccs_fit', so the displayed value is the solver input.
    None when the model is 'fixed' or collisions are off."""
    col = spec.collisions
    if not getattr(col, "enabled", False):
        return None
    if str(getattr(col, "sigma_model", "fixed")) != "ccs_fit":
        return None
    fit_class = str(col.ccs_fit_class)
    row = _fit_row(fit_class)
    parts = []
    for mz in spec.source.mz_list:
        ccs, extrapolated, _row = ccs_fit_A2(mz, fit_class)
        flag = " EXTRAPOLATED" if extrapolated else ""
        parts.append(f"m/z {float(mz):g}: {ccs:.1f} A^2{flag}")
    return (f"collisions: hard-sphere cross sections from the {CCS_FIT_GAS} "
            f"CCS fit, class {fit_class!r} (CCS = {row['A_A2']:.3f} * "
            f"(m/z)^{row['B']:.4f}, fitted m/z {row['mz_min']:.0f} to "
            f"{row['mz_max']:.0f}, scatter "
            f"{100 * row['resid_sd_frac']:.1f} %, room-temperature data "
            f"used at {float(col.T_k):g} K): " + "; ".join(parts))


def low_field_mobility_m2Vs(collisions, mz, charge):
    """Low-field ion mobility K (m^2 / (V s)) from the Mason-Schamp
    relation, using the SAME cross section the hard-sphere kernel flies:

        K = (3 / 16) (q / N) sqrt(2 pi / (mu k T)) / Omega

    q = |z| e, N = P / (k T) the gas number density, mu the ion-gas reduced
    mass and Omega = ion_sigma_m2(collisions, mz, charge). It is the
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
    m_ion = float(mz) * abs(int(charge)) * KG_AMU
    m_gas = gas_mass(collisions.gas) * KG_AMU
    mu = m_ion * m_gas / (m_ion + m_gas)
    q = abs(int(charge)) * E_CHG
    sigma = ion_sigma_m2(collisions, mz, charge)
    return (3.0 / 16.0) * (q / n_gas) * math.sqrt(
        2.0 * math.pi / (mu * KB * T)) / sigma
