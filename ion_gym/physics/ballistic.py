"""
ballistic.py — field-free (ballistic) motion along one axis, added to a
deck flown in the other two.

When the forces along an axis are negligible (a field-free drift, or an
axis the electrodes do not act on), motion along it is exactly ballistic:
x(t) = x0 + v * t, with v set by the kinetic energy carried on that axis.
That lets a 2-D (two-axis) flight be extended to three dimensions without
a new solve: the third axis rides on top, and nothing in the two flown
axes changes.

GENERIC by design (D6b, 2026-10-16; PI: "That is basic physics. To the
degree possible code should be agnostic"). This was viz/deck_v.py, an
oa-TOF module carrying that instrument's numbers as defaults (a 22-34 eV
energy window, a 16 mm slit footprint, a 0.64 eV spread) and a retyped
96.485. Every value is now a REQUIRED argument -- an instrument's numbers
live with that instrument's study -- and the unit constant is
physics.constants.ACC.

Units: positions mm, time us, energy eV, mass Da (m/z at charge 1),
speed mm/us.
"""
from __future__ import annotations

import numpy as np

from ion_gym.physics.constants import ACC


def speed_from_energy(energy_ev, mz):
    """Speed (mm/us) carried by kinetic energy energy_ev (eV) on one axis,
    for an ion of mass mz (Da): v = sqrt(2 * ACC * K / m). A negative
    energy is REFUSED (the old helper clipped it to zero silently)."""
    k = np.asarray(energy_ev, float)
    if np.any(k < 0):
        raise ValueError(f"speed_from_energy: negative energy "
                         f"{float(k.min()):g} eV")
    return np.sqrt(2.0 * ACC * k / float(mz))


def add_ballistic_axis(deck, mz, energy_ev, *, rng, center_mm,
                       footprint_mm, energy_sigma_ev):
    """Extend an (N, k) deck flown in two axes with a ballistic third.

    Per ion: energy_i = energy_ev + N(0, energy_sigma_ev); start position
    uniform over footprint_mm centred on center_mm. Returns
    dict(deck, pos0_mm, energy_ev, speed_mm_us). Draw order (energies,
    then positions) is fixed, so a seeded rng reproduces a deck."""
    deck = np.asarray(deck, float)
    n = len(deck)
    energy = energy_ev + rng.normal(0.0, energy_sigma_ev, n)
    pos0 = center_mm + footprint_mm * (rng.random(n) - 0.5)
    return dict(deck=deck, pos0_mm=pos0, energy_ev=energy,
                speed_mm_us=speed_from_energy(energy, mz))


def position_at(extended, t_us):
    """Ballistic position (mm) at time t_us (per-ion array or scalar)."""
    return (extended["pos0_mm"]
            + extended["speed_mm_us"] * np.asarray(t_us, float))


def transverse_walk(energy_transverse_ev, energy_axial_ev, path_mm):
    """Sideways displacement over a flight of length path_mm, for one ion
    carrying energy_transverse_ev across and energy_axial_ev along the
    path: dZ = sqrt(Kt / Ka) * L (the mass cancels)."""
    return (np.sqrt(np.asarray(energy_transverse_ev, float)
                    / energy_axial_ev) * path_mm)
