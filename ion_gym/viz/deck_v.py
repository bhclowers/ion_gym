"""
deck_v.py  --  W3d: the FULL-DIMENSIONAL deck
=============================================
The certified deck is finite in (s, u) but a POINT AT REST in v -- and v is
where Kv lives: the 22-34 eV of retained beam energy that walks the packet
sideways to the detector and decides which trim-strip territory each ion
crosses.  A v = 0 deck cannot express the physics the trim strips exist to
null, so Kv is treated as a RANGE like
T_s; a point source could be misleading).

WHAT THIS MODULE ADDS, and its license to add it kinematically:
  * v0  : uniform over the extracted FOOTPRINT.  The slit slot is 16 mm
          long (the narrowest along the column -- it gates what is
          extracted), so the footprint is +/-8 mm about the launch line
          unless overridden.
  * vv  : sqrt(2*ACC*Kv/m), with Kv drawn per-ion from the window value
          plus a Gaussian dKv (~1-2 eV gas-dynamic; default
          sigma 0.64 eV ~ 1.5 eV FWHM).
  * propagation: BALLISTIC.  Gate B certified the length-plane axis forces
          are ~zero in the beam footprint (wall screening, the 3900x
          exhibit); stack, drift and mirror all act in (s, u).  So
          v(t) = v0 + vv*t exactly -- no new solve exists to be wrong.
NOTHING in the (s, u) dynamics changes: the width-plane fly is reused as-is
and v rides on top.  That separation is the certified composition rule;
test_W3d gates its consequences rather than assuming them.
"""
from __future__ import annotations

import numpy as np

ACC = 96.485
KV_WINDOW = (22.0, 34.0)          # eV, the certified bounded parameter
DKV_SIGMA_EV = 0.64               # ~1.5 eV FWHM gas-dynamic spread
V_FOOT_MM = 16.0                  # slit slot length: the extraction gate


def vv_of(Kv_eV, mz):
    return np.sqrt(2.0 * ACC * np.maximum(np.asarray(Kv_eV, float), 0.0)
                   / mz)


def extend_deck(deck_su, mz, Kv_eV, *, rng, v_center=0.0,
                v_foot_mm=V_FOOT_MM, dKv_sigma_eV=DKV_SIGMA_EV):
    """(N,5) width-plane deck [s,u,vs,vu,tob] -> dict with the v columns.

    Kv is drawn per-ion: Kv_i = Kv_eV + N(0, dKv_sigma).  Positions v0 are
    uniform over the footprint about v_center (the launch line; the caller
    owns the chamber frame)."""
    deck_su = np.asarray(deck_su, float)
    n = len(deck_su)
    Kv = Kv_eV + rng.normal(0.0, dKv_sigma_eV, n)
    v0 = v_center + v_foot_mm * (rng.random(n) - 0.5)
    return dict(su=deck_su, v0=v0, Kv=Kv, vv=vv_of(Kv, mz))


def v_at(deckv, t_us):
    """Ballistic v at time t (per-ion arrays or scalar t)."""
    return deckv["v0"] + deckv["vv"] * np.asarray(t_us, float)


def detector_walk(Kv_eV, Kx_eV, L_path_mm):
    """The design identity dZ = sqrt(Kv/Kx) * L."""
    return np.sqrt(np.asarray(Kv_eV, float) / Kx_eV) * L_path_mm
