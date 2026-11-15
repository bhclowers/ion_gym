"""Regenerate the SDS diffusion-jump statistics from the published procedure.

ion_gym's SDS model draws each diffusion jump's length from tabulated
statistics of hard-sphere random walks (ion_gym/physics/sds_jump_icdf.dat).
This tool produces that table: an independent Monte Carlo of the procedure
published in

  A. D. Appelhans, D. A. Dahl, Int. J. Mass Spectrom. 244 (2005) 1-14,
  doi:10.1016/j.ijms.2005.03.010, Section 2.5 (the "random" collection).

THE PROCEDURE
-------------
* Each ion starts with a Maxwell-Boltzmann velocity (mean speed 1) in a
  random direction.
* Free paths are Exp(1): mean free path 1 (see the step model below).
* Each collision is an elastic hard-sphere impact on a gas molecule drawn
  from a Maxwell-Boltzmann distribution at the ion's temperature, so
  v_avg_gas / v_avg_ion = sqrt(m_ion / m_gas).
* After N_COLLISIONS collisions the ion's distance r from its start is
  recorded, for N_IONS ions at each mass ratio in MASS_RATIOS.
* The table is the inverse CDF of r in 0.1 % probability steps: 1002
  points per curve (minimum, 0.1 % .. 99.9 %, maximum).

TWO CHOICES THE PAPER LEAVES OPEN (both implemented)
----------------------------------------------------
* Step model: the paper's text randomizes each TIME step (segment v t,
  t ~ Exp(1)); a distance reading (segment length ~ Exp(1) along the
  velocity) matches "mean free path normalized to one". --step-model
  selects; distance is the default. The time model inflates r by the
  speed-time variance (about +23 % at mass ratio 1).
* Gas partner draw: plain Maxwell-Boltzmann, or weighted by relative
  speed (the collision flux an ion actually meets). --flux-weighted
  selects the latter.

THE SHIPPED TABLE was made with the flux-weighted partner draw, the
distance step, seed 20260803 and the full N (its header states this):

  python tools/regen_sds_tables.py --flux-weighted

Usage:
  python tools/regen_sds_tables.py --probe                # timing only
  python tools/regen_sds_tables.py --n-ions 20000         # reduced N
  python tools/regen_sds_tables.py --flux-weighted        # shipped recipe

Writes <out-dir>/sds_jump_icdf<tag>.dat and the per-ion radii
rjump_raw<tag>.npy (for convergence checks). It never writes into
ion_gym/physics unless --out-dir points there. A full run takes hours;
--probe projects the cost on this machine first.
"""
import argparse
import math
import pathlib
import time

import numpy as np
from numba import njit, prange

N_IONS = 100_000            # per mass ratio
N_COLLISIONS = 100_000      # per ion
MASS_RATIOS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
N_POINTS = 1002             # ICDF points (0.1% steps + endpoints)
SHIPPED_SEED = 20260803     # the seed the shipped table was made with

# Both rejection samplers below accept with probability of order one
# (Marsaglia pi/4; the flux weighting at least ~1/7), so this many tries
# without an acceptance means the inputs are broken, not unlucky.
MAX_REJECTION_TRIES = 10_000

# ---------------------------------------------------------------- MC core


@njit(cache=True, fastmath=False, inline="always")
def _mb_velocity(sig1d):
    """Maxwell-Boltzmann velocity vector: three Gaussians at the 1-D
    sigma. For the ion's own initialization the paper normalizes the
    AVERAGE SPEED to 1: <|v|> = sig1d*sqrt(8/pi) -> sig1d = sqrt(pi/8)."""
    return (np.random.normal() * sig1d, np.random.normal() * sig1d,
            np.random.normal() * sig1d)


@njit(cache=True, fastmath=False, inline="always")
def _hs_scatter(vx, vy, vz, gx, gy, gz, m_ratio):
    """One elastic hard-sphere collision, ion mass = m_ratio * gas
    mass. Uses the EXACT equivalence: uniform-impact-parameter
    hard-sphere scattering is ISOTROPIC in the center-of-mass frame
    (standard kinetic-theory theorem), so the outgoing relative
    velocity is |v_rel| times a uniform random direction -- the same
    physics as the runtime kernel's asin(sqrt(U)) construction, at a
    fraction of the cost. Marsaglia sphere sampling.

    Returns (vx, vy, vz, ok); ok is False only if the bounded sampler
    found no point (the caller reports it -- a raise here would stop the
    caller's prange loop from running in parallel)."""
    rx, ry, rz = vx - gx, vy - gy, vz - gz
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < 1e-14:
        return vx, vy, vz, True
    # uniform direction on the sphere (Marsaglia), bounded
    a = 0.0
    b = 0.0
    ss = 2.0
    for _try in range(MAX_REJECTION_TRIES):
        a = 2.0 * np.random.random() - 1.0
        b = 2.0 * np.random.random() - 1.0
        ss = a * a + b * b
        if ss < 1.0:
            break
    if ss >= 1.0:
        return vx, vy, vz, False
    root = 2.0 * math.sqrt(1.0 - ss)
    nx = a * root
    ny = b * root
    nz = 1.0 - 2.0 * ss
    # COM frame: v_ion' = v_com + (m_gas/(m_ion+m_gas)) |v_rel| n_hat
    w = 1.0 / (m_ratio + 1.0)                # m_gas/(m_ion+m_gas)
    cx = vx - w * rx                          # v_com components:
    cy = vy - w * ry                          # v_ion - mu/m_ion*v_rel
    cz = vz - w * rz                          # == v_com (algebra)
    return (cx + w * rlen * nx, cy + w * rlen * ny, cz + w * rlen * nz,
            True)


@njit(cache=True, parallel=True)
def _walk_rjump_kernel(n_ions, n_coll, m_ratio, flux_weighted, seed,
                       distance_step):
    """The 'random' collection: |r| after n_coll collisions for n_ions
    ions, and a per-ion flag set when a bounded rejection sampler gave
    up (that ion's r is NaN). Units: ion mean speed = 1, mean free path
    = 1. distance_step=True: segment length l ~ Exp(1) along the current
    velocity direction. distance_step=False (the paper's wording):
    segment = v * t, t ~ Exp(1). No raise in here: an exit from the
    prange loop would make numba run it serially."""
    out = np.empty(n_ions)
    failed = np.zeros(n_ions, np.uint8)
    sig_ion = math.sqrt(math.pi / 8.0)           # <|v|>=1
    sig_gas = sig_ion * math.sqrt(m_ratio)       # equal T: vbar_gas =
    #                                              vbar_ion*sqrt(mi/mg)
    for i in prange(n_ions):
        np.random.seed(seed + i)
        vx, vy, vz = _mb_velocity(sig_ion)
        x = y = z = 0.0
        for _c in range(n_coll):
            t = -math.log(1.0 - np.random.random())   # Exp(1)
            if distance_step:
                sp = math.sqrt(vx * vx + vy * vy + vz * vz)
                if sp < 1e-14:
                    sp = 1e-14
                x += vx / sp * t
                y += vy / sp * t
                z += vz / sp * t
            else:
                x += vx * t
                y += vy * t
                z += vz * t
            if flux_weighted:
                # rejection: accept partner ~ relative speed, bounded
                scale = (math.sqrt(vx * vx + vy * vy + vz * vz)
                         + 5.196152422706632 * sig_gas)
                gx = gy = gz = 0.0
                accepted = False
                for _try in range(MAX_REJECTION_TRIES):
                    gx = np.random.normal() * sig_gas
                    gy = np.random.normal() * sig_gas
                    gz = np.random.normal() * sig_gas
                    rx, ry, rz = vx - gx, vy - gy, vz - gz
                    rl = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if np.random.random() < rl / scale:
                        accepted = True
                        break
                if not accepted:
                    failed[i] = 1
                    break
            else:
                gx = np.random.normal() * sig_gas
                gy = np.random.normal() * sig_gas
                gz = np.random.normal() * sig_gas
            vx, vy, vz, ok = _hs_scatter(vx, vy, vz, gx, gy, gz, m_ratio)
            if not ok:
                failed[i] = 2
                break
        if failed[i]:
            out[i] = np.nan
        else:
            out[i] = math.sqrt(x * x + y * y + z * z)
    return out, failed


def _walk_rjump(n_ions, n_coll, m_ratio, flux_weighted, seed,
                distance_step=True):
    """|r| after n_coll collisions for n_ions ions (see the kernel).
    Raises if any ion's bounded rejection sampler gave up."""
    out, failed = _walk_rjump_kernel(n_ions, n_coll, m_ratio,
                                     flux_weighted, seed, distance_step)
    if failed.any():
        n_partner = int(np.sum(failed == 1))
        n_sphere = int(np.sum(failed == 2))
        raise RuntimeError(
            f"{n_partner + n_sphere} of {n_ions} ions stopped at mass "
            f"ratio {m_ratio:g} (seed {seed}): the bounded rejection "
            f"samplers ({MAX_REJECTION_TRIES} tries) gave up -- "
            f"{n_partner} flux-weighted partner draws, {n_sphere} "
            f"sphere-direction draws. Acceptance is of order one, so "
            f"this means broken inputs or a broken random generator.")
    return out


def icdf_1002(r):
    """The shipped table format: 1002 monotone quantile points of r,
    0.1% steps with both endpoints (min, 0.1%..99.9%, max)."""
    r = np.sort(np.asarray(r))
    qs = np.concatenate([[0.0], np.arange(1, 1001) / 1000.0, [1.0]])
    return np.quantile(r, qs)


def write_jump_table(out_path, curves, n_ions, n_coll, variant, seed):
    with open(out_path, "w") as f:
        f.write(
            "; sds_jump_icdf.dat — SDS diffusion-jump inverse-CDF statistics\n"
            "; REGENERATED by tools/regen_sds_tables.py: this file is an\n"
            "; ion_gym artifact — an independent Monte Carlo of the "
            "PUBLISHED procedure of\n"
            "; Appelhans & Dahl, Int. J. Mass Spectrom. 244 (2005) "
            "1-14, Sec. 2.5 ('random'\n"
            "; collection): MB-initialized ions (mean speed 1), "
            "Exp(1) free paths (MFP=1),\n"
            "; elastic hard-sphere collisions with equal-temperature "
            "MB gas partners\n"
            f"; (partner draw: {variant}); r from origin recorded "
            f"after {n_coll} collisions\n"
            f"; for {n_ions} ions per curve (seeds {seed}).\n"
            "; Curves: mass_ion/mass_gas = 1, 10, 100, 1000, 10000; "
            "each entry is a 0.1%\n"
            "; probability step (1002 points: min, 0.1%..99.9%, "
            "max).\n")
        for k, (mr, q) in enumerate(zip(MASS_RATIOS, curves), start=1):
            f.write(f"; MBMR{k}: mass_ion/mass_gas = {mr:g}\n")
            for v in q:
                f.write(f"{v:.6g}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Regenerate ion_gym's SDS jump statistics "
                    "(Appelhans & Dahl 2005, Sec. 2.5).")
    ap.add_argument("--probe", action="store_true",
                    help="time a short run and project the full cost")
    ap.add_argument("--n-ions", type=int, default=N_IONS)
    ap.add_argument("--n-collisions", type=int, default=N_COLLISIONS,
                    help="collisions per ion (the published procedure "
                    "uses 100000; fewer is for testing only)")
    ap.add_argument("--flux-weighted", action="store_true",
                    help="collision-flux-weighted partner draw, as the "
                    "shipped table (default: plain MB)")
    ap.add_argument("--step-model", choices=["distance", "time"],
                    default="distance",
                    help="what Exp(1) randomizes: DISTANCE (default) or "
                    "TIME per the paper's wording (about +23%% high at "
                    "mass ratio 1)")
    ap.add_argument("--seed", type=int, default=SHIPPED_SEED)
    ap.add_argument("--out-dir", default="sds_regen_out",
                    help="staging dir (default sds_regen_out/; the "
                    "tool never overwrites ion_gym/physics unless you "
                    "point it there deliberately)")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, so several runs "
                    "can coexist")
    args = ap.parse_args(argv)
    if args.n_ions < 2 or args.n_collisions < 1:
        ap.error(f"--n-ions must be >= 2 and --n-collisions >= 1 "
                 f"(got {args.n_ions}, {args.n_collisions})")
    distance = args.step_model == "distance"

    if args.probe:
        t0 = time.time()
        _walk_rjump(64, 2000, 100.0, args.flux_weighted, args.seed,
                    distance)
        _ = time.time() - t0            # jit compile
        t0 = time.time()
        _walk_rjump(512, 20_000, 100.0, args.flux_weighted, args.seed,
                    distance)
        dt = time.time() - t0
        per_evt = dt / (512 * 20_000)
        full = (per_evt * args.n_ions * args.n_collisions
                * len(MASS_RATIOS))
        print(f"probe: {per_evt*1e9:.1f} ns/collision "
              f"-> full run ({args.n_ions} ions x {args.n_collisions} x "
              f"{len(MASS_RATIOS)} ratios) ~ {full/60:.1f} min")
        return 0

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variant = (("flux-weighted MB" if args.flux_weighted
                else "plain MB")
               + f", {args.step_model}-step Exp(1)")
    curves = []
    raws = []
    for k, mr in enumerate(MASS_RATIOS):
        t0 = time.time()
        r = _walk_rjump(args.n_ions, args.n_collisions, mr,
                        args.flux_weighted, args.seed + k * 1_000_000,
                        distance)
        curves.append(icdf_1002(r))
        raws.append(r)
        print(f"  ratio {mr:>7g}: median r = {np.median(r):9.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    tag = f"_{args.tag}" if args.tag else ""
    write_jump_table(out / f"sds_jump_icdf{tag}.dat", curves, args.n_ions,
                     args.n_collisions, variant, args.seed)
    # per-ion raw radii for convergence checks (split-sample scatter
    # needs the sample, not just the quantiles)
    np.save(out / f"rjump_raw{tag}.npy", np.vstack(raws))
    print(f"wrote {out}/sds_jump_icdf{tag}.dat ({len(curves)} curves) and "
          f"rjump_raw{tag}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
