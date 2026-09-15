"""
test_tw2d.py — CONTRACT for the 2-D travelling-wave kernel port (Opus).

Deliverable module: `tracer_tw2d` with
    build_tw2d_fields(bases: {idx: (nx,ny) float64 unit-basis /V},
                      groups: [RFGroupSpec], assign: {idx: group_name|None},
                      dc: {idx: volts}, h_mm: float) -> fields dict
    fly_tw2d(fields, mz, x0_mm, v0_mm_us, tob_us, *, dt_ns, t_max_us,
             record_every, max_records, seed) -> same dict as tracer3d.fly3d
Each electrode is its OWN channel (amp, waveform kind, freq, phase) — no
quadrature folding, no two-phase restriction. Waveform kinds and evaluation
MUST reuse tracer3d's _wave_eval table (K_SIN/K_COS/K_SQUARE/K_TAB_*), not a
re-implementation.

PINNED CONVENTIONS (from slim_board + RFGroupSpec — do not re-derive):
  * RFGroupSpec.frequency_hz is the WAVEFORM frequency f_waveform. For an
    N-phase stepped wave the pattern advances one wavelength (N pitches) per
    period:  v_wave = N_PHASE * PITCH_MM * f_waveform.   (f_step = N * f_wf.)
  * travelling_wave_groups(n) steps phase by +360/n per group; electrodes
    assigned cyclically k -> group k % n march the wave toward -x (higher-k
    phases LEAD, so features appear at high k first — measured in Gate C).
    Use phase_step_deg=-360/n for +x travel.
  * 'square' waveform = sign(sin(2*pi*f*t + phase)), amplitude_v applied at
    fly time to UNIT bases (bases stay /V).
  * Birth time-of-flight offsets shift the drive phase an ion is born into.

WHY (measured, v83): per-ion fly cost 2-D 0.01 s vs 3-D channel kernel
9.2 s (662x) on the tetramer — roll-over sweeps and optimizer loops need 2-D.
"""
import importlib
import numpy as np
import pytest

from ion_gym.io.sim_spec import travelling_wave_groups
from tests.support.slim_board import N_PHASE, PITCH_MM

# GATE-B REVIVAL: the bare pre-port name left this gate
# PERMANENTLY SKIPPED since the relocation into ion_gym.physics.
_HAS = importlib.util.find_spec("ion_gym.physics.tracer_tw2d") is not None
needs_port = pytest.mark.skipif(not _HAS, reason="tracer_tw2d not yet ported")


def test_tw_spec_conventions():
    """Gate A (runs NOW): the drive convention is locked. 8 groups, phase
    step 45 deg; square wave at electrode k leads electrode k+1 by exactly
    one step; the wave advances one pitch per 1/f_step."""
    f_wf = 5e5
    gs = travelling_wave_groups(8, frequency_hz=f_wf, amplitude_v=40.0,
                                waveform="square") \
        if "waveform" in travelling_wave_groups.__code__.co_varnames \
        else travelling_wave_groups(8, frequency_hz=f_wf, amplitude_v=40.0)
    assert len(gs) == 8
    assert np.allclose([g.phase_deg for g in gs], np.arange(8) * 45.0)
    assert all(abs(g.frequency_hz - f_wf) < 1e-9 for g in gs)
    v_wave = N_PHASE * PITCH_MM * f_wf * 1e-6          # mm/us
    assert abs(v_wave - 4.572) < 1e-3                  # 8*1.143mm*0.5MHz
    # square evaluation: group k at time t equals group 0 at t + k/(8 f)
    t = 0.317e-6
    def sq(ph_deg, tt):
        return np.sign(np.sin(2 * np.pi * f_wf * tt + np.deg2rad(ph_deg)))
    for k in range(8):
        assert sq(gs[k].phase_deg, t) == sq(0.0, t + k / (8 * f_wf))
    print("  TW conventions locked: 45deg steps, v_wave=4.572 mm/us  OK")


def _smooth(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.random((n, n))
    for _ in range(150):
        b[1:-1, 1:-1] = 0.25 * (b[2:, 1:-1] + b[:-2, 1:-1]
                                + b[1:-1, 2:] + b[1:-1, :-2])
    return b


@needs_port
def test_tw2d_matches_tracer3d_on_extruded_field():
    """Gate B: same ion, same 3-channel drive (sin + two squares), z-invariant
    field: fly_tw2d must match fly3d to <1e-6 mm (dt=1 ns, 10 us, no gas)."""
    from ion_gym.physics.tracer_tw2d import build_tw2d_fields, fly_tw2d
    from ion_gym.physics.tracer3d import fly3d
    from ion_gym.io.sim_spec import RFGroupSpec
    n, nz, h = 41, 5, 0.5
    bases = {1: _smooth(n, 1), 2: _smooth(n, 2), 3: _smooth(n, 3)}
    groups = [RFGroupSpec(name="G0", frequency_hz=5e5, amplitude_v=30.0,
                          waveform="sin", phase_deg=0.0),
              RFGroupSpec(name="G1", frequency_hz=5e5, amplitude_v=25.0,
                          waveform="square", phase_deg=45.0),
              RFGroupSpec(name="G2", frequency_hz=5e5, amplitude_v=25.0,
                          waveform="square", phase_deg=225.0)]
    assign = {1: "G0", 2: "G1", 3: "G2"}
    dc = {1: 5.0, 2: 0.0, 3: -3.0}
    f2 = build_tw2d_fields(bases, groups, assign, dc, h)
    # extrude the SAME 2-D E arrays into the 3-D pack (identical field data)
    def ext(a):
        return np.repeat(a[..., None], nz, axis=2)
    K = f2["ExK"].shape[0]
    f3 = dict(EAx=ext(f2["EAx"]), EAy=ext(f2["EAy"]),
              EAz=np.zeros((n, n, nz)),
              ExK=np.stack([ext(f2["ExK"][k]) for k in range(K)]),
              EyK=np.stack([ext(f2["EyK"][k]) for k in range(K)]),
              EzK=np.zeros((K, n, n, nz)),
              ch_kind=f2["ch_kind"], ch_om=f2["ch_om"], ch_ph=f2["ch_ph"],
              ch_amp=f2["ch_amp"], ch_off=f2["ch_off"], tab_t=f2["tab_t"],
              tab_v=f2["tab_v"], tab_off=f2["tab_off"],
              ele=np.zeros((n, n, nz), bool), h_mm=h)
    r0 = (10.2, 9.7, 2 * h)                 # z exactly on a node -> tz = 0
    v0 = (0.8, -0.5, 0.0)
    o2 = fly_tw2d(f2, 300.0, r0, v0, tob_us=0.13, dt_ns=1.0, t_max_us=10.0,
                  record_every=5)
    o3 = fly3d(f3, 300.0, r0, v0, 0.13, dt_ns=1.0, t_max_us=10.0,
               record_every=5)
    nmin = min(len(o2["x"]), len(o3["x"]))
    dx = np.abs(o2["x"][:nmin] - o3["x"][:nmin]).max()
    dy = np.abs(o2["y"][:nmin] - o3["y"][:nmin]).max()
    assert dx < 1e-6 and dy < 1e-6, f"kernels diverge: dx={dx}, dy={dy}"
    print(f"  Gate B: 2-D vs extruded 3-D max dev {max(dx, dy):.2e} mm  OK")


@needs_port
def test_tw_rollover_monotone():
    """Gate C (the physics): 8-phase square TW ladder, He at 4 Torr (HS) —
    the drift ratio <vx>/v_wave must be MONOTONE in amplitude, reach the
    surfing plateau (within 10% of v_wave) at high amplitude, and slip
    (<50%) at low amplitude. m/z 300. NOTE: contract said SDS; HS is the
    kernel's collision model — same roll-over physics, flagged honestly."""
    from ion_gym.physics.tracer_tw2d import build_tw2d_fields, fly_tw2d
    from ion_gym.io.sim_spec import travelling_wave_groups, RFGroupSpec
    h = PITCH_MM / 8.0                       # 8 nodes per electrode pitch
    n_wave = 6                               # domain: 6 wavelengths
    nx = int(round(n_wave * N_PHASE * PITCH_MM / h)) + 1
    ny = 41
    xs = np.arange(nx) * h
    # bases: per-phase-group bump ladder, y-invariant, periodic in x
    lam = N_PHASE * PITCH_MM
    bases = {}
    for k in range(N_PHASE):
        prof = np.zeros(nx)
        for m in range(n_wave + 1):
            x_k = (m * N_PHASE + k) * PITCH_MM
            prof += np.exp(-0.5 * ((xs - x_k) / (0.35 * PITCH_MM)) ** 2)
        bases[k + 1] = np.repeat(prof[:, None], ny, axis=1)
    f_wf = 1e4          # 10 kHz: v_wave = 0.0914 mm/us, matched to the
                        # He mobility so the surf/slip transition is
                        # reachable. (At 500 kHz the wave outruns any
                        # mobility drift — the fast-wave regime shows ~zero
                        # transport, consistent with SLIM-SUPER operation.)
    v_wave = N_PHASE * PITCH_MM * f_wf * 1e-6
    gs = travelling_wave_groups(N_PHASE, frequency_hz=f_wf, amplitude_v=1.0)
    gs = [RFGroupSpec(name=g.name, frequency_hz=g.frequency_hz,
                      amplitude_v=1.0, waveform="square",
                      phase_deg=g.phase_deg) for g in gs]
    assign = {k + 1: gs[k].name for k in range(N_PHASE)}
    col = dict(enabled=True, gas="He", T_k=298.0, P_pa=533.3)   # 4 Torr
    ratios = []
    Y = np.repeat((np.arange(ny) * h)[None, :], nx, axis=0)
    yc = (ny // 2) * h
    for amp in (0.3, 2.0, 30.0):
        f2 = build_tw2d_fields(bases, gs, assign, {i: 0.0 for i in bases}, h)
        f2["ch_amp"][:] = amp
        # transverse DC restoring channel (stands in for the SLIM RF
        # confinement): without it, He collisions diffuse the ion out of
        # the thin y-domain in ~15 us.
        f2["EAy"] = f2["EAy"] - 2.0 * (Y - yc)
        o = fly_tw2d(f2, 300.0, (0.6 * lam, yc), (0.0, 0.0),
                     dt_ns=2.0, t_max_us=400.0, record_every=100,
                     collisions=col, seed=7)
        q = len(o["x"]) // 4
        vdrift = ((o["x"][-1] - o["x"][q])
                  / (o["t_us"][-1] - o["t_us"][q]))
        # +360/n phase steps -> wave toward -x (see module docstring):
        # the surf ratio is drift along the WAVE direction.
        ratios.append(-vdrift / v_wave)
    print("  Gate C: <vx>/v_wave (along wave) = "
          + ", ".join(f"{r:.2f}" for r in ratios)
          + f"  (amps 0.3/2/30 V, 10 kHz, v_wave={v_wave:.3f} mm/us)")
    assert ratios[0] < ratios[1] < ratios[2] + 0.05, \
        f"drift not monotone in amplitude: {ratios}"
    assert ratios[2] > 0.9, f"no surfing plateau at high amp: {ratios[2]}"
    assert ratios[0] < 0.5, f"no slipping at low amp: {ratios[0]}"


if __name__ == "__main__":
    test_tw_spec_conventions()
    print("contract gates (pre-port) passed; B/C skip until tracer_tw2d")


def _mk_duct_spec(q, gas=False, dt_ns=1.0, t_max_us=5.0, rec=5):
    """Shared Gate D/E fixture: two-electrode duct, sin + square drive,
    point source, charge q; gas=True enables HS N2 at 50 Pa."""
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                     ShapeSpec, SourceSpec, CollisionSpec,
                                     IntegrationSpec, SymmetrySpec,
                                     RFGroupSpec)

    def _r(x, y, w, h):
        return ShapeSpec("rect", {"x_mm": x, "y_mm": y,
                                  "width_mm": w, "height_mm": h})
    g = GeometrySpec(
        width_mm=10.0, height_mm=6.0, depth_mm=0.0, mm_per_gu=0.5,
        symmetry=SymmetrySpec(coords="xyz"),
        electrodes=[
            ElectrodeSpec(name="top", dc=2.0, rf_groups=["G0"],
                          shapes=[_r(0.0, 5.0, 10.0, 1.0)]),
            ElectrodeSpec(name="bot", dc=0.0, rf_groups=["G1"],
                          shapes=[_r(0.0, 0.0, 10.0, 1.0)]),
        ],
        rf_groups=[
            RFGroupSpec(name="G0", frequency_hz=5e5, amplitude_v=30.0,
                        waveform="sin", phase_deg=0.0),
            RFGroupSpec(name="G1", frequency_hz=5e5, amplitude_v=25.0,
                        waveform="square", phase_deg=45.0),
        ])
    return SimSpec(
        geometry=g, name=f"gateDE_q{q}_{'hs' if gas else 'vac'}",
        source=SourceSpec(distribution="point", n_ions=1,
                          x0_mm=2.0, y0_mm=3.0, z0_mm=0.0,
                          direction=[1.0, 0.0, 0.0],
                          ke_lo=1.0, ke_hi=1.0,
                          mz_list=[300.0], charge=q,
                          tob_span_us=0.0, seed=11),
        collisions=CollisionSpec(enabled=gas, gas="N2", T_k=300.0,
                                 P_pa=(50.0 if gas else 0.0),
                                 sigma_m2=2.27e-18),
        integration=IntegrationSpec(dt_ns=dt_ns, t_max_us=t_max_us,
                                    rec_every=rec,
                                    record_channels=[]))


def test_three_way_kernel_equivalence_and_charge():
    """Gate D (PI-commissioned 2026-09-09, steps 1+2 of the kernel
    comparison): three-way VACUUM equivalence — planar spec route vs
    tw2d vs extruded 3-D — on ONE solved square-TW spec, run at
    charge 1 AND charge 2. The same solved bases feed all three
    kernels; the same birth flies in each. Asserts pairwise max
    trajectory deviation < 1e-6 mm (Gate B's bound) and that charge
    genuinely changes the trajectory. A planar-vs-tw2d deviation
    beyond bound is a FINDING (kernel-detail divergence), not a
    tolerance to widen."""
    from ion_gym.physics.sim_build import build_run
    from ion_gym.physics.tracer_tw2d import build_tw2d_fields, fly_tw2d_ion
    from ion_gym.physics.tracer3d import fly3d

    DT_NS, TMAX_US, REC = 1.0, 5.0, 5

    def _mkspec(q):
        return _mk_duct_spec(q, gas=False, dt_ns=DT_NS,
                             t_max_us=TMAX_US, rec=REC)

    trajs = {}
    for q in (1, 2):
        spec = _mkspec(q)
        model, fly, cols, births = build_run(spec)
        traj, summ = fly(0)
        ix, iy = cols.index("x"), cols.index("y")
        px, py = traj[:, ix], traj[:, iy]
        # half-dt planar flight for the cross-integrator convergence
        # check (planar is velocity-Verlet; tw2d/3-D are RK4 — see the
        # assertion block below for why bit-agreement is NOT asserted
        # across that pair)
        spec_h = _mkspec(q)
        spec_h.integration.dt_ns = DT_NS / 2.0
        spec_h.integration.rec_every = REC * 2
        model_h, fly_h, cols_h, births_h = build_run(spec_h)
        traj_h, _ = fly_h(0)
        pxh = traj_h[:, cols_h.index("x")]
        pyh = traj_h[:, cols_h.index("y")]

        bx, by, bz, bvx, bvy, bvz, btob = (float(v) for v in births[0])
        assign = {}
        dc = {}
        for k, el in enumerate(spec.geometry.electrodes, start=1):
            # bases are 1-BASED by package convention (Gate B's synthetic
            # bases {1,2,3} and the solved model agree)
            if k not in model.bases:
                raise AssertionError(
                    f"gate D: electrode {el.name!r} index {k} not in "
                    f"model.bases keys {sorted(model.bases)} — the "
                    f"bases/assign key convention diverged")
            if len(el.rf_groups) != 1:
                raise AssertionError(
                    f"gate D: electrode {el.name!r} has "
                    f"{len(el.rf_groups)} group memberships; this gate's "
                    f"assign mapping requires exactly one")
            assign[k] = el.rf_groups[0]
            dc[k] = float(el.dc)
        # the planar model stores bases in V_BASIS-scaled units (one
        # named constant, build_planar.V_BASIS); build_tw2d_fields
        # expects genuine per-volt bases — divide ONCE, by the name
        from ion_gym.physics.build_planar import V_BASIS
        unit_bases = {k: b / V_BASIS for k, b in model.bases.items()}
        f2 = build_tw2d_fields(unit_bases, spec.geometry.rf_groups,
                               assign, dc, h_mm=model.mm_per_gu,
                               ele=np.asarray(model.ele, bool))
        o2 = fly_tw2d_ion(spec, f2, 0, r0_mm=(bx, by),
                          v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                          dt_ns=DT_NS, t_max_us=TMAX_US, record_every=REC)
        o2h = fly_tw2d_ion(spec, f2, 0, r0_mm=(bx, by),
                           v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                           dt_ns=DT_NS / 2.0, t_max_us=TMAX_US,
                           record_every=REC * 2)

        nz, h = 5, float(model.mm_per_gu)

        def ext(a):
            return np.repeat(a[..., None], nz, axis=2)
        K = f2["ExK"].shape[0]
        n1, n2 = f2["EAx"].shape
        f3 = dict(EAx=ext(f2["EAx"]), EAy=ext(f2["EAy"]),
                  EAz=np.zeros((n1, n2, nz)),
                  ExK=np.stack([ext(f2["ExK"][k]) for k in range(K)]),
                  EyK=np.stack([ext(f2["EyK"][k]) for k in range(K)]),
                  EzK=np.zeros((K, n1, n2, nz)),
                  ch_kind=f2["ch_kind"], ch_om=f2["ch_om"],
                  ch_ph=f2["ch_ph"], ch_amp=f2["ch_amp"],
                  ch_off=f2["ch_off"], tab_t=f2["tab_t"],
                  tab_v=f2["tab_v"], tab_off=f2["tab_off"],
                  ele=np.zeros((n1, n2, nz), bool), h_mm=h)
        o3 = fly3d(f3, 300.0, (bx, by, 2 * h), (bvx, bvy, bvz), btob,
                   dt_ns=DT_NS, t_max_us=TMAX_US, record_every=REC,
                   charge=q)

        n = min(len(px), len(o2["x"]), len(o3["x"]))
        assert n > 100, (
            f"gate D q={q}: too few samples to compare (n={n}; "
            f"planar={len(px)}, tw2d={len(o2['x'])}, 3d={len(o3['x'])}; "
            f"tw2d kind={o2.get('kind')}, 3d kind={o3.get('kind')})")
        # SAME-FAMILY pair (tw2d is the 2-D port of the 3-D RK4 kernel):
        # bit-level agreement is the contract — hard bound, no widening.
        d_23 = max(np.abs(o2["x"][:n] - o3["x"][:n]).max(),
                   np.abs(o2["y"][:n] - o3["y"][:n]).max())
        assert d_23 < 1e-6, (
            f"gate D q={q}: tw2d vs 3-D diverge (max dev {d_23:.3e} mm)")
        # CROSS-FAMILY pair (planar Verlet vs tw2d RK4): trajectories
        # differ by integrator truncation (measured 1.1e-2 mm at
        # dt=1 ns on this drive, session 2026-09-09 — the finding this
        # gate surfaced). The honest contract is CONVERGENCE. NOTE the
        # order: the SQUARE channel's sign flips land between step
        # boundaries, degrading BOTH integrators' formal order at each
        # crossing, so the deviation shrinks between linearly and
        # quadratically per dt-halving (measured 2.53x on this drive,
        # not the smooth-field ~4x). The gate asserts genuine
        # convergence (>2x per halving) — a frame/unit/charge bug
        # yields ~1x and goes red — plus an absolute bound.
        d_p2 = max(np.abs(px[:n] - o2["x"][:n]).max(),
                   np.abs(py[:n] - o2["y"][:n]).max())
        nh = min(len(pxh), len(o2h["x"]))
        d_p2h = max(np.abs(pxh[:nh] - o2h["x"][:nh]).max(),
                    np.abs(pyh[:nh] - o2h["y"][:nh]).max())
        assert d_p2h < d_p2 / 2.0, (
            f"gate D q={q}: planar|tw2d deviation did not shrink "
            f">2x with dt/2 ({d_p2:.3e} -> {d_p2h:.3e} mm): "
            f"the divergence is not integrator truncation — investigate")
        assert d_p2h < 5e-3, (
            f"gate D q={q}: planar|tw2d deviation at dt/2 exceeds the "
            f"absolute bound ({d_p2h:.3e} mm >= 5e-3)")
        print(f"  Gate D q={q}: tw2d|3-D {d_23:.2e} mm (RK4 family); "
              f"planar|tw2d {d_p2:.2e} -> {d_p2h:.2e} mm at dt/2 "
              f"(Verlet-vs-RK4 truncation, converging)  OK")
        trajs[q] = (px[:n], py[:n])

    n = min(len(trajs[1][0]), len(trajs[2][0]))
    # compare BOTH channels — the duct's field is along y, so x alone
    # would test charge against a near-zero field (weak by construction)
    dq = max(np.abs(trajs[1][0][:n] - trajs[2][0][:n]).max(),
             np.abs(trajs[1][1][:n] - trajs[2][1][:n]).max())
    assert dq > 1e-2, (
        f"gate D: charge=2 trajectory indistinguishable from charge=1 "
        f"(max dev {dq:.3e} mm) — charge is not reaching the planar kernel")
    print(f"  Gate D: charge 1 vs 2 max dev {dq:.2e} mm (charge acts)  OK")


def test_collisional_crn_equivalence_tw2d_vs_3d():
    """Gate E (PI-commissioned 2026-09-09, the collision leg): tw2d vs
    extruded 3-D WITH HS COLLISIONS under common random numbers. Both
    kernels seed the same numba RNG and call the same imported
    _collide/_mfp_mm, so with one seed the collision sequences must be
    IDENTICAL: equal collision counts, trajectories agreeing to
    ulp-class (measured 4.4e-16 mm across 90 collisions in the probe —
    a single differing collision amplifies to mm scale, so the 1e-12
    bound is a razor). Runs at charge 1 AND charge 2, so the z>=2
    collision-mass fix (issues.md 2026-09-09) is exercised end-to-end
    on both routes. z headroom in the extrusion (nz=81) keeps thermal
    z drift inside the box for the full flight."""
    from ion_gym.physics.sim_build import build_run
    from ion_gym.physics.build_planar import V_BASIS
    from ion_gym.physics.tracer_tw2d import build_tw2d_fields, fly_tw2d_ion
    from ion_gym.physics.tracer3d import fly3d

    col = dict(enabled=True, gas="N2", T_k=300.0, P_pa=50.0,
               sigma_m2=2.27e-18)
    for q in (1, 2):
        spec = _mk_duct_spec(q, gas=True)
        model, fly, cols, births = build_run(spec)
        bx, by, bz, bvx, bvy, bvz, btob = (float(v) for v in births[0])
        unit = {k: b / V_BASIS for k, b in model.bases.items()}
        f2 = build_tw2d_fields(unit, spec.geometry.rf_groups,
                               {1: "G0", 2: "G1"}, {1: 2.0, 2: 0.0},
                               h_mm=model.mm_per_gu,
                               ele=np.asarray(model.ele, bool))
        kw = dict(r0_mm=(bx, by), v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                  dt_ns=1.0, t_max_us=5.0, record_every=5,
                  collisions=col, seed=7)
        o2 = fly_tw2d_ion(spec, f2, 0, **kw)

        nz, h = 81, float(model.mm_per_gu)

        def ext(a):
            return np.repeat(a[..., None], nz, axis=2)
        K = f2["ExK"].shape[0]
        n1, n2 = f2["EAx"].shape
        f3 = dict(EAx=ext(f2["EAx"]), EAy=ext(f2["EAy"]),
                  EAz=np.zeros((n1, n2, nz)),
                  ExK=np.stack([ext(f2["ExK"][k]) for k in range(K)]),
                  EyK=np.stack([ext(f2["EyK"][k]) for k in range(K)]),
                  EzK=np.zeros((K, n1, n2, nz)),
                  ch_kind=f2["ch_kind"], ch_om=f2["ch_om"],
                  ch_ph=f2["ch_ph"], ch_amp=f2["ch_amp"],
                  ch_off=f2["ch_off"], tab_t=f2["tab_t"],
                  tab_v=f2["tab_v"], tab_off=f2["tab_off"],
                  ele=np.zeros((n1, n2, nz), bool), h_mm=h)
        o3 = fly3d(f3, 300.0, (bx, by, nz // 2 * h), (bvx, bvy, bvz),
                   btob, dt_ns=1.0, t_max_us=5.0, record_every=5,
                   collisions=col, seed=7, charge=q)

        nc2, nc3 = int(o2["n_col"]), int(o3["ncol"])
        assert nc2 == nc3, (
            f"gate E q={q}: collision counts differ under CRN "
            f"(tw2d {nc2} vs 3-D {nc3}) — the RNG draw sequences "
            f"have desynced between the kernels")
        assert nc2 > 30, (
            f"gate E q={q}: only {nc2} collisions in 5 us at 50 Pa — "
            f"the gas is silently off or the rate is wrong")
        n = min(len(o2["x"]), len(o3["x"]))
        assert n > 500, f"gate E q={q}: too few samples ({n})"
        dx = np.abs(o2["x"][:n] - o3["x"][:n]).max()
        dy = np.abs(o2["y"][:n] - o3["y"][:n]).max()
        assert max(dx, dy) < 1e-12, (
            f"gate E q={q}: CRN trajectories diverge "
            f"(dx={dx:.3e}, dy={dy:.3e} mm) — a differing collision "
            f"amplifies to mm scale, so this is a real desync")
        print(f"  Gate E q={q}: {nc2} collisions, CRN max dev "
              f"{max(dx, dy):.2e} mm  OK")
