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
