"""
test_multidrive3d
-----------------
Gate suite for the 3-D multi-drive channel kernel (tracer3d, affine
waveform). Mirrors test_multidrive.py (2-D) with 3-D analytic ground truth,
plus the transport gate the flat slice structurally cannot produce.

Gates
  1  backward BIT-EXACT vs tracer3d_v61_frozen (single sin channel)
  2  two-frequency, axis-separated (multi-freq + no cross-axis contamination)
  3  table drive: hold (step) and linear (ramp) kinematics
  4  square drive: triangle-wave velocity
  5  traveling-wave TRANSPORT: potential well and a captured ion both advance
     at v_phase = lambda*f  (the 3-D-only observable)

Run:  python3 -m pytest test_multidrive3d.py -q
"""

import numpy as np
import pytest

from ion_gym.physics.tracer3d import (fly3d, build_field_aware_3d, K_SIN, K_COS, K_SQUARE,
                      K_TAB_HOLD, K_TAB_LIN)
from ion_gym.physics.tracer3d_v61_frozen import fly3d as fly3d_v61

E_CHG = 1.602176634e-19
AMU = 1.66053906660e-27


# --------------------------------------------------------------- helpers
def _uniform_axis_field(shape, axis, val=1.0):
    """(Ex,Ey,Ez) uniform basis: `val` on `axis`, zero elsewhere."""
    E = [np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    E[axis][...] = val
    return E[0], E[1], E[2]


def _qm(mz_Da):
    return E_CHG / (mz_Da * AMU)


# ============================================================= GATE 1
def test_gate1_backward_bit_exact():
    """Single sin channel must reproduce the frozen v61 kernel bit-for-bit.
    Smooth non-trivial EA/EB, non-zero tob (absolute clock), no metal."""
    n = 21
    h = 0.5
    xs = (np.arange(n) - (n - 1) / 2) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    # smooth potentials -> electrode-aware field (no metal: pure central diff)
    ele = np.zeros((n, n, n), bool)
    phiA = 0.3 * X + 0.05 * (X * X - Y * Y) + 0.02 * Z
    phiB = 0.1 * (X * X - Y * Y) + 0.03 * X * Y
    EA = build_field_aware_3d(phiA, ele, h)
    EB = build_field_aware_3d(phiB, ele, h)

    fields = dict(EAx=EA[0], EAy=EA[1], EAz=EA[2],
                  EBx=EB[0], EBy=EB[1], EBz=EB[2],
                  ele=ele, h_mm=h, rf_V=7.3, dc_V=-1.2, om_rad_us=2.5)

    kw = dict(dt_ns=1.0, t_max_us=8.0, record_every=3, max_records=20000)
    new = fly3d(fields, 40.0, (0.4, -0.3, 0.1), (0.5, -0.2, 0.05), 1.234, **kw)
    old = fly3d_v61(fields, 40.0, (0.4, -0.3, 0.1), (0.5, -0.2, 0.05), 1.234, **kw)

    assert new["kind"] == old["kind"]
    assert new["x"].shape == old["x"].shape
    for key in ("x", "y", "z", "vx", "vy", "vz", "t_us"):
        assert np.array_equal(new[key], old[key]), f"{key} differs"
    assert new["tof_us"] == old["tof_us"]
    assert new["KE_eV"] == old["KE_eV"]


# ============================================================= GATE 2
def test_gate2_two_frequency_axis_separated():
    """Uniform Ex driven at f_LO, uniform Ey at f_HI, independent channels.
    Closed form per axis:  x(t) = (a/W)[t - sin(W t)/W],  a = qm*1e3*A."""
    n = 31
    h = 1.0
    shape = (n, n, 5)
    ele = np.zeros(shape, bool)
    EAx = np.zeros(shape); EAy = np.zeros(shape); EAz = np.zeros(shape)

    ExK0, EyK0, EzK0 = _uniform_axis_field(shape, 0, 1.0)   # x-axis basis
    ExK1, EyK1, EzK1 = _uniform_axis_field(shape, 1, 1.0)   # y-axis basis
    ExK = np.stack([ExK0, ExK1]); EyK = np.stack([EyK0, EyK1])
    EzK = np.stack([EzK0, EzK1])

    A0, A1 = 0.8, 0.5                         # V/mm amplitudes
    w0, w1 = 0.30, 1.10                       # rad/us  (f_LO, f_HI)
    fields = dict(
        EAx=EAx, EAy=EAy, EAz=EAz, ExK=ExK, EyK=EyK, EzK=EzK,
        ch_kind=np.array([K_SIN, K_SIN], np.int64),
        ch_om=np.array([w0, w1]), ch_ph=np.zeros(2),
        ch_amp=np.array([A0, A1]), ch_off=np.zeros(2),
        tab_t=np.zeros(0), tab_v=np.zeros(0), tab_off=np.zeros(3, np.int64),
        ele=ele, h_mm=h)

    mz = 50.0
    cx = (n - 1) / 2 * h
    o = fly3d(fields, mz, (cx, cx, 2 * h), (0.0, 0.0, 0.0), 0.0,
              dt_ns=0.5, t_max_us=6.0, record_every=1, max_records=40000)

    qm = _qm(mz)
    W0 = w0 * 1e6; W1 = w1 * 1e6            # rad/s
    a0 = qm * 1e3 * A0; a1 = qm * 1e3 * A1  # m/s^2
    t = (o["t_us"] - o["t_us"][0]) * 1e-6   # s
    x_pred = (a0 / W0) * (t - np.sin(W0 * t) / W0) * 1e3 + cx   # mm
    y_pred = (a1 / W1) * (t - np.sin(W1 * t) / W1) * 1e3 + cx
    ex = np.abs(o["x"] - x_pred).max() / (np.abs(x_pred - cx).max() + 1e-30)
    ey = np.abs(o["y"] - y_pred).max() / (np.abs(y_pred - cx).max() + 1e-30)
    assert ex < 1e-4, f"x-axis f_LO rel err {ex:.2e}"
    assert ey < 1e-4, f"y-axis f_HI rel err {ey:.2e}"
    # z untouched -> exactly still
    assert np.allclose(o["z"], 2 * h, atol=1e-12)


# ============================================================= GATE 3
def test_gate3_table_hold_and_linear():
    n = 21; h = 1.0; shape = (n, n, 5)
    ele = np.zeros(shape, bool)
    EAx = np.zeros(shape); EAy = np.zeros(shape); EAz = np.zeros(shape)
    ExK0, EyK0, EzK0 = _uniform_axis_field(shape, 0, 1.0)
    ExK = ExK0[None]; EyK = EyK0[None]; EzK = EzK0[None]
    mz = 30.0; qm = _qm(mz)
    cx = (n - 1) / 2 * h

    # ---- HOLD: field 0 until t_sw, then V1 ; rest-then-constant-accel
    t_sw = 2.0; V1 = 0.6
    fields = dict(EAx=EAx, EAy=EAy, EAz=EAz, ExK=ExK, EyK=EyK, EzK=EzK,
                  ch_kind=np.array([K_TAB_HOLD], np.int64),
                  ch_om=np.zeros(1), ch_ph=np.zeros(1),
                  ch_amp=np.ones(1), ch_off=np.zeros(1),
                  tab_t=np.array([0.0, t_sw]), tab_v=np.array([0.0, V1]),
                  tab_off=np.array([0, 2], np.int64), ele=ele, h_mm=h)
    o = fly3d(fields, mz, (cx, cx, 2 * h), (0, 0, 0), 0.0,
              dt_ns=1.0, t_max_us=5.0, record_every=1, max_records=40000)
    t = (o["t_us"] - o["t_us"][0]) * 1e-6
    a = qm * 1e3 * V1
    xh = np.where(t < t_sw * 1e-6, 0.0, 0.5 * a * (t - t_sw * 1e-6) ** 2) * 1e3 + cx
    eh = np.abs(o["x"] - xh).max() / (np.abs(xh - cx).max() + 1e-30)
    assert eh < 6e-4, f"table-hold rel err {eh:.2e}"

    # ---- LINEAR: field ramps 0->V1 over [0,T]; cubic x = (qm*1e3*V1/T) t^3/6
    T = 4.0
    fields = dict(EAx=EAx, EAy=EAy, EAz=EAz, ExK=ExK, EyK=EyK, EzK=EzK,
                  ch_kind=np.array([K_TAB_LIN], np.int64),
                  ch_om=np.zeros(1), ch_ph=np.zeros(1),
                  ch_amp=np.ones(1), ch_off=np.zeros(1),
                  tab_t=np.array([0.0, T]), tab_v=np.array([0.0, V1]),
                  tab_off=np.array([0, 2], np.int64), ele=ele, h_mm=h)
    o = fly3d(fields, mz, (cx, cx, 2 * h), (0, 0, 0), 0.0,
              dt_ns=1.0, t_max_us=T, record_every=1, max_records=40000)
    t = (o["t_us"] - o["t_us"][0]) * 1e-6
    Ts = T * 1e-6
    xl = (qm * 1e3 * V1 / Ts) * t ** 3 / 6.0 * 1e3 + cx
    el = np.abs(o["x"] - xl).max() / (np.abs(xl - cx).max() + 1e-30)
    assert el < 2e-5, f"table-linear rel err {el:.2e}"


# ============================================================= GATE 4
def test_gate4_square_triangle_velocity():
    """sign(sin) drive from rest: velocity is a triangle wave. Check the
    analytic corners v(T/2)=a0*T/2, v(T)=0, x(T/2)=0.5*a0*(T/2)^2."""
    n = 41; h = 1.0; shape = (n, n, 5)
    ele = np.zeros(shape, bool)
    EAx = np.zeros(shape); EAy = np.zeros(shape); EAz = np.zeros(shape)
    ExK0, EyK0, EzK0 = _uniform_axis_field(shape, 0, 1.0)
    ExK = ExK0[None]; EyK = EyK0[None]; EzK = EzK0[None]
    mz = 100.0; qm = _qm(mz)
    cx = (n - 1) / 2 * h

    f_us = 1.0                                # 1 cycle per us
    W = 2 * np.pi * f_us                      # rad/us
    A = 0.2                                   # V/mm (small -> stays in box)
    T = 1.0 / f_us                            # us period
    fields = dict(EAx=EAx, EAy=EAy, EAz=EAz, ExK=ExK, EyK=EyK, EzK=EzK,
                  ch_kind=np.array([K_SQUARE], np.int64),
                  ch_om=np.array([W]), ch_ph=np.zeros(1),
                  ch_amp=np.array([A]), ch_off=np.zeros(1),
                  tab_t=np.zeros(0), tab_v=np.zeros(0),
                  tab_off=np.array([0, 0], np.int64), ele=ele, h_mm=h)
    o = fly3d(fields, mz, (cx, cx, 2 * h), (0, 0, 0), 0.0,
              dt_ns=0.5, t_max_us=T, record_every=1, max_records=40000)
    t = (o["t_us"] - o["t_us"][0]) * 1e-6
    a0 = qm * 1e3 * A                          # m/s^2
    Th = 0.5 * T * 1e-6                         # half period, s
    # nearest samples to T/2 and T
    ih = int(np.argmin(np.abs(t - Th)))
    itop = len(t) - 1
    v_half = o["vx"][ih] * 1e3                  # mm/us -> m/s  (vx stored mm/us)
    v_full = o["vx"][itop] * 1e3
    x_half = (o["x"][ih] - cx) * 1e-3           # mm -> m
    assert abs(v_half - a0 * Th) / (a0 * Th) < 3e-3, "v(T/2) off"
    assert abs(v_full) / (a0 * Th) < 3e-3, "v(T) not ~0"
    assert abs(x_half - 0.5 * a0 * Th ** 2) / (0.5 * a0 * Th ** 2) < 5e-3, "x(T/2) off"


# ============================================================= GATE 5
def _tw_fields(lam_mm, f_us, V0, shape, h, mz):
    """Traveling wave phi = V0 cos(kx - W t) as a quadrature channel pair.
    Ex = -dphi/dx = V0 k sin(kx - W t)
       = cos(W t)*[V0 k sin(kx)] + sin(W t)*[-V0 k cos(kx)]
    Returns fields dict and v_phase (mm/us)."""
    nx, ny, nz = shape
    k = 2 * np.pi / lam_mm
    W = 2 * np.pi * f_us
    xs = np.arange(nx) * h
    X = xs[:, None, None] * np.ones(shape)
    b0x = V0 * k * np.sin(k * X)         # cos-weight channel
    b1x = -V0 * k * np.cos(k * X)        # sin-weight channel
    zeros = np.zeros(shape)
    ExK = np.stack([b0x, b1x]); EyK = np.stack([zeros, zeros])
    EzK = np.stack([zeros, zeros])
    fields = dict(
        EAx=np.zeros(shape), EAy=np.zeros(shape), EAz=np.zeros(shape),
        ExK=ExK, EyK=EyK, EzK=EzK,
        ch_kind=np.array([K_COS, K_SIN], np.int64),
        ch_om=np.array([W, W]), ch_ph=np.zeros(2),
        ch_amp=np.ones(2), ch_off=np.zeros(2),
        tab_t=np.zeros(0), tab_v=np.zeros(0), tab_off=np.zeros(3, np.int64),
        ele=np.zeros(shape, bool), h_mm=h)
    return fields, lam_mm * f_us, k, W


@pytest.mark.slow  # minutes-scale
def test_gate5_traveling_wave_transport():
    """The 3-D-only gate. A traveling potential well and a captured ion both
    advance along x at v_phase = lambda*f."""
    lam = 8.0; f_us = 0.0375           # -> v_phase = 0.30 mm/us (~37.5 kHz)
    V0 = 6.0                            # deep well (V) -> robust capture
    h = 0.2; nx = 301; ny = nz = 5
    shape = (nx, ny, nz)
    mz = 100.0
    fields, v_phase, k, W = _tw_fields(lam, f_us, V0, shape, h, mz)

    # (a) analytic: potential minimum of cos(kx - W t) advances at W/k
    v_phase_check = (W / k)             # (rad/us)/(rad/mm) = mm/us
    assert abs(v_phase_check - v_phase) / v_phase < 1e-12

    # (b) ion: start at rest at a t=0 well minimum (kx = pi -> x = lam/2 + n*lam)
    x0 = 2.5 * lam                      # = 20 mm, an interior minimum
    cy = (ny - 1) / 2 * h
    t_max = 30.0
    o = fly3d(fields, mz, (x0, cy, cy), (0.0, 0.0, 0.0), 0.0,
              dt_ns=0.5, t_max_us=t_max, record_every=4, max_records=200000)

    # ion must not have splatted/left; it should ride the wave in +x
    assert o["kind"] == 2, f"ion did not survive (kind={o['kind']})"
    disp = o["x"][-1] - o["x"][0]
    dt = (o["t_us"][-1] - o["t_us"][0])
    v_ion = disp / dt                  # mm/us, mean
    # captured ion tracks the well within the slosh amplitude
    assert disp > 0.5 * v_phase * dt, "no net forward transport"
    rel = abs(v_ion - v_phase) / v_phase
    assert rel < 0.15, f"transport speed {v_ion:.4f} vs v_phase {v_phase:.4f} (rel {rel:.2%})"


# ============================================================= GATE 6
@pytest.mark.slow  # minutes-scale
def test_gate6_collision_thermalization():
    """HS in 3-D must drive an ensemble to the gas temperature: field-free
    ions started at rest relax to <KE> = (3/2) kT (equipartition), independent
    of ion mass. This certifies the collision hook the roll-back sweep uses."""
    n = 11; h = 1.0; shape = (n, n, n)
    fields = dict(EAx=np.zeros(shape), EAy=np.zeros(shape), EAz=np.zeros(shape),
                  ele=np.zeros(shape, bool), h_mm=h)
    col = dict(gas="N2", T_k=273.0, P_pa=133.28, sigma_m2=2.27e-18)
    cx = (n - 1) / 2 * h
    KB = 1.3806505e-23
    kT32_eV = 1.5 * KB * col["T_k"] / E_CHG

    kes = []
    for s in range(48):
        o = fly3d(fields, 100.0, (cx, cx, cx), (0.0, 0.0, 0.0), 0.0,
                  dt_ns=2.0, t_max_us=40.0, record_every=200,
                  max_records=4000, collisions=col, seed=s + 1)
        assert o["ncol"] > 50, f"too few collisions ({o['ncol']}) to thermalize"
        # time-average KE over the second half (equilibrium) of each ion
        v2 = o["vx"] ** 2 + o["vy"] ** 2 + o["vz"] ** 2   # (mm/us)^2
        half = len(v2) // 2
        ke_eV = 0.5 * (100.0 * AMU) * (v2[half:].mean() * 1e6) / E_CHG
        kes.append(ke_eV)
    mean_ke = float(np.mean(kes))
    rel = abs(mean_ke - kT32_eV) / kT32_eV
    assert rel < 0.20, (f"thermalized <KE>={mean_ke*1e3:.2f} meV vs "
                        f"(3/2)kT={kT32_eV*1e3:.2f} meV (rel {rel:.1%})")


if __name__ == "__main__":
    for fn in [test_gate1_backward_bit_exact,
               test_gate2_two_frequency_axis_separated,
               test_gate3_table_hold_and_linear,
               test_gate4_square_triangle_velocity,
               test_gate5_traveling_wave_transport,
               test_gate6_collision_thermalization]:
        fn(); print("PASS", fn.__name__)
