"""Declarative ion-source definition: FORMAT x DISTRIBUTION.

Two orthogonal choices, following the pattern of an established
reference tool:

  FORMAT       -- what currency a quantity is expressed in. A velocity is
                  a vector, or a direction and a speed, or a direction and
                  a kinetic energy. Choosing the format changes WHICH
                  SCALARS EXIST, nothing else.
  DISTRIBUTION -- how each scalar varies across the packet. Applied
                  per-scalar, independently, with NO scalar privileged.

Why this shape rather than named per-axis modes: an earlier proposal gave
x/y/z each a physical "mode" (thermal / divergence / sigma_v). It was
refused, correctly -- that encoded ONE instrument's mapping into the
schema, and "if the instrument is reoriented or the geometry is different
that all goes out the window". Format-and-distribution carries no such
assumption: every scalar is declared the same way whatever it is called
and whatever direction it points.

TEMPERATURE IS DEMOTED, deliberately. It is not a property a source HAS;
it is one way of GENERATING a velocity distribution -- gaussian per
component with sigma = sqrt(kT/m). It therefore lives as a distribution
kind (`maxwellian`), not as a field of the source. A scalar temperature
still works as isotropic shorthand and expands to exactly that, so decks
and banked results predating this module are unaffected.

BEAM ENVELOPE is the one format that spans position AND velocity, because
a converging beam is a CORRELATION between them and cannot be expressed by
declaring the two independently. It is stated in ordinary geometry --
size, divergence, and where the focus is -- rather than in Twiss
parameters, on the grounds that most users do not know what a Twiss
parameter is. The mapping is exact and both directions are provided, so
anyone who thinks in beta/alpha can still read them.
"""

import math

import numpy as np

E_CHG = 1.602176634e-19
KG_AMU = 1.66053906660e-27
K_B = 1.380649e-23

POSITION_FORMATS = ("xyz", "cylindrical", "beam_envelope")
VELOCITY_FORMATS = ("velocity_vector", "direction_speed", "direction_ke",
                    "direction_momentum", "beam_envelope")
DISTRIBUTIONS = ("single", "uniform", "gaussian", "grid", "list",
                 "maxwellian")

# Which scalars each format requires. The UI renders exactly these fields,
# so this table is the single authority for "what does this format ask
# for" -- a UI that hardcoded its own list would drift from the generator
# and show fields nothing reads.
FORMAT_SCALARS = {
    "xyz": ("x_mm", "y_mm", "z_mm"),
    "cylindrical": ("r_mm", "theta_deg", "z_mm"),
    "velocity_vector": ("vx", "vy", "vz"),
    "direction_speed": ("dir_x", "dir_y", "dir_z", "speed_mm_per_us"),
    "direction_ke": ("dir_x", "dir_y", "dir_z", "ke_ev"),
    "direction_momentum": ("dir_x", "dir_y", "dir_z", "momentum_amu_mm_us"),
    # beam_envelope declares PLANES, not scalars; see EnvelopePlane.
    "beam_envelope": (),
}


class BeamSpecError(ValueError):
    """Raised for a source declaration that cannot be honoured as written.

    Never a fallback: a source silently corrected to something samplable
    produces a packet the user did not ask for, and every number derived
    from it inherits that substitution without saying so.
    """


def _rng(seed):
    """Seeded generator. A random distribution WITHOUT a seed is refused
    upstream -- an unseeded packet is not reproducible, and this project
    retired Monte-Carlo objectives precisely because seed scatter was
    being mistaken for signal."""
    return np.random.default_rng(int(seed))


def sample_scalar(dist, n, *, name="scalar", mass_amu=None):
    """Draw `n` values of one scalar from its declared distribution.

    `dist` is a dict with a `kind` and that kind's parameters. Unknown
    kinds and missing parameters REFUSE by name rather than defaulting:
    the distribution is a physical claim, and a silently substituted one
    is a wrong answer wearing the right shape.
    """
    if not isinstance(dist, dict):
        # A bare number is the obvious shorthand for "single", and
        # accepting it costs nothing while refusing it would make simple
        # decks verbose for no gain.
        dist = {"kind": "single", "value": float(dist)}
    kind = dist.get("kind")
    if kind not in DISTRIBUTIONS:
        raise BeamSpecError(
            f"{name}: unknown distribution kind {kind!r}; "
            f"one of {list(DISTRIBUTIONS)}")

    def need(key):
        if key not in dist:
            raise BeamSpecError(
                f"{name}: distribution {kind!r} requires {key!r}")
        return dist[key]

    if kind == "single":
        return np.full(n, float(need("value")))

    if kind == "uniform":
        lo, hi = float(need("min")), float(need("max"))
        if not lo <= hi:
            raise BeamSpecError(f"{name}: uniform min ({lo}) > max ({hi})")
        if lo == hi:
            return np.full(n, lo)
        return _rng(need("seed")).uniform(lo, hi, n)

    if kind == "gaussian":
        mu, sd = float(need("mean")), float(need("sigma"))
        if sd < 0:
            raise BeamSpecError(f"{name}: gaussian sigma must be >= 0")
        if sd == 0:
            return np.full(n, mu)
        return _rng(need("seed")).normal(mu, sd, n)

    if kind == "maxwellian":
        # ONE VELOCITY COMPONENT of a Maxwellian at temperature T:
        # gaussian, zero-mean, sigma = sqrt(kT/m). This is the ONLY place
        # temperature enters the source, and it enters as a generator.
        t_k = float(need("temperature_k"))
        if t_k < 0:
            raise BeamSpecError(f"{name}: temperature must be >= 0 K")
        if mass_amu is None:
            raise BeamSpecError(
                f"{name}: a maxwellian needs the ion mass to convert "
                f"temperature to a velocity spread; none was supplied")
        if t_k == 0:
            return np.zeros(n)
        # m/s -> mm/us is 1e-3
        sigma = math.sqrt(K_B * t_k / (float(mass_amu) * KG_AMU)) * 1.0e-3
        return _rng(need("seed")).normal(0.0, sigma, n)

    if kind == "grid":
        # DETERMINISTIC and first-class, not an afterthought: this project
        # retired Monte-Carlo FWHM as an objective (~40% seed scatter) in
        # favour of a deterministic ray stencil. A source that can only
        # draw randomly cannot express that stencil.
        lo, hi = float(need("min")), float(need("max"))
        if n == 1:
            return np.full(1, 0.5 * (lo + hi))
        return np.linspace(lo, hi, n)

    if kind == "list":
        vals = np.asarray(need("values"), dtype=float)
        if vals.size == 0:
            raise BeamSpecError(f"{name}: list distribution has no values")
        # Cycled, so a short list is a repeating pattern rather than a
        # length error -- and the cycling is stated, not silent.
        return vals[np.arange(n) % vals.size]

    raise BeamSpecError(f"{name}: distribution {kind!r} is declared but "
                        f"not implemented")


def envelope_to_twiss(size_mm, divergence_mrad, focus_mm):
    """(size, divergence, focus distance) -> (emittance, beta, alpha).

    THE ORDINARY-LANGUAGE FORM IS THE DECLARED ONE: most
    users do not know what a Twiss parameter is, but everyone can state a
    beam's half-width, its half-angle, and how far away it comes to a
    focus. The three are exactly equivalent, so nothing is given up.

    `focus_mm`: 0 at a waist, positive when the waist is DOWNSTREAM (a
    converging beam), negative when it is behind (diverging). At distance
    d from a waist, <y y'> = d * sigma_prime^2, and that correlation IS
    what alpha encodes -- it is the quantity a per-scalar declaration
    cannot express, which is why this format exists at all.
    """
    s = float(size_mm)
    sp = float(divergence_mrad) * 1.0e-3      # mrad -> rad
    d = float(focus_mm)
    if s < 0 or sp < 0:
        raise BeamSpecError(
            f"beam envelope: size ({s}) and divergence ({divergence_mrad}) "
            f"must be >= 0")
    if sp == 0.0:
        # A parallel beam: zero divergence, hence zero emittance. beta is
        # undefined (the ellipse is a horizontal line), and reporting a
        # number for it would be inventing one.
        return 0.0, float("inf") if s > 0 else 0.0, 0.0
    # Waist size follows from projecting the declared plane back by d.
    s_w2 = s * s - (d * sp) ** 2
    if s_w2 == 0.0:
        # A POINT SOURCE: zero size at the waist, finite divergence.
        # Degenerate but entirely legitimate -- emittance is zero and the
        # phase-space ellipse collapses to a vertical line. Found by
        # deriving from a certified reference packet, which is born at a
        # point in one plane; the guard below had treated `<= 0` as
        # impossible when only `< 0` is, and so refused a real beam.
        # beta is 0 in the limit (size^2 / eps with both -> 0); alpha is
        # 0 because a point has no position spread to correlate.
        return 0.0, 0.0, 0.0
    if s_w2 < 0:
        raise BeamSpecError(
            f"beam envelope: a beam of size {s} mm and divergence "
            f"{divergence_mrad} mrad cannot have its focus {d} mm away -- "
            f"the implied waist size is not real. |focus| must be < "
            f"{s / sp:.3f} mm for these values.")
    s_w = math.sqrt(s_w2)
    eps = s_w * sp                    # mm*rad; invariant at the waist
    beta = s * s / eps
    alpha = d * sp * sp / eps
    return eps, beta, alpha


def sample_envelope_plane(n, size_mm, divergence_mrad, focus_mm, *,
                          seed, kind="gaussian"):
    """Draw (q, q') pairs for one plane with the declared correlation.

    Returns position in mm and angle in RADIANS. The correlation is
    applied as a drift from the waist, which is both the physically
    honest construction and the one that makes `focus_mm` mean what it
    says: draw an UPRIGHT distribution at the waist, then propagate it
    the declared distance. Drawing a correlated pair directly would give
    the same covariance while making the sign convention easy to get
    backwards.
    """
    sp = float(divergence_mrad) * 1.0e-3
    d = float(focus_mm)
    _eps, _beta, _alpha = envelope_to_twiss(size_mm, divergence_mrad, d)
    s_w = math.sqrt(max(0.0, float(size_mm) ** 2 - (d * sp) ** 2))
    if kind == "gaussian":
        r = _rng(seed)
        q_w = r.normal(0.0, s_w, n) if s_w > 0 else np.zeros(n)
        qp = r.normal(0.0, sp, n) if sp > 0 else np.zeros(n)
    elif kind == "uniform":
        # BOUNDED, and that boundedness is physical. A packet drawn
        # uniformly has hard edges; describing it as gaussian puts ions in
        # tails that do not exist, and the error GROWS WITH N because the
        # tails are sampled more finely -- measured on a certified
        # reference packet: R +41% at n=200 and -85% at n=1000 from the same
        # declaration. An n-dependent answer from a fixed description is
        # proof the description is wrong.
        # Half-width chosen so the sampled sd matches the declared one.
        r = _rng(seed)
        h_w = s_w * math.sqrt(3.0)
        h_p = sp * math.sqrt(3.0)
        q_w = r.uniform(-h_w, h_w, n) if s_w > 0 else np.zeros(n)
        qp = r.uniform(-h_p, h_p, n) if sp > 0 else np.zeros(n)
    elif kind == "grid":
        # Deterministic stencil: the corners and centre of the phase-space
        # ellipse rather than draws from it.
        m = max(1, int(round(math.sqrt(n))))
        a = np.linspace(-1.0, 1.0, m)
        qq, pp = np.meshgrid(a, a)
        q_w = (qq.ravel()[:n] * s_w)
        qp = (pp.ravel()[:n] * sp)
        if q_w.size < n:                     # pad by cycling, stated
            reps = int(np.ceil(n / max(1, q_w.size)))
            q_w = np.tile(q_w, reps)[:n]
            qp = np.tile(qp, reps)[:n]
    else:
        raise BeamSpecError(
            f"beam envelope: plane distribution {kind!r} is not supported; "
            f"use 'gaussian' or 'grid'")
    # propagate BACK from the waist to the declared plane: the waist is
    # `focus_mm` downstream, so this plane is at -d relative to it.
    q = q_w + d * qp
    return q, qp


def _norm_dirs(dx, dy, dz, *, name):
    v = np.stack([dx, dy, dz], axis=1)
    nrm = np.linalg.norm(v, axis=1)
    if np.any(nrm == 0):
        raise BeamSpecError(
            f"{name}: a direction of zero length has no direction; "
            f"{int(np.sum(nrm == 0))} of {len(nrm)} ions declared one")
    return v / nrm[:, None]


def speed_from_ke(ke_ev, mass_amu):
    """KE (eV) -> speed (mm/us) for a given mass."""
    ke_j = np.asarray(ke_ev, dtype=float) * E_CHG
    if np.any(ke_j < 0):
        raise BeamSpecError("kinetic energy must be >= 0 eV")
    return np.sqrt(2.0 * ke_j / (float(mass_amu) * KG_AMU)) * 1.0e-3


def ke_from_speed(speed_mm_per_us, mass_amu):
    """Speed (mm/us) -> KE (eV). The inverse, so a format switch can
    CONVERT rather than clear -- switching from direction+KE to a velocity
    vector should show the equivalent components, not blanks."""
    v = np.asarray(speed_mm_per_us, dtype=float) * 1.0e3   # -> m/s
    return 0.5 * float(mass_amu) * KG_AMU * v * v / E_CHG


def generate_packet(decl, *, mass_amu, n=None):
    """Realise a declared source into ion rows.

    Returns a list of [x, y, z, vx, vy, vz, tob_us] -- the row shape the
    instrument document's `beam.ions` already uses, so a regenerated
    packet drops straight in.

    The declaration carries its own `n`, its formats and its
    distributions; `n` here overrides it for a one-off. Every refusal is
    by name, and nothing is defaulted silently: an under-specified source
    is a question for the user, not a guess for this function.
    """
    if not isinstance(decl, dict):
        raise BeamSpecError("source declaration must be a mapping")
    n = int(n if n is not None else decl.get("n", 0))
    if n < 1:
        raise BeamSpecError(f"a packet needs at least 1 ion; got n = {n}")
    pf = decl.get("position_format", "xyz")
    vf = decl.get("velocity_format", "velocity_vector")
    if pf not in POSITION_FORMATS:
        raise BeamSpecError(f"unknown position_format {pf!r}; "
                            f"one of {list(POSITION_FORMATS)}")
    if vf not in VELOCITY_FORMATS:
        raise BeamSpecError(f"unknown velocity_format {vf!r}; "
                            f"one of {list(VELOCITY_FORMATS)}")
    if (pf == "beam_envelope") != (vf == "beam_envelope"):
        # A beam envelope is a JOINT statement about position and
        # velocity in a plane; half of it is not a declaration.
        raise BeamSpecError(
            "beam_envelope describes position AND velocity together, so "
            "both formats must select it (position_format="
            f"{pf!r}, velocity_format={vf!r})")
    sc = decl.get("scalars") or {}

    def draw(key):
        if key not in sc:
            raise BeamSpecError(
                f"format requires scalar {key!r}, which the declaration "
                f"does not provide")
        return sample_scalar(sc[key], n, name=key, mass_amu=mass_amu)

    if pf == "beam_envelope":
        planes = decl.get("planes") or {}
        if not planes:
            raise BeamSpecError(
                "beam_envelope declares no planes; each plane needs "
                "size_mm, divergence_mrad and focus_mm")
        pos = {a: np.zeros(n) for a in ("x", "y", "z")}
        ang = {a: np.zeros(n) for a in ("x", "y", "z")}
        for ax, p in planes.items():
            if ax not in ("x", "y", "z"):
                raise BeamSpecError(
                    f"beam_envelope: unknown plane {ax!r}; planes are "
                    f"'x', 'y', 'z'")
            for k in ("size_mm", "divergence_mrad", "focus_mm"):
                if k not in p:
                    raise BeamSpecError(
                        f"beam_envelope plane {ax!r} requires {k!r}")
            q, qp = sample_envelope_plane(
                n, p["size_mm"], p["divergence_mrad"], p["focus_mm"],
                seed=p.get("seed", decl.get("seed", 0)),
                kind=p.get("distribution", "gaussian"))
            # MEAN ANGLE IS DESIGN, NOT NOISE. A plane's mean divergence
            # angle can be deliberately non-zero -- in a planar MRT the
            # constant z-velocity IS the drift that produces the zigzag,
            # and the K_z resonance depends on it. An earlier version centred the
            # angles, which silently deleted that drift and lost every
            # ion. `mean_angle_mrad` and `center_mm` default to 0, so a
            # plain beam is unaffected.
            pos[ax] = q + float(p.get("center_mm", 0.0))
            ang[ax] = qp + float(p.get("mean_angle_mrad", 0.0)) * 1.0e-3
        axis = decl.get("axis")
        if axis not in ("x", "y", "z"):
            raise BeamSpecError(
                "beam_envelope needs `axis`: the propagation direction "
                "the divergences are angles RELATIVE TO. Without it an "
                "angle has no reference and the packet is undefined.")
        ke = draw("ke_ev") if "ke_ev" in sc else None
        if ke is None:
            raise BeamSpecError(
                "beam_envelope needs `ke_ev`: divergence sets the "
                "transverse angles, energy sets how fast the beam travels "
                "along the axis; neither implies the other.")
        spd = speed_from_ke(ke, mass_amu)
        # small-angle: v_transverse = v * angle, v_axial = v * cos ~ v
        # Axial sign: the beam may travel in the NEGATIVE axis direction
        # (vx < 0 on a real analyzer). `axis_sign` declares it; angles are
        # measured relative to the direction of travel either way.
        _sgn = float(decl.get("axis_sign", 1.0))
        vel = {}
        for a in ("x", "y", "z"):
            vel[a] = spd * (_sgn if a == axis else _sgn * ang[a])
        # the axial coordinate carries the declared offset, if any
        off = sc.get(f"{axis}_mm")
        if off is not None:
            pos[axis] = sample_scalar(off, n, name=f"{axis}_mm",
                                      mass_amu=mass_amu)
        x, y, z = pos["x"], pos["y"], pos["z"]
        vx, vy, vz = vel["x"], vel["y"], vel["z"]
    else:
        if pf == "xyz":
            x, y, z = draw("x_mm"), draw("y_mm"), draw("z_mm")
        else:
            r = draw("r_mm")
            th = np.radians(draw("theta_deg"))
            x, y, z = r * np.cos(th), r * np.sin(th), draw("z_mm")
        if vf == "velocity_vector":
            vx, vy, vz = draw("vx"), draw("vy"), draw("vz")
        else:
            d = _norm_dirs(draw("dir_x"), draw("dir_y"), draw("dir_z"),
                           name=vf)
            if vf == "direction_speed":
                spd = draw("speed_mm_per_us")
            elif vf == "direction_ke":
                spd = speed_from_ke(draw("ke_ev"), mass_amu)
            else:
                spd = draw("momentum_amu_mm_us") / float(mass_amu)
            vx, vy, vz = d[:, 0] * spd, d[:, 1] * spd, d[:, 2] * spd

    tob = (sample_scalar(sc["tob_us"], n, name="tob_us", mass_amu=mass_amu)
           if "tob_us" in sc else np.zeros(n))
    return [[float(x[i]), float(y[i]), float(z[i]),
             float(vx[i]), float(vy[i]), float(vz[i]), float(tob[i])]
            for i in range(n)]


def describe(decl, *, mass_amu):
    """One compact line per declared plane, for the UI's derived read-out.

    Emittance is surfaced because it is the INVARIANT: size and divergence
    trade against each other as a beam drifts while their product at the
    waist does not. Twiss appears here, derived and read-only, so the
    vocabulary is available to anyone who wants it without being a
    prerequisite for anyone who does not.
    """
    out = []
    for ax, p in (decl.get("planes") or {}).items():
        eps, beta, alpha = envelope_to_twiss(
            p["size_mm"], p["divergence_mrad"], p["focus_mm"])
        out.append(
            f"{ax}: size {float(p['size_mm']):.3g} mm · divergence "
            f"{float(p['divergence_mrad']):.3g} mrad · focus "
            f"{float(p['focus_mm']):.4g} mm  →  ε {eps * 1e3:.4g} mm·mrad "
            f"(β {beta:.4g} m, α {alpha:.4g})")
    return out


def _shape_of(v, *, uniform_kurtosis=-0.8):
    """Read a sample's distribution shape from its excess kurtosis.

    Uniform is -1.2, gaussian is 0; the threshold sits between them. This
    is a two-way classification, deliberately: it exists to stop the
    derivation ASSUMING gaussian, not to identify arbitrary distributions.
    Anything genuinely other will show up as a large residual, which is
    reported rather than hidden.
    """
    v = np.asarray(v, dtype=float)
    sd = float(np.std(v))
    if sd <= 0 or v.size < 4:
        return "gaussian"
    z = (v - float(np.mean(v))) / sd
    k = float(np.mean(z ** 4)) - 3.0
    return "uniform" if k < uniform_kurtosis else "gaussian"

def derive_declaration(rows, *, mass_amu, axis=None, seed=7):
    """Fit a `beam_envelope` declaration to an EXISTING packet.

    WHY THIS EXISTS: a packet declared as literal rows cannot be
    resampled -- there is no distribution to draw more from. The
    motivating case is exactly that: a certified beam of 200 literal rows,
    and building a statistically useful Impact Analysis histogram wants
    thousands of ions FROM THE SAME DISTRIBUTION, not from a new one
    invented in the panel.

    Deriving turns the rows into a DESCRIPTION of the beam they sample:
    per-plane size, divergence and focus distance, plus the axial energy.
    Regenerating from it at any N then draws the same beam more finely.

    THE FIT IS LINEAR AND SAYS SO. Three numbers per plane describe an
    ellipse. A packet carrying real aberration is a curved phase-space
    filament, and no three numbers capture it -- `residual_frac` is
    returned per plane so the caller can see how well the ellipse
    actually describes the rows rather than assuming it does. A large
    residual is a finding, not a failure to hide.

    The propagation axis is inferred as the component carrying the
    largest mean speed unless given: that is the direction the beam
    travels, which is what divergences are angles relative to.
    """
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] < 6:
        raise BeamSpecError(
            "derive_declaration needs rows of [x,y,z,vx,vy,vz,(tob)]; "
            f"got shape {a.shape}")
    n = a.shape[0]
    if n < 3:
        raise BeamSpecError(
            f"cannot fit a distribution to {n} ion(s); at least 3 are "
            f"needed for a spread to mean anything")
    pos = {"x": a[:, 0], "y": a[:, 1], "z": a[:, 2]}
    vel = {"x": a[:, 3], "y": a[:, 4], "z": a[:, 5]}
    if axis is None:
        axis = max(("x", "y", "z"), key=lambda k: abs(float(np.mean(vel[k]))))
    v_ax = vel[axis]
    if np.any(v_ax == 0):
        raise BeamSpecError(
            f"{int(np.sum(v_ax == 0))} ion(s) have zero velocity along the "
            f"propagation axis {axis!r}; their divergence angle is "
            f"undefined, so the beam cannot be described this way")
    speed = np.linalg.norm(a[:, 3:6], axis=1)
    ke = ke_from_speed(speed, mass_amu)

    planes, resid = {}, {}
    for ax in ("x", "y", "z"):
        if ax == axis:
            continue
        _cen = float(np.mean(pos[ax]))
        q = pos[ax] - _cen
        qp_all = vel[ax] / v_ax                  # angle, radians
        _mean_ang = float(np.mean(qp_all))       # DESIGN, not noise
        qp = qp_all - _mean_ang
        s = float(np.std(q))
        sp = float(np.std(qp))
        if sp <= 0.0:
            planes[ax] = {"size_mm": s, "divergence_mrad": 0.0,
                          "focus_mm": 0.0, "seed": int(seed),
                          "center_mm": _cen,
                          "mean_angle_mrad": _mean_ang * 1.0e3}
            resid[ax] = 0.0
            continue
        # <q q'> = d * sigma'^2 at distance d from the waist: the SAME
        # relation the forward direction uses, inverted.
        cov = float(np.mean(q * qp))
        d = cov / (sp * sp)
        planes[ax] = {"size_mm": s, "divergence_mrad": sp * 1.0e3,
                      "focus_mm": d, "seed": int(seed),
                      "center_mm": _cen,
                      "mean_angle_mrad": _mean_ang * 1.0e3,
                      # SHAPE IS READ FROM THE DATA, not assumed. Excess
                      # kurtosis near -1.2 is the uniform signature, near
                      # 0 the gaussian one; guessing wrong is what made R
                      # depend on n.
                      "distribution": _shape_of(q)}
        # How much of q the linear model does NOT explain. Correlation
        # coefficient r; residual fraction sqrt(1 - r^2) is what is left
        # over after the best straight line through phase space.
        r = cov / (s * sp) if s > 0 else 0.0
        r = max(-1.0, min(1.0, r))
        resid[ax] = float(math.sqrt(max(0.0, 1.0 - r * r)))

    ke_mean, ke_sd = float(np.mean(ke)), float(np.std(ke))
    if ke_sd <= 0:
        ke_dist = {"kind": "single", "value": ke_mean}
    elif _shape_of(ke) == "uniform":
        h = ke_sd * math.sqrt(3.0)
        ke_dist = {"kind": "uniform", "min": ke_mean - h,
                   "max": ke_mean + h, "seed": int(seed)}
    else:
        ke_dist = {"kind": "gaussian", "mean": ke_mean, "sigma": ke_sd,
                   "seed": int(seed)}
    decl = {"n": int(n), "seed": int(seed),
            "position_format": "beam_envelope",
            "velocity_format": "beam_envelope",
            "axis": axis,
            "axis_sign": (-1.0 if float(np.mean(v_ax)) < 0 else 1.0),
            "planes": planes,
            "scalars": {"ke_ev": ke_dist,
                        f"{axis}_mm": {"kind": "single",
                                       "value": float(np.mean(pos[axis]))}},
            "derived_from": {"n_rows": int(n),
                             "residual_frac": resid,
                             "note": "fitted to an existing packet; a "
                                     "linear phase-space ellipse cannot "
                                     "represent aberration"}}
    if a.shape[1] >= 7:
        tob = a[:, 6]
        decl["scalars"]["tob_us"] = (
            {"kind": "single", "value": float(np.mean(tob))}
            if float(np.std(tob)) <= 0 else
            {"kind": "gaussian", "mean": float(np.mean(tob)),
             "sigma": float(np.std(tob)), "seed": int(seed)})
    return decl
