# GLOSSARY — ion_gym

Terms of art used across the modules, gates, and documentation. When a
number in a definition and a solve disagree, **the solve wins** — this file
is orientation, not authority.

---

## Run messages

**Numerical advisory** — a `[ADVISORY]`-prefixed note that a *numerical
accuracy* consideration applies to the run being built: a curved surface
voxelized at the declared pitch carries an O(h) surface-position term, a
point source has no declared extent to check derivatives against, a
collision timestep sits near the mean free time. Three properties are
the contract: an advisory **never
blocks** (contrast a *refusal*, which names what did not line up and
stops); it states the **measured or derivable magnitude** where one
exists, not vibes; and it prints **once per unique message per
process** — every emitter routes through one deduplicating door, so
advisories inform without nagging. If a certified number depends on the
advised effect, the advisory tells you what to demonstrate (an
h-plateau, a measured spread, a dt ladder) rather than asking you to
trust.

**Detector plane** — where the score lives. Optimize for the endpoint
where ions actually land; every field-array face, crop, and seam is
internal plumbing, invisible to any objective.

---

## Physics terms

**Space focus** — the plane where ions born at different positions along
the extraction axis arrive simultaneously to first order (deeper ions gain
more energy and catch up). Closed-form for two fields (Wiley & McLaren
1955); `ion_gym/physics/w1_wiley.py` generalises it exactly to any plate
table. Slot penetration (below) moves the focus away from the ideal
plate-table value, so a focus distance is finished on the solved field.

**Source temperature (`temperature_k`)** — the Ion Source tab's
`temperature (K)`: an isotropic Maxwell–Boltzmann thermal spread of the
birth velocities. Each lab axis receives an independent Gaussian kick
with σ = √(kT/m), SUPERPOSED on the directed beam KE, the per-axis
`dv_fwhm_ms`, and `v_drift_mm_us` (a beam AT a temperature).
**0 K = off**: an idealized, perfectly cold beam. In a TOF that deletes
the turnaround term (next entry) — usually the resolution-limiting
aberration of an orthogonal-acceleration source — so a 0 K TOF resolution
is optimistic by construction and must say so when quoted. Real beams are
rarely isotropic (drift and transverse axes sit at different effective
temperatures); for such a source declare per-axis `dv_fwhm_ms` instead.

**Turnaround time** — the irreducible TOF spread: thermal twins ±v_s are
identical after extraction except for Δt = 2·m·v_s/(q·E_fill). No downstream
optic can remove it, so it sets a ceiling on resolving power,
Rp_max ≈ T/(2Δt), for a given source temperature and flight time T.

**Slot penetration** — field leaking through plate apertures, pulling the
on-axis potential off the electrode voltages. It is large whenever the
apertures are not thin compared with the plate thickness and the plate
gaps. Consequence: plate-table analytics are for scaling and sanity;
**every operating point is finished on the solved field**.

**Emittance** — conserved phase-space area (size × divergence).
Conservation is why a beam that is "small but divergent" grows to
millimetres over a long drift, and why optics can trade shape but never
shrink the product.

**Rp (resolving power)** — `Rp = median(T) / (2·FWHM)` with a robust FWHM
of the arrival-time distribution. A quoted Rp without its operating point
(deck, filter/aperture, flight length, tune, convention) is not a number.

---

## Assembly / model machinery

**FA (field assembly)** — one solved planar field: geometry mask +
per-electrode basis fields on that instance's own grid. Solved once, cached
on disk keyed by **geometry alone** (voltage retunes reuse the solve).

**Assembly (multi-FA)** — declarative multi-field composition: instances =
{spec_file, origin, rotation, priority, voltage_scale}. Fields are NEVER
meshed/resampled/blended — the kernel locates the owning instance,
transforms the ion in, samples bilinearly on that instance's own grid, and
rotates E back. Rigid transforms only; relative electrode positions never
change. Outside every instance: exactly field-free.

**field_extent_mm (crop)** — an instance-level crop of the region where its
field acts: the flight truncates the field where a composed model does,
instead of splatting there or flying the leftover live fringe. A FIELD
crop, not a geometry edit — moves no electrode, re-solves nothing; refuses
to reach outside the solved box.

**Seam vs wall** — a seam is a face where the flight truncates field; it is
only legitimate where that field is dead. A face made of metal is a WALL —
ions terminate on it, nothing is truncated — and must never be scored as a
dead seam (it reads |E| = 0 for the wrong reason: the sample is inside a
conductor). Measure |E| over the vacuum fraction only, one grid unit
INSIDE the face (ON the face the gradient stencil reads a flattering zero).

**is_grid** — an electrode that is a Dirichlet boundary in the solve but
TRANSPARENT to ions (mesh, gridless-mirror entrance column, detector face,
gauge-pinning drift copper). Without the flag, every such plane is a wall.

**Fates** — every ray terminates in a named status (`flyer_plane`:
EXIT / HIT / LOST / STUCK). No silent ray fates.

**Gate** — a test that certifies one claim against an analytic or
independent reference, refuse-with-diagnostic on failure, numbers LOCKED so
they cannot drift unnoticed. Subsystem certifications are never quoted as
instrument performance.
