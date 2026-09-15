# GLOSSARY — Fable / ion_gym

Terms of art used across the modules, gates, and handoffs. Numbers quoted are
from the certified contract (`outputs/instrument_spec_v103.json`) or the gate
that measured them; the citation says which. When a number here and a solve
disagree, **the solve wins** — this file is orientation, not authority.

---

## The instruments

**Fable** — the instrument under design: compact PCB-based planar orthogonal-
acceleration TOF (oa-TOF); gridless transaxial reflectron (Bimurzaev /
Verenchikov style); 16 mm analyzer sandwich gap; target analyte m/z 524;
grounded source side, flight side depressed to −4 kV.

**V2 stack** — the current **frozen** revision of Fable's extraction /
acceleration column. Defining change from V1: **plate thickness 1.6 → 0.6 mm
metal**, which cut exit emittance ~4× (plate-channel aspect ratio was the
dominant aberration source). Configuration of record
(`immersion_stack.configure_v2()`):

| n | node | role            | front–back (mm, s-frame) | slot L × W (mm) |
|---|------|-----------------|--------------------------|-----------------|
| 0 | A    | GND shield      | −0.60 – 0.00             | solid           |
| 1 | B    | PUSH (solid)    | 3.00 – 3.60              | solid           |
| 2 | A    | GND slit        | 6.60 – 7.20              | 16.0 × 1.1      |
| 3 | C    | PULL            | 8.10 – 8.70              | 18.0 × 1.5      |
| 4 | M1   | immersion 1     | 12.70 – 13.30            | 18.0 × 3.0      |
| 5 | M2   | immersion 2     | 17.30 – 17.90            | 18.0 × 3.5      |
| 6 | M3   | immersion 3     | 21.90 – 22.50            | 20.0 × 4.0      |
| 7 | F    | EXIT (−4 kV)    | 26.50 – 27.10            | 20.0 × 4.0      |
| 8 | F    | drift entrance  | 29.60 – 30.20            | 20.0 × 4.0      |

STACK_END = 30.20, S_BEAM = 5.10, S_EXIT = 34.20 (all s-frame, mm).
Certified tune: Vp = 360.76 V, grading f = (0.0910, 0.3409, 0.5180) →
extract-state nodes A 0 / B +360.8 / C −360.8 / M1 −691.9 / M2 −1601.4 /
M3 −2245.8 / F −4000.0 V. **Geometry is frozen; changes need explicit budget
authorization.** The slots in these plates are the instrument's physical
apertures (per the PI: apertures = the slots; the reflectron is gridless; one
grid sits in front of the MCP as a transmission factor only).

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

**Reference stack (GB2605116B)** — the 12-plate reference extraction from
the patent, modeled in the internal `gb_stack` module: 0.5 mm plates, ±944 V push/pull, −8600 V
flight, 60.25 mm long, stadium (obround) slots. It is the extraction of the
**2 m instrument** that physically achieves **Rp ≈ 30,000 with a two-stage
reflectron**. Role: *method validation only* — the one stack where geometry,
voltages, and physical performance are all known. The 30k headline is NOT a
reproduction target (the two-stage mirror has no drawings); only the
stack-side terms are checked (W1e, W2).

**Older target-chamber stack** — a previous compact-frame instrument that
physically achieved **Rp ≈ 4,500**. No geometry or voltages survive. Role:
*existence proof* that 4,500-class is achievable at this flight-length class,
and therefore **the floor Fable must clear** — a lower bound, not the goal.

---

## Frames and axes

**s / u / v** — s = extraction (TOF) axis; u = width-plane transverse
coordinate (across the 16 mm gap); v = along the slot length (the original
beam direction; ballistic in the width-plane model — the length-plane 2-D
slice is INVALID, off ~3900× from wall screening, Gate B).

**s-frame vs FA x-frame** — `immersion_stack` puts plate 0's front at
−PLATE_T; the exported field array puts the whole stack at a margin inside its domain.
Offset = **3.6 mm** (x = s + 3.6); it closes on the exit plane
(34.2 + 3.6 = 37.8 = `bounds.x_max`). Derive it from the two geometries,
never assume it (W1 found a fan launched 3.6 mm inside the push plate).

**S_BEAM** — nominal beam line: mid fill-gap between the push plate and the
ground slit, where the injected beam sits when the pulse fires (5.10 mm,
s-frame). The birth slab is ±0.5 mm around it.

**S_EXIT** — the hand-off plane, 4 mm past the last plate (34.20 s-frame /
37.80 FA-frame). The composed model stops using the stack field here; the
bench reproduces that with a **field-extent crop** (see below). NOT a dead
seam: 1.32 V/mm live, φ +8.7 V off the flight gauge, 1.29 V across the beam
(M-FA2).

**Detector plane** — where the score lives. Doctrine: optimize for
the endpoint where ions actually land; every FA face, crop, and seam is
internal plumbing, invisible to any objective.

---

## Voltage program

**Vp** — push/pull pulse amplitude: B → +Vp, C → −Vp in extract state.
Certified 360.76 V.

**f1, f2, f3 (grading)** — fractions placing M1–M3 on the pull → drift drop:
`V_Mi = −Vp + fi·(V_drift + Vp)`. They set the space-focus distance (the f1
knob: +66…+101 mm flown bracket) and — as M-FA3 exposed — the exit
divergence, which the old objective never priced.

**extract / fill** — the two pulser states. Fill: B = 0, C ≈ +13.4 V (the
nulled C\*, cancelling leak through the pull slot) while the beam drifts in
along v. Extract: B/C pulse to ±Vp. M1–M3 and F are DC in both states.

**Kv / Kz** — beam energy along v entering the pusher: a bounded design
parameter, 22–34 eV, with ΔKv ≈ 1–2 eV (gas-dynamic) as a mandatory ledger
row. Axial energy must stay cold (~22 eV) through the transfer optics; all
acceleration belongs to the OA stage. Detector v-offset scales as
√(Kv/Kx)·L.

**Voltage-scaling invariance (mirror gauge)** — the mirror tune is quoted at
KE 4000 eV reference and scales ×(K_deck/4000). The composed model applies
this by scaling *time*; a bench flies real ions so it must scale the
*voltages* (flown beam arrives at 4166 eV → ×1.0415). Skipping it costs ~3×
in Rp (M-FA3).

**Certified mirror tune** — V_M 3835 / V_R 4465 V (at the 4000 eV gauge).
`mirror_plane`'s module constants (3794 / 4505) are the ORIGINAL zone
voltages, not the tune — `mirror_export.spec()` takes v_m/v_r explicitly and
stamps them into the spec name so a wrong-tune run is visible on its face.

**Trim strips** — the R-zone segmentation in the certified spec (≥8
v-segments, ≤7.5 mm pitch, ±10 V, per-Kv nulled lookup). Present in the
contract, never yet flown on the bench; a free parameter set for the Frame-3
reflectron optimization.

---

## Physics terms

**Space focus** — the plane where ions born at different s0 arrive
simultaneously to first order (deeper ions gain more energy and catch up).
Closed-form for two fields (Wiley & McLaren 1955); `w1_wiley.py` generalises
it exactly to any plate table. V2 at the certified tune: **84.1 mm** past
S_EXIT (ideal plate table) vs **119.0 mm** (solved field / flown) — the
+34.9 mm shift IS slot penetration (W1c).

**Source temperature (`temperature_k`)** — the Ion Source tab's
`temperature (K)`: an isotropic Maxwell–Boltzmann thermal spread of the
birth velocities. Each lab axis receives an independent Gaussian kick
with σ = √(kT/m), SUPERPOSED on the directed beam KE, the per-axis
`dv_fwhm_ms`, and `v_drift_mm_us` (a beam AT a temperature; superposition
semantics 2026-08-11 — before that, T > 0 silently replaced the beam).
**0 K = off**: an idealized, perfectly cold beam. In a TOF that deletes
the turnaround term (next entry) — usually the resolution-limiting
aberration of an OA source — so a 0 K TOF resolution is optimistic by
construction and must say so when quoted. Real beams are rarely
isotropic (drift and transverse axes sit at different effective
temperatures); for an OA-MRT declare per-axis `dv_fwhm_ms` instead.

**Turnaround time** — the irreducible TOF spread: thermal twins ±v_s are
identical after extraction except for Δt = 2·m·v_s/(q·E_fill). No downstream
optic can remove it. Flown **1.276 ns** (12 K, m/z 524; analytic 1.243 ns) —
sets the hard ceiling Rp_max ≈ T/(2Δt) ≈ **11,000-class** for this source
temperature and flight length (W1b/W1d).

**Slot penetration** — field leaking through the plate apertures, pulling
the on-axis potential off the node voltages (up to ~100 V in V2, because
3–4 mm slots against 0.6 mm plates and 4 mm gaps are NOT thin). Consequence,
binding: plate-table analytics are for scaling and sanity; **every operating
point is finished on the solved field** (W1c).

**Emittance** — conserved phase-space area (size × divergence). V2 exit:
σ_u ≈ 0.037 mm × 4.63 mrad ≈ 0.17 mm·mrad. Conservation is why "small but
divergent" becomes ±2 mm after 425 mm of drift, and why optics can trade
shape but never shrink the product.

**Waist (w)** — in `coopt_E4`, the beam half-size the composed model ASSUMES
at the mirror via an ideal re-imaging optic (`layout.waist_w_mm`, a free
co-optimised parameter, chosen from (0.08, 0.13, 0.2, 0.3)). M-FA3's finding:
no solved FA delivers it — the certified Rp 6,126 is conditioned on transfer
optics that were never designed. Abolished from all future objectives: the
beam at the mirror is whatever the flown stack + drift produce.

**Rp (resolving power)** — certified convention, reused verbatim everywhere:
`Rp = median(T) / (2·hypot(FWHM_robust, T_FLOOR))`, FWHM by
`trade_table_E4.robust_fwhm`. A quoted Rp without its operating point (deck,
filter/aperture, D, tune, convention) is not a number.

**D (external drift)** — the TOTAL external flight path, out + back:
certified 850.31 mm (one-way stack→mirror 425.15 mm — what fits the 635 mm
chamber). It is the **argmax of a scan**, and simultaneously the
chamber-matched maximum: no retune headroom.

**Turnaround ceiling / budget arithmetic** — at T ≈ 29 µs and Δt_turn ≈
1.28 ns per twin (≈2.6 ns width), the source-temperature ceiling is
~11,000-class; the 4,500 floor therefore has ~2.4× headroom that transverse
transport + mirror must not eat.

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
field acts (option (a)): the bench truncates where the composed model
truncates (S_EXIT), instead of splatting there (`bounds_mode='inherit'`) or
flying the leftover 2.9 mm of live fringe (`'seam'`). A FIELD crop, not a
geometry edit — moves no electrode, re-solves nothing; refuses to reach
outside the solved box.

**Seam vs wall** — a seam is a face where the bench truncates field; it is
only legitimate where that field is dead. A face made of metal is a WALL —
ions terminate on it, nothing is truncated — and must never be scored as a
dead seam (it reads |E| = 0 for the wrong reason: the sample is inside a
conductor). `seam_report()` classifies every face (`metal_frac`, `is_seam`)
and measures |E| over the vacuum fraction only, one grid unit INSIDE the
face (ON the face the gradient stencil reads a flattering zero).

**is_grid** — an electrode that is a Dirichlet boundary in the solve but
TRANSPARENT to ions (mesh, gridless-mirror entrance column, detector face,
gauge-pinning drift copper). Silently ignored by the builder until
2026-07-12 — every such plane was a wall.

**Fates** — every ray terminates in a named status: EXIT / HIT / LOST /
STUCK. No silent ray fates.

**Composed model** — the certified pipeline (`coopt_E4` / `spec_loader`):
stack flown to S_EXIT, then mirror TOF **interpolated** from a precomputed
energy fan + u-correction, then D/v drift with an assumed waist. The thing
the flown bench replaces — and interrogates.

**Gate** — a pytest that certifies one claim against an analytic or
independent reference, refuse-with-diagnostic on failure, numbers LOCKED so
they cannot drift unnoticed. Subsystem certifications are never quoted as
instrument performance.

**W1 / W2 / W3** — the validation ladder for the stack frame: W1 = our
geometry vs its own analytics (DONE, green); W2 = reference-stack-side terms at
2 m (turnaround at ±944 V, exit emittance/divergence as the proven-design
reference, one-sided check that the stack is not the 30k bottleneck); W3 =
deck audit (SIG, slab, u-spread vs the cooler's actual output — the one
physical input all three instruments share). Then: derive Fable's stack
point on the solved field; then Frame-3 reflectron optimization (V_M, V_R,
trim strips) scored at the detector, ≥4,500 floor through real apertures.
Prerequisite for Frame 3: mirror-export dT/dK < FWHM budget (currently
4.0 ns vs 1.82 ns — gate I4 refuses the composition question until fixed).
