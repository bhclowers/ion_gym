<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# Examples

Every deck that ships in `examples/`, with the opening of its own notes. Load any of them from the dashboard (Load > Deck > Example) or with `SimSpec.from_json`.

### CDMS ELIT (Gated)

`examples/cdms_elit_gated_rz.json`

REFERENCE: trap design and dimensions from J. A. Hogan & M. F. Jarrold, 'Optimized Electrostatic Linear Ion Trap for Charge Detection Mass Spectrometry', J. Am. Soc. Mass Spectrom. 29, 2086-2095 (2018), DOI 10.1007/s13361-018-2007-x (the cylindrical trap); voltages and beamline from a laboratory beamline drawing. …

### CDMS ELIT

`examples/cdms_elit_rz.json`

Electrostatic linear ion trap for CDMS, r-z symmetry, built from J. A. Hogan & M. F. Jarrold, J. Am. Soc. Mass Spectrom. 29, 2086-2095 (2018), DOI 10.1007/s13361-018-2007-x (cylindrical trap): mirror electrodes l=4.57 mm, ID=13.97 mm, spacing delta=0.127 mm; grounded shields 3.81 mm thick with EXACT beveled bore 13.97 …

### drift_tube_mason_schamp

`examples/drift_tube_mason_schamp.json`

Certified example: Mason-Schamp absolute-rate referee (all four gates PASS). Stacked-ring DC guide, NO RF: 28 rings w 0.5 / pitch 3 / r_in 8 / depth 6 mm (Bohnhorst-2016 ratios w/p 0.167, d/p 2.0), ladder 16.2->0 V = 0.2 V/mm (6.9 Td). Gas: C60 m/z 720 in He, 1.000 Torr, 298 K, sigma 1.25e-18 m^2 from measured Gillig …

### einzel lens (r-z, three-tube) — Abdulla & Hussein 2020 geometry, decelerating mode

`examples/einzel_round_r-z.json`

Three-tube einzel lens, cylindrically symmetric, solved in r-z. Geometry after Abdulla & Hussein (2020): three coaxial tubes each 25 mm long, bores 30/24/30 mm diameter, with 6 mm axial overlap between adjacent tubes -- the tubes interleave rather than butting end-to-end, which keeps the gap fields short. Outer tubes …

### hexapole — native 3-D (6 round rods, extruded, no STL)

`examples/hexapole_native_3d.json`

NATIVE ion_gym 3-D hexapole: 6 round rods, radius 1.498 mm, rod-centre radius 4.29 mm at 30+k*60 deg; inscribed field radius r0 = 2.792 mm; length 50 mm. AUTHORED IN SIGNED WORLD COORDINATES, centred on the origin: rods at +-R about (0,0), domain [-7.8,+7.8] mm in x,y via origin_mm=[-7.8,-7.8], z in [0,50]. Native …

### Ion Funnel (r-z)

`examples/ion_funnel_rz.json`

Stacked-ring ion funnel with a conductance-limit exit, r-z symmetric. Geometry of record: 18 ring electrodes, thickness 0.6 mm, gap 1.4 mm (pitch 2.0 mm), inner diameter tapering 12.5 mm down to 2.50 mm, wall (OD-ID)/2 = 2.00 mm on every ring. The last element is a DC-only conductance limit 'conductance_limit': …

### Kingdon trap (2-D cylindrical, compact)

`examples/kingdon_trap_rz.json`

Kingdon trap: a charged inner rod (-2300 V) on the axis of a grounded can, r-z symmetric. The radial field is logarithmic - no radial minimum - so nothing is trapped by the field alone: ions are held by ANGULAR MOMENTUM, orbiting the rod (launch direction has a tangential component). Compact geometry: the earlier …

### Orbitrap (r-z, quadro-logarithmic) -- 10 mm outer, 3.5 kV

`examples/orbitrap_rz.json`

Orbitrap, cylindrically symmetric, solved in r-z (x = trap axis, y = radius; the kernel carries full 3-D Cartesian state, so the orbital motion is real). Both electrodes are truncated equipotentials of the quadro-logarithmic field U = (k/2)(z^2 - (r^2-R2^2)/2 + Rm^2 ln(r/R2)): central spindle through (R1, 0) at -Vc, …

### Paul trap (r-z, hyperbolic) — Finnigan ITMS commercial geometry, He cooling

`examples/paul_trap_r-z_he_cooling.json`

3-D quadrupole ion trap, cylindrically symmetric, solved in r-z. Geometry after the Finnigan ITMS/ITD as documented in the two trap papers: ring inner radius r0 = 10.00 mm; endcap apex at z0 = 7.83 mm -- the COMMERCIAL geometry, i.e. endcaps moved outward from the theoretical 7.07 mm WITHOUT reshaping, so the surfaces …

### Quadrupole (2-D analytic rods)

`examples/quadrupole_analytic_rods_2d.json`

Four-rod RF quadrupole built entirely from analytic circle shapes — no STL files. Rod radius 4.408 mm at r0 = 3.84 mm (rod/r0 = 1.148, the classic cylindrical approximation); RF 241.34 V zero-to-peak at 2.0 MHz puts m/z 100 at q = 0.40 (stable) and m/z 30 at q = 1.33 (unstable). Both masses are flown together: the …

### quadrupole — STL rods, full 3-D (120 mm)

`examples/quadrupole_stl_rods_full_3-d.json`

The four-rod quadrupole as built: 8.67 mm rods, 120 mm long, solved in full 3-D so the axial transport is real rather than assumed. STL files live in examples/quad_assembly/ (resolved relative to this spec); stl_manifest carries their sha256 so a run verifies the meshes are unchanged. Ions are injected at the entrance …

### quadrupole (STL rods, axial transport)

`examples/quadrupole_stl_rods_transport.json`

Four-rod RF quadrupole (STL rod meshes), flown as a 2-D cross-section with axial transport. Two masses are flown together: the stable m/z stays bounded while the unstable m/z grows and strikes a rod — the Mathieu stable/unstable demonstration in one flight. The rod meshes are pristine origin-centred CAD exports; their …

### reflectron TOF (oa, refined: WM-matched, mirror-imaged)

`examples/reflectron_tof_oa_refined_wm_matched.json`

Domain trimmed to the active region: x sliced to [55, 95] mm of the original frame and re-zeroed (x' = x - 55); 0-55 mm was field-free vacuum. All x positions in the notes below are the OLD frame. Source plates are clipped at the new left edge; certification below proves the beam-region field is unchanged. REFINED …

### reflectron TOF (orthogonal acceleration)

`examples/reflectron_tof_oa_with_detector.json`

Domain trimmed to the active region: x sliced to [55, 95] mm of the original frame and re-zeroed (x' = x - 55); 0-55 mm was field-free vacuum. All x positions in the notes below are the OLD frame. Source plates are clipped at the new left edge; certification below proves the beam-region field is unchanged. …

### reflectron TOF (r-z, hard mirror — defocused contrast)

`examples/reflectron_tof_r-z_hard_mirror.json`

Reflectron TOF (r-z), HARD-mirror contrast case. Same drift, source, and back voltage as the time-focused variant but a short 12-ring mirror: the steep ladder reflects with little energy-dependent penetration, so the return-time spread is far worse. Load both variants side by side — mirror LENGTH is the time-focusing …

### reflectron TOF (r-z, time-focused mirror, vacuum)

`examples/reflectron_tof_r-z_time-focused_mirror.json`

Reflectron TOF (r-z), time-focused mirror, vacuum. Operating point: 160 mm field-free drift + 97-ring linear-ladder mirror (V_back = 1000 V); KE 760–840 eV (±5% about 800 eV), m/z 100. The 2% radial launch slope separates the outbound and reflected legs into a visible chevron in the xy view. Compare with the …

### rf-confining drift cell (Allen & Bush 2016) id 5 mm

`examples/rf_drift_cell_allen_bush.json`

Allen & Bush 2016 geometry: 56 rings x 0.5 mm, pitch 1.5 mm, i.d. 5 mm, r_out 6 mm (ours); RF 200 Vpp @ 2.8 MHz alternating; E/P 0.507 V/cm/Torr -> 0.17745 V/mm, ladder 14.640 V over 82.5 mm; 3.5 Torr He 298 K. ION: GRGDS [M+2H]2+, mass 492.5 Da (mz_list holds the ion MASS in Da on this route; m/z 246.3), z=+2, after …

### surround SLIM (mini, native extruded shapes)

`examples/slim_surround_mini_3-d_native_shapes.json`

NATIVE extruded inline shapes on the shapes3d route — no geometry-file import, no STL: x-y cross-sections (circle rods, rect rails, one cutout bore) each carrying extrude {axis,lo_mm,hi_mm}, z-mirrored about the plane at 0. The SS7p3 surround-SLIM architecture in miniature; stored half solves, display and flight read …

### SLIM 2D

`examples/slim_tetramer_confinement_2-d_acrossxgap.json`

Across-channel x gap confinement plane; ions travel into the screen. RF rails (opposite phase) confine ions in the gap. 100.0 V @ 800 kHz, guard 7.5 V, 2.5 Torr N2.

### SLIM 3D

`examples/slim_tetramer_full_3-d_transport_generic.json`

SLIM tetramer, full 3-D, authored natively as inline extruded shapes (no geometry files). Axes: x = transport (37 mm), y = board gap (mirror axis; half-domain 0..2.05 mm with grid.mirror='y' giving two boards and a 2.75 mm gap between inner faces at y = ±1.375 mm), z = board width (13 mm, five rungs). RF rails confine …
