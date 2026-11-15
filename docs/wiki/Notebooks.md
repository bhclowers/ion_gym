<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# Notebooks

The teaching series (00 to 11) builds from the Laplace equation to a multi-stage instrument. The V series checks the physics against known answers.

### 00 — The Laplace equation: why every instrument in this series starts with it, and how a computer actually solves it

`notebooks/00_laplace_numerics.ipynb`

Every notebook after this one does the same first step: *solve the field*. Before trusting that step, you should know what equation is being solved, why it is the right equation, and what the computer is actually doing while you wait. This notebook builds all three, from Coulomb's law to a working solver you write …

### The Einzel Lens: solve, fly, and tune

`notebooks/01_einzel_lens.ipynb`

**Purpose.** Walk the `ion_gym` code path module by module, outside the GUI, on the simplest shipped device: the round einzel lens (r-z, native primitives, no external simulator). Each stage is a separate, plain code block, deliberately kept independent so an optimization loop (CMA-ES or otherwise) can later call the …

### The RF Quadrupole: oscillations, mass filtering, and the Mathieu diagram

`notebooks/02_quadrupole.ipynb`

**Purpose.** The einzel notebook was electrostatics: solve once, fly. This one adds **time**: a radio-frequency drive that confines ions with a field that has no static minimum at all. We look at the CAD meshes in 3-D, fly ions released *off-axis* so their oscillations show, watch two masses meet opposite fates in the …

### The Ion Funnel: moving ions through a gas

`notebooks/03_ion_funnel.ipynb`

* The funnel geometry: a stack of rings whose apertures shrink, driven by RF on alternating rings and a DC ladder along the stack. * Why the RF confines rather than ejects — the pseudopotential wall, seen in the solved field. * **Integration time versus record time**: `dt_ns` sets the physics, `rec_every` sets only …

### Field Export: any device, one drive cycle

`notebooks/04_field_export.ipynb`

* Loading a geometry from JSON and reading its electrodes and groups. * Declaring **per-electrode voltages** and **per-group frequencies** interactively — nothing is inherited silently from the file. * Choosing **solver parameters**: the field method (electrode-aware vs plain gradient) and the working precision …

### Trajectory Files: save once, analyse anywhere

`notebooks/05_trajectory_files.ipynb`

* **Where these files live** and what is inside one. * Loading a run back and plotting the paths — without re-flying anything. * **Kinetic energy along each axis** ($x$, $y$, $z$) — derived exactly from the stored channels, which answers a natural worry: only the *net* KE is stored as a channel, but the full velocity …

### 06 — Why real Paul traps are the "wrong" shape: holes, stretched endcaps, and the sign of the field error

`notebooks/06_paul_trap_geometry.ipynb`

**Who this is for.** A 1st- or 2nd-year chemistry or physics graduate student who has met the quadrupole (notebook 02) and is now looking at a 3-D trap.

### Time-lag (delayed-extraction) autotune

`notebooks/07_time_lag_autotune.ipynb`

**Purpose.** Derive and demonstrate Wiley–McLaren time-lag focusing on the refined oa-TOF example — *outside* the GUI (co-tuning is a demonstration, not a UI feature). The GUI exposes the knobs (pulse delay τ on the Voltages tab, source temperature on the Ion Source tab). **This notebook shows why the knobs must move …

### 14 — Authoring geometry natively: the shapes route

`notebooks/08_shapes_io.ipynb`

**Who this is for.** Anyone who wants to build an instrument in `ion_gym` and has been asking "where do I draw the electrodes?"

### 09 — DRIFT-OPT: the stacked-ring PCB drift-tube design matrix

`notebooks/09_drift_tube_matrix.ipynb`

A drift-tube optimisation study, twice amended: automated design of a stacked-ring PCB drift tube with the **literature as referee**. Architecture: rings are PCB boards (axial width = board thickness), insulating spacer boards with **recessed ID** between them. Stock set W7 = {0.4, 0.5, 0.8, 1.0, 1.2, 1.6, 2.4} mm …

### 27 — The Orbitrap: orbital trapping, image current, and the Fourier route to m/z

`notebooks/10_orbitrap.ipynb`

In 1923 Kingdon showed that ions attracted to a thin charged wire can be trapped on stable orbits around it. Knight (1981) added shaped end electrodes to superimpose an axial potential well, and Makarov (2000) perfected the idea into the **Orbitrap**: a spindle-shaped central electrode inside a barrel-shaped outer …

### 11 — A multi-FA instrument, end to end, through the API

`notebooks/11_multi_fa_instrument_api.ipynb`

This notebook flies a complete two-stage ion-optical instrument — an **ion funnel** feeding a **hexapole ion guide** — entirely from Python, with no dashboard in the loop. If you have never met either device:

### V01 · The field solver against closed-form Laplace solutions

`notebooks/V01_fields_vs_laplace.ipynb`

**Purpose.** First notebook of the **validation series**: demonstrate, against *analytic* electrostatics (not against another simulator), that the `ion_gym` field solve is correct, and quantify its error and convergence order at working resolutions. Three cases with closed forms: parallel plates (linear potential), …

### V02 · Mathieu stability & secular frequencies

`notebooks/V02_mathieu_stability.ipynb`

**Purpose.** V03 anchored the direct-RF integrator against exact Floquet theory at four q points and one stability bracket. V02 is the **enumerated coverage** around those anchors: a dense q ladder along a=0, the a≠0 axis that a DC quadrupole offset opens (untouched by V03), an m/z ladder set by one calibration, and …

### V03 · Pseudopotential vs direct RF

`notebooks/V03_pseudopotential_vs_rf.ipynb`

**Purpose.** `pe_surface` draws the Dehmelt effective potential — a *visualization concept*. The tracer always integrates the real time-varying field. This notebook proves the picture is quantitatively trustworthy where its own docstring claims validity (adiabatic regime, Mathieu q ≲ 0.4), measures exactly how it …

### V04 · Time of flight

`notebooks/V04_time_of_flight.ipynb`

**Purpose.** Certify the pipeline's **timing fidelity** — the quantity every future resolution claim rests on. Four contracts: (1) flight time in a uniform field against the closed form, with the dt-convergence table, (2) the √m calibration law, (3) the Wiley–McLaren two-stage space focus against the piecewise …

### V05 · Thermalization against Maxwell-Boltzmann — three flight paths

`notebooks/V05_thermalization.ipynb`

**Purpose.** Validate collisional thermalization on ALL THREE flight paths, each against the analytic truth of its own collision contract. Ions flown in neutral gas with **no fields** must relax to the gas temperature. What "relax" means depends on the path's documented model:

### V06 · Transport and the Einstein relation

`notebooks/V06_transport_einstein.ipynb`

**Purpose.** Validate ion transport in gas: drift velocity linear in field (mobility K), diffusion from mean-square displacement (D), and the **Einstein closure D/K = kT/q** — the fluctuation-dissipation statement that ties V05's thermal statistics to directed transport. HS is tested against kinetic theory and the …

### ion_gym — general simulation workbench

`notebooks/sim_workbench.ipynb`

One interactive app for **any** geometry, driven by `build_run(spec)`: the planar (Cartesian) einzel, the round (r-z) einzel, the IMS drift tube, and the funnel all load into the same interface. Nothing external needed to run (only the shipped funnel uses pre-solved bases).
