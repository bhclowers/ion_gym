<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 12. Limitations

The following are properties of the ion_gym approach, not defects. Knowing them is the difference between a simulation you can defend and one you cannot.

- **Electrostatics only.** Laplace's equation with conductors at fixed potentials. No magnetic fields. No dielectric materials. No space charge, so the results are single-ion results: they do not describe a beam dense enough for ion-ion repulsion to matter, and they will not reproduce space-charge limits on transmission or trapping capacity.

- **A uniform grid.** Everything is discretized at one pitch. Accuracy is governed by that pitch, and geometric detail below roughly two cells is not resolved. There is no local refinement, so a fine feature inside a large domain forces a fine pitch everywhere and the cost grows as the cube of the refinement in three dimensions.

- **Voxelized surfaces.** A curved or angled electrode becomes a staircase of grid cells. This matters most where the surface field matters most — near sharp features and in narrow gaps — and it is a discretization error that no choice of gradient method removes.

- **A fixed time step.** The integrator does not adapt. The error of a Runge-Kutta step grows steeply with how much the force changes within the step, so the step must be chosen against the fastest motion present — the RF period in a driven device, the sharpest field gradient in a static one — not against the transit time. An adaptive integrator would detect a too-coarse step and shrink it; this one will integrate an undersampled oscillation into a smooth, plausible, wrong trajectory. Verifying convergence is therefore the user's responsibility, and the check is mechanical: halve Δt, re-fly, and confirm the observable you are reporting does not move (section 7.1).

- **Statistical collisions. The gas model is a diffusion-and-mobility treatment valid in the many-collisions-per-step regime. A single uniform drift of the gas can be declared, but there is no computed flow field: jets, expansions, and pressure gradients across one region are out of scope, and a step change in pressure between the stages of an assembly is an idealization of a differential-pumping aperture, not a simulation of one.**

- **The pseudopotential is a picture.** As described in section 8 it is an adiabatic approximation used for display. Its RF term is built from the field gradient within the displayed plane, so on a slice through a structure that varies steeply out of that plane it understates the true confinement.

- **Symmetry is a declaration.** A mirror plane is applied because the specification asserts it. Asserting a symmetry the voltages do not actually have produces a field that is wrong in a way the solver cannot detect.
