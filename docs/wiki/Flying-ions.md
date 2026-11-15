<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 7. Flying ions

## 7.1 The integrator

Trajectories are integrated with the classical fourth-order Runge-Kutta method, the same scheme commonly used for trajectory integration **\[1\]**. The field is sampled at the ion's position by trilinear interpolation between the eight surrounding grid nodes, which is what makes the grid pitch the dominant accuracy parameter: the interpolated field is smooth, but it is smooth between samples of a field that was itself computed on that grid.

The time step is declared rather than adapted. This is a current design choice that could be modified in the future, though the present approach is clean and valid. The practical consequence is that the step must be checked against the fastest motion in the problem — for an RF device, against the RF period, not against the transit time. A step that comfortably resolves the drift will silently misrepresent the micromotion.

The reason is worth spelling out. A Runge-Kutta step assumes the force varies smoothly ACROSS the step: its error per step grows steeply with how much the field changes within one Δt. Motion in a 3D RF trap or a tight lens is oscillatory, and sampling an oscillation with too few points per cycle does not merely blur it — it aliases it, producing a trajectory that is systematically wrong (spurious heating or damping, a shifted secular frequency) while looking perfectly smooth. That is why the recommended rule is ~20 steps per RF period, counted against the FASTEST drive present, and why a vacuum TOF drift tolerates a far coarser step than the same ion inside an RF device.

Convergence in Δt is checked the same way convergence is always checked: halve the step, re-fly, and compare the observable you actually care about (a time-of-flight, a transmitted fraction, a spot size). If the answer moves, the first step was too coarse; if it does not, you have measured — not assumed — that the step is converged. The toolkit will not do this for you, deliberately: which observable must be converged is a physics judgment, not a solver setting. However, there are recommendations returned to the user in the UI that estimate the time step needed.

## 7.2 Time-varying drives

Electrodes are grouped, and each group carries a waveform: sine, cosine, square with an adjustable duty cycle, or an arbitrary tabulated shape. The field seen by an ion is the static field plus the sum of each group's field weighted by its waveform at that instant. This is a full time-domain treatment — the RF is integrated as it actually varies, not approximated — and it is what makes traveling-wave devices expressible: a traveling wave is a set of groups with a phase progression.

## 7.3 Collisions with background gas

Two gas models are available, and they occupy opposite ends of the pressure axis. HS treats collisions as discrete events: between collisions the ion flies the electric field unperturbed; at each step a collision probability is computed from the hard-sphere cross-section, the gas density and the relative speed, and when a collision fires, a gas atom is drawn from the thermal distribution and the pair scatter elastically. Every kick is individually resolved, which is exactly right when collisions are sparse — a collision cell at tens of millitorr, a funnel entrance — and increasingly expensive and unnecessary as pressure rises and thousands of collisions occur per RF cycle.

SDS — the Statistical Diffusion Simulation of Appelhans and Dahl \[4\] — is the other end of the axis. Rather than tracking individual collisions, SDS applies two effects per time step: a viscous damping of the acceleration toward the local mobility-limited drift velocity, and a random-walk displacement drawn from tabulated diffusion statistics scaled by the local mean free path. This is a faithful port of the published model, and its assumptions are inherited with it — in particular it is a statistical treatment valid when many collisions occur per time step, not a single-collision model.

The choice between them is therefore a physics statement, not a preference: HS where individual collisions are the phenomenon (energy relaxation, collision counting, low pressure), SDS where the gas acts as a continuum drag-and-diffusion bath (mobility-dominated transport at roughly a Torr and above). Running SDS at a pressure so low that few collisions occur per step violates its founding assumption, and running HS deep into the mobility regime is paying per-collision cost for a statistically determined answer.

Both models read the same declaration. Pressure is authored in torr (P_torr — the canonical pressure field; the derived pascal field of older files is retired), temperature in kelvin, and the gas is chosen by name from the shared table — He, H2, N2, air, Ar, CO2, Kr, Xe — or characterized explicitly by cross-section (HS) and molecular diameter (SDS). A uniform drift of the gas can be declared as a flow vector (collisions.flow_mm_us, in mm/µs): collisions are then sampled in the moving gas frame on every route, which is how a drift tube's counterflow or a funnel's carrier flow enters the physics. A planar (2-D) model refuses a flow component along its unsolved axis rather than silently projecting it away.

Two measured properties of these models are worth knowing before quoting numbers from them. HS timing observables carry a small O(Δt) bias — roughly 1.5–3% per nanosecond of step on reference devices — so demonstrate a Δt plateau before certifying a drift time. SDS reproduces realistic mobilities, but its diffusion does not close the Einstein relation: D/K measures 2.1–2.3 × kT/q on this implementation, a property of the published model itself (empirically fitted drift paired with geometric displacement tables), so SDS spreading estimates carry that factor by construction. Notebook V06 measures both statements against the shipped kernels and asserts bands around them, so a silent change to either model fails a notebook rather than a reader.

Bookkeeping follows the physics: under HS each ion's collision count is recorded (n_col); under SDS it is deliberately absent — the model has no discrete collision events to count, and recording 0 would read as a vacuum flight. The application likewise derives the two step-size ceilings — resolving the fastest drive, and sampling collisions no more often than about one per step — from the declared gas and drive, and warns when the declared Δt violates either; a deck flying SDS below its validity regime (too few collisions per RF cycle) gets a named advisory quoting the number it computed.

## 7.4 Units

One unit convention runs through the whole toolkit. Mixing these up is the most common source of results that are wrong by exactly a factor of a thousand:

| **Quantity**           | **Unit**                                                                                |
|------------------------|-----------------------------------------------------------------------------------------|
| Length                 | millimeters (mm)                                                                        |
| Time                   | microseconds (µs)                                                                       |
| Velocity               | mm/µs (numerically equal to km/s)                                                       |
| Potential              | volts (V)                                                                               |
| Electric field         | volts per millimeter (V/mm)                                                             |
| Energy                 | electronvolts (eV)                                                                      |
| Mass                   | daltons (Da); the ion mass list carries masses, and charge is a separate signed integer |
| Pressure / temperature | Torr / kelvin                                                                           |

The physical constants behind all of this live in one module (ion_gym.physics.constants — CODATA 2018 and exact SI-2019 values): every kernel, builder and notebook reads the same e, kB, NA and atomic mass constant, so a quantity computed two ways cannot disagree in the fifth digit because two files remembered different vintages.

## 7.5 Stations: detectors and apertures

A flight that nothing measures is a demonstration, not an experiment. A station is a declared plane in the instrument — carried by the specification like everything else (the top-level stations list) — at a position on one axis, with optional rectangular windows over the other two. There are two kinds, named for what they are physically:

- detect — a measuring plane. Its on_hit field says what a hit does: "pass" logs the crossing and lets the ion continue (the arrival time is the first in-window crossing, interpolated exactly between the recorded points that bracket it); "splat" is an absorbing detector face — the ion ends there, fate 6 (station detect).

- impact_plane — a physical plate. An ion crossing inside the window passes through the aperture; outside it, the ion splats on the plate, fate 5 (station impact plane). An empty window is a solid wall.

Mind the inversion between the two kinds, because it is the one honest trap in this vocabulary: a detect window is where ions are absorbed (a detector patch sitting on the beam), while an impact_plane window is where they are not (a hole in a plate). The same numbers mean opposite fates under the two kinds, and clearing the window turns a plate into a wall — not into a full-plane detector. The station editor's reference text states this beside the controls.

Stations are honored in the kernel on every route — planar, r-z, full 3-D, and every stage of a multi-stage instrument, where a stage's stations are authored in that stage's own frame, exactly as when the stage flies alone. Termination is at the exact interpolated crossing, not at the next recorded sample. Fates 5 and 6 take part in the same fate vocabulary, colors and statistics as every other outcome, and the Impact Analysis tab reads arrivals at a detect station the same way for a single stage and for an assembly.

## 7.6 Beyond the solved domain: the ballistic extension

Field-free flight is a straight line, and a straight line's crossing of a plane is exact algebra. So when a flight leaves the solved domain through a face with a declared bound or station beyond it — the detector far downstream of a lens, say — the trajectory is extended by one analytically exact record to honor the declaration: a reachable terminating bound ends the flight exactly on its plane; a wall or detector patch beyond the domain stops the ion with its own fate; a pass detector simply gets its crossing. No field is invented to do this, and the extension refuses by name when its premise fails: when background gas is declared (field-free is not collision-free), or when the escape face still carries residual field above a stated ceiling (the fringe has not decayed there — pad the domain instead).

A deck that declares nothing beyond its domain is untouched by any of this: leaving the box is then simply the fate the flight reports.
