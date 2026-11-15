# Velocity Verlet against RK4 on the manuscript quadrupole

Measured for the ion_gym framework manuscript. Script `integrator_compare.py` beside this file, banked output `integrator_compare.out.txt`. Environment: Linux container, 2 cores, Python 3.13.18, numpy 2.5.3, numba 0.68.1. Container wall time for the whole probe is 16 s. Classifications and frequencies are machine independent. Wall times are not and none are quoted here as results.

## Question

The production planar and r-z routes advance ions with the velocity Verlet method and the 3-D route uses RK4. The question is whether the two schemes give different results, and whether to unify on one scheme. The per-route schemes are kept, and the choice is presented as demonstrated insensitivity.

## Method

Every flight pair uses the same solved planar bases, the same drive channels, and the identical birth state. The Verlet flight runs through the production route (`build_run`, `build_planar._fly_planar`). The RK4 flight runs the same unit bases through `tracer_tw2d.fly_tw2d_ion`, the kernel that is bit-parity checked against the 3-D RK4 kernel in `tests/test_tw2d.py` Gate B. Operating points are the Figure 4 production values of `make_fig4_stability.py`: m/z 100, 1 MHz, r0 3 mm, rods 1.148 r0, 0.1 mm grid, v_per_q 94.761010 V from the solved-field calibration, zone 3,1 band [4.587228, 4.595481] at duty 0.835.

## Results

| Comparison | Flights per integrator | Outcome |
| --- | --- | --- |
| A. zone 3,1 band scan, 17 voltages x 16 phases, dt 0.1 ns, 50 us | 272 | identical, 0 of 272 cells differ. 4 of 16 phases transmit at each of the three in-band voltages, 0 of 16 at the 14 outside |
| A. starting-offset ladder 20, 10, 5, 2.5 um at the band centre | 64 | Verlet 2, 4, 7, 10 of 16 (the manuscript values). RK4 1, 4, 7, 9. Two cells differ, see E |
| B. 13 x 13 (a, q) grid of zone 1, sinusoidal, dt 1 ns, 80 us | 169 | both integrators agree with the transfer-matrix chart at all 169 points and with each other |
| C. secular ladder q = 0.10 to 0.80, dt 1 ns, 160 us | 5 | the two integrators agree to within 7e-6 relative. Both sit 1.4e-4 to 5.7e-3 from the transfer matrix, the manuscript's 0.6 percent bound |

Totals: 505 paired classification flights, 503 agree, and the five frequency pairs agree to within one part in 1e5.

### D. Step-halving ladders (end position at fixed lab time)

Sinusoidal drive, a = 0, q = 0.5, 80 us, production dt 1 ns. RK4 self-converges from 2.6e-8 mm (4 to 2 ns) to 6.0e-11 mm (0.5 to 0.25 ns), close to its formal order at the first halving. Verlet self-converges at the 1e-4 to 1e-5 mm level. The cross-integrator gap at the production step is 3.1e-5 mm. Both are orders of magnitude below any quoted quantity.

Digital drive, zone 3,1 band centre, transmitted phase 0.3125, 50 us, production dt 0.1 ns. RK4 self-convergence stalls at 4.3e-3 then 4.2e-3 mm per halving, worse than Verlet at 5.5e-4 then 7.0e-5 mm. The waveform switching edges inside the step cap the order of both methods and remove RK4's formal advantage entirely. At dt 0.4 ns the ion is lost on both integrators, consistent with the manuscript's timing-sensitivity finding. End positions on this orbit amplify small differences by large growth factors, so the classification, and not the end position, is the robust observable here.

### E. The two disagreeing cells

Both disagreements are single ions at the band centre whose orbits graze the loss radius, and neither is a converged answer at the production step.

| x0 (um) | phase | ideal-field peak (mm) | Verlet, dt 0.1 | RK4, dt 0.1 | at dt 0.05 |
| --- | --- | --- | --- | --- | --- |
| 20.0 | 0.4375 | 0.907 | transmitted | lost | both transmitted |
| 2.5 | 0.9375 | 1.341 | transmitted | lost | Verlet lost, RK4 transmitted |

The 20 um cell resolves at the halved step to the Verlet production answer. The 2.5 um cell swaps on both integrators at the halved step, so its single-ion outcome is step sensitive on either scheme. Consequence for the text: the offset-ladder count 10 of 16 at 2.5 um carries a one-count step sensitivity, and the 1.36 to 1.50 mm undecided band stated with Figure 4 brackets the second cell but not the first, whose ideal-field peak of 0.907 mm understates how closely the real-field orbit approaches the rods.

## What the manuscript states

Trajectory Integration names the scheme per route, the two-evaluation waveform timing, the SDS exponential relaxation factor, and the 503 of 505 result with the grazing-orbit explanation. The Introduction sentence now reads "the same field solution and trajectory integration serve every calculation" in place of "and integrator". The Narrow Digital Zones section states that the fourth-order integrator converged no better than the second-order method under the digital drive. The SI list gains this comparison as an item.

## Standing conclusions

The per-route schemes stay for this paper. On smooth drives both schemes are converged far past the reported precision at the production steps. On the digital drive the switching edges cap the order of any integrator, so a higher-order scheme buys nothing where the narrow-zone numbers live. A future unification, if wanted, targets the kick-drift-kick family because the SDS drag cannot be stepped with plain RK4 at SDS-scale steps, and the RK4 kernels then remain as the independent cross-check. One known asymmetry is left open: the 3-D SDS kernel applies one damped kick per step while the planar and r-z SDS paths use two half-kicks.
