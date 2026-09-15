# Related and Complementary Software

This list is not exhaustive, and inclusion does not imply numerical
equivalence or feature parity. The packages below solve overlapping but
generally distinct problems; they are listed to document the
methodological context of ion_gym and to help a reader pick the right
tool for a given problem. Where scopes overlap, specialized tools and
analytic references should be used as validation baselines — ion_gym's
own validation notebooks take exactly that stance (field accuracy
against analytic potentials, Mathieu stability, Wiley–McLaren TOF
timing, collisional thermalization, transport/Einstein closure).

## Ion-optics and instrument trajectory platforms

### SIMION

SIMION is the widely used general-purpose platform for calculating
electrostatic fields and flying charged particles through user-defined
electrode geometries, with decades of deployed history across ion
optics and mass spectrometer development. It remains the de facto
reference environment in the field; see the
[SIMION user manual](https://simion.com/info/manual.html) for its
documented scope.

ion_gym occupies adjacent ground with different trade-offs: it is
open-source (GPL-3) and Python-native, geometry is authored as
declarative JSON rather than through a GUI/PA workflow, and a solve
produces cached per-electrode basis fields so that a voltage change is
a re-weight rather than a re-solve — the property its optimization and
interactive-workbench workflows are built on. ion_gym's SDS collision
model implements the algorithm of Appelhans & Dahl as published (see
"Physics provenance" in the README); its data tables are independently
regenerated ion_gym artifacts.

### IDSimF

IDSimF is an open-source C++ framework for simulating nonrelativistic
molecular ion dynamics in mass spectrometry and ion mobility
spectrometry: modular trajectory solvers, electric fields, ion–neutral
collision models, space-charge interactions, and reaction kinetics.

At the framework level it is among the closest open projects to
ion_gym. The centers of gravity differ: IDSimF is strong precisely
where ion_gym currently stops (space charge, reaction chemistry),
while ion_gym adds solve-from-geometry field generation, an
interactive Panel/Plotly workbench and geometry editor, and a graduate
teaching track. Results from the two should not be assumed
interchangeable without model-specific validation.

### ICARION

ICARION is a C++17/CUDA framework for multi-domain ion dynamics —
IMS/TIMS, quadrupole, Orbitrap, TOF, LQIT and FT-ICR domains with
stochastic and deterministic collision models, ion–neutral reactions,
and strongly provenance-stamped HDF5 output. As of its v1.1.x
releases it has no field solver or optimizer (fields are analytical
or imported arrays; both are on its roadmap), which makes it very
nearly complementary to ion_gym: ICARION is deepest in collision and
reaction physics, ion_gym in geometry-to-field workflows and
interactive design.

## Collision cross section and mobility calculators

MOBCAL and its parallelized successors MobCal-MPI / MobCal-MPI 2.0,
IMoS, Collidoscope, and CoSIMS compute molecular transport quantities —
collision cross sections and mobilities — from molecular structures and
ion–neutral interaction potentials. These are inputs to, not
alternatives for, instrument simulation: an instrument framework such
as ion_gym propagates ions through fields and buffer gas given
transport properties of the kind these tools predict (ion_gym's SDS
model consumes mobility/CCS data per species). For structure-level CCS
questions, use these tools directly.

## Scope of ion_gym

ion_gym's goal is one continuous, scriptable path from a declarative
geometry to flown ions: JSON geometry (2-D planar, r-z, full-3-D
extruded/STL) → multigrid field solve with cached per-electrode bases →
RF/DC/traveling-wave drives → trajectory integration in vacuum or
buffer gas (HS, SDS) → framework rendering and an interactive
workbench — with the same machinery serving research design work and
graduate teaching notebooks.

It is deliberately not a universal simulator. It currently has no
space-charge interaction (ions fly independently), no reaction
chemistry, no magnetic fields, and no GPU backend; the field solver is
a finite-difference electrostatic solver on a regular grid, not
BEM/FMM. Where those capabilities matter, the packages above are the
right tools, and where scopes overlap, treat ion_gym as one instrument
in a validation triangle with analytic references and specialized
software rather than as ground truth.

## References

- D. A. Dahl, "SIMION for the personal computer in reflection,"
  *Int. J. Mass Spectrom.* **200** (2000) 3–25.
  DOI: [10.1016/S1387-3806(00)00305-5](https://doi.org/10.1016/S1387-3806(00)00305-5)
- A. D. Appelhans and D. A. Dahl, "SIMION ion optics simulations at
  atmospheric pressure," *Int. J. Mass Spectrom.* **244** (2005) 1–14.
  DOI: [10.1016/j.ijms.2005.03.010](https://doi.org/10.1016/j.ijms.2005.03.010)
- M. Rajkovic, S. Benter, M. Hammelrath, M. Thinius, T. Benter, and
  W. Wißdorf, "IDSimF: An Open-Source Framework for the Simulation of
  Molecular Ion Dynamics in Mass Spectrometry and Ion Mobility
  Spectrometry," *J. Am. Soc. Mass Spectrom.* **35** (2024) 1451–1460.
  DOI: [10.1021/jasms.4c00054](https://doi.org/10.1021/jasms.4c00054)
- ICARION — Ion Collision And Reaction IntegratiON.
  Software: [github.com/ICARION-Project/ICARION](https://github.com/ICARION-Project/ICARION);
  release archive DOI: [10.5281/zenodo.20599037](https://doi.org/10.5281/zenodo.20599037)
- M. F. Mesleh, J. M. Hunter, A. A. Shvartsburg, G. C. Schatz, and
  M. F. Jarrold, "Structural Information from Ion Mobility
  Measurements: Effects of the Long-Range Potential," *J. Phys. Chem.*
  **100** (1996) 16082–16086.
  DOI: [10.1021/jp961623v](https://doi.org/10.1021/jp961623v)
- A. Haack, C. Ieritano, and W. S. Hopkins, "MobCal-MPI 2.0: an
  accurate and parallelized package for calculating field-dependent
  collision cross sections and ion mobilities," *Analyst* **148**
  (2023) 3257–3273.
  DOI: [10.1039/D3AN00545C](https://doi.org/10.1039/D3AN00545C)
- S. A. Ewing, M. T. Donor, J. W. Wilson, and J. S. Prell,
  "Collidoscope: An Improved Tool for Computing Collisional
  Cross-Sections with the Trajectory Method," *J. Am. Soc. Mass
  Spectrom.* **28** (2017) 587–596.
  DOI: [10.1007/s13361-017-1594-2](https://doi.org/10.1007/s13361-017-1594-2)
- C. A. Myers, R. J. D'Esposito, D. Fabris, S. V. Ranganathan, and
  A. A. Chen, "CoSIMS: An Optimized Trajectory-Based Collision
  Simulator for Ion Mobility Spectrometry," *J. Phys. Chem. B* **123**
  (2019) 4347–4357.
  DOI: [10.1021/acs.jpcb.9b01018](https://doi.org/10.1021/acs.jpcb.9b01018)
