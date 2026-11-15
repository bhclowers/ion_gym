<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 1. What this toolkit is

ion_gym computes electrostatic, electrodynamic, and radio-frequency fields for ion optics simulations. It is a Python toolkit: geometry, voltages, drive waveforms, gas conditions and the ion source are all declared in a JSON specification file, solved with its own field solver, and integrated with its own trajectory kernel. It does not require, wrap, or call any external ion-optics package.

The design goal is simplicity and reproducibility. A specification file plus its geometry resources is a complete, portable description of a simulation: the same file gives the same field and the same trajectories on another machine. Every quantity the solver uses is declared somewhere you can read, rather than being set in a graphical interface and lost.

If you are learning the physics as well as the tool, start with the notebooks that ship beside this manual: 00–11 build from the Laplace equation to a complete multi-stage instrument, each runnable end to end, and V01–V06 validate the toolkit's physics against known answers — field solutions against analytic cases, Mathieu stability, thermalization, transport. This manual is the reference those notebooks lean on; they are the course.

## 1.1 What it is not

Being explicit about the boundaries is more useful than a feature list:

- It is not a general electromagnetic solver. It solves Laplace's equation for electrostatics on a uniform Cartesian or cylindrical grid. ***There are no magnetic fields, no space charge, and no dielectrics.***

<!-- -->

- It is not a replacement for established commercial ion-optics packages. Where the two disagree, that disagreement is a finding to be investigated, not a result.

- It is not a CAD package. Though a shape editor exists as an ancillary control (section 13.1), geometry is generally built from declared primitives (e.g., circles, rectangles, and polygons) or imported as STL meshes produced elsewhere.

- It does not model gas dynamics. Collisions use a statistical model at a declared pressure and temperature; a uniform gas-flow vector can be declared in the specification and is honored on every route, but there is no computed flow field (section 7.3).

## 1.2 The two ways to drive it

Everything below can be done either from the browser-based application or from a script. The application is a viewer and an editor for the same specification files the scripts consume; it holds no configuration of its own. ***Anything you can set in the interface is a field in the file.***
