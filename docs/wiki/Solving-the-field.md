<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 5. Solving the field

Why a grid at all? Between the electrodes the potential obeys Laplace's equation, and for any realistic electrode shape that boundary-value problem has no closed-form solution — the geometry IS the problem. The classical response is to replace the continuous domain with a regular lattice of sample points, write the equation as a relation between each point and its neighbors (each node relaxes toward the average of its neighbors — the discrete statement that a harmonic function has no interior extrema), and iterate until the lattice stops changing. Everything downstream inherits the choice: the pitch of that lattice is the resolution of the physics, which is why the specification declares it explicitly instead of inferring it.


## 5.1 Per-electrode bases and superposition

The field is not solved once for a set of voltages. Instead, Laplace's equation is solved once per electrode, with that electrode at a reference potential and every other electrode grounded. Each solution is stored normalized to unit electrode potential, giving a dimensionless response function — the fraction of that electrode's voltage seen at each point.

The field for any set of voltages is then the weighted sum of these bases. This has a consequence worth exploiting: changing a voltage is not a re-solve. Sweeping a lens voltage, retuning a stack, or optimizing a set of potentials re-weights bases that are already in memory, which is essentially free. Changing the geometry or the grid pitch is a re-solve, because the bases themselves depend on those.

Because a solve is expensive and a geometry is often reused, solved bases are cached on disk under a key derived from everything that affects them — the geometry, the pitch, the declared symmetry, and the mesh hashes. Any change to those produces a different key and a fresh solve.

## 5.2 The iterative solvers

Laplace's equation is discretized on the uniform grid and relaxed iteratively. The base method is successive over-relaxation (SOR), the classical accelerated relaxation scheme analyzed by Young **\[3\]**, applied in a red-black ordering so that each half-sweep is independent and can be executed in parallel. The relaxation factor is chosen from the grid dimensions rather than left to the user.

SOR alone converges slowly on large grids because it removes short-wavelength error (components of the residual that vary over a few grid cells) quickly and long-wavelength error (components that vary over a large fraction of the domain) very slowly. The multigrid solver addresses this by relaxing on a hierarchy of coarser grids, where the long-wavelength error is short-wavelength, and prolongating the result upward — the multi-level idea introduced by Brandt **\[2\]**. The implementation here is cascadic: it solves coarse-to-fine, using each level's solution as the next level's starting guess, and the final answer is produced by relaxing on the finest grid to the requested tolerance. This matters for a subtle reason — the fine-grid solution is the one the ions fly through, so it is solved directly rather than reconstructed from a correction.

Convergence is tested against a declared tolerance on the maximum change per sweep (watch the terminal console during a large solve to see this in action). If the solver reaches its iteration cap without meeting the tolerance it raises an error. It does not return the unconverged field. This is deliberate: an unconverged field that looks plausible is far more dangerous than a failed solve, because nothing downstream can detect it.
