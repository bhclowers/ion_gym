<!-- Generated from the ion_gym user manual and the shipped decks and notebooks. Edits made here are overwritten. -->

# 6. From potential to field

The solver produces a potential. The trajectory kernel needs an electric field, E = −∇φ, evaluated on the same grid. How that gradient is taken is a choice with real consequences, and it is a declared option.

## 6.1 The two methods

**Plain gradient** takes a central difference everywhere: the field at a node is computed from its two neighbors along each axis. This is second-order accurate and is the correct thing to do anywhere in the vacuum between electrodes.

**Electrode-aware** differs only at metal surfaces. At a node that lies on a conductor and has vacuum on exactly one side along an axis, the central difference straddles the electrode — and because the potential inside a conductor is flat, the half of the stencil inside the metal contributes nothing. The result is a surface field roughly half its true value. The electrode-aware method replaces that one derivative with a one-sided difference taken into the vacuum, never across the metal interior. Vacuum nodes are untouched, so the two methods differ only on the one-cell skin where the metal faces the gap.


## 6.2 When each is appropriate

| **Situation**                     | **Use**         | **Why**                                                                                                                                                                                                                                                                   |
|-----------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ions born on an electrode surface | electrode-aware | The ion's first acceleration happens in exactly the cells where the plain gradient is wrong. A surface-born ion is mis-accelerated by roughly one percent of the gap energy without the correction — small, but systematic, and it maps directly onto arrival-time error. |
| Ions grazing or striking surfaces | electrode-aware | Whenever the trajectory samples the surface skin, the surface field is part of the answer.                                                                                                                                                                                |
| Transport through a central gap   | plain gradient  | If ions stay away from the electrodes — the usual case for a guide, a funnel, or a transport stack — the correction touches cells the ions never visit. It is then pure cost.                                                                                             |
| Large grids where speed matters   | plain gradient  | The correction is a per-metal-node pass, so its cost scales with the electrode surface area on the grid.                                                                                                                                                                  |

The default is electrode-aware, because being right at surfaces is the safer default and the situations that need it are the ones where being wrong is least visible. Choose the plain gradient deliberately, when you can say why the ions do not sample the surfaces.

A related option controls the **stored precision** of the drive-field arrays. The gradient is always computed in double precision; the option decides whether the result is stored as 32-bit or 64-bit. Single precision halves the memory of the field arrays, which becomes the binding constraint on large three-dimensional grids, at a rounding-level cost that is negligible against the discretization error already present. It is stated as an option rather than chosen for you because a frozen reference result stored at one precision will not reproduce bit-for-bit at the other.
