# rods: the four-rod quadrupole, exported one part at a time

These four files are the same rods as `examples/quad_assembly/`, exported from CAD one part at a time instead of from the assembly. Each export used its own part frame, so the placement the rods had in the assembly was lost. Imported together, the four rods land almost on top of one another. Their centres sit within 1.5 mm of each other, while each rod is 8.67 mm across. They are not a working example of STL import. They are kept to show what this export mistake looks like and to check that ion_gym refuses it.

## The meshes

| File | Centre (x, y) as exported | RF pairing |
|---|---|---|
| `Rod_1.STL` | (4.33, 5.71) mm | with rod 3 |
| `Rod_2.STL` | (5.81, 5.71) mm | with rod 4 |
| `Rod_3.STL` | (5.81, 4.33) mm | with rod 1 |
| `Rod_4.STL` | (4.33, 4.33) mm | with rod 2 |

Each rod is 8.67 mm across, with a 45 degree mounting flat on its outer face, and 120 mm long (z from 0 to 120 mm). Each is stored as a 100-face mesh.

## What happens on import

In the STL upload tab the four files load and preview, but Commit is refused because the electrodes interpenetrate. The message explains that the STLs were most likely exported in part frames and should be re-exported from the assembly. The voxelizer refuses overlapping electrodes, so these rods can never be flown in the wrong place without warning.

`tests/test_stl_rods.py` checks that refusal. It then moves each rod in code to its diagonal position at r0 = 3.84 mm and checks that the solved field is a quadrupole.

## The fix

Export one STL per electrode from the assembly. `examples/quad_assembly/` holds the result for these same rods, and those files import untouched.

## Related folders

- `examples/quad_assembly/` holds the same rods exported from the assembly, the reference example of STL import.
- `examples/quad_stl_rods/` holds synthetic rods 1 mm thick in z, used by the 2-D transport deck and notebook 02.
