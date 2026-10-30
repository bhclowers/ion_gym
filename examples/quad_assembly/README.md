# quad_assembly: the four-rod quadrupole, exported from the assembly

These are the four rods of a real quadrupole, exported from the CAD assembly with one STL per electrode. Each file keeps the assembly frame, so the rods arrive at their true positions and import without any placement correction. This is the convention ion_gym expects for STL electrodes, and these files are the reference example of it.

## The meshes

| File | Electrode | RF pairing |
|---|---|---|
| `Quad_Assembly_-_Rod_A-1.STL` | rod 1 | with rod 3 |
| `Quad_Assembly_-_Rod_B-2.STL` | rod 2 | with rod 4 |
| `Quad_Assembly_-_Rod_C-3.STL` | rod 3 | with rod 1 |
| `Quad_Assembly_-_Rod_D-4.STL` | rod 4 | with rod 2 |

Each rod is 8.67 mm across and 120 mm long (z from 0 to 120 mm), stored as a 100-face mesh. Together the four span x from 0.10 to 20.40 mm and y from 0.10 to 20.19 mm. The quadrupole axis sits near (x, y) = (10.25, 10.15) mm, and the nearest mesh vertex of each rod lies 3.87 mm from it. The faceted surface can sit slightly closer than its vertices, so treat 3.87 mm as an upper bound on r0.

## Where they are used

- `examples/quadrupole_stl_rods_full_3-d.json`, the full 3-D quadrupole deck.
- `internal/studies/digital_quad/decks/quadrupole_digital_3d.json`, the digital-drive variant.
- `tests/test_stl_assembly.py` and `tests/test_stl3d.py`, which check that the files import untouched as four separate electrodes.

## Using them in the STL upload tab

1. Drop all four files on the drop area.
2. Set the pitch. The tab reports the grid size, memory, and solve time for the chosen pitch before anything is solved.
3. Tick "solve full 3-D" to solve along the rods' length. Left unticked, the tab solves a 2-D slice through the middle of the rods.
4. Give rods 1 and 3 one RF group and rods 2 and 4 the other, and set any DC values.
5. Press Preview to check placement and units in 3-D, then Commit.

## Related folders

- `examples/rods/` holds the same rods exported one part at a time. Those files lose the assembly placement and are kept to show that failure.
- `examples/quad_stl_rods/` holds synthetic rods 1 mm thick in z, used by the 2-D transport deck and notebook 02.
