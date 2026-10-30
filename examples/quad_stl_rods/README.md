# quad_stl_rods -- pristine CAD frame (L-295)

These four rod meshes are stored exactly as CAD exports them: the
quadrupole axis (the mean of the four rod centres) is at the ORIGIN.

Placement into the solve domain is DECLARED, not baked: the deck's
`geometry.frame_offset_mm` states the CAD->build translation and
`ion_gym.io.stl_resolve.load_mesh` -- the single mesh-ingest point every
STL route loads through -- applies it. Re-exporting these rods from CAD
is therefore safe: as long as the array stays origin-centred, no hidden
state exists to lose.

History: before 2026-09-04 the translation was baked into the mesh bytes
by the deck-authoring path, and the one time pristine exports shipped,
half the quadrupole fell outside the domain and notebook 02 failed
end-to-end. The bake path is deleted.

## Related folders

These rods are synthetic and only 1 mm thick in z (z from -0.5 to 0.5 mm), because the deck that uses them, `examples/quadrupole_stl_rods_transport.json`, solves a 2-D cross-section. Notebook 02 uses the same deck. For a full-length example of STL import, use the other sets.

- `examples/quad_assembly/` holds a real four-rod quadrupole, 120 mm long, exported from the CAD assembly. It is the reference example of STL import.
- `examples/rods/` holds the same real rods exported one part at a time. Those files lose the assembly placement and are kept to show that failure.
