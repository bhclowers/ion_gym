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
