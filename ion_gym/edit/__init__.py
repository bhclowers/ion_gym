"""Geometry editing for single-FA spec documents.

Scope: editable routes are planar, rz, shapes3d.
stl2d/stl3d refuse (meshes belong to CAD); scene3d is parked with
no fixtures; multi-FA documents refuse pending a stage-export path.

The editor gate proves no-op byte identity across
every in-scope shipped deck, named refusals for everything else, and
patch-confinement proofs.  Seconds, headless.
"""
from ion_gym.edit.policy import (EditPolicy, EditRefusal,
                                 GHOST_EXTENT_FRACTION, POLICIES,
                                 clamps_for, policy_for_document)
from ion_gym.edit.session import (EDITOR_SCENE_SCHEMA, EditSession,
                                  structural_diff)

__all__ = [
    "EditPolicy", "EditRefusal", "EditSession", "EDITOR_SCENE_SCHEMA",
    "GHOST_EXTENT_FRACTION", "POLICIES", "clamps_for",
    "policy_for_document", "structural_diff",
]
