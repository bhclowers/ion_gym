"""ion_gym.projects — project packages.

An instrument project (a lab's own design: certified voltages, geometry
constants, trade tables)
may live PHYSICALLY under internal/projects/ — dev checkouts only, deleted
from every shareable cut. This __path__ extension is the ONE sanctioned
bridge: in a dev checkout such a project imports normally; in
a shared checkout internal/ does not exist, the path is simply absent,
and any import fails loudly with the standard ModuleNotFoundError. Its
gate corpus lives beside it in the same internal tree.
"""
import pathlib as _pathlib

_internal = (_pathlib.Path(__file__).resolve().parent.parent.parent
             / "internal" / "projects")
if _internal.is_dir():
    __path__.append(str(_internal))
