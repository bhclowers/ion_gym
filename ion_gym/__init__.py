"""ion_gym — independent ion-optics simulation toolkit.

Copyright (C) 2026 Brian Clowers / Washington State University.
Licensed under the GNU General Public License v3 (see LICENSE).

Support for this work was provided in part by NIGMS R35GM161833.

Importing this package has NO side effects: no solves, no I/O, no path
mutation. Subpackages (physics, io, viz, ui, cad, examples, projects) are
imported explicitly by their absolute dotted paths.
"""

# SINGLE VERSION AUTHORITY (the version shows in the
# Config tab). This string IS the release number -- the vNNN in the
# delivered zip name. pyproject.toml derives from it (dynamic
# version, tool.setuptools.dynamic), and the Config tab displays it,
# so the number on the zip, the installed metadata, and the UI can
# never disagree. Bump it when cutting a zip, nowhere else --
# and the release packer REFUSES to cut a zip
# whose target tag disagrees with this string, so the hand
# bump cannot be forgotten again (this once read 344
# while the zip series had reached 384, and the UI Config
# tab faithfully displayed the stale number for 40 cuts).
# (Found stale and split once: __init__ said 148.2 while
# pyproject said 149.0 and the zip series was at v328.)
# Version = the cut number of the zip this tree ships in; bumped AT CUT
# TIME so the Config tab (which reads this live) always matches the
# artifact the user downloaded. A tree whose version trails its zip
# name reads as "did my upload take?" in the browser.
__version__ = "542"
