"""Board visualization: schematic frames of a position, and a demo animation.

Two modules: :mod:`entente.viz.svg` renders schematic board frames as SVG
strings from map *data* alone (stdlib only -- always importable), and
:mod:`entente.viz.gif` rasterizes the same scenes into an animated GIF
(requires Pillow from the ``viz`` dependency group; import it lazily, not
from here, so the default environment keeps working).
"""

from entente.viz.svg import load_centroids, render_board, render_record

__all__ = ["load_centroids", "render_board", "render_record"]
