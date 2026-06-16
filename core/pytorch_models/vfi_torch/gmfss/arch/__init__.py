"""GMFSS Fortuna architecture modules.

Architecture files ported from https://github.com/98mxr/GMFSS_Fortuna
via ComfyUI-Frame-Interpolation reference implementation.
"""

from .softsplat import softsplat
from .GMFSS_Fortuna_union_arch import Model as GMFSSUnionModel
from .GMFSS_Fortuna_arch import Model as GMFSSModel

__all__ = [
    "softsplat",
    "GMFSSUnionModel",
    "GMFSSModel",
]
