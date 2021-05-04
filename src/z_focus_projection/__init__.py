"""Z Focus Projection."""
import logging

from .z_projection import (
    DEFAULT_CANNY_THRESHOLDS,
    DEFAULT_MOVING_AVG_WINDOW,
    DEFAULT_PLOT_SIZE,
    HyperStack,
    ZStack,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())


__all__ = [
    "HyperStack",
    "ZStack",
    "DEFAULT_PLOT_SIZE",
    "DEFAULT_CANNY_THRESHOLDS",
    "DEFAULT_MOVING_AVG_WINDOW",
]
