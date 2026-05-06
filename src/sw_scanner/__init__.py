"""
sw_scanner — Solar Wind Scanner using Jensen-Shannon divergence.

A tool for detecting non-Gaussian structures in solar wind magnetic field
data by scanning time series with variable-width windows and computing
Jensen-Shannon divergence from a normal distribution.
"""

__version__ = "0.1.0"

from .scanner import SolarWindScanner
from .lib import (
    round_up_to_minute,
    round_down_to_minute,
    calc_xinds,
    js_distance,
)
from .filters import hampel

__all__ = [
    "SolarWindScanner",
    "round_up_to_minute",
    "round_down_to_minute",
    "calc_xinds",
    "js_distance",
    "hampel",
]
