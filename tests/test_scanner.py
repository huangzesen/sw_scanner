"""Basic smoke tests for the sw_scanner package."""

import numpy as np
import pytest


def test_import_package():
    """The package should import without errors."""
    import sw_scanner
    assert sw_scanner.__version__ == "0.1.0"


def test_import_scanner():
    """The main scanner module should import."""
    from sw_scanner.scanner import SolarWindScanner
    assert callable(SolarWindScanner)


def test_import_ndims():
    """The multi-dimensional scanner module should import."""
    from sw_scanner.ndims import SolarWindScanner as NdimScanner
    assert callable(NdimScanner)


def test_import_reduced():
    """The reduced scanner module should import."""
    from sw_scanner.reduced import SolarWindScanner as ReducedScanner
    assert callable(ReducedScanner)


def test_import_lib():
    """The library module should import."""
    from sw_scanner.lib import (
        round_up_to_minute,
        round_down_to_minute,
        calc_xinds,
        js_distance,
        f,
    )
    assert callable(round_up_to_minute)
    assert callable(round_down_to_minute)
    assert callable(calc_xinds)
    assert callable(js_distance)
    assert callable(f)


def test_import_filters():
    """The filters module should import."""
    from sw_scanner.filters import hampel
    assert callable(hampel)


def test_js_distance_basic():
    """js_distance should return a finite divergence for Gaussian-like data."""
    from sw_scanner.lib import js_distance

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10000)
    js_div, outside_count = js_distance(x, n_sigma=3, nbins=50)

    # A Gaussian sampled should have very small JS divergence from itself
    assert np.isfinite(js_div)
    assert js_div < 0.05, f"Expected small JS divergence for Gaussian data, got {js_div}"
    assert isinstance(outside_count, (int, np.integer))


def test_js_distance_non_gaussian():
    """js_distance should return larger divergence for non-Gaussian data."""
    from sw_scanner.lib import js_distance

    rng = np.random.default_rng(42)
    # Exponential distribution is very non-Gaussian
    x = rng.exponential(1.0, 10000)
    js_div, _ = js_distance(x, n_sigma=3, nbins=50)

    assert np.isfinite(js_div)
    assert js_div > 0.1, f"Expected large JS divergence for exponential data, got {js_div}"


def test_round_up_to_minute():
    """round_up_to_minute should round datetime64 values up."""
    from sw_scanner.lib import round_up_to_minute

    # 12:30:45 should round up to 12:31:00
    dt = np.datetime64('2024-01-01T12:30:45', 'ns')
    result = round_up_to_minute(dt)
    expected = np.datetime64('2024-01-01T12:31:00', 'ns')
    assert result == expected


def test_round_down_to_minute():
    """round_down_to_minute should round datetime64 values down."""
    from sw_scanner.lib import round_down_to_minute

    # 12:30:45 should round down to 12:30:00
    dt = np.datetime64('2024-01-01T12:30:45', 'ns')
    result = round_down_to_minute(dt)
    expected = np.datetime64('2024-01-01T12:30:00', 'ns')
    assert result == expected


def test_f_linear():
    """f(x, a, b) should return a*x + b."""
    from sw_scanner.lib import f

    assert f(1, 2, 3) == 5  # 2*1 + 3
    assert f(0, 2, 3) == 3
    assert f(-1, 2, 3) == 1


def test_public_api_exports():
    """All advertised public names should be importable from sw_scanner."""
    from sw_scanner import (
        SolarWindScanner,
        round_up_to_minute,
        round_down_to_minute,
        calc_xinds,
        js_distance,
        hampel,
    )
    assert callable(SolarWindScanner)
    assert callable(round_up_to_minute)
    assert callable(round_down_to_minute)
    assert callable(calc_xinds)
    assert callable(js_distance)
    assert callable(hampel)
