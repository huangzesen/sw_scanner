# sw-scanner

**Solar Wind Scanner** — detecting non-Gaussian structures in solar wind magnetic field data using Jensen-Shannon divergence.

## Overview

`sw-scanner` scans solar wind time series with variable-width sliding windows and computes the Jensen-Shannon divergence of the magnetic field magnitude distribution from a Gaussian. This reveals intervals where the solar wind deviates from normality — a signature of coherent structures such as shocks, discontinuities, and turbulence intermittency.

The scanner is parallelized (via `multiprocessing.Pool`) and supports several analysis modes:

| Mode | Description |
|------|-------------|
| `fit` | Fit an r-dependence power law, normalize, then compute JS divergence |
| `no_fit` | Compute JS divergence without distance normalization |
| `self_fit` | Normalize by a rolling-mean smooth of the time series (reduced module) |
| `hist_moments_no_fit` | Compute only histogram moments (mean, std, skew, kurtosis) |
| `all_inclusive` | Compute all JS variants (raw, log10, scaled, scaled-log10) |
| `find_nan` | Scan for NaN coverage only |

## Installation

```bash
pip install sw-scanner
```

With optional dependencies for the Hampel filter or reduced/ndims modules:

```bash
pip install sw-scanner[all]      # numba + pandas
pip install sw-scanner[dev]       # + pytest for testing
```

## Quick Start

```python
import numpy as np
from sw_scanner import SolarWindScanner, calc_xinds

# Prepare your data
# Btot: 1-D numpy array of magnetic field magnitude [nT]
# time_index: numpy datetime64 array of timestamps
# Dist_au: 1-D numpy array of heliocentric distance [AU]

# Build the time grid and index mapping
xinds_info = calc_xinds(time_index, t_step=np.timedelta64(1, 'm'))
xgrid = xinds_info['xgrid']
xinds = xinds_info['xinds']

# Define scanning windows
wins = np.array([
    np.timedelta64(30, 'm'),
    np.timedelta64(60, 'm'),
    np.timedelta64(120, 'm'),
])

# Scanner settings
settings = {
    'wins': wins,
    'step': np.timedelta64(1, 'm'),
    'xgrid': xgrid,
    'xinds': xinds,
    'n_sigma': 3,
    'capsize': 5000,
    'normality_mode': 'fit',
    'divergence': {
        'js': {'nbins': 50},
    },
}

# Run the scanner
scans = SolarWindScanner(
    Btot,
    Dist_au,
    settings=settings,
    verbose=True,
    Ncores=4,
    return_scans=True,
)
```

## Modules

### `sw_scanner.scanner` — Main Scanner

The primary 1-D scanner. Scans `Btot` (magnetic field magnitude) with variable windows, optionally normalizing by heliocentric distance (`Dist_au`).

**Main function:** `SolarWindScanner(Btot, Dist_au, settings, ...)`

### `sw_scanner.ndims` — Multi-Dimensional Scanner

Extends the scanner to vector magnetic field components (`Bvec`).

**Main function:** `SolarWindScanner(Btot, Dist_au, Bvec, settings, ...)`

### `sw_scanner.reduced` — Reduced Scanner

A variant with adaptive step sizing (step scales with window width) and self-fit normalization using a rolling mean.

**Main function:** `SolarWindScanner(Btot, Dist_au, time_index, settings, ...)`

### `sw_scanner.lib` — Utility Functions

- `round_up_to_minute(datetime)` — Round datetime64 up to nearest minute
- `round_down_to_minute(datetime)` — Round datetime64 down to nearest minute
- `calc_xinds(index, t_step)` — Build a uniform time grid and map it to data indices
- `js_distance(x, n_sigma, nbins)` — Compute JS divergence of `x` from a Gaussian
- `f(x, a, b)` — Linear function `a*x + b` (used for distance fitting)

### `sw_scanner.filters` — Signal Filters

- `hampel(arr, window_size, n, parallel, return_indices)` — Hampel filter for outlier detection (requires `numba`)

## Requirements

- Python >= 3.10
- numpy
- scipy
- tqdm

Optional:
- numba (for Hampel filter)
- pandas (for reduced/ndims modules and DataFrame output)

## License

MIT
