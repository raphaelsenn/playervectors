from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

# A pitch is described by its x-range and y-range, e.g. ((0, 105), (0, 68)).
PitchSize = tuple[tuple[float, float], tuple[float, float]]

MINUTES_PER_MATCH = 90.0


def action_heatmap(
    x: Sequence[float],
    y: Sequence[float],
    grid: tuple[int, int],
    pitch_size: PitchSize,
    sigma: float,
    minutes_played: float,
) -> np.ndarray:
    """
    Build a normalised, smoothed heatmap of action locations.

    Parameters
    ----------
    x, y
        Coordinates of the actions. Rows with a missing (NaN) coordinate are ignored.
    grid
        Number of cells as ``(n_rows, n_cols)``. Rows run along the pitch width (y),
        columns along the pitch length (x).
    pitch_size
        ``((x_min, x_max), (y_min, y_max))`` of the coordinate system.
    sigma
        Standard deviation (in grid cells) of the Gaussian smoothing.
    minutes_played
        Total minutes the player was on the pitch. If it is 0 or unknown, the raw
        counts are used without normalisation.

    Returns
    -------
    np.ndarray of shape ``grid``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    has_location = np.isfinite(x) & np.isfinite(y)
    x, y = x[has_location], y[has_location]

    if len(x) == 0:
        return np.zeros(grid, dtype=float)

    # 1. Count. NOTE: `histogram2d` puts its first argument on the rows, so we pass y
    #    first to get an array that "looks like" the pitch: rows = width, cols = length.
    (x_range, y_range) = pitch_size
    counts, _, _ = np.histogram2d(y, x, bins=grid, range=[y_range, x_range])

    # 2. Normalise to actions per 90 minutes.
    if minutes_played > 0:
        counts *= MINUTES_PER_MATCH / minutes_played

    # 3. Smooth.
    return gaussian_filter(counts, sigma=sigma)
