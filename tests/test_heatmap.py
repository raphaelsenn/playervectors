import numpy as np
import pytest

from playervectors import action_heatmap

# A 10 x 10 pitch split into 10 x 10 cells: every cell is exactly 1 x 1.
PITCH = ((0.0, 10.0), (0.0, 10.0))
GRID = (10, 10)


def heatmap(x, y, sigma=0.0, minutes_played=90.0):
    return action_heatmap(x, y, grid=GRID, pitch_size=PITCH, sigma=sigma, minutes_played=minutes_played)


class TestActionHeatmap:
    def test_shape_matches_grid(self):
        assert heatmap([1.0, 2.0], [3.0, 4.0]).shape == GRID

    def test_no_actions_gives_zeros(self):
        result = heatmap([], [])
        assert result.shape == GRID
        assert not result.any()

    def test_action_lands_in_correct_cell(self):
        # Rows follow y and columns follow x.
        result = heatmap([1.5], [8.5])
        assert result[8, 1] == 1.0
        assert result.sum() == 1.0

    def test_counts_multiple_actions(self):
        result = heatmap([1.5, 1.5, 5.5], [1.5, 1.5, 5.5])
        assert result[1, 1] == 2.0
        assert result[5, 5] == 1.0

    def test_missing_coordinates_are_ignored(self):
        result = heatmap([1.5, np.nan, 3.5], [1.5, 2.5, np.nan])
        assert result.sum() == 1.0

    def test_normalised_per_90_minutes(self):
        one_match = heatmap([5.5], [5.5], minutes_played=90.0)
        two_matches = heatmap([5.5], [5.5], minutes_played=180.0)
        np.testing.assert_allclose(two_matches, one_match / 2)

    def test_zero_minutes_keeps_raw_counts(self):
        assert heatmap([5.5], [5.5], minutes_played=0.0).sum() == 1.0

    def test_smoothing_spreads_but_keeps_total(self):
        result = heatmap([5.5], [5.5], sigma=1.0)
        assert result[5, 5] < 1.0          # the peak is spread out ...
        assert result[5, 6] > 0.0          # ... onto neighbouring cells ...
        assert result.sum() == pytest.approx(1.0)  # ... without losing mass
