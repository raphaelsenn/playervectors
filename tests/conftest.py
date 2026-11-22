"""Shared fixtures: a tiny, synthetic event-stream dataset."""

import numpy as np
import pandas as pd
import pytest

PITCH = ((0.0, 105.0), (0.0, 68.0))


@pytest.fixture
def events() -> pd.DataFrame:
    """
    Six players, each acting around their own "home" position on the pitch,
    so that NMF has some structure to find.
    """
    rng = np.random.default_rng(0)
    home_positions = {
        1: (20, 34), 2: (50, 10), 3: (50, 58),
        4: (80, 34), 5: (95, 20), 6: (95, 48),
    }

    rows = []
    for player_id, (x, y) in home_positions.items():
        for action, n in [("shot", 5), ("cross", 5), ("dribble", 10), ("pass", 30)]:
            for _ in range(n):
                start_x, start_y = rng.normal([x, y], 5)
                rows.append({
                    "player_id": player_id,
                    "type_name": action,
                    "start_x": np.clip(start_x, 0, 105),
                    "start_y": np.clip(start_y, 0, 68),
                    "end_x": np.clip(start_x + rng.normal(10, 5), 0, 105),
                    "end_y": np.clip(start_y + rng.normal(0, 5), 0, 68),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def minutes_played() -> dict[int, float]:
    return {player_id: 90.0 * player_id for player_id in range(1, 7)}


@pytest.fixture
def player_names() -> dict[int, str]:
    return {player_id: f"Player {player_id}" for player_id in range(1, 7)}
