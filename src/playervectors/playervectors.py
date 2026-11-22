"""
Unofficial implementation of

    Decroos & Davis (2019), "Player Vectors: Characterizing Soccer Players'
    Playing Style from Match Event Streams".

Authors: Raphael Senn & Sebastian Tim Wehmeier
Initial Coding: 20.12.2024
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from playervectors.heatmap import PitchSize, action_heatmap

# Configuration used in the paper.
DEFAULT_ACTIONS = ["shot", "cross", "dribble", "pass"]
DEFAULT_COMPONENTS = [4, 4, 5, 5]

DEFAULT_COLUMNS = {
    "player_id": "player_id",
    "action_type": "type_name",
    "x_start": "start_x",
    "y_start": "start_y",
    "x_end": "end_x",
    "y_end": "end_y",
}


class PlayerVectors:
    """
    Compute player vectors from event-stream data.

    Parameters
    ----------
    grid
        Heatmap resolution as ``(n_rows, n_cols)``: rows along the pitch width,
        columns along the pitch length.
    pitch_size
        ``((x_min, x_max), (y_min, y_max))`` of your coordinate system.
    sigma
        Gaussian smoothing of the heatmaps, in grid cells.
    actions
        Action types to use, as they appear in the action-type column.
    components
        Number of NMF components (prototype heatmaps) per action type.
    with_end_coordinates
        Per action type: if True, the heatmap of where the action *ended* is
        appended to the heatmap of where it started. This is useful for passes,
        where the destination says a lot about style.
    init_nmf, max_iter_nmf, random_state
        Passed on to ``sklearn.decomposition.NMF``.

    Attributes (available after ``fit``)
    ------------------------------------
    players_ : list
        Player ids, in the order used for the columns of every ``M`` and ``H``.
    M_matrices_ : dict[str, np.ndarray]
        Per action: the heatmap matrix, shape ``(n_cells, n_players)``.
    W_factors_ : dict[str, np.ndarray]
        Per action: the prototype heatmaps, shape ``(n_cells, k)``.
    H_factors_ : dict[str, np.ndarray]
        Per action: the player weights, shape ``(k, n_players)``.
    player_vectors_ : dict[player_id, np.ndarray]
        The final player vectors, each of length ``vector_length``.
    """

    def __init__(
        self,
        grid: tuple[int, int] = (50, 50),
        pitch_size: PitchSize = ((0.0, 105.0), (0.0, 68.0)),
        sigma: float = 3.0,
        actions: list[str] | None = None,
        components: list[int] | None = None,
        with_end_coordinates: list[bool] | None = None,
        init_nmf: str = "nndsvd",
        max_iter_nmf: int = 1000,
        random_state: int = 0,
    ) -> None:
        if actions is None:
            actions = DEFAULT_ACTIONS
        if components is None:
            components = DEFAULT_COMPONENTS if actions == DEFAULT_ACTIONS else [4] * len(actions)
        if with_end_coordinates is None:
            with_end_coordinates = [False] * len(actions)

        if not (len(actions) == len(components) == len(with_end_coordinates)):
            raise ValueError(
                "actions, components and with_end_coordinates must have the same length."
            )

        self.grid = tuple(grid)
        self.pitch_size = pitch_size
        self.sigma = sigma
        self.actions = list(actions)
        self.components = list(components)
        self.with_end_coordinates = list(with_end_coordinates)
        self.init_nmf = init_nmf
        self.max_iter_nmf = max_iter_nmf
        self.random_state = random_state

        self.player_names: dict = {}

    @property
    def vector_length(self) -> int:
        """Length of a player vector: the total number of NMF components."""
        return sum(self.components)

    def fit(
        self,
        df_events: pd.DataFrame,
        minutes_played: dict,
        player_names: dict | None = None,
        column_names: dict[str, str] | None = None,
        verbose: bool = False,
    ) -> PlayerVectors:
        """
        Learn the prototype heatmaps and compute a vector for every player.

        Parameters
        ----------
        df_events
            One row per action (SPADL-like format, see ``DEFAULT_COLUMNS``).
        minutes_played
            ``player_id -> total minutes played``, used to normalise per 90 minutes.
        player_names
            Optional ``player_id -> name``, used for plotting and ``to_frame``.
        column_names
            Overrides for ``DEFAULT_COLUMNS`` if your DataFrame uses other names.
        verbose
            Print the matrix shapes for every action.
        """
        columns = DEFAULT_COLUMNS | (column_names or {})
        self._check_columns(df_events, columns)
        self.player_names = player_names or {}

        df = df_events[df_events[columns["action_type"]].isin(self.actions)]
        if df.empty:
            raise ValueError(f"No rows found for the actions {self.actions}.")

        self.players_ = sorted(df[columns["player_id"]].dropna().unique().tolist())
        self.M_matrices_, self.W_factors_, self.H_factors_ = {}, {}, {}

        for action, k, use_end in zip(self.actions, self.components, self.with_end_coordinates):
            df_action = df[df[columns["action_type"]] == action]

            # Step 1: one (flattened) heatmap per player  ->  M
            M = self._build_matrix(df_action, minutes_played, columns, use_end)

            # Step 2: compress the heatmaps with NMF  ->  M ~= W @ H
            W, H = self._factorise(M, k)

            self.M_matrices_[action] = M
            self.W_factors_[action] = W
            self.H_factors_[action] = H

            if verbose:
                print(f"{action}: M={M.shape}, W={W.shape}, H={H.shape}")

        # Step 3: stack the H matrices; column j is then the vector of player j.
        H_all = np.vstack([self.H_factors_[action] for action in self.actions])
        self.player_vectors_ = {
            player_id: H_all[:, j] for j, player_id in enumerate(self.players_)
        }
        return self

    def _build_matrix(
        self,
        df_action: pd.DataFrame,
        minutes_played: dict,
        columns: dict[str, str],
        use_end: bool,
    ) -> np.ndarray:
        """Return M with one flattened heatmap per player as its columns."""
        actions_by_player = dict(list(df_action.groupby(columns["player_id"])))
        no_actions = df_action.iloc[0:0]  # empty frame for players without this action

        heatmap_columns = []
        for player_id in self.players_:
            df_player = actions_by_player.get(player_id, no_actions)
            minutes = float(minutes_played.get(player_id, 0.0))

            def heatmap(x_col: str, y_col: str) -> np.ndarray:
                return action_heatmap(
                    x=df_player[columns[x_col]],
                    y=df_player[columns[y_col]],
                    grid=self.grid,
                    pitch_size=self.pitch_size,
                    sigma=self.sigma,
                    minutes_played=minutes,
                ).ravel()

            column = heatmap("x_start", "y_start")
            if use_end:
                column = np.concatenate([column, heatmap("x_end", "y_end")])
            heatmap_columns.append(column)

        return np.column_stack(heatmap_columns)

    def _factorise(self, M: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
        """Run NMF on M and return (W, H)."""
        if not np.any(M):
            # Nobody performed this action: NMF is undefined, so return zeros.
            return np.zeros((M.shape[0], n_components)), np.zeros((n_components, M.shape[1]))

        # NNDSVD initialisation needs n_components <= min(M.shape).
        init = self.init_nmf
        if init.startswith("nndsvd") and n_components > min(M.shape):
            init = "random"

        model = NMF(
            n_components=n_components,
            init=init,
            max_iter=self.max_iter_nmf,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            W = model.fit_transform(M)
        return W, model.components_

    def _check_columns(self, df: pd.DataFrame, columns: dict[str, str]) -> None:
        needed = ["player_id", "action_type", "x_start", "y_start"]
        if any(self.with_end_coordinates):
            needed += ["x_end", "y_end"]

        missing = [columns[name] for name in needed if columns[name] not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def to_frame(self) -> pd.DataFrame:
        """Player vectors as a DataFrame with columns player_id, player_name, z_1, ..., z_n."""
        weights = pd.DataFrame(
            list(self.player_vectors_.values()),
            columns=[f"z_{i + 1}" for i in range(self.vector_length)],
        )
        ids = list(self.player_vectors_.keys())
        weights.insert(0, "player_name", [self.player_names.get(p) for p in ids])
        weights.insert(0, "player_id", ids)
        return weights
