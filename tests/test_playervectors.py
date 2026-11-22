import numpy as np
import pytest

from playervectors import PlayerVectors

GRID = (10, 10)
N_CELLS = GRID[0] * GRID[1]
N_PLAYERS = 6


def make_model(**kwargs) -> PlayerVectors:
    """A small, fast model: two actions, two components each."""
    params = dict(grid=GRID, sigma=1.0, actions=["shot", "pass"], components=[2, 3])
    params.update(kwargs)
    return PlayerVectors(**params)


class TestInit:
    def test_paper_defaults(self):
        pv = PlayerVectors()
        assert pv.actions == ["shot", "cross", "dribble", "pass"]
        assert pv.components == [4, 4, 5, 5]
        assert pv.with_end_coordinates == [False] * 4
        assert pv.vector_length == 18

    def test_custom_actions_default_to_four_components(self):
        pv = PlayerVectors(actions=["shot", "pass", "tackle"])
        assert pv.components == [4, 4, 4]

    def test_vector_length_is_sum_of_components(self):
        assert make_model(components=[2, 7]).vector_length == 9

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            PlayerVectors(actions=["shot", "pass"], components=[4])
        with pytest.raises(ValueError):
            PlayerVectors(actions=["shot", "pass"], with_end_coordinates=[True])


class TestFit:
    def test_fit_returns_self(self, events, minutes_played):
        pv = make_model()
        assert pv.fit(events, minutes_played) is pv

    def test_players_are_sorted(self, events, minutes_played):
        pv = make_model().fit(events, minutes_played)
        assert pv.players_ == [1, 2, 3, 4, 5, 6]

    def test_matrix_shapes(self, events, minutes_played):
        pv = make_model().fit(events, minutes_played)

        assert pv.M_matrices_["shot"].shape == (N_CELLS, N_PLAYERS)
        assert pv.W_factors_["shot"].shape == (N_CELLS, 2)
        assert pv.H_factors_["shot"].shape == (2, N_PLAYERS)
        assert pv.H_factors_["pass"].shape == (3, N_PLAYERS)

    def test_end_coordinates_double_the_heatmap(self, events, minutes_played):
        pv = make_model(with_end_coordinates=[False, True]).fit(events, minutes_played)

        assert pv.M_matrices_["shot"].shape == (N_CELLS, N_PLAYERS)
        assert pv.M_matrices_["pass"].shape == (2 * N_CELLS, N_PLAYERS)

    def test_every_player_gets_a_vector(self, events, minutes_played):
        pv = make_model().fit(events, minutes_played)

        assert set(pv.player_vectors_) == set(range(1, 7))
        for vector in pv.player_vectors_.values():
            assert vector.shape == (pv.vector_length,)

    def test_vector_is_concatenation_of_h_columns(self, events, minutes_played):
        pv = make_model().fit(events, minutes_played)

        player_3 = pv.players_.index(3)
        expected = np.concatenate([pv.H_factors_["shot"][:, player_3], pv.H_factors_["pass"][:, player_3]])
        np.testing.assert_array_equal(pv.player_vectors_[3], expected)

    def test_factors_are_non_negative(self, events, minutes_played):
        pv = make_model().fit(events, minutes_played)

        for action in pv.actions:
            assert (pv.W_factors_[action] >= 0).all()
            assert (pv.H_factors_[action] >= 0).all()

    def test_nmf_approximates_m(self, events, minutes_played):
        pv = make_model(components=[2, 5]).fit(events, minutes_played)

        M, W, H = pv.M_matrices_["pass"], pv.W_factors_["pass"], pv.H_factors_["pass"]
        relative_error = np.linalg.norm(M - W @ H) / np.linalg.norm(M)
        assert relative_error < 0.5

    def test_fit_is_deterministic(self, events, minutes_played):
        a = make_model().fit(events, minutes_played)
        b = make_model().fit(events, minutes_played)

        for player_id in a.player_vectors_:
            np.testing.assert_array_equal(a.player_vectors_[player_id], b.player_vectors_[player_id])

    def test_action_nobody_performed_gives_zero_weights(self, events, minutes_played):
        pv = make_model(actions=["shot", "tackle"]).fit(events, minutes_played)

        assert not pv.H_factors_["tackle"].any()

    def test_custom_column_names(self, events, minutes_played):
        renamed = events.rename(columns={"type_name": "action", "player_id": "pid"})
        pv = make_model().fit(
            renamed, minutes_played, column_names={"action_type": "action", "player_id": "pid"}
        )

        assert len(pv.player_vectors_) == N_PLAYERS

    def test_missing_column_raises(self, events, minutes_played):
        with pytest.raises(ValueError, match="Missing columns"):
            make_model().fit(events.drop(columns="start_x"), minutes_played)

    def test_end_columns_only_required_when_used(self, events, minutes_played):
        without_end = events.drop(columns=["end_x", "end_y"])

        make_model().fit(without_end, minutes_played)  # fine
        with pytest.raises(ValueError, match="Missing columns"):
            make_model(with_end_coordinates=[False, True]).fit(without_end, minutes_played)

    def test_no_matching_actions_raises(self, events, minutes_played):
        with pytest.raises(ValueError, match="No rows"):
            make_model(actions=["tackle"], components=[2]).fit(events, minutes_played)


class TestToFrame:
    def test_columns_and_rows(self, events, minutes_played, player_names):
        df = make_model().fit(events, minutes_played, player_names).to_frame()

        assert list(df.columns) == ["player_id", "player_name", "z_1", "z_2", "z_3", "z_4", "z_5"]
        assert len(df) == N_PLAYERS

    def test_names_and_values_match(self, events, minutes_played, player_names):
        pv = make_model().fit(events, minutes_played, player_names)
        row = pv.to_frame().set_index("player_id").loc[4]

        assert row["player_name"] == "Player 4"
        np.testing.assert_allclose(row.filter(like="z_").to_numpy(dtype=float), pv.player_vectors_[4])

    def test_unknown_names_are_empty(self, events, minutes_played):
        df = make_model().fit(events, minutes_played).to_frame()

        assert df["player_name"].isna().all()
