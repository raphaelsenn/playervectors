import matplotlib

matplotlib.use("Agg")  # draw off-screen, no window needed

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

pytest.importorskip("databallpy")

from playervectors import PlayerVectors
from playervectors.visualize import (
    _layout_component_panels,
    _readable_name,
    plot_player_weights,
    plot_principal_components,
    plot_weight_distribution,
)


@pytest.fixture
def fitted(events, minutes_played, player_names) -> PlayerVectors:
    pv = PlayerVectors(
        grid=(10, 10),
        actions=["shot", "pass"],
        components=[3, 2],
        with_end_coordinates=[False, True],
    )
    return pv.fit(events, minutes_played, player_names)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestPlots:
    def test_plot_principal_components(self, fitted):
        assert isinstance(plot_principal_components(fitted), Figure)

    def test_plot_principal_components_needs_enough_axes(self, fitted):
        _, axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="Need at least"):
            plot_principal_components(fitted, ax=axes)

    def test_plot_weight_distribution(self, fitted):
        assert isinstance(plot_weight_distribution(fitted), Figure)

    def test_plot_player_weights(self, fitted):
        fig = plot_player_weights(fitted, player_id=1)
        assert fig.axes[0].get_title() == "Weights for Player 1"

    def test_plot_player_weights_unknown_player(self, fitted):
        with pytest.raises(KeyError):
            plot_player_weights(fitted, player_id=999)


class TestLayout:
    def test_one_panel_per_component_plus_one_per_end_map(self, fitted):
        # 3 shot components (1 panel each) + 2 pass components (2 panels each)
        panels = _layout_component_panels(fitted, n_cols=4)
        assert len(panels) == 3 + 2 * 2

    def test_start_end_pairs_stay_on_one_row(self, fitted):
        panels = _layout_component_panels(fitted, n_cols=4)
        cells = [cell for cell, _, _ in panels]

        # shots fill cells 0-2; the first pass pair would straddle cells 3/4,
        # so it moves to the next row.
        assert cells == [0, 1, 2, 4, 5, 6, 7]

    def test_titles_use_global_component_numbers(self, fitted):
        titles = [title for _, _, title in _layout_component_panels(fitted, n_cols=4)]

        assert titles[0] == "Component 1 (shot)"
        assert titles[3] == "Component 4 (pass, start)"
        assert titles[4] == "Component 4 (pass, end)"
        assert titles[-1] == "Component 5 (pass, end)"


class TestReadableName:
    def test_decodes_escaped_accents(self):
        assert _readable_name("Andr\\u00e9s") == "Andrés"

    def test_leaves_normal_names_alone(self):
        assert _readable_name("Thomas Müller") == "Thomas Müller"
