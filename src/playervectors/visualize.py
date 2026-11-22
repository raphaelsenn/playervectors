"""
Plots for a fitted `PlayerVectors` model.

* ``plot_principal_components`` – the prototype heatmaps (columns of every W).
* ``plot_weight_distribution``  – how the weights of all players are spread per component.
* ``plot_player_weights``       – the player vector of a single player.

Component numbers in all plots match the ``z_1 ... z_n`` entries of a player vector.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from databallpy.visualize import plot_soccer_pitch

from playervectors.playervectors import PlayerVectors


def plot_principal_components(
    player_vectors: PlayerVectors,
    ax: np.ndarray | list[Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    n_cols: int = 4,
) -> Figure:
    """
    Draw every learned prototype heatmap on a pitch.

    Actions fitted with end coordinates get two panels per component (start and
    end), which are always kept next to each other on the same row.
    """
    panels = _layout_component_panels(player_vectors, n_cols)
    n_cells = panels[-1][0] + 1

    if ax is None:
        n_rows = -(-n_cells // n_cols)  # ceiling division
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize or (4 * n_cols, 4 * n_rows))
    axes = np.asarray(ax).reshape(-1)
    fig = axes[0].figure

    if len(axes) < n_cells:
        raise ValueError(f"Need at least {n_cells} axes, got {len(axes)}.")

    for a in axes:
        a.axis("off")  # cells that stay empty remain blank
    for cell, heatmap, title in panels:
        _plot_heatmap_on_pitch(fig, axes[cell], heatmap, title, player_vectors.pitch_size)

    fig.tight_layout()
    return fig


def plot_weight_distribution(
    player_vectors: PlayerVectors,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (16, 8),
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> Figure:
    """Boxplot of the weights of all players, one box per component."""
    weights = player_vectors.to_frame().filter(like="z_")

    fig, ax = _figure_and_axes(ax, figsize)
    sns.boxplot(data=weights, ax=ax)

    ax.set_title("Weight distribution per component")
    ax.set_xlabel("Component")
    ax.set_ylabel("Weight")
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    return fig


def plot_player_weights(
    player_vectors: PlayerVectors,
    player_id,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    ylim: float | None = 0.6,
) -> Figure:
    """Scatter plot of one player's vector: one dot per component."""
    if player_id not in player_vectors.player_vectors_:
        raise KeyError(f"Unknown player_id: {player_id}")

    vector = player_vectors.player_vectors_[player_id]
    name = _readable_name(player_vectors.player_names.get(player_id, str(player_id)))
    component_numbers = np.arange(1, len(vector) + 1)

    fig, ax = _figure_and_axes(ax, figsize)
    ax.scatter(component_numbers, vector, color="red", edgecolors="black")

    ax.set_xticks(component_numbers)
    ax.set_xlabel("Component")
    ax.set_ylabel("Weight")
    ax.set_title(f"Weights for {name}")
    if ylim is not None:
        ax.set_ylim(0.0, ylim)

    fig.tight_layout()
    return fig


def _layout_component_panels(
    player_vectors: PlayerVectors, n_cols: int
) -> list[tuple[int, np.ndarray, str]]:
    """
    Decide which grid cell every heatmap goes into.

    Returns a list of ``(cell_index, heatmap, title)``.
    """
    grid = player_vectors.grid
    cells_per_map = grid[0] * grid[1]

    panels = []
    cell = 0
    number = 1  # global component number, matches z_<number>

    for action, use_end in zip(player_vectors.actions, player_vectors.with_end_coordinates):
        W = player_vectors.W_factors_[action]

        for k in range(W.shape[1]):
            column = W[:, k]
            if use_end:
                maps = [
                    (column[:cells_per_map].reshape(grid), f"Component {number} ({action}, start)"),
                    (column[cells_per_map:].reshape(grid), f"Component {number} ({action}, end)"),
                ]
            else:
                maps = [(column.reshape(grid), f"Component {number} ({action})")]

            # Start a new row if the start/end pair would not fit on this one.
            if cell % n_cols + len(maps) > n_cols:
                cell += n_cols - cell % n_cols

            for heatmap, title in maps:
                panels.append((cell, heatmap, title))
                cell += 1
            number += 1

    if not panels:
        raise ValueError("No components to plot.")
    return panels


def _plot_heatmap_on_pitch(
    fig: Figure,
    ax: Axes,
    heatmap: np.ndarray,
    title: str,
    pitch_size: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    # databallpy always draws the pitch centred at (0, 0), so we centre the
    # heatmap as well, whatever coordinate system the data was in.
    (x_min, x_max), (y_min, y_max) = pitch_size
    half_length, half_width = (x_max - x_min) / 2, (y_max - y_min) / 2

    ax.imshow(
        heatmap,
        extent=[-half_length, half_length, -half_width, half_width],
        origin="lower",
        cmap="Blues",
        alpha=0.7,
    )
    plot_soccer_pitch(
        field_dimen=(2 * half_length, 2 * half_width), pitch_color="white", fig=fig, ax=ax
    )

    ax.set_xlim(-half_length, half_length)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def _figure_and_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _readable_name(name: str) -> str:
    """Wyscout stores accents as literal escapes, e.g. 'Andr\\u00e9s' -> 'Andrés'."""
    if "\\u" in name:
        return name.encode("latin-1", "backslashreplace").decode("unicode_escape")
    return name
