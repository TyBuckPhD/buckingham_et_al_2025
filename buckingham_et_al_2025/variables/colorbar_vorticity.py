import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class ColorbarVorticity:
    """
    Class for creating a customized vorticity colormap and corresponding colorbar for visualizing
    absolute vorticity data.

    This class generates a discrete colormap (ListedColormap) based on specified levels and colors,
    applying an optional alpha (opacity) value to all colors. If no levels or colors are provided, default
    values are used based on the vorticity type. The class also provides a method to add a horizontal
    colorbar to a plot, with ticks set to display every other level.

    Attributes:
      vorticity_type (str): Specifies the vorticity type ("Type 1" or "Type 2"); used to determine default levels.
      levels (array-like): List or array of level boundaries for the colormap.
      colors (list): List of color strings or values used for constructing the colormap.
      alpha (float): Opacity value (between 0.0 and 1.0) applied to all colors.
      under_color (RGBA tuple or str): Color for data values below the lowest level.
      over_color (RGBA tuple or str): Color for data values above the highest level.
      orientation (str): Orientation of the colorbar (e.g., "horizontal").
      cmap (ListedColormap): The constructed colormap.
      norm (BoundaryNorm): The normalization object mapping data values to colormap intervals.

    Methods:
      _apply_alpha(color, alpha):
          Converts a color to an RGBA tuple with the specified alpha.
      _get_default_colors():
          Generates a default list of colors for the colormap based on the number of intervals defined by levels.
      _create_colormap():
          Constructs the ListedColormap and corresponding BoundaryNorm based on levels and colors, and sets the
          under and over colors.
      add_colorbar(plot, ax, pad=0.02):
          Adds a horizontal colorbar to the given Axes object using the generated colormap and normalization.
          The colorbar displays every other tick from the defined levels with formatted labels.
    """

    def __init__(
        self,
        vorticity_type="Type 1",
        levels=None,
        colors=None,
        alpha=1.0,
        under_color="darkblue",
        over_color="black",
        orientation="horizontal",
    ):
        if levels is None:
            if vorticity_type == "Type 1":
                self.levels = np.arange(-18, 18 + 3, 3)
            elif vorticity_type == "Type 2":
                self.levels = np.arange(-18, 18 + 3, 3)
            else:
                raise ValueError(
                    "Invalid vorticity type. Choose either 'Type 1' or 'Type 2'."
                )
        else:
            self.levels = levels

        self.alpha = alpha  # Opacity input (0.0 to 1.0)

        if colors is None:
            self.colors = self._get_default_colors()
        else:
            self.colors = colors

        self.under_color = self._apply_alpha(under_color, self.alpha)
        self.over_color = self._apply_alpha(over_color, self.alpha)
        self.orientation = orientation
        self.cmap = None
        self.norm = None

        self._create_colormap()

    def _apply_alpha(self, color, alpha):
        """Convert a color to RGBA with specified alpha."""
        rgba = mcolors.to_rgba(color, alpha=alpha)
        return rgba

    def _get_default_colors(self):
        num_intervals = len(self.levels) - 1
        zero_index = np.abs(self.levels).argmin()

        num_neg_colors = zero_index - 1
        num_pos_colors = num_intervals - zero_index - 1

        cmap_neg = plt.get_cmap("Blues_r", num_neg_colors + 2)
        neg_colors = [
            self._apply_alpha(cmap_neg(i), self.alpha)
            for i in range(1, num_neg_colors + 1)
        ]

        white_colors = [
            self._apply_alpha("#FFFFFF", self.alpha),
            self._apply_alpha("#FFFFFF", self.alpha),
        ]

        cmap_pos = plt.get_cmap("Greys", num_pos_colors + 2)
        pos_colors = [
            self._apply_alpha(cmap_pos(i), self.alpha)
            for i in range(1, num_pos_colors + 1)
        ]

        colors = neg_colors + white_colors + pos_colors

        return colors

    def _create_colormap(self):
        self.cmap = mcolors.ListedColormap(self.colors, name="Vorticity")
        self.cmap.set_under(self.under_color)
        self.cmap.set_over(self.over_color)
        self.norm = mcolors.BoundaryNorm(
            self.levels, ncolors=self.cmap.N, clip=False
        )

    def add_colorbar(self, plot, ax, pad=0.02):
        """
        Add a colorbar that always displays every other tick, preserving the full color range.
        """
        cbar = plt.colorbar(
            plot,
            ax=ax,
            orientation="horizontal",
            pad=pad,
            shrink=0.85,
            aspect=20,
            extend="both",
        )

        # Select every other level for ticks
        displayed_levels = [
            level for i, level in enumerate(self.levels) if i % 2 == 0
        ]

        # Apply these chosen ticks
        cbar.set_ticks(displayed_levels)

        # Create labels for the displayed ticks
        tick_labels = [
            f"{lvl:.0f}" if not np.isclose(lvl, 0.0) else "0"
            for lvl in displayed_levels
        ]
        cbar.set_ticklabels(tick_labels)

        # Set label and formatting
        label = "Absolute Vorticity ($10^{-3}$ s$^{-1}$)"
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        return cbar
