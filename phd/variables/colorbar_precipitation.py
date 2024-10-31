import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class ColorbarPrecipitation:
    def __init__(self, levels=None, colors=None, under_color=(0, 0, 0, 0), over_color='#808080'):
        """
        Initializes the PrecipitationColormap class with default or custom levels and colors.

        Parameters:
            levels: Optional list of levels for the colormap.
            colors: Optional list of colors corresponding to the levels.
            under_color: Color for values below the minimum level.
            over_color: Color for values above the maximum level.
        """
        # Default levels and colors if none provided
        if levels is None:
            self.levels = [0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 32, 64, 96]
        else:
            self.levels = levels

        if colors is None:
            self.colors = [
                '#92ccff',  # Light blue
                '#4ABBFF',  # Blue
                '#03A0FF',  # Dark blue
                '#02DAC6',  # Turquoise
                '#86FAA9',  # Light green
                '#FBFF10',  # Yellow
                '#F3AE2C',  # Orange
                '#FF0000',  # Red
                '#FF55AD',  # Pink
                '#E3E3E3',  # Light gray
                '#ADAAAA'   # Gray
            ]
        else:
            self.colors = colors

        self.under_color = under_color
        self.over_color = over_color
        self.cmap = None
        self.norm = None

        self._create_colormap()

    def _create_colormap(self):
        """
        Creates the colormap and normalization based on levels and colors.
        """
        self.cmap = mcolors.ListedColormap(self.colors, name='RainfallRate')
        self.norm = mcolors.BoundaryNorm(self.levels, ncolors=self.cmap.N, clip=False)
        self.cmap.set_under(self.under_color)
        self.cmap.set_over(self.over_color)

    def add_colorbar(self, rain_plot, ax, label='Precipitation rate (mm hr$^{-1}$)'):
        """
        Adds a customized colorbar to the precipitation plot.

        Parameters:
            rain_plot: The QuadContourSet returned by contourf.
            ax: The Axes object to which the colorbar will be added.
            label: Label for the colorbar.

        Returns:
            cbar: The colorbar object.
        """
        # Create the colorbar
        cbar = plt.colorbar(
            rain_plot, ax=ax, orientation='horizontal', pad=0.02,
            shrink=0.7, aspect=30, extend='both'
        )
        # Set the ticks and labels
        cbar_ticks = self.levels
        cbar_ticklabels = [str(level) for level in self.levels]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.set_label(label)
        return cbar