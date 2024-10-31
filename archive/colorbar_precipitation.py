import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class ColorbarPrecipitation:
    def __init__(self, levels=None, colors=None, under_color='none', over_color='#808080'):
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
            self.levels = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 
                           10, 12, 14, 20, 28, 38, 48, 60, 76, 96]
        else:
            self.levels = levels

        if colors is None:
            self.colors = [
                '#92ccff',  # Light blue
                '#4aa8ff',  # Medium blue
                '#0087FF',  # Blue
                '#02DAC6',  # Turquoise
                '#009304',  # Green
                '#4EDD51',  # Light green
                '#86FAA9',  # Pale green
                '#FBFF10',  # Yellow
                '#EFD614',  # Gold
                '#FFCA49',  # Pale orange
                '#F3AE2C',  # Orange
                '#E3911E',  # Dark orange
                '#FF0000',  # Red
                '#E60000',  # Redder
                '#860000',  # Dark red
                '#8A0000',  # Light marroon
                '#530000',  # Marroon
                '#883131',  # Burnt
                '#C77070',  # Pale burnt
                '#F1A9A9',  # Pale pink
                '#FEF0F0',  # Off white
                '#DACEFF'  # Lilac
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

    def add_colorbar(self, rain_plot, ax, label='Rainfall Rate (mm/hr)'):
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