import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class ColorbarVorticity:
    def __init__(self, vorticity_type='Type 1', levels=None, colors=None, under_color='black', over_color='black'):
        if levels is None:
            if vorticity_type == 'Type 1':
                self.levels = np.arange(-0.018, 0.018 + 0.003, 0.003)
            elif vorticity_type == 'Type 2':
                self.levels = np.arange(-0.012, 0.012 + 0.002, 0.002)
            else:
                raise ValueError("Invalid vorticity type. Choose either 'Type 1' or 'Type 2'.")
        else:
            self.levels = levels

        if colors is None:
            self.colors = self._get_default_colors()
        else:
            self.colors = colors

        self.under_color = under_color
        self.over_color = over_color
        self.cmap = None
        self.norm = None

        self._create_colormap()

    def _get_default_colors(self):
        num_intervals = len(self.levels) - 1
        zero_index = np.abs(self.levels).argmin()

        num_neg_colors = zero_index - 1
        num_pos_colors = num_intervals - zero_index - 1

        cmap_neg = plt.get_cmap('Blues_r', num_neg_colors + 2)
        neg_colors = [cmap_neg(i) for i in range(1, num_neg_colors + 1)]

        white_colors = ['#FFFFFF', '#FFFFFF']

        cmap_pos = plt.get_cmap('Reds', num_pos_colors + 2)
        pos_colors = [cmap_pos(i) for i in range(1, num_pos_colors + 1)]

        colors = neg_colors + white_colors + pos_colors

        return colors

    def _create_colormap(self):
        self.cmap = mcolors.ListedColormap(self.colors, name='Vorticity')
        self.cmap.set_under(self.under_color)
        self.cmap.set_over(self.over_color)
        self.norm = mcolors.BoundaryNorm(self.levels, ncolors=self.cmap.N, clip=False)

    def add_colorbar(self, plot, ax, label='Vorticity'):
        cbar = plt.colorbar(
            plot, ax=ax, orientation='horizontal', pad=0.02,
            shrink=0.7, aspect=30, extend='both'
        )
        cbar.set_ticks(self.levels)
        tick_labels = []
        for level in self.levels:
            if np.isclose(level, 0.0):
                tick_labels.append('0')
            else:
                tick_labels.append(f"{level:.3f}")
        cbar.set_ticklabels(tick_labels)
        cbar.set_label(label)
        return cbar