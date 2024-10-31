import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from phd.variables.calculate_precipitation import CalculatePrecipitation
from phd.variables.calculate_vorticity import CalculateVorticity
from phd.variables.colorbar_precipitation_light import ColorbarPrecipitation
from phd.utils.label_and_scale import add_corner_label, add_length_scale
from phd.utils.timer import Timer

class RainfallRatePlotter:
    def __init__(self, filename, delta, min_lat, max_lat, min_lon, max_lon, text_lines, event_type):
        self.filename = filename
        self.delta = delta
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.text_lines = text_lines
        self.event_type = event_type
        self.base_filename = None
        self.current_time = None
        self.previous_filename = None
        self.Rainfall_Rate_surface = None
        self.lats = None
        self.lons = None
        self.crs = ccrs.Mercator()

        self._initialize_filenames()

    def _initialize_filenames(self):
        """ Parse the current and previous file names. """
        match = re.search(r'wrfout_d\d\d_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', self.filename)
        if not match:
            raise ValueError("Filename does not match expected format.")
        self.base_filename = match.group()
        datetime_str = '_'.join(self.base_filename.split('_')[2:4])
        self.current_time = pd.to_datetime(datetime_str, format='%Y-%m-%d_%H-%M-%S')
        self.previous_time = self.current_time - pd.Timedelta(minutes=self.delta)
        previous_datetime_str = self.previous_time.strftime('%Y-%m-%d_%H-%M-%S')
        base_parts = self.base_filename.split('_')
        self.previous_filename = os.path.join(os.path.dirname(self.filename), f"{base_parts[0]}_{base_parts[1]}_{previous_datetime_str}.nc")

    def _check_files_exist(self):
        """ Ensure that both current and previous WRF files exist. """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        if not os.path.isfile(self.previous_filename):
            raise FileNotFoundError(f"File not found: {self.previous_filename}")

    def _calculate_rainfall_rate(self):
        """ Calculate the rainfall rate from WRF output files. """
        precip_calc = CalculatePrecipitation(self.filename, self.previous_filename, self.delta)
        precip_calc.run()
        self.Rainfall_Rate_surface = precip_calc.Rainfall_Rate_surface
        self.lats = precip_calc.lats
        self.lons = precip_calc.lons

    def _calculate_absolute_vorticity(self):
        """ Calculate absolute vorticity from WRF output. """
        avort_calc = CalculateVorticity(self.filename)
        self.abs_vorticity, _, _ = avort_calc.run()

    def plot_rainfall_rate_vorticity(self, ax):
        """ Plot rainfall rate and absolute vorticity on the provided axis. """
        # Set up map features
        ax.add_feature(cfeature.LAND, facecolor='#F3EFE3', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#CDEDFF', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, color='gray', zorder=2)

        # Set extent
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())

        # Plot precipitation
        precip_cmap = ColorbarPrecipitation()
        rain_plot = ax.contourf(self.lons, self.lats, self.Rainfall_Rate_surface.squeeze(),
                                levels=precip_cmap.levels, cmap=precip_cmap.cmap, norm=precip_cmap.norm,
                                transform=ccrs.PlateCarree(), extend='both', zorder=1)

        # Plot vorticity contours
        if self.event_type == "Type 1":
            levels_vort = np.linspace(-18e-3, 18e-3, 12)
        elif self.event_type == "Type 2":
            levels_vort = np.linspace(-12e-3, 12e-3, 12)
            
        levels_vort = levels_vort[levels_vort != 0]  # Remove zero
        ax.contour(self.lons, self.lats, self.abs_vorticity, levels=levels_vort, colors='black', linewidths=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)

        # Add corner label (before scale)
        add_corner_label(ax, self.text_lines, self.event_type)

        # Add the length scale last, ensuring it is drawn last and thus on top
        add_length_scale(ax)

        return rain_plot, precip_cmap

    @Timer
    def run(self, ax):
        """ Run all steps and generate the plot. """
        self._check_files_exist()
        self._calculate_rainfall_rate()
        self._calculate_absolute_vorticity()
        return self.plot_rainfall_rate_vorticity(ax)

    @staticmethod
    def create_plots(plot_configs, fig_filename='figure_comparison_with_colorbar.png'):
        """ Static method to create and display plots for multiple configurations. """
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(11, 8), subplot_kw={'projection': ccrs.Mercator()})
        plt.subplots_adjust(wspace=0.05)

        # Ensure both subplots are exactly the same size
        for ax in axs:
            ax.set_aspect('auto')

        # Run the plotter for each configuration
        all_plots = []
        precip_cmap = None  # Placeholder for colormap
        for ax, config in zip(axs, plot_configs):
            plotter = RainfallRatePlotter(**config)
            rain_plot, precip_cmap = plotter.run(ax)
            all_plots.append(rain_plot)

        # Add the colorbar
        if precip_cmap:
            precip_cmap.add_colorbar(all_plots[0], axs)

        # Save and display the figure
        plt.savefig(fig_filename, dpi=200)
        plt.show()


if __name__ == "__main__":
    # Plot configurations
    plot_configs = [
        {
            "filename": '/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2011-11-29_10-50-00.nc',
            "delta": 10,
            "min_lat": 52.5,
            "max_lat": 54,
            "min_lon": -6.6,
            "max_lon": -4.4,
            "text_lines": ["(a) Modelled:", " 29 Nov 2011"],
            "event_type": "Type 1"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/2005-11-24/wrfout_d03_2005-11-24_14-50-00.nc',
            "delta": 10,
            "min_lat": 51.5,
            "max_lat": 53.0,
            "min_lon": -0.3,
            "max_lon": 1.9,
            "text_lines": ["(b) Modelled:", " 24 Nov 2005"],
            "event_type": "Type 2"
        }
    ]

    # Generate and display plots
    RainfallRatePlotter.create_plots(plot_configs)