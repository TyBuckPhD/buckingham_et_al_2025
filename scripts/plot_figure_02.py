import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from buckingham_et_al_2025.variables.colorbar_precipitation import (
    ColorbarPrecipitation,
)
from buckingham_et_al_2025.variables.get_variables_wrf import GetVariablesWRF
from buckingham_et_al_2025.utils.label_and_scale import (
    add_corner_label,
    add_length_scale,
)
from buckingham_et_al_2025.utils.timer import Timer


class RainfallRatePlotter:
    """
    Class for plotting rainfall rate and vorticity data from WRF model outputs with an inset zoom feature.

    This class processes a WRF netCDF file to extract rainfall rate data and computes absolute vorticity
    (at a specified height) using a dedicated WRF data processor. It then generates a plot that displays the
    rainfall rate as filled contours with overlaid vorticity contours. An inset zoom plot is also created to show
    a detailed view of a specified sub-region. The inset bounds can either be provided explicitly or computed
    by default.

    Attributes:
      filename (str): Path to the input WRF data file.
      delta (int): A parameter to adjust analysis resolution or processing (usage context specific).
      min_lat (float): Minimum latitude for the main plot extent.
      max_lat (float): Maximum latitude for the main plot extent.
      min_lon (float): Minimum longitude for the main plot extent.
      max_lon (float): Maximum longitude for the main plot extent.
      text_lines (list of str): Lines of text used to annotate the plot.
      event_type (str): Identifier for the event type, which influences vorticity contour levels.
      vorticity_height (int): Height (in meters) at which the absolute vorticity is calculated.
      Rainfall_Rate_surface (ndarray): 2D array containing the rainfall rate data.
      lats (ndarray): 2D array of latitude values corresponding to the data grid.
      lons (ndarray): 2D array of longitude values corresponding to the data grid.
      abs_vorticity (ndarray): 2D array containing computed absolute vorticity values.
      crs (Cartopy CRS): Cartopy Mercator projection used for plotting.
      min_lat_inset (float): Minimum latitude for the inset plot extent.
      max_lat_inset (float): Maximum latitude for the inset plot extent.
      min_lon_inset (float): Minimum longitude for the inset plot extent.
      max_lon_inset (float): Maximum longitude for the inset plot extent.
      wrf_data_processor (GetVariablesWRF): Instance for extracting variables from the WRF file.

    Methods:
      _check_file_exist:
        Validates that the input file exists; raises an error if not found.

      _load_lat_lons:
        Loads the latitude and longitude arrays using the WRF data processor.

      _load_rainfall_rate:
        Loads the rainfall rate data from the WRF file at a specified time index.

      _load_absolute_vorticity:
        Computes the absolute vorticity from the WRF data at the given height and time index.

      plot_rainfall_rate_vorticity:
        Generates a plot that overlays rainfall rate contours and vorticity contours on the main axis,
        and includes an inset zoom plot of a specified sub-region.

      _get_default_inset_extent:
        Calculates default inset boundaries if specific inset bounds are not provided.

      run:
        Executes all steps (file check, data loading, plotting) and returns the generated plot.

      create_plots (staticmethod):
        Creates and displays plots for multiple configurations, arranging them in a single figure,
        and saves the resulting figure to disk.
    """

    def __init__(
        self,
        filename,
        delta,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        text_lines,
        event_type,
        min_lat_inset=None,
        max_lat_inset=None,
        min_lon_inset=None,
        max_lon_inset=None,
        vorticity_height=500,
    ):
        self.filename = filename
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.text_lines = text_lines
        self.event_type = event_type
        self.vorticity_height = (
            vorticity_height  # Height for vorticity calculation
        )
        self.Rainfall_Rate_surface = None
        self.lats = None
        self.lons = None
        self.abs_vorticity = None
        self.crs = ccrs.Mercator()

        # Inset parameters
        self.min_lat_inset = min_lat_inset
        self.max_lat_inset = max_lat_inset
        self.min_lon_inset = min_lon_inset
        self.max_lon_inset = max_lon_inset

        # Instantiate the WRF data processor for variable loading
        self.wrf_data_processor = GetVariablesWRF(self.filename)

    def _check_file_exist(self):
        """Ensure the input file exists."""
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")

    def _load_lat_lons(self):
        """Load latitude and longitude using the WRF data processor."""
        self.lats, self.lons = self.wrf_data_processor.get_lat_lons()

    def _load_rainfall_rate(self):
        """Load rainfall rate from radar reflectivity using the WRF data processor."""
        self.Rainfall_Rate_surface = self.wrf_data_processor.get_rainfall_rate(
            timeidx=5
        )

    def _load_absolute_vorticity(self):
        """Calculate absolute vorticity using the WRF data processor."""
        self.abs_vorticity = self.wrf_data_processor.get_absolute_vorticity(
            height=self.vorticity_height, timeidx=5
        )

    def plot_rainfall_rate_vorticity(self, ax):
        """Plot rainfall rate and absolute vorticity on the provided axis with an inset zoom plot."""
        # Set up map features
        ax.add_feature(cfeature.LAND, color="#F3EFE3", zorder=0)
        ax.add_feature(cfeature.OCEAN, color="#CDEDFF", zorder=0)
        ax.add_feature(
            cfeature.COASTLINE.with_scale("10m"),
            linewidth=0.5,
            color="gray",
            zorder=2,
        )

        # Set extent
        ax.set_extent(
            [self.min_lon, self.max_lon, self.min_lat, self.max_lat],
            crs=ccrs.PlateCarree(),
        )

        # Plot precipitation
        precip_cmap = ColorbarPrecipitation()
        rain_plot = ax.contourf(
            self.lons,
            self.lats,
            self.Rainfall_Rate_surface.squeeze(),
            levels=precip_cmap.levels,
            cmap=precip_cmap.cmap,
            norm=precip_cmap.norm,
            transform=ccrs.PlateCarree(),
            extend="both",
            zorder=1,
        )

        # Plot vorticity contours
        if self.event_type == "Type 1":
            levels_vort = np.linspace(-18e-3, 18e-3, 12)
        elif self.event_type == "Type 2":
            levels_vort = np.linspace(-12e-3, 12e-3, 12)
        else:
            levels_vort = np.linspace(-12e-3, 12e-3, 12)  # Default levels

        levels_vort = levels_vort[levels_vort != 0]  # Remove zero
        ax.contour(
            self.lons,
            self.lats,
            self.abs_vorticity,
            levels=levels_vort,
            colors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Add corner label (before scale)
        add_corner_label(ax, self.text_lines, self.event_type)

        # Add the length scale last, ensuring it is drawn last and thus on top
        add_length_scale(ax)

        # --- Add inset plot (zoomed-in section) ---
        inset_extent = (
            [
                self.min_lon_inset,
                self.max_lon_inset,
                self.min_lat_inset,
                self.max_lat_inset,
            ]
            if self.min_lon_inset
            and self.max_lon_inset
            and self.min_lat_inset
            and self.max_lat_inset
            else self._get_default_inset_extent()
        )

        width, height = 0.3, 0.3  # Relative size of the inset
        inset_left = 0.96 - width
        inset_bottom = 0.03

        # Create inset
        inset_ax = ax.inset_axes(
            [inset_left, inset_bottom, width, height],
            transform=ax.transAxes,
            projection=ccrs.Mercator(),
        )
        inset_ax.set_extent(inset_extent, crs=ccrs.PlateCarree())

        # Plot inset features
        inset_ax.add_feature(cfeature.LAND, color="#F3EFE3", zorder=0)
        inset_ax.add_feature(cfeature.OCEAN, color="#CDEDFF", zorder=0)
        inset_ax.add_feature(
            cfeature.COASTLINE.with_scale("10m"),
            linewidth=0.5,
            color="gray",
            zorder=2,
        )
        inset_ax.contourf(
            self.lons,
            self.lats,
            self.Rainfall_Rate_surface.squeeze(),
            levels=precip_cmap.levels,
            cmap=precip_cmap.cmap,
            norm=precip_cmap.norm,
            transform=ccrs.PlateCarree(),
            extend="both",
            zorder=1,
        )
        inset_ax.contour(
            self.lons,
            self.lats,
            self.abs_vorticity,
            levels=levels_vort,
            colors="black",
            linewidths=1.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Remove tick marks and labels from inset
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        # Add a thicker black border for the inset
        inset_ax.spines["geo"].set_linewidth(2)

        # Add a rectangle on the main plot to show the inset location
        rect = patches.Rectangle(
            (inset_extent[0], inset_extent[2]),
            inset_extent[1] - inset_extent[0],
            inset_extent[3] - inset_extent[2],
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)

        return rain_plot, precip_cmap

    def _get_default_inset_extent(self):
        inset_margin_lon = (self.max_lon - self.min_lon) * 0.2
        inset_margin_lat = (self.max_lat - self.min_lat) * 0.2
        inset_min_lon = self.min_lon + inset_margin_lon
        inset_max_lon = self.max_lon - inset_margin_lon
        inset_min_lat = self.min_lat + inset_margin_lat
        inset_max_lat = self.max_lat - inset_margin_lat
        return [inset_min_lon, inset_max_lon, inset_min_lat, inset_max_lat]

    @Timer
    def run(self, ax):
        """Run all steps and generate the plot."""
        self._check_file_exist()
        self._load_lat_lons()
        self._load_rainfall_rate()
        self._load_absolute_vorticity()
        return self.plot_rainfall_rate_vorticity(ax)

    @staticmethod
    def create_plots(plot_configs):
        """Static method to create and display plots for multiple configurations."""
        num_plots = len(plot_configs)
        fig, axs = plt.subplots(
            1,
            num_plots,
            figsize=(11, 8),
            subplot_kw={"projection": ccrs.Mercator()},
        )
        plt.subplots_adjust(wspace=0.05)

        if num_plots == 1:
            axs = [axs]

        for ax in axs:
            ax.set_aspect("auto")

        all_plots = []
        precip_cmap = None
        for ax, config in zip(axs, plot_configs):
            plotter = RainfallRatePlotter(**config)
            rain_plot, precip_cmap = plotter.run(ax)
            all_plots.append(rain_plot)

        # Add the colorbar
        if precip_cmap:
            precip_cmap.add_colorbar(all_plots[0], axs)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig_filename = os.path.join(script_dir, "../figures/figure_04.png")

        # Save and display the figure
        plt.savefig(
            fig_filename, dpi=200, bbox_inches="tight", pad_inches=0.05
        )
        plt.show()


if __name__ == "__main__":
    # Plot configurations
    plot_configs = [
        {
            "filename": "/Volumes/Samsung_T5/phd_data/wrfout_d03_2011-11-29_10-00-00.nc",
            "delta": 10,
            "min_lat": 52.5,
            "max_lat": 54.0,
            "min_lon": -6.6,
            "max_lon": -4.4,
            "text_lines": ["(a) Modelled:", "29 Nov 2011"],
            "event_type": "Type 1",
            # Inset bounds
            "min_lat_inset": 53.2,
            "max_lat_inset": 53.48,
            "min_lon_inset": -5.75,
            "max_lon_inset": -5.35,
        },
        {
            "filename": "/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2005-11-24_14-00-00.nc",
            "delta": 10,
            "min_lat": 51.5,
            "max_lat": 53.0,
            "min_lon": -0.3,
            "max_lon": 1.9,
            "text_lines": ["(b) Modelled:", "24 Nov 2005"],
            "event_type": "Type 2",
            # Inset bounds
            "min_lat_inset": 52.02,
            "max_lat_inset": 52.24,
            "min_lon_inset": 0.06,
            "max_lon_inset": 0.4,
        },
    ]

    # Generate and display plots
    RainfallRatePlotter.create_plots(plot_configs)
