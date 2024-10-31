import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from phd.variables.calculate_precipitation import CalculatePrecipitation
from phd.variables.colorbar_precipitation_light import ColorbarPrecipitation
from phd.utils.label_and_scale import add_corner_label, add_length_scale
from phd.utils.timer import Timer
from phd.ukmo_composites.ukmo_composite_file_converter import UKMOCompositeProcessor

class RainfallRatePlotter:
    def __init__(self, filename, delta=None, min_lat=None, max_lat=None, min_lon=None, max_lon=None, text_lines=None, event_type=None):
        self.filename = filename
        self.delta = delta  # Used to distinguish between WRF and UKMO configurations
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.text_lines = text_lines
        self.event_type = event_type

        # Initialize attributes for WRF files only
        if self.delta is not None:
            self.base_filename = None
            self.current_time = None
            self.previous_filename = None
            self.Rainfall_Rate_surface = None
            self.lats = None
            self.lons = None
            self._initialize_filenames()

    def _initialize_filenames(self):
        """ Parse the current and previous file names for WRF files. """
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
        """ Ensure that files exist for both UKMO and WRF configurations. """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        if self.delta is not None and not os.path.isfile(self.previous_filename):
            raise FileNotFoundError(f"File not found: {self.previous_filename}")

    def _calculate_rainfall_rate(self):
        """ Calculate the rainfall rate from WRF output files. """
        precip_calc = CalculatePrecipitation(self.filename)
        precip_calc.run()
        self.Rainfall_Rate_surface = precip_calc.rain_rate
        self.lats = precip_calc.lats
        self.lons = precip_calc.lons

    def plot(self, ax):
        """ Plot either UKMO composite data or WRF rainfall data based on the configuration. """
        precip_cmap = ColorbarPrecipitation()
        ax.add_feature(cfeature.LAND, color='#F3EFE3', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='#CDEDFF', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, color='gray', zorder=2)
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())

        if self.delta is None:
            # Plot UKMO composite data
            processor = UKMOCompositeProcessor(self.filename)
            processor.ingest_file()
            rain_plot = ax.pcolormesh(
                processor.lon_grid, processor.lat_grid, processor.radar_array,
                cmap=precip_cmap.cmap, norm=precip_cmap.norm, transform=ccrs.PlateCarree()
            )
        else:
            # Plot WRF rainfall rate data
            rain_plot = ax.contourf(
                self.lons, self.lats, self.Rainfall_Rate_surface.squeeze(),
                levels=precip_cmap.levels, cmap=precip_cmap.cmap, norm=precip_cmap.norm,
                transform=ccrs.PlateCarree(), extend='both', zorder=1
            )

        add_corner_label(ax, self.text_lines, self.event_type)
        add_length_scale(ax)
        return rain_plot, precip_cmap

    @Timer
    def run(self, ax):
        """ Run all steps and generate the plot for either UKMO or WRF data. """
        self._check_files_exist()
        if self.delta is not None:
            self._calculate_rainfall_rate()
        return self.plot(ax)

    @staticmethod
    def create_plots(configs, fig_filename='figures/figure_01.png'):
        """ Static method to create a 2x2 grid plot for UKMO and WRF data based on configurations. """
        fig, axs = plt.subplots(2, 2, figsize=(8, 14), subplot_kw={'projection': ccrs.Mercator()})
        plt.subplots_adjust(wspace=0.05, hspace=0.03)

        all_plots = []
        precip_cmap = None
        for ax, config in zip(axs.ravel(), configs):
            plotter = RainfallRatePlotter(**config)
            rain_plot, precip_cmap = plotter.run(ax)
            all_plots.append(rain_plot)
            ax.set_aspect('auto')

        if precip_cmap:
            precip_cmap.add_colorbar(all_plots[0], axs.ravel().tolist())

        plt.savefig(fig_filename, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.show()


if __name__ == "__main__":
    # Configuration for plots (UKMO and WRF combined without a plot type)
    configs = [
        {
            "filename": '/Volumes/Samsung_T5/phd_data/ukmo_composite_data/ukmo_20111129/metoffice-c-band-rain-radar_uk_201111291400_1km-composite.dat',
            "min_lat": 50.5,
            "max_lat": 55.5,
            "min_lon": -5.5,
            "max_lon": -0.5,
            "text_lines": ["(UKMO Composite 1)"],
            "event_type": "Type 1"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/ukmo_composite_data/ukmo_20051124/metoffice-c-band-rain-radar_uk_200511241450_1km-composite.dat',
            "min_lat": 50.5,
            "max_lat": 54.0,
            "min_lon": -2.5,
            "max_lon": 1.8,
            "text_lines": ["(UKMO Composite 2)"],
            "event_type": "Type 2"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2011-11-29_14-00-00.nc',
            "delta": 10,
            "min_lat": 50.5,
            "max_lat": 55.5,
            "min_lon": -5.5,
            "max_lon": -0.5,
            "text_lines": ["(WRF Modelled 1)"],
            "event_type": "Type 1"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/2005-11-24/wrfout_d03_2005-11-24_14-50-00.nc',
            "delta": 10,
            "min_lat": 50.5,
            "max_lat": 54.0,
            "min_lon": -2.5,
            "max_lon": 1.8,
            "text_lines": ["(WRF Modelled 2)"],
            "event_type": "Type 2"
        }
    ]

    # Generate and display plots
    RainfallRatePlotter.create_plots(configs)
