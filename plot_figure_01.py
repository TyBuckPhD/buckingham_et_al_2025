import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from phd.variables.colorbar_precipitation import ColorbarPrecipitation
from phd.variables.get_variables_wrf import GetVariablesWRF
from phd.utils.label_and_scale import add_corner_label, add_length_scale
from phd.utils.timer import Timer
from phd.ukmo_composites.ukmo_composite_file_converter import UKMOCompositeProcessor

class RainfallRatePlotter:
    def __init__(self, filename, min_lat=None, max_lat=None, min_lon=None, max_lon=None, text_lines=None, event_type=None):
        self.filename = filename
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.text_lines = text_lines
        self.event_type = event_type
        self.Rainfall_Rate_surface = None
        self.lats = None
        self.lons = None
        self.processor = None

        # Determine if file is a WRF or UKMO format
        self.is_wrf_file = self.filename.endswith('.nc')

        # Instantiate GetVariablesWRF if the file is WRF
        if self.is_wrf_file:
            self.wrf_data_processor = GetVariablesWRF(self.filename)

    def _check_file_exist(self):
        """ Ensure the input file exists. """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")

    def _load_lat_lons_wrf(self):
        self.lats, self.lons = self.wrf_data_processor.get_lat_lons()

    def _load_rainfall_rate_wrf(self):
        self.Rainfall_Rate_surface = self.wrf_data_processor.get_rainfall_rate(timeidx=0)

    def _load_rainfall_rate_ukmo(self):
        """ Load and process UKMO composite data from a .dat file. """
        self.processor = UKMOCompositeProcessor(self.filename)
        self.processor.ingest_file()
        self.Rainfall_Rate_surface = self.processor.radar_array
        self.lats = self.processor.lat_grid
        self.lons = self.processor.lon_grid

    def plot(self, ax):
        """ Plot either UKMO composite data or WRF rainfall data based on the configuration. """
        precip_cmap = ColorbarPrecipitation()
        ax.add_feature(cfeature.LAND, color='#F3EFE3', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='#CDEDFF', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, color='gray', zorder=2)
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())

        # Plot radar data based on file type
        rain_plot = ax.pcolormesh(
            self.lons, self.lats, self.Rainfall_Rate_surface,
            cmap=precip_cmap.cmap, norm=precip_cmap.norm,
            transform=ccrs.PlateCarree()
        )

        add_corner_label(ax, self.text_lines, self.event_type)
        add_length_scale(ax)
        return rain_plot, precip_cmap

    @Timer
    def run(self, ax):
        """ Run all steps and generate the plot for either UKMO or WRF data. """
        self._check_file_exist()
        if self.is_wrf_file:
            self._load_lat_lons_wrf()
            self._load_rainfall_rate_wrf()
        else:
            self._load_rainfall_rate_ukmo()
        return self.plot(ax)

    @staticmethod
    def create_plots(configs, fig_filename='figures/figure_01.png'):
        """ Static method to create a 2x2 grid plot for UKMO and WRF data based on configurations. """
        fig, axs = plt.subplots(2, 2, figsize=(9, 14), subplot_kw={'projection': ccrs.Mercator()})
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
            "min_lat": 51.5,
            "max_lat": 55.0,
            "min_lon": -5.1,
            "max_lon": -0.1,
            "text_lines": ["(a) Composite:", " 1400 UTC", " 29 Nov 2011"],
            "event_type": "Type 1"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/ukmo_composite_data/ukmo_20051124/metoffice-c-band-rain-radar_uk_200511241500_1km-composite.dat',
            "min_lat": 50.5,
            "max_lat": 54.0,
            "min_lon": -3.0,
            "max_lon": 2.0,
            "text_lines": ["(b) Composite:", " 1400 UTC", " 24 Nov 2005"],
            "event_type": "Type 2"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/wrfout_d03_2011-11-29_14-00-00.nc',
            "min_lat": 51.5,
            "max_lat": 55.0,
            "min_lon": -5.1,
            "max_lon": -0.1,
            "text_lines": ["(c) Modelled:", " 1400 UTC", " 29 Nov 2011"],
            "event_type": "Type 1"
        },
        {
            "filename": '/Volumes/Samsung_T5/phd_data/wrfout_d03_2005-11-24_15-00-00.nc',
            "min_lat": 50.5,
            "max_lat": 54.0,
            "min_lon": -3.0,
            "max_lon": 2.0,
            "text_lines": ["(d) Modelled:", " 1400 UTC", " 24 Nov 2005"],
            "event_type": "Type 2"
        }
    ]

    # Generate and display plots
    RainfallRatePlotter.create_plots(configs)
