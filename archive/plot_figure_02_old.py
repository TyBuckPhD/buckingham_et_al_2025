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
    def __init__(self, filename, delta, min_lat=None, max_lat=None, min_lon=None, max_lon=None, text_lines=None, event_type=None, labels=True):
        self.filename = filename
        self.delta = delta
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.text_lines = text_lines
        self.event_type = event_type
        self.labels = labels

        self.fpath, self.fname = os.path.split(filename)
        self.base_filename = None
        self.current_time = None
        self.previous_time = None
        self.previous_filename = None

        self.Rainfall_Rate_surface = None
        self.lats = None
        self.lons = None
        self.crs = ccrs.Mercator()

    def _parse_filename(self):
        match = re.search(r'wrfout_d\d\d_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', self.fname)
        if not match:
            print("Filename does not match expected format.")
            return False
        self.base_filename = match.group()
        datetime_str = '_'.join(self.base_filename.split('_')[2:4])
        self.current_time = pd.to_datetime(datetime_str, format='%Y-%m-%d_%H-%M-%S')
        return True

    def _get_previous_filename(self):
        self.previous_time = self.current_time - pd.Timedelta(minutes=self.delta)
        previous_datetime_str = self.previous_time.strftime('%Y-%m-%d_%H-%M-%S')
        base_parts = self.base_filename.split('_')
        self.previous_fname = f"{base_parts[0]}_{base_parts[1]}_{previous_datetime_str}.nc"
        self.previous_filename = os.path.join(self.fpath, self.previous_fname)

    def _check_files_exist(self):
        if not os.path.isfile(self.filename):
            print(f"File not found: {self.filename}")
            return False
        if not os.path.isfile(self.previous_filename):
            print(f"File not found: {self.previous_filename}")
            return False
        return True

    def _calculate_rainfall_rate(self):
        precip_calc = CalculatePrecipitation(
            current_filename=self.filename,
            previous_filename=self.previous_filename,
            delta=self.delta
        )
        precip_calc.run()
        self.Rainfall_Rate_surface = precip_calc.Rainfall_Rate_surface
        self.lats = precip_calc.lats
        self.lons = precip_calc.lons

    def _calculate_absolute_vorticity(self):
        avort_calc = CalculateVorticity(self.filename)
        abs_vorticity, _, _ = avort_calc.run()
        self.abs_vorticity = abs_vorticity

    def plot_rainfall_rate_vorticity(self):
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': self.crs})

        # Add pale beige land and light blue sea
        ax.add_feature(cfeature.LAND, facecolor='#F3EFE3')  # Pale beige land color
        ax.add_feature(cfeature.OCEAN, facecolor='#CDEDFF')  # Very light blue sea color

        ax.add_feature(
            cfeature.COASTLINE.with_scale('50m'),
            linewidth=0.5, color='gray', zorder=2
        )

        # Set the extent based on user input or calculated from data
        extent = [
            self.min_lon if self.min_lon is not None else float(self.lons.min()),
            self.max_lon if self.max_lon is not None else float(self.lons.max()),
            self.min_lat if self.min_lat is not None else float(self.lats.min()),
            self.max_lat if self.max_lat is not None else float(self.lats.max()),
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Create an instance of the PrecipitationColormap class
        precip_cmap = ColorbarPrecipitation()

        # Plot the precipitation data
        rain_plot = ax.contourf(
            self.lons, self.lats, self.Rainfall_Rate_surface.squeeze(),
            levels=precip_cmap.levels, cmap=precip_cmap.cmap, norm=precip_cmap.norm,
            transform=ccrs.PlateCarree(), extend='both', zorder=1
        )

        # Plot absolute vorticity contours on top of precipitation
        levels_vort = np.linspace(-30e-3, 30e-3, 21)
        levels_vort = levels_vort[levels_vort != 0]
        ax.contour(
            self.lons, self.lats, self.abs_vorticity,
            levels=levels_vort, colors='black',
            linewidths=3.0, transform=ccrs.PlateCarree(), zorder=3
        )

        if self.labels:
            # Add the customized colorbar
            precip_cmap.add_colorbar(rain_plot, ax)

            # Add the corner label
            add_corner_label(ax, self.text_lines, self.event_type)
            
            # Add the length scale
            add_length_scale(ax)

        plt.savefig('figure_02.png', dpi=200)
        plt.show()

    @Timer
    def run(self):
        if not self._parse_filename():
            return
        self._get_previous_filename()
        if not self._check_files_exist():
            return
        self._calculate_rainfall_rate()
        self._calculate_absolute_vorticity()
        self.plot_rainfall_rate_vorticity()

if __name__ == "__main__":
    filename = '/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2011-11-29_10-50-00.nc'
    plotter = RainfallRatePlotter(
        filename,
        delta=10,
        min_lat=53.1,
        max_lat=53.6,
        min_lon=-5.9,
        max_lon=-5.2,
        text_lines=["(a) Modelled:", " 29 Nov 2011"],
        event_type="Type 1",
        labels=False
    )
    plotter.run()
