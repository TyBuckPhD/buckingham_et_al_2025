import struct
import array
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
from phd.variables.colorbar_precipitation import ColorbarPrecipitation

class UKMOCompositeProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.transformer_osgb_to_utm = Transformer.from_crs("EPSG:27700", "EPSG:32630", always_xy=True)
        self.transformer_utm_to_latlon = Transformer.from_crs("EPSG:32630", "EPSG:4326", always_xy=True)
        self.radar_array = None
        self.header = None
        self.utm_corners = None
        self.grid_size = None
        self.lat_grid = None
        self.lon_grid = None

    def ingest_file(self):
        """ Reads and processes UK Met Office rainfall radar composites from NIMROD files. """
        with open(self.filepath, "rb") as file_id:
            if struct.unpack(">l", file_id.read(4))[0] != 512:
                raise ValueError("Unexpected initial record length")
            
            gen_ints = array.array("h")
            gen_reals = array.array("f")
            spec_reals = array.array("f")
            characters = array.array("b")
            spec_ints = array.array("h")

            gen_ints.fromfile(file_id, 31)
            gen_reals.fromfile(file_id, 28)
            spec_reals.fromfile(file_id, 45)
            characters.fromfile(file_id, 56)
            spec_ints.fromfile(file_id, 51)

            for data in [gen_ints, gen_reals, spec_reals, spec_ints]:
                data.byteswap()
            
            if struct.unpack(">l", file_id.read(4))[0] != 512:
                raise ValueError("Unexpected secondary record length")
            
            nrows, ncols = gen_ints[15], gen_ints[16]
            self.grid_size = (nrows, ncols)
            cellsize, nodata_value = gen_reals[3], gen_reals[6]
            ytlcorner, xtlcorner = gen_reals[2], gen_reals[4]

            yllcorner_osgb = ytlcorner - (nrows * cellsize)
            xllcorner_osgb = xtlcorner
            xtrcorner_osgb = xtlcorner + (ncols * cellsize)
            ytrcorner_osgb = ytlcorner

            self.utm_corners = (
                self.transformer_osgb_to_utm.transform(xllcorner_osgb, yllcorner_osgb),
                self.transformer_osgb_to_utm.transform(xtrcorner_osgb, ytrcorner_osgb),
            )

            array_size = nrows * ncols
            if struct.unpack(">l", file_id.read(4))[0] != array_size * 2:
                raise ValueError("Unexpected data record length")
            
            data = array.array("h")
            data.fromfile(file_id, array_size)
            data.byteswap()

            if struct.unpack(">l", file_id.read(4))[0] != array_size * 2:
                raise ValueError("Unexpected final record length")

            self.radar_array = np.reshape(data, (nrows, ncols)) / 32.0
            self.radar_array = np.flipud(self.radar_array)

            self.header = (f'NCols {ncols}\nNRows {nrows}\n'
                           f'xllcorner {self.utm_corners[0][0]}\nyllcorner {self.utm_corners[0][1]}\n'
                           f'cellsize {cellsize}\nNODATA_value {nodata_value}')
        
        self._create_latlon_grid()

    def _create_latlon_grid(self):
        """ Converts UTM coordinates to a lat/lon grid for plotting. """
        if not self.utm_corners or not self.grid_size:
            raise ValueError("Grid boundaries or size are missing. Please run ingest_file() first.")

        xllcorner_utm, yllcorner_utm = self.utm_corners[0]
        xtrcorner_utm, ytrcorner_utm = self.utm_corners[1]
        nrows, ncols = self.grid_size

        x_res = (xtrcorner_utm - xllcorner_utm) / (ncols - 1)
        y_res = (ytrcorner_utm - yllcorner_utm) / (nrows - 1)

        utm_x = xllcorner_utm + np.arange(ncols) * x_res
        utm_y = yllcorner_utm + np.arange(nrows) * y_res
        utm_x_grid, utm_y_grid = np.meshgrid(utm_x, utm_y)

        self.lon_grid, self.lat_grid = self.transformer_utm_to_latlon.transform(utm_x_grid, utm_y_grid)

    def plot_radar_composite(self, extent=None, output_file="composite.png"):
        """ Plot the radar composite data with an optional extent and save to file. """
        if self.radar_array is None or self.lat_grid is None or self.lon_grid is None:
            self.ingest_file()

        precip_cmap = ColorbarPrecipitation()

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})

        # Use provided extent or calculate from the grid
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            min_lon, max_lon = self.lon_grid.min(), self.lon_grid.max()
            min_lat, max_lat = self.lat_grid.min(), self.lat_grid.max()
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, color='#F3EFE3', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='#CDEDFF', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, color='black', zorder=2)

        # Plot radar data
        rain_plot = ax.pcolormesh(
            self.lon_grid, self.lat_grid, self.radar_array,
            cmap=precip_cmap.cmap, norm=precip_cmap.norm,
            transform=ccrs.PlateCarree()
        )

        # Add colorbar
        precip_cmap.add_colorbar(rain_plot, ax)

        # Save and display the plot
        plt.savefig(output_file)
        plt.show()
