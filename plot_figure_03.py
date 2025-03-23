import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import re
import cartopy.crs as ccrs
from datetime import datetime
from shapely.geometry import Polygon
from matplotlib.path import Path
from scipy.spatial.distance import pdist, squareform
from pyproj import Geod
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.label_and_scale import add_length_scale
from phd.utils.timer import Timer

class VorticityPlotter:
    """
    Class for computing and plotting collapsed vorticity data from WRF files with an inset plot 
    and a centralized colorbar.

    This class processes a set of WRF files located in a specified directory that fall within a 
    given time range. It computes the collapsed (maximum absolute) vorticity over time, applies a mask 
    based on a threshold, and then plots the vorticity data. The plot includes a main map and an inset 
    zoomed view defined by a polygon. Additionally, the inset plot displays markers with connecting 
    dashed lines (and calculates distances between these markers), and the figure features a centralized 
    horizontal colorbar.

    Attributes:
      directory (str): Path to the directory containing WRF files.
      start_time (str): Start time (formatted as "HH:MM:SS") for selecting files.
      end_time (str): End time (formatted as "HH:MM:SS") for selecting files.
      vorticity_threshold (float): Threshold for masking vorticity values (absolute value).
      vorticity_type (str): Type of vorticity ('Type 1' or 'Type 2') determining contour levels.
      lon_corners (list of float): Longitudes defining the corners of the polygon for the inset.
      lat_corners (list of float): Latitudes defining the corners of the polygon for the inset.
      polygon_coords (list of tuples): List of (lon, lat) tuples for the polygon.
      marker_lons (list of float): Longitudes of markers to plot in the inset.
      marker_lats (list of float): Latitudes of markers to plot in the inset.
      vorticity_collapsed (np.ndarray): Collapsed vorticity data computed over time.
      lats (np.ndarray): Array of latitude values from the WRF data.
      lons (np.ndarray): Array of longitude values from the WRF data.
      times (list of str): List of time strings extracted from the processed files.
      geod (pyproj.Geod): Instance of Geod for geodetic calculations.

    Methods:
      compute_collapsed_vorticity:
        Processes WRF files in the specified directory within the time range, computes the absolute 
        vorticity for each, masks values below the threshold, and collapses the data over time.
      
      _define_levels:
        Defines contour levels for plotting based on the selected vorticity type.
      
      _create_colormap:
        Creates a custom colormap and normalization based on the defined contour levels.
      
      _calculate_distances:
        Calculates distances (in kilometers) between consecutive marker pairs using geodetic calculations.
      
      plot_vorticity_with_inset:
        Plots the collapsed vorticity on a main map and creates an inset zoom plot defined by a polygon. 
        The inset includes markers, dashed connecting lines, and a centralized horizontal colorbar.
      
      run:
        Executes the computation of collapsed vorticity and plots the resulting figure, returning the 
        distances between marker pairs.
    """
    
    def __init__(self, directory, start_time, end_time):
        """
        Initializes the VorticityPlotter with necessary parameters.
        
        Parameters:
        - directory (str): Path to the directory containing WRF files.
        - start_time (str): Start time in "HH:MM:SS" format.
        - end_time (str): End time in "HH:MM:SS" format.
        """
        self.directory = directory
        self.start_time = start_time
        self.end_time = end_time
        
        # Hardcoded parameters
        self.vorticity_threshold = 3
        self.vorticity_type = 'Type 1'  # or 'Type 2'
        
        # Hardcoded polygon corners (lon, lat)
        self.lon_corners = [-6.25, -6.35, -5.16, -5.06]
        self.lat_corners = [53.13, 53.40, 53.57, 53.30]
        self.polygon_coords = list(zip(self.lon_corners, self.lat_corners))
        
        # Hardcoded marker coordinates
        self.marker_lons = [-6.17, -6.07, -6.02, -5.91, -5.87, -5.75, 
                            -5.70, -5.58, -5.54, -5.40, -5.35, -5.21]
        self.marker_lats = [53.19, 53.33, 53.21, 53.36, 53.23, 53.38, 
                            53.25, 53.41, 53.28, 53.44, 53.32, 53.48]
        
        # Data attributes
        self.vorticity_collapsed = None
        self.lats = None
        self.lons = None
        self.times = []
        
        # Geodetic calculator
        self.geod = Geod(ellps='WGS84')
    
    @Timer
    def compute_collapsed_vorticity(self):
        """
        Computes the collapsed vorticity data by processing WRF files within the specified time range.
        
        Returns:
        - vorticity_collapsed (np.ndarray): Collapsed vorticity data.
        - lats (np.ndarray): Latitude values.
        - lons (np.ndarray): Longitude values.
        - times (list): List of time strings.
        """
        # Convert start and end times to datetime objects
        start_time_dt = datetime.strptime(self.start_time, "%H:%M:%S")
        end_time_dt = datetime.strptime(self.end_time, "%H:%M:%S")

        # Load all WRF files in the directory
        files = sorted(glob.glob(f"{self.directory}/wrfout_d03_*.nc"))
        pattern = r"wrfout_d03_\d{4}-\d{2}-\d{2}_(\d{2}-\d{2}-\d{2})\.nc"

        # Select files based on the time range
        selected_files = []
        for file in files:
            match = re.search(pattern, file)
            if match:
                # Parse the time from the filename
                file_time_str = match.group(1).replace("-", ":")
                file_time_dt = datetime.strptime(file_time_str, "%H:%M:%S")
                if start_time_dt <= file_time_dt <= end_time_dt:
                    selected_files.append(file)

        # Check if no files are selected
        if not selected_files:
            raise ValueError(f"No files found in the specified time range {self.start_time} to {self.end_time}.")

        # Initialize lists to store data and metadata
        vorticity_data = []
        times = []
        lats = None
        lons = None

        # Calculate absolute vorticity for each file and apply the mask
        for file in selected_files:
            variables = GetVariablesWRF(file)
            vorticity = variables.get_absolute_vorticity()

            # Mask values below the threshold
            masked_vorticity = np.where(np.abs(vorticity) > self.vorticity_threshold, vorticity, np.nan)
            vorticity_data.append(masked_vorticity)

            # Record latitude/longitude once and store time
            if lats is None or lons is None:
                lats, lons = variables.get_lat_lons()
                # Convert lats and lons to NumPy arrays
                lats = lats.values
                lons = lons.values
            time_str = re.search(r"_(\d{2}-\d{2}-\d{2})\.nc", file).group(1)
            times.append(time_str.replace("-", ":"))

        # Stack vorticity data along the time axis
        vorticity_stacked = np.stack(vorticity_data, axis=0)  # Shape: (time, lat, lon)

        # Compute the maximum absolute vorticity over time
        vorticity_collapsed = np.nanmax(np.abs(vorticity_stacked), axis=0)
        vorticity_collapsed = np.nan_to_num(vorticity_collapsed, nan=0.0)

        # Assign to instance attributes
        self.vorticity_collapsed = vorticity_collapsed
        self.lats = lats
        self.lons = lons
        self.times = times

        return self.vorticity_collapsed, self.lats, self.lons, self.times
    
    def _define_levels(self):
        """
        Defines the contour levels based on the vorticity type.

        Returns:
        - levels (list): List of contour levels.
        """
        if self.vorticity_type == 'Type 1':
            levels = np.arange(-18, 21, 3)  # From -18 to 18 inclusive with step 3
        elif self.vorticity_type == 'Type 2':
            levels = np.arange(-12, 14, 2)  # From -12 to 12 inclusive with step 2
        else:
            raise ValueError("Invalid vorticity type. Choose either 'Type 1' or 'Type 2'.")
        levels = [int(level) for level in levels]
        return levels
    
    def _create_colormap(self, levels):
        """
        Creates a custom colormap based on the specified levels.

        Parameters:
        - levels (list): List of contour levels.

        Returns:
        - cmap (ListedColormap): Custom colormap.
        - norm (BoundaryNorm): Normalization for the colormap.
        """
        # Define colors
        num_intervals = len(levels) - 1
        zero_index = np.abs(levels).argmin()

        num_neg_colors = zero_index - 1
        num_pos_colors = num_intervals - zero_index - 1

        cmap_neg = plt.get_cmap('Blues_r', num_neg_colors + 2)
        neg_colors = [cmap_neg(i) for i in range(1, num_neg_colors + 1)]

        white_colors = ['#FFFFFF', '#FFFFFF']

        cmap_pos = plt.get_cmap('Greys', num_pos_colors + 2)
        pos_colors = [cmap_pos(i) for i in range(1, num_pos_colors + 1)]

        colors = neg_colors + white_colors + pos_colors

        # Create the colormap
        cmap = mcolors.ListedColormap(colors, name='Vorticity')
        cmap.set_under('darkblue')
        cmap.set_over('black')
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)

        return cmap, norm
    
    def _calculate_distances(self):
        """
        Calculates distances between consecutive marker pairs.

        Returns:
        - distances_info (list of tuples): Each tuple contains (lon1, lat1, lon2, lat2, distance_km)
        """
        distances_info = []
        for i in range(0, len(self.marker_lons), 2):
            if i + 1 < len(self.marker_lons):
                lon1, lat1 = self.marker_lons[i], self.marker_lats[i]
                lon2, lat2 = self.marker_lons[i + 1], self.marker_lats[i + 1]

                # Calculate forward and back azimuths, and distance
                az12, az21, distance = self.geod.inv(lon1, lat1, lon2, lat2)

                # Convert distance from meters to kilometers
                distance_km = distance / 1000.0
                distances_info.append((lon1, lat1, lon2, lat2, distance_km))
        return distances_info
    
    @Timer
    def plot_vorticity_with_inset(self, output_filename='figure_03.png'):
        """
        Plots the collapsed vorticity data with an inset and a centralized colorbar.

        Parameters:
        - output_filename (str): Filename to save the figure.

        Returns:
        - distances_km (list): List of distances between marker pairs in kilometers.
        """
        if self.vorticity_collapsed is None:
            raise ValueError("Vorticity data not computed. Call compute_collapsed_vorticity() first.")

        # Define contour levels
        levels = self._define_levels()

        # Create colormap and normalization
        cmap, norm = self._create_colormap(levels)

        # Define the extent for the main plot
        main_extent = [-7, -4.4, 52.8, 54.0]  # [lon_min, lon_max, lat_min, lat_max]

        # Create a shapely polygon for the inset
        inset_polygon = Polygon(self.polygon_coords)

        # Extract polygon corners
        lon_corners, lat_corners = zip(*self.polygon_coords)

        # Define two points along the main axis of the polygon (furthest points)
        coords = np.array([lon_corners, lat_corners]).T
        dist_matrix = squareform(pdist(coords))
        i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        point1 = (coords[i, 0], coords[i, 1])
        point2 = (coords[j, 0], coords[j, 1])

        # Compute central point
        central_longitude = (point1[0] + point2[0]) / 2
        central_latitude = (point1[1] + point2[1]) / 2

        # Compute azimuth between point1 and point2
        azi1, azi2, dist = self.geod.inv(point1[0], point1[1], point2[0], point2[1])
        azimuth = azi1  # azimuth from point1 to point2

        # Define the Oblique Mercator projection using the computed parameters
        proj_inset = ccrs.ObliqueMercator(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            azimuth=azimuth,
            scale_factor=1.0,
            false_easting=0.0,
            false_northing=0.0
        )

        # Create the main plot and the inset plot
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # Main plot
        ax_main = fig.add_subplot(gs[0], projection=ccrs.Mercator())
        ax_main.set_extent(main_extent, crs=ccrs.PlateCarree())
        add_length_scale(ax_main)

        # Plot the vorticity data using contourf
        ax_main.contourf(
            self.lons,
            self.lats,
            self.vorticity_collapsed,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both',
            transform=ccrs.PlateCarree()
        )

        # Plot the polygon on the main plot
        ax_main.add_geometries(
            [inset_polygon],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='black',
            linewidth=2
        )

        # Mask the data outside the polygon
        # Flatten lons and lats
        lon_flat = self.lons.flatten()
        lat_flat = self.lats.flatten()

        # Create an array of (lon, lat) points
        points = np.vstack((lon_flat, lat_flat)).T

        # Create a Path object from the polygon coordinates
        polygon_path = Path(self.polygon_coords)

        # Test which points are inside the polygon
        mask_flat = polygon_path.contains_points(points)

        # Reshape mask to the shape of lons and lats
        mask = mask_flat.reshape(self.lons.shape)

        # Mask the data outside the polygon
        inset_vorticity = np.ma.array(self.vorticity_collapsed, mask=~mask)

        # Inset plot
        ax_inset = fig.add_subplot(gs[1], projection=proj_inset)

        # Transform the polygon to the inset projection to get bounds
        inset_polygon_proj = proj_inset.project_geometry(inset_polygon, ccrs.PlateCarree())
        minx, miny, maxx, maxy = inset_polygon_proj.bounds

        # Adjust the bounds to make the extent slightly smaller
        buffer_percentage = 0.025  # Adjust this value as needed
        delta_x = (maxx - minx) * buffer_percentage
        delta_y = (maxy - miny) * buffer_percentage
        minx += delta_x
        maxx -= delta_x
        miny += delta_y
        maxy -= delta_y

        # Additional reduction for maxy (northern bound)
        additional_maxy_reduction = (maxy - miny) * 0.025  # Reduce maxy by an additional percentage
        maxy -= additional_maxy_reduction

        # Ensure that maxy is still greater than miny
        if maxy <= miny:
            raise ValueError("After adjustment, maxy is not greater than miny. Adjust the reduction amounts.")

        # Set the adjusted extent for the inset plot
        ax_inset.set_extent([minx, maxx, miny, maxy], crs=proj_inset)

        # Plot the inset data (masked)
        cf_inset = ax_inset.contourf(
            self.lons,
            self.lats,
            inset_vorticity,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both',
            transform=ccrs.PlateCarree()
        )

        # Plot markers
        ax_inset.plot(
            self.marker_lons,
            self.marker_lats,
            'o',
            markersize=10,
            color='black',
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # Draw dashed lines between pairs and calculate distances
        distances_info = self._calculate_distances()
        distances_km = []  # To store distances for returning

        for lon1, lat1, lon2, lat2, distance_km in distances_info:
            distances_km.append(distance_km)
            
            # Plot dashed line between the two points
            ax_inset.plot(
                [lon1, lon2],
                [lat1, lat2],
                linestyle='--',
                color='black',
                transform=ccrs.PlateCarree(),
                zorder=2
            )

        # Adjust layout to make room for colorbar
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the colorbar

        # Adjust the widths of the axes to be the same
        pos_main = ax_main.get_position()
        pos_inset = ax_inset.get_position()
        # Set the left position and width of ax_inset to match ax_main
        ax_inset.set_position([pos_main.x0, pos_inset.y0, pos_main.width, pos_inset.height])

        # Create an axes for the colorbar at the desired position
        width = 0.60
        left = (1 - width) / 2 
        cbar_ax = fig.add_axes([left, 0.03, width, 0.022])  # [left, bottom, width, height] - adjusted 'bottom' to bring colorbar closer

        # Add the colorbar
        cbar = fig.colorbar(
            cf_inset,
            cax=cbar_ax,
            orientation='horizontal',
            extend='both'
        )

        # Set colorbar tick labels
        cbar.set_ticks(levels)
        tick_labels = []
        for level in levels:
            if np.isclose(level, 0.0):
                tick_labels.append('0')
            else:
                tick_labels.append(f"{level}")
        cbar.ax.set_xticklabels(tick_labels)
        cbar.set_label('Absolute Vorticity (10$^{-3}$ s$^{-1}$)')

        # Save and display the figure
        fig_filename = f'figures/{output_filename}'
        plt.savefig(fig_filename, dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        
        return distances_km
    
    def run(self, output_filename='figure_03.png'):
        self.compute_collapsed_vorticity()
        distances_km = self.plot_vorticity_with_inset(output_filename)
        return distances_km

if __name__ == "__main__":
    directory = "/Volumes/Samsung_T5/phd_data/2011-11-29"
    start_time = "08:00:00"
    end_time = "12:30:00"

    plotter = VorticityPlotter(directory=directory, start_time=start_time, end_time=end_time)
    distances_km = plotter.run(output_filename='figure_03.png')
