import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.variables.colorbar_vorticity import ColorbarVorticity
from phd.core.backward_trajectories import BackwardParticleTrajectories
from phd.utils.timer import Timer

class ParticleTrajectoryAnalysis:
    def __init__(self, config_path, x0_range, y0_range, z0_range, vorticity_threshold=9, buffer=10, levels=None):
        self.config_path = config_path
        self.x0_range = x0_range
        self.y0_range = y0_range
        self.z0_range = z0_range
        self.vorticity_threshold = vorticity_threshold
        self.buffer = buffer

        # Initialize GetVariablesWRF
        self.gvw = GetVariablesWRF(config_path=config_path)

        # Get latitudes and longitudes
        self.lats, self.lons = self.gvw.get_lat_lons()

        # Get absolute vorticity data
        self.avo = self.gvw.get_absolute_vorticity()

        # Use the last time step of absolute vorticity
        self.vorticity_data = self.avo[-1, :, :]
        
        # Initialize vorticity colormap
        self.vorticity_cmap = ColorbarVorticity(vorticity_type='Type 1', levels=levels)

        # Initialize variables for seeds
        self.x_seeds = None
        self.y_seeds = None
        self.z_seeds = None
        self.lon_seeds = None
        self.lat_seeds = None
        self.seed_vorticity = None

        # Initialize variables for trajectories
        self.traj_lon = None
        self.traj_lat = None
        self.traj_z = None
        self.traj_avo = None
        self.traj_stretching = None
        self.traj_tilting = None

        # Selected trajectory indices
        self.selected_indices = None

    @Timer
    def plot_seed_locations_and_filter(self, labels=False):
        """
        Plot seed locations on top of the zoomed-in vorticity field and filter seeds based on vorticity values.
        """
        # Generate seed locations from x0_range, y0_range, z0_range
        x_seeds, y_seeds, z_seeds = np.meshgrid(
            np.linspace(self.x0_range[0], self.x0_range[1], self.x0_range[1] - self.x0_range[0] + 1),
            np.linspace(self.y0_range[0], self.y0_range[1], self.y0_range[1] - self.y0_range[0] + 1),
            self.z0_range
        )
        x_seeds = x_seeds.flatten().astype(int)
        y_seeds = y_seeds.flatten().astype(int)
        z_seeds = z_seeds.flatten().astype(int)

        seed_indices = (y_seeds, x_seeds)
        lon_seeds = self.lons.values[seed_indices]
        lat_seeds = self.lats.values[seed_indices]

        # Extrapolate vorticity values for seed locations
        seed_vorticity = self.vorticity_data.values[y_seeds, x_seeds]

        # Filter seeds based on vorticity threshold
        valid_seeds = seed_vorticity >= self.vorticity_threshold
        x_seeds = x_seeds[valid_seeds]
        y_seeds = y_seeds[valid_seeds]
        z_seeds = z_seeds[valid_seeds]
        lon_seeds = lon_seeds[valid_seeds]
        lat_seeds = lat_seeds[valid_seeds]
        seed_vorticity = seed_vorticity[valid_seeds]

        print(f"Filtered {len(valid_seeds) - np.sum(valid_seeds)} seeds below the threshold of {self.vorticity_threshold}.")
        print(f"{np.sum(valid_seeds)} seeds remain.")

        # Add buffer to indices
        min_x = max(0, np.min(x_seeds) - self.buffer)
        max_x = min(self.lons.shape[1] - 1, np.max(x_seeds) + self.buffer)
        min_y = max(0, np.min(y_seeds) - self.buffer)
        max_y = min(self.lats.shape[0] - 1, np.max(y_seeds) + self.buffer)

        # Clip lat/lon and vorticity data
        clipped_lons = self.lons[min_y:max_y + 1, min_x:max_x + 1]
        clipped_lats = self.lats[min_y:max_y + 1, min_x:max_x + 1]
        clipped_vorticity = self.vorticity_data.values[min_y:max_y + 1, min_x:max_x + 1]

        # Plot the filtered seeds on the clipped region
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw={'projection': ccrs.Mercator()})

        # Set plot extent with a slight buffer around the seeds
        plot_buffer = -0.01
        lat_min_zoom = np.min(clipped_lats) - plot_buffer
        lat_max_zoom = np.max(clipped_lats) + plot_buffer
        lon_min_zoom = np.min(clipped_lons) - plot_buffer
        lon_max_zoom = np.max(clipped_lons) + plot_buffer
        ax.set_extent([lon_min_zoom, lon_max_zoom, lat_min_zoom, lat_max_zoom], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)

        # Plot the clipped vorticity data
        vorticity_plot = ax.contourf(
            clipped_lons,
            clipped_lats,
            clipped_vorticity,
            levels=self.vorticity_cmap.levels,
            cmap=self.vorticity_cmap.cmap,
            norm=self.vorticity_cmap.norm,
            extend='both',
            transform=ccrs.PlateCarree()
        )
        self.vorticity_cmap.add_colorbar(vorticity_plot, ax, pad=0.07)

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        # Plot seed locations with indices and vorticity values
        for i, (lon, lat, vort) in enumerate(zip(lon_seeds, lat_seeds, seed_vorticity)):
            ax.scatter(lon, lat, color='purple', transform=ccrs.PlateCarree(), label='Seed {}'.format(i) if i == 0 else "")
            if labels:
                ax.text(lon, lat, f"{i}", transform=ccrs.PlateCarree(), fontsize=9, color='black')

        plt.savefig('test_seeds.png', dpi=200)
        plt.show()

        # Set instance variables
        self.x_seeds = x_seeds
        self.y_seeds = y_seeds
        self.z_seeds = z_seeds
        self.lon_seeds = lon_seeds
        self.lat_seeds = lat_seeds
        self.seed_vorticity = seed_vorticity

    def validate_and_filter_seeds(self):
        print("Enter indices of seeds to drop (comma-separated) or type 'proceed' to continue, 'exit' to quit.")

        while True:
            user_input = input("Input: ").strip()
            if user_input.lower() == 'proceed':
                break
            elif user_input.lower() == 'exit':
                print("Exiting the simulation.")
                exit()
            else:
                try:
                    drop_indices = list(map(int, user_input.split(',')))
                    self.x_seeds = np.delete(self.x_seeds, drop_indices)
                    self.y_seeds = np.delete(self.y_seeds, drop_indices)
                    self.z_seeds = np.delete(self.z_seeds, drop_indices)
                    self.lon_seeds = np.delete(self.lon_seeds, drop_indices)
                    self.lat_seeds = np.delete(self.lat_seeds, drop_indices)
                    self.seed_vorticity = np.delete(self.seed_vorticity, drop_indices)
                    print(f"Removed seeds at indices: {drop_indices}")
                except ValueError:
                    print("Invalid input. Please enter a valid list of indices or type 'proceed'.")

        print(f"Remaining seeds: {len(self.x_seeds)}")

    def compute_trajectories(self, dt=10, grid_spacing=200):
        simulator = BackwardParticleTrajectories(
            config_path=self.config_path,
            dt=dt,
            grid_spacing=grid_spacing,
            x0=self.x_seeds,
            y0=self.y_seeds,
            z0=self.z_seeds
        )
        traj_lon, traj_lat, traj_z, traj_avo, traj_stretching, traj_tilting = simulator.compute_trajectories()

        # Store trajectory data
        self.traj_lon = traj_lon
        self.traj_lat = traj_lat
        self.traj_z = traj_z
        self.traj_avo = traj_avo
        self.traj_stretching = traj_stretching
        self.traj_tilting = traj_tilting

    def plot_trajectories_with_vorticity(self, buffer=0.01):
        """
        Plot trajectories on top of vorticity data and initial seed markers.
        Trajectories are plotted as black lines.
        """
        traj_lon = self.traj_lon
        traj_lat = self.traj_lat

        # Determine trajectory bounds
        lon_min = np.min(traj_lon) - buffer
        lon_max = np.max(traj_lon) + buffer
        lat_min = np.min(traj_lat) - buffer
        lat_max = np.max(traj_lat) + buffer

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)

        # Mask vorticity data outside plot limits
        mask = (self.lons >= lon_min) & (self.lons <= lon_max) & (self.lats >= lat_min) & (self.lats <= lat_max)
        vorticity_masked = np.ma.masked_where(~mask, self.vorticity_data)

        # Plot vorticity data using vorticity_cmap object
        vorticity_plot = ax.contourf(
            self.lons,
            self.lats,
            vorticity_masked,
            levels=self.vorticity_cmap.levels,
            cmap=self.vorticity_cmap.cmap,
            norm=self.vorticity_cmap.norm,
            extend='both',
            transform=ccrs.PlateCarree()
        )
        self.vorticity_cmap.add_colorbar(vorticity_plot, ax, pad=0.07)

        # Plot gridlines with dotted style
        gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5, color='gray')
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        # Plot trajectories as black lines
        for lon, lat in zip(traj_lon, traj_lat):
            # Ensure that trajectory data are numpy arrays
            lon = lon.flatten()
            lat = lat.flatten()

            # Plot the trajectory
            ax.plot(lon, lat, color='purple', linewidth=0.3, transform=ccrs.PlateCarree())

        plt.savefig('test_trajectories.png', dpi=200)
        plt.show()

    def plot_trajectories_3d_with_vorticity(self, contour_value=9, buffer=0.01):
        """
        Plot 3D particle trajectories with a filled grey vorticity contour at the starting height.
        Trajectories are colored according to absolute vorticity values along them.
        The plot limits are set to focus on the trajectories, and vorticity contours are clipped to these limits.
        """
        traj_lon = self.traj_lon
        traj_lat = self.traj_lat
        traj_z = self.traj_z
        traj_avo = self.traj_avo

        # Determine trajectory bounds
        lon_min = traj_lon.min() - buffer
        lon_max = traj_lon.max() + buffer
        lat_min = traj_lat.min() - buffer
        lat_max = traj_lat.max() + buffer

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')

        # Plot trajectories with colors according to traj_avo
        # Normalize the vorticity values for colormap
        norm = plt.Normalize(traj_avo.min(), traj_avo.max())
        cmap = cm.seismic

        for lon, lat, z, avo in zip(traj_lon, traj_lat, traj_z, traj_avo):
            # Create line segments
            points = np.array([lon, lat, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a Line3DCollection
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(avo[:-1])  # Set the array of vorticity values
            lc.set_linewidth(2)
            ax.add_collection3d(lc)

        # Contour vorticity at the trajectory's starting height
        z_surface = traj_z[0, -1]  # Starting height of the trajectories
        vorticity_max = self.vorticity_data.max()

        # Mask vorticity data outside the plot limits
        mask = (self.lons >= lon_min) & (self.lons <= lon_max) & (self.lats >= lat_min) & (self.lats <= lat_max)
        vorticity_masked = np.ma.masked_where(~mask, self.vorticity_data)

        # Generate the filled contour using masked vorticity data
        ax.contourf(
            self.lons, self.lats, vorticity_masked,
            levels=[contour_value, vorticity_max],
            zdir='z',
            offset=z_surface,
            colors='grey',
            antialiased=True
        )

        # Customize gridlines
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray')  # Dotted gridlines

        # Set axes pane colors and remove edges
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor('white')
            axis.pane.set_edgecolor('white')

        # Rotate the z-axis label and adjust padding
        ax.set_zlabel("Altitude (m)", rotation=90)
        ax.zaxis.labelpad = 20  # Increase labelpad to move label further from the axis

        # Set axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set plot limits to focus on the trajectories
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add colorbar for trajectory vorticity
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(traj_avo)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Trajectory Absolute Vorticity')

        plt.savefig('test_3d.png', dpi=200)
        plt.show()

    def plot_trajectories_3d_with_tilting_stretching(self, contour_value=9, buffer=0.01):
        """
        Plot 3D particle trajectories colored by tilting and stretching terms, with a bivariate colormap.
        """
        traj_lon = self.traj_lon
        traj_lat = self.traj_lat
        traj_z = self.traj_z
        traj_stretching = self.traj_stretching
        traj_tilting = self.traj_tilting

        # Determine trajectory bounds
        lon_min = traj_lon.min() - buffer
        lon_max = traj_lon.max() + buffer
        lat_min = traj_lat.min() - buffer
        lat_max = traj_lat.max() + buffer

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')

        # Rotate the plot slightly clockwise
        # ax.view_init(elev=30, azim=-80)

        # Loop over each trajectory
        for lon, lat, z, stretching, tilting in zip(traj_lon, traj_lat, traj_z, traj_stretching, traj_tilting):
            # Ensure that trajectory data are numpy arrays
            lon = np.array(lon)
            lat = np.array(lat)
            z = np.array(z)
            stretching = np.array(stretching)
            tilting = np.array(tilting)

            # Compute magnitude and angle
            magnitude = np.sqrt(stretching**2 + tilting**2)
            magnitude_norm = magnitude / np.max(magnitude)  # Normalize magnitude
            angle = np.arctan2(stretching, tilting)  # arctan2(y, x)
            hue = (angle + np.pi) / (2 * np.pi)  # Map angle to [0, 1]

            # Set saturation and value
            saturation = np.ones_like(hue)  # Full saturation
            value = magnitude_norm  # Use normalized magnitude

            # Stack HSV components
            HSV = np.stack((hue, saturation, value), axis=-1)

            # Convert HSV to RGB
            RGB = hsv_to_rgb(HSV)

            # Create line segments
            points = np.array([lon, lat, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create Line3DCollection with the RGB colors
            lc = Line3DCollection(segments, colors=RGB[:-1], linewidth=2)
            ax.add_collection3d(lc)

        # Contour vorticity at the trajectory's starting height
        z_surface = traj_z[0, -1]

        # Mask vorticity data outside plot limits
        mask = (self.lons >= lon_min) & (self.lons <= lon_max) & (self.lats >= lat_min) & (self.lats <= lat_max)
        vorticity_masked = np.ma.masked_where(~mask, self.vorticity_data)

        # Generate the filled contour
        ax.contourf(
            self.lons, self.lats, vorticity_masked,
            levels=[contour_value, self.vorticity_data.max()],
            zdir='z',
            offset=z_surface,
            colors='grey',
            antialiased=True
        )

        # Customize gridlines
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

        # Set axes pane colors
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor('white')
            axis.pane.set_edgecolor('white')

        # Set axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (m)")

        # Adjust axis label padding
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10

        # Set plot limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_zlim(traj_z.min(), traj_z.max())

        # Create the bivariate colormap
        tilting_range = 5e-5
        stretching_range = 5e-5
        tilting_vals = np.linspace(-tilting_range, tilting_range, 256)
        stretching_vals = np.linspace(-stretching_range, stretching_range, 256)
        T_grid, S_grid = np.meshgrid(tilting_vals, stretching_vals)
        magnitude_grid = np.sqrt(S_grid**2 + T_grid**2)
        max_magnitude = np.sqrt(tilting_range**2 + stretching_range**2)
        magnitude_norm_grid = magnitude_grid / max_magnitude
        angle_grid = np.arctan2(S_grid, T_grid)
        hue_grid = (angle_grid + np.pi) / (2 * np.pi)
        saturation_grid = np.ones_like(hue_grid)
        value_grid = magnitude_norm_grid
        HSV_grid = np.stack((hue_grid, saturation_grid, value_grid), axis=-1)
        RGB_grid = hsv_to_rgb(HSV_grid)

        # Plot the bivariate colormap as an inset axes
        bbox = (0, 0.85, 0.25, 0.25)
        ax_inset = inset_axes(ax, width="100%", height="100%", loc='lower left',
                              bbox_to_anchor=bbox, bbox_transform=ax.transAxes, borderpad=0)

        extent = [tilting_vals.min() * 1e5, tilting_vals.max() * 1e5,
                  stretching_vals.min() * 1e5, stretching_vals.max() * 1e5]
        ax_inset.imshow(RGB_grid, origin='lower', extent=extent, aspect='auto')
        ax_inset.set_xlabel('Tilting ($\\times 10^{-5}$)', fontsize=8)
        ax_inset.set_ylabel('Stretching ($\\times 10^{-5}$)', fontsize=8)
        ax_inset.set_aspect('equal')
        ticks = np.linspace(-5, 5, 5)
        ax_inset.set_xticks(ticks)
        ax_inset.set_yticks(ticks)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)

        plt.savefig('test_3d_bivariate.png', dpi=200)
        plt.show()

    def select_trajectories(self):
        """
        Allows the user to select specific trajectory indices for further analysis.
        """
        print("Enter indices of trajectories to analyze (comma-separated), or type 'exit' to quit.")
        while True:
            user_input = input("Input: ").strip()
            if user_input.lower() == 'exit':
                print("Exiting the selection.")
                exit()
            else:
                try:
                    selected_indices = list(map(int, user_input.split(',')))
                    max_index = len(self.x_seeds) - 1
                    if all(0 <= idx <= max_index for idx in selected_indices):
                        self.selected_indices = selected_indices
                        print(f"Selected trajectories: {self.selected_indices}")
                        break
                    else:
                        print(f"Please enter indices between 0 and {max_index}.")
                except ValueError:
                    print("Invalid input. Please enter a valid list of indices or type 'exit'.")

    def plot_selected_trajectories_side_by_side(self, contour_value=9, buffer=0.01):

        if self.selected_indices is None:
            print("No trajectories selected. Please run select_trajectories() first.")
            return
    
        selected_indices = self.selected_indices
    
        traj_lon = self.traj_lon[selected_indices]
        traj_lat = self.traj_lat[selected_indices]
        traj_z = self.traj_z[selected_indices]
        traj_stretching = self.traj_stretching[selected_indices]
        traj_tilting = self.traj_tilting[selected_indices]
    
        # Determine bounds from selected trajectories
        lon_min = np.min(traj_lon) - buffer
        lon_max = np.max(traj_lon) + buffer
        lat_min = np.min(traj_lat) - buffer
        lat_max = np.max(traj_lat) + buffer
    
        fig = plt.figure(figsize=(7, 14), facecolor='white')
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
        ax2d = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax3d = fig.add_subplot(gs[1], projection='3d', facecolor='white')
    
        # ---------- Left plot (2D) ----------
        ax2d.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
        # Mask vorticity data outside plot limits
        mask = (self.lons >= lon_min) & (self.lons <= lon_max) & (self.lats >= lat_min) & (self.lats <= lat_max)
        vorticity_masked = np.ma.masked_where(~mask, self.vorticity_data)
    
        # Plot vorticity data
        vorticity_plot = ax2d.contourf(
            self.lons,
            self.lats,
            vorticity_masked,
            levels=self.vorticity_cmap.levels,
            cmap=self.vorticity_cmap.cmap,
            norm=self.vorticity_cmap.norm,
            extend='both',
            transform=ccrs.PlateCarree()
        )
    
        # No black contour line at contour_value on the 2D plot (removed)
    
        # Set axis labels and numeric formatting
        ax2d.set_xlabel("Longitude")
        ax2d.set_ylabel("Latitude")
        # Remove any legends
        # Just rely on cartopy's automatic ticks and formatting
        # Force numeric formatting without degrees symbol
        ax2d.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax2d.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
        # Plot selected trajectories in black, thicker lines
        for (lon, lat) in zip(traj_lon, traj_lat):
            ax2d.plot(lon, lat, color='black', linewidth=1.5, transform=ccrs.PlateCarree())
            # Plot seed location as a black circle
            seed_lon = lon[-1]
            seed_lat = lat[-1]
            ax2d.plot(seed_lon, seed_lat, 'o', color='black', transform=ccrs.PlateCarree())
    
        # ---------- Right plot (3D) ----------
        ax3d.set_xlabel("Longitude", labelpad=15)
        ax3d.set_ylabel("Latitude", labelpad=15)
        ax3d.set_zlabel("Altitude (m)", labelpad=15)
        ax3d.set_xlim(lon_min, lon_max)
        ax3d.set_ylim(lat_min, lat_max)
        ax3d.set_zlim(100, 500)
    
        tilting_range = 5e-5
        stretching_range = 5e-5
        for lon, lat, z, stretching, tilting in zip(traj_lon, traj_lat, traj_z, traj_stretching, traj_tilting):
            lon = np.array(lon)
            lat = np.array(lat)
            z = np.array(z)
            stretching = np.array(stretching)
            tilting = np.array(tilting)
    
            magnitude = np.sqrt(stretching**2 + tilting**2)
            mag_max = np.max(magnitude) if np.max(magnitude) != 0 else 1e-5
            magnitude_norm = magnitude / mag_max
            angle = np.arctan2(stretching, tilting)
            hue = (angle + np.pi) / (2 * np.pi)
            saturation = np.ones_like(hue)
            value = magnitude_norm
            HSV = np.stack((hue, saturation, value), axis=-1)
            RGB = hsv_to_rgb(HSV)
    
            points = np.array([lon, lat, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, colors=RGB[:-1], linewidth=2)
            ax3d.add_collection3d(lc)
    
        ax3d.grid(True, linestyle=':', linewidth=0.5, color='gray')
    
        # 3D filled contour to show front location
        # Example from previous code snippet
        z_surface = traj_z[0, -1]  # starting height of the trajectories
        mask_3d = (self.lons >= lon_min) & (self.lons <= lon_max) & (self.lats >= lat_min) & (self.lats <= lat_max)
        vorticity_masked_3d = np.ma.masked_where(~mask_3d, self.vorticity_data)
    
        ax3d.contourf(
            self.lons, self.lats, vorticity_masked_3d,
            levels=[contour_value, self.vorticity_data.max()],
            zdir='z',
            offset=z_surface,
            colors='grey',
            antialiased=True
        )
        self.vorticity_cmap.add_colorbar(vorticity_plot, ax3d, pad=0.07)
        
        # Bivariate colormap in the top-left corner of the 3D plot, larger
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cbar_width = "25%"
        cbar_height = "25%"
        cbar_ax = inset_axes(ax3d, width=cbar_width, height=cbar_height,
                             loc='upper left', borderpad=1.5)
    
        T_vals = np.linspace(-tilting_range, tilting_range, 256)
        S_vals = np.linspace(-stretching_range, stretching_range, 256)
        T_grid, S_grid = np.meshgrid(T_vals, S_vals)
        magnitude_grid = np.sqrt(S_grid**2 + T_grid**2)
        max_magnitude = np.sqrt(tilting_range**2 + stretching_range**2)
        magnitude_norm_grid = magnitude_grid / max_magnitude
        angle_grid = np.arctan2(S_grid, T_grid)
        hue_grid = (angle_grid + np.pi) / (2 * np.pi)
        saturation_grid = np.ones_like(hue_grid)
        value_grid = magnitude_norm_grid
        HSV_grid = np.stack((hue_grid, saturation_grid, value_grid), axis=-1)
        RGB_grid = hsv_to_rgb(HSV_grid)
    
        extent = [T_vals.min()*1e5, T_vals.max()*1e5,
                  S_vals.min()*1e5, S_vals.max()*1e5]
        cbar_ax.imshow(RGB_grid, origin='lower', extent=extent, aspect='auto')
        cbar_ax.set_xlabel('Tilting ($\\times 10^{-5}$)', fontsize=8)
        cbar_ax.set_ylabel('Stretching ($\\times 10^{-5}$)', fontsize=8)
        cbar_ax.set_aspect('equal')
        ticks = np.linspace(-5, 5, 5)
        cbar_ax.set_xticks(ticks)
        cbar_ax.set_yticks(ticks)
        cbar_ax.tick_params(axis='both', which='major', labelsize=8)
    
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
    
        plt.tight_layout()
        plt.savefig('selected_trajectories_side_by_side.png', dpi=200, bbox_inches='tight')
        plt.show()
            
    def plot_vorticity_components(self):
        """
        Plots time series of absolute vorticity, stretching, tilting, and height for selected trajectories.
        """
        if self.selected_indices is None:
            print("No trajectories selected. Please run select_trajectories() first.")
            return

        num_trajs = len(self.selected_indices)
        num_cols = 2
        num_rows = (num_trajs + 1) // num_cols  # Ceiling division

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))
        axes = axes.flatten()

        for idx, ax in zip(self.selected_indices, axes):
            time = np.arange(self.traj_avo.shape[1])  # Assuming time steps are regular and start from 0

            # Extract data for the trajectory
            avo = self.traj_avo[idx]
            stretching = self.traj_stretching[idx]
            tilting = self.traj_tilting[idx]
            total = stretching + tilting
            height = self.traj_z[idx]  # Height

            # Plotting
            ax.plot(time, avo, label='Absolute Vorticity')
            ax.plot(time, stretching, label='Stretching x $10^4$')
            ax.plot(time, tilting, label='Tilting x $10^4$')
            ax.plot(time, total, label='Stretching + Tilting')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('s$^-1$ x $10^-5$')
            ax2 = ax.twinx()
            ax2.plot(time, height, label='Height', color='grey', linestyle='--')
            ax2.set_ylabel('Height (m)')
            ax.set_title(f'Trajectory {idx}')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

        # Remove unused subplots
        for ax in axes[num_trajs:]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig('vorticity_components.png', dpi=200)
        plt.show()
        
        return self.traj_avo, self.traj_stretching, self.traj_tilting

# Usage
if __name__ == "__main__":
    config_path = 'inputs/type1_high_resolution_config.json'
    x0_range = [250, 350]
    y0_range = [190, 210]
    z0_range = [5]
    vorticity_threshold = 9
    buffer = 10

    # Create instance of ParticleTrajectoryAnalysis
    pta = ParticleTrajectoryAnalysis(
        config_path=config_path,
        x0_range=x0_range,
        y0_range=y0_range,
        z0_range=z0_range,
        vorticity_threshold=vorticity_threshold,
        buffer=buffer
    )

    # Plot seed locations and filter seeds
    pta.plot_seed_locations_and_filter(labels=True)

    # Validate and filter seeds interactively
    pta.validate_and_filter_seeds()

    # Compute trajectories
    pta.compute_trajectories()

    # Plot trajectories with vorticity
    pta.plot_trajectories_with_vorticity()

    # Plot 3D trajectories with vorticity
    pta.plot_trajectories_3d_with_vorticity()

    # Plot 3D trajectories with tilting and stretching
    pta.plot_trajectories_3d_with_tilting_stretching()

    # After the 3D plots, plot seed locations again with labels
    pta.plot_seed_locations_and_filter(labels=True)

    # Select trajectories of interest
    pta.select_trajectories()
    
    pta.plot_selected_trajectories_side_by_side()

    # Plot time series of vorticity components for selected trajectories
    avo, stretch, tilt = pta.plot_vorticity_components()