import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import interpn, RegularGridInterpolator
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.timer import Timer

class BackwardParticleTrajectories:
    def __init__(self, config_path, dt, grid_spacing, x0=None, y0=None, z0=None):
        self.config_path = config_path
        self.dt = dt
        self.grid_spacing = grid_spacing
        self.x_seeds = x0
        self.y_seeds = y0
        self.z_seeds = z0
        self._load_data()

    def _load_data(self):
        """Load WRF data and initialize required variables."""
        gvw = GetVariablesWRF(config_path=self.config_path)
        self.lats, self.lons = gvw.get_lat_lons()
        self.u, self.v, self.w = gvw.get_wind_components()
        self.height_agl = gvw.get_geopotential_height()
        self.absolute_vorticity, self.stretching, self.tilting = gvw.get_full_absolute_vorticity()

        # Grid spacings
        self.dx = self.dy = self.grid_spacing

        # Dimensions
        self.nt, self.nz, self.ny, self.nx = self.u.shape

        # Assign physical coordinates to the DataArrays
        self.x_coords = np.arange(self.nx) * self.dx  # x in meters
        self.y_coords = np.arange(self.ny) * self.dy  # y in meters
        self.u = self.u.assign_coords(
            {'west_east': self.x_coords, 'south_north': self.y_coords})
        self.v = self.v.assign_coords(
            {'west_east': self.x_coords, 'south_north': self.y_coords})
        self.w = self.w.assign_coords(
            {'west_east': self.x_coords, 'south_north': self.y_coords})

        # Ensure height_agl has the 'Time' dimension
        if 'Time' not in self.height_agl.dims:
            self.height_agl = self.height_agl.expand_dims('Time', axis=0)

        # Broadcast height_agl to match u's dimensions
        self.height_agl, self.u = xr.broadcast(self.height_agl, self.u)
        self.u = self.u.assign_coords({'z': self.height_agl})
        self.v = self.v.assign_coords({'z': self.height_agl})
        self.w = self.w.assign_coords({'z': self.height_agl})

        # Extract z-coordinate levels
        self.z_coords = self.height_agl.isel(
            Time=0, south_north=0, west_east=0).values

        # Convert input indices to coords in metres
        self.x0 = self.x_coords[self.x_seeds]
        self.y0 = self.y_coords[self.y_seeds]
        self.z0 = self.z_coords[self.z_seeds]

    @Timer
    def compute_trajectories(self):
        """
        Compute backward trajectories for multiple particles over time.
    
        Returns:
        - lon_traj, lat_traj, z_traj: Arrays of trajectories [n_particles, nt].
        - zeta_abs_traj, stretching_traj, tilting_traj: Arrays of variables along trajectories.
        """
        n_particles = len(self.x0)

        # Initialize arrays for trajectories
        x_positions = np.zeros((n_particles, self.nt))
        y_positions = np.zeros((n_particles, self.nt))
        z_positions = np.zeros((n_particles, self.nt))

        zeta_abs_traj = np.zeros((n_particles, self.nt))
        stretching_traj = np.zeros((n_particles, self.nt))
        tilting_traj = np.zeros((n_particles, self.nt))

        # Set initial positions at the final timestep
        x_positions[:, -1] = self.x0
        y_positions[:, -1] = self.y0
        z_positions[:, -1] = self.z0

        points = (self.z_coords, self.y_coords, self.x_coords)
        z_min = self.z_coords[0]  # First model level

        # Reverse time-stepping
        for t_idx in range(self.nt - 1, -1, -1):  # Include t_idx = 0
            print(f"Backward time step: {t_idx}")

            x_p = x_positions[:, t_idx]
            y_p = y_positions[:, t_idx]
            z_p = z_positions[:, t_idx]

            xi = np.vstack([z_p, y_p, x_p]).T

            u_t = self.u.isel(Time=t_idx).values
            v_t = self.v.isel(Time=t_idx).values
            w_t = self.w.isel(Time=t_idx).values
            zeta_abs_t = self.absolute_vorticity.isel(Time=t_idx).values
            stretching_t = self.stretching.isel(Time=t_idx).values
            tilting_t = self.tilting.isel(Time=t_idx).values

            # Use negative velocities for backward trajectories
            u_p = interpn(points, -u_t, xi, method='linear',
                          bounds_error=False, fill_value=np.nan)
            v_p = interpn(points, -v_t, xi, method='linear',
                          bounds_error=False, fill_value=np.nan)
            w_p = interpn(points, -w_t, xi, method='linear',
                          bounds_error=False, fill_value=np.nan)
            zeta_p = interpn(points, zeta_abs_t, xi, method='linear',
                             bounds_error=False, fill_value=np.nan)
            stretching_p = interpn(
                points, stretching_t, xi, method='linear', bounds_error=False, fill_value=np.nan)
            tilting_p = interpn(
                points, tilting_t, xi, method='linear', bounds_error=False, fill_value=np.nan)

            valid_particles = ~np.isnan(u_p) & ~np.isnan(v_p) & ~np.isnan(w_p)

            # Store interpolated variables at current time index
            zeta_abs_traj[valid_particles, t_idx] = zeta_p[valid_particles]
            stretching_traj[valid_particles,
                            t_idx] = stretching_p[valid_particles]
            tilting_traj[valid_particles, t_idx] = tilting_p[valid_particles]

            if t_idx > 0:
                # Update positions for the next time step (t_idx - 1)
                x_positions[valid_particles, t_idx -
                            1] = x_p[valid_particles] + u_p[valid_particles] * self.dt
                y_positions[valid_particles, t_idx -
                            1] = y_p[valid_particles] + v_p[valid_particles] * self.dt
                z_positions[valid_particles, t_idx - 1] = np.maximum(
                    z_p[valid_particles] + w_p[valid_particles] * self.dt, z_min)

                # Handle invalid particles by keeping their positions unchanged
                x_positions[~valid_particles, t_idx -
                            1] = x_positions[~valid_particles, t_idx]
                y_positions[~valid_particles, t_idx -
                            1] = y_positions[~valid_particles, t_idx]
                z_positions[~valid_particles, t_idx -
                            1] = z_positions[~valid_particles, t_idx]

        # Interpolate to lat lon.
        positions = np.stack(
            [y_positions.flatten(), x_positions.flatten()], axis=-1)
        lat_interp = RegularGridInterpolator(
            (self.y_coords, self.x_coords), self.lats.values, method='linear', bounds_error=False)
        lon_interp = RegularGridInterpolator(
            (self.y_coords, self.x_coords), self.lons.values, method='linear', bounds_error=False)
        self.lat_traj = lat_interp(positions).reshape(len(self.x0), -1)
        self.lon_traj = lon_interp(positions).reshape(len(self.x0), -1)
        self.z_traj = z_positions

        return self.lon_traj, self.lat_traj, self.z_traj, zeta_abs_traj, stretching_traj, tilting_traj

    def plot_trajectories(self, buffer=0.01):
        # Calculate dynamic extent based on trajectories
        lat_min, lat_max = np.min(self.lat_traj) - \
            buffer, np.max(self.lat_traj) + buffer
        lon_min, lon_max = np.min(self.lon_traj) - \
            buffer, np.max(self.lon_traj) + buffer

        # Plot map
        plt.figure(figsize=(10, 12))
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot trajectories
        for i in range(len(self.x0)):
            ax.plot(self.lon_traj[i], self.lat_traj[i],
                    marker='o', transform=ccrs.PlateCarree())

        plt.show()

    def plot_heights(self):
        """Plot particle heights over time."""
        time = np.arange(self.nt) * self.dt / 3600
        plt.figure(figsize=(12, 6))
        for i in range(len(self.x0)):
            plt.plot(time, self.z_traj[i])

        plt.title('Height of Particles Over Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Height (m)')
        plt.grid(True)
        plt.show()
