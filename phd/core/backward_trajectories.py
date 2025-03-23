import numpy as np
import xarray as xr
import dask.array as da
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.timer import Timer

class BackwardParticleTrajectories:
    def __init__(self, config_path, dt, grid_spacing, x0=None, y0=None, z0=None, f=0.0):
        self.config_path = config_path
        self.dt = dt
        self.grid_spacing = grid_spacing
        self.x_seeds = x0
        self.y_seeds = y0
        self.z_seeds = z0
        self.f = f  # Coriolis parameter if needed
        self._load_data()
        self._compute_velocity_gradients()
        
    @Timer
    def _load_data(self):
        """Load WRF data and initialize required variables."""
        gvw = GetVariablesWRF(config_path=self.config_path)
        self.lats, self.lons = gvw.get_lat_lons()
        self.u, self.v, self.w = gvw.get_wind_components()
        self.height_agl = gvw.get_geopotential_height()

        # Horizontal grid spacing
        self.dx = self.dy = self.grid_spacing
        self.dz = gvw.get_vertical_resolution().data

        # Dimensions
        self.nt, self.nz, self.ny, self.nx = self.u.shape

        # Assign physical coordinates
        self.x_coords = np.arange(self.nx) * self.dx
        self.y_coords = np.arange(self.ny) * self.dy

        self.u = self.u.assign_coords({'west_east': self.x_coords, 'south_north': self.y_coords})
        self.v = self.v.assign_coords({'west_east': self.x_coords, 'south_north': self.y_coords})
        self.w = self.w.assign_coords({'west_east': self.x_coords, 'south_north': self.y_coords})

        # Ensure height_agl has Time dim
        if 'Time' not in self.height_agl.dims:
            self.height_agl = self.height_agl.expand_dims('Time', axis=0)

        # Broadcast height to match u
        self.height_agl, self.u = xr.broadcast(self.height_agl, self.u)
        self.u = self.u.assign_coords({'z': self.height_agl})
        self.v = self.v.assign_coords({'z': self.height_agl})
        self.w = self.w.assign_coords({'z': self.height_agl})

        # Vertical coordinates
        self.z_coords = self.height_agl.isel(Time=0, south_north=0, west_east=0).values

        # Convert seed indices to coordinates
        self.x0 = self.x_coords[self.x_seeds]
        self.y0 = self.y_coords[self.y_seeds]
        self.z0 = self.z_coords[self.z_seeds]

    @Timer
    def _compute_velocity_gradients(self):
        du_dx = da.gradient(self.u.data, self.dx, axis=3)
        du_dy = da.gradient(self.u.data, self.dy, axis=2)
        du_dz = da.gradient(self.u.data, axis=1) / self.dz
    
        dv_dx = da.gradient(self.v.data, self.dx, axis=3)
        dv_dy = da.gradient(self.v.data, self.dy, axis=2)
        dv_dz = da.gradient(self.v.data, axis=1) / self.dz
    
        dw_dx = da.gradient(self.w.data, self.dx, axis=3)
        dw_dy = da.gradient(self.w.data, self.dy, axis=2)
        dw_dz = da.gradient(self.w.data, axis=1) / self.dz

        self.du_dx = xr.DataArray(du_dx, dims=self.u.dims, coords=self.u.coords)
        self.du_dy = xr.DataArray(du_dy, dims=self.u.dims, coords=self.u.coords)
        self.du_dz = xr.DataArray(du_dz, dims=self.u.dims, coords=self.u.coords)

        self.dv_dx = xr.DataArray(dv_dx, dims=self.v.dims, coords=self.v.coords)
        self.dv_dy = xr.DataArray(dv_dy, dims=self.v.dims, coords=self.v.coords)
        self.dv_dz = xr.DataArray(dv_dz, dims=self.v.dims, coords=self.v.coords)

        self.dw_dx = xr.DataArray(dw_dx, dims=self.w.dims, coords=self.w.coords)
        self.dw_dy = xr.DataArray(dw_dy, dims=self.w.dims, coords=self.w.coords)
        self.dw_dz = xr.DataArray(dw_dz, dims=self.w.dims, coords=self.w.coords)

    @Timer
    def compute_trajectories(self):
        """
        Compute backward trajectories using a predictor-corrector (two-step) integration approach
        without rotation, then produce vorticity anomaly (vorticity minus its baseline at the final time)
        and cumulative integrals for stretching and tilting.
    
        Returns:
        - lon_traj, lat_traj, z_traj: Particle positions [n_particles, nt].
        - zeta_abs_anomaly: Instantaneous vorticity minus the baseline vorticity at the final (starting) timestep.
        - stretching_cumtrapz: Cumulative integrated stretching [n_particles, nt]
        - tilting_cumtrapz: Cumulative integrated tilting [n_particles, nt]
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
    
        points_3d = (self.z_coords, self.y_coords, self.x_coords)
        z_min = self.z_coords[0]
    
        def interp_3d(field_3d, xi):
            return interpn(points_3d, field_3d, xi, method='linear', bounds_error=False, fill_value=np.nan)
    
        for t_idx in range(self.nt - 1, 0, -1):
            print(f"Processing backward time step {t_idx}/{self.nt - 1}")
    
            x_p = x_positions[:, t_idx]
            y_p = y_positions[:, t_idx]
            z_p = z_positions[:, t_idx]
            xi = np.vstack([z_p, y_p, x_p]).T
    
            # Current and next time fields
            u_t = self.u.isel(Time=t_idx).values
            v_t = self.v.isel(Time=t_idx).values
            w_t = self.w.isel(Time=t_idx).values
    
            u_next = self.u.isel(Time=t_idx - 1).values
            v_next = self.v.isel(Time=t_idx - 1).values
            w_next = self.w.isel(Time=t_idx - 1).values
    
            # Interpolate velocities at current time (predictor)
            u_p = interp_3d(-u_t, xi)
            v_p = interp_3d(-v_t, xi)
            w_p = interp_3d(-w_t, xi)
    
            valid = ~np.isnan(u_p) & ~np.isnan(v_p) & ~np.isnan(w_p)
    
            # Predictor step
            x_pred = x_p[valid] + u_p[valid] * self.dt
            y_pred = y_p[valid] + v_p[valid] * self.dt
            z_pred = np.maximum(z_p[valid] + w_p[valid] * self.dt, z_min)
    
            # Interpolate velocities at next time step using predicted position
            xi_pred = np.vstack([z_pred, y_pred, x_pred]).T
            u_p_next = interp_3d(-u_next, xi_pred)
            v_p_next = interp_3d(-v_next, xi_pred)
            w_p_next = interp_3d(-w_next, xi_pred)
    
            # Corrector step
            x_positions[valid, t_idx - 1] = x_p[valid] + 0.5*(u_p[valid] + u_p_next)*self.dt
            y_positions[valid, t_idx - 1] = y_p[valid] + 0.5*(v_p[valid] + v_p_next)*self.dt
            z_positions[valid, t_idx - 1] = np.maximum(z_p[valid] + 0.5*(w_p[valid] + w_p_next)*self.dt, z_min)
    
            x_positions[~valid, t_idx - 1] = x_positions[~valid, t_idx]
            y_positions[~valid, t_idx - 1] = y_positions[~valid, t_idx]
            z_positions[~valid, t_idx - 1] = z_positions[~valid, t_idx]
    
            # Interpolate gradients and compute vorticity terms at current positions
            du_dy_t = self.du_dy.isel(Time=t_idx).values
            du_dz_t = self.du_dz.isel(Time=t_idx).values
            dv_dx_t = self.dv_dx.isel(Time=t_idx).values
            dv_dz_t = self.dv_dz.isel(Time=t_idx).values
            dw_dx_t = self.dw_dx.isel(Time=t_idx).values
            dw_dy_t = self.dw_dy.isel(Time=t_idx).values
            dw_dz_t = self.dw_dz.isel(Time=t_idx).values
    
            du_dy_p = interp_3d(du_dy_t, xi)
            du_dz_p = interp_3d(du_dz_t, xi)
            dv_dx_p = interp_3d(dv_dx_t, xi)
            dv_dz_p = interp_3d(dv_dz_t, xi)
            dw_dx_p = interp_3d(dw_dx_t, xi)
            dw_dy_p = interp_3d(dw_dy_t, xi)
            dw_dz_p = interp_3d(dw_dz_t, xi)
    
            omega_z = dv_dx_p - du_dy_p
            omega_z_abs = omega_z + self.f
            stretching_p = omega_z_abs * dw_dz_p
            tilting_p = -((dw_dx_p * dv_dz_p) - (dw_dy_p * du_dz_p))
    
            zeta_abs_traj[valid, t_idx] = omega_z_abs[valid]
            stretching_traj[valid, t_idx] = stretching_p[valid]
            tilting_traj[valid, t_idx] = tilting_p[valid]
    
        # At t_idx=0:
        x_p = x_positions[:, 0]
        y_p = y_positions[:, 0]
        z_p = z_positions[:, 0]
        xi = np.vstack([z_p, y_p, x_p]).T
    
        du_dy_0 = self.du_dy.isel(Time=0).values
        du_dz_0 = self.du_dz.isel(Time=0).values
        dv_dx_0 = self.dv_dx.isel(Time=0).values
        dv_dz_0 = self.dv_dz.isel(Time=0).values
        dw_dx_0 = self.dw_dx.isel(Time=0).values
        dw_dy_0 = self.dw_dy.isel(Time=0).values
        dw_dz_0 = self.dw_dz.isel(Time=0).values
    
        du_dy_p = interp_3d(du_dy_0, xi)
        du_dz_p = interp_3d(du_dz_0, xi)
        dv_dx_p = interp_3d(dv_dx_0, xi)
        dv_dz_p = interp_3d(dv_dz_0, xi)
        dw_dx_p = interp_3d(dw_dx_0, xi)
        dw_dy_p = interp_3d(dw_dy_0, xi)
        dw_dz_p = interp_3d(dw_dz_0, xi)
    
        valid = np.ones(n_particles, dtype=bool)
        omega_z = dv_dx_p - du_dy_p
        omega_z_abs = omega_z + self.f
        stretching_p = omega_z_abs * dw_dz_p
        tilting_p = -((dw_dx_p * dv_dz_p) - (dw_dy_p * du_dz_p))
    
        zeta_abs_traj[valid, 0] = omega_z_abs[valid]
        stretching_traj[valid, 0] = stretching_p[valid]
        tilting_traj[valid, 0] = tilting_p[valid]
    
        # Interpolate lat/lon for final trajectories
        positions = np.stack([y_positions.flatten(), x_positions.flatten()], axis=-1)
        lat_interp = RegularGridInterpolator((self.y_coords, self.x_coords), self.lats.values, method='linear', bounds_error=False)
        lon_interp = RegularGridInterpolator((self.y_coords, self.x_coords), self.lons.values, method='linear', bounds_error=False)
    
        self.lat_traj = lat_interp(positions).reshape(len(self.x0), -1)
        self.lon_traj = lon_interp(positions).reshape(len(self.x0), -1)
        self.z_traj = z_positions
    
        # Define the baseline vorticity as the value at the final (starting) timestep: t_idx = nt-1
        # Then create a vorticity anomaly by subtracting this baseline from all timesteps
        baseline_vort = zeta_abs_traj[:, 0]  # shape: [n_particles]
        zeta_abs_anomaly = zeta_abs_traj - baseline_vort[:, None]
    
        # Integrate stretching and tilting cumulatively over time using cumulative_trapezoid
        stretching_cumtrapz = cumulative_trapezoid(stretching_traj, dx=self.dt, axis=1, initial=0)
        tilting_cumtrapz = cumulative_trapezoid(tilting_traj, dx=self.dt, axis=1, initial=0)
    
        return (self.lon_traj, self.lat_traj, self.z_traj,
                zeta_abs_anomaly, stretching_cumtrapz, tilting_cumtrapz)