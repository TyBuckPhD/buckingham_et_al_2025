import xarray as xr
import numpy as np

class CalculateVorticity:
    def __init__(self, filename, target_height=500):
        self.filename = filename
        self.target_height = target_height  # Target height in meters AGL
        self.ds = None
        self.lats = None
        self.lons = None
        self.height_agl = None
        self.U_500m = None
        self.V_500m = None

    def load_dataset(self):
        self.ds = xr.open_dataset(self.filename)
        self.lats = self.ds['XLAT'].squeeze()
        self.lons = self.ds['XLONG'].squeeze()

    def calculate_geopotential_height(self):
        """
        Calculates geopotential height in meters above mean sea level (MSL) at mass grid points.
        """
        PH = self.ds['PH'].squeeze()  # Perturbation geopotential
        PHB = self.ds['PHB'].squeeze()  # Base-state geopotential
        g = 9.81  # Gravity acceleration (m s-2)

        geopotential = PH + PHB  # Total geopotential (m2 s-2)

        # Compute geopotential height at mass levels by averaging staggered levels
        geopotential = 0.5 * (geopotential[:-1, :, :] + geopotential[1:, :, :])

        height = geopotential / g  # Geopotential height (m)

        return height  # Dimensions: ('bottom_top', 'south_north', 'west_east')

    def calculate_height_agl(self, height):
        """
        Calculates height above ground level (AGL) in meters.
        """
        HGT = self.ds['HGT'].squeeze()  # Terrain height (m)

        # Subtract terrain height from geopotential height to get height AGL
        self.height_agl = height - HGT  # Broadcasts HGT along 'bottom_top'

        if 'bottom_top_stag' in self.height_agl.dims:
            self.height_agl = self.height_agl.rename({'bottom_top_stag': 'bottom_top'})

    def unstagger_winds(self):
        """
        Unstaggers the U and V wind components.
        """
        U_stag = self.ds['U'].squeeze()
        V_stag = self.ds['V'].squeeze()

        # Unstagger U (average over west_east_stag dimension)
        U = 0.5 * (U_stag[:, :, :-1] + U_stag[:, :, 1:])
        # Unstagger V (average over south_north_stag dimension)
        V = 0.5 * (V_stag[:, :-1, :] + V_stag[:, 1:, :])

        if 'west_east_stag' in U.dims:
            U = U.rename({'west_east_stag': 'west_east'})
        if 'south_north_stag' in V.dims:
            V = V.rename({'south_north_stag': 'south_north'})

        return U, V

    def get_closest_height_level(self):
        """
        Find the closest level in the 'bottom_top' dimension to the target height.
        """
        height_diff = np.abs(self.height_agl - self.target_height)
        closest_level = height_diff.argmin(dim='bottom_top')  # Find the index of the closest height level
        return closest_level

    def interpolate_to_height(self, U, V):
        """
        Interpolates U and V to the target height AGL using xarray's native interpolation
        method along the vertical (bottom_top) dimension.
        """
        closest_level = self.get_closest_height_level()

        # Interpolate U and V to the target height (e.g., 500m AGL)
        self.U_500m = U.isel(bottom_top=closest_level)
        self.V_500m = V.isel(bottom_top=closest_level)

    def calculate_vorticity(self):
        """
        Calculates absolute vorticity at the target height.
        """
        # Compute distances between grid points
        Re = 6371000  # Earth's radius in meters
        lat_rad = np.deg2rad(self.lats)
        lon_rad = np.deg2rad(self.lons)

        lat_rad = lat_rad.values
        lon_rad = lon_rad.values

        # Calculate differences in lat/lon
        dlat = np.gradient(lat_rad, axis=0)
        dlon = np.gradient(lon_rad, axis=1)

        # Compute dy and dx in meters
        dy = dlat * Re
        dx = dlon * Re * np.cos(lat_rad)

        # For simplicity, use average grid spacing
        dx_mean = np.mean(dx)
        dy_mean = np.mean(dy)

        # Compute gradients
        dudy, dudx = np.gradient(self.U_500m.values, dy_mean, dx_mean, edge_order=2)
        dvdy, dvdx = np.gradient(self.V_500m.values, dy_mean, dx_mean, edge_order=2)

        # Calculate relative vorticity
        rel_vorticity = dvdx - dudy

        # Compute Coriolis parameter
        f = 2 * 7.2921e-5 * np.sin(np.deg2rad(self.lats.values))

        # Calculate absolute vorticity
        abs_vorticity = rel_vorticity + f

        return abs_vorticity

    def run(self):
        # Step 1: Load dataset
        self.load_dataset()

        # Step 2: Calculate geopotential height
        height = self.calculate_geopotential_height()

        # Step 3: Calculate height above ground level (AGL)
        self.calculate_height_agl(height)

        # Step 4: Unstagger winds
        U, V = self.unstagger_winds()

        # Step 5: Interpolate winds to target height
        self.interpolate_to_height(U, V)

        # Step 6: Calculate vorticity
        abs_vorticity = self.calculate_vorticity()

        return abs_vorticity, self.lats, self.lons
    
filename = '/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2011-11-29_15-00-00.nc'
cv = CalculateVorticity(filename)
av, lats, lons = cv.run()