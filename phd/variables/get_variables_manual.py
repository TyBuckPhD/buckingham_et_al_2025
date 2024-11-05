import numpy as npimport xarray as xrclass GetVariablesWRF:    def __init__(self, filename):        self.filename = filename        self.absolute_vorticity = None        self.lats = None        self.lons = None    def get_lat_lons(self):        """Retrieve latitude and longitude coordinates."""        with xr.open_dataset(self.filename) as ds:            self.lats = ds['XLAT']            self.lons = ds['XLONG']        return self.lats, self.lons    def de_stagger(self, var, axis):        """De-stagger a variable along a specified axis using xarray, automatically renaming the dimension."""        destaggered_var = 0.5 * (var.isel({axis: slice(1, None)}) +                                 var.isel({axis: slice(None, -1)}))        new_dim_name = axis.replace('_stag', '')        return destaggered_var.rename({axis: new_dim_name})    def compute_geopotential_height(self, ds):        """Calculate full geopotential height from perturbation and base geopotential, retaining DataArray properties."""        PH = self.de_stagger(ds['PH'], axis='bottom_top_stag')        PHB = self.de_stagger(ds['PHB'], axis='bottom_top_stag')        return (PH + PHB) / 9.81  # Convert to meters    def find_closest_height_level(self, height_field, target_height):        """Find the index of the closest height level to the target height at each lat/lon."""        height_diff = np.abs(height_field - target_height)        height_idx = height_diff.argmin(dim='bottom_top')  # Index for the closest height level        return height_idx    def extract_variable_at_height(self, var, height_idx):        """Extract a variable (e.g., U or V) at the specified height level for each lat/lon."""        return var.isel(bottom_top=height_idx)    def calculate_coriolis_parameter(self, lats):        """Calculate Coriolis parameter based on latitude in degrees."""        omega = 7.2921159e-5  # Earth's rotation rate (rad/s)        lat_radians = np.radians(lats)        return 2 * omega * np.sin(lat_radians)    def get_wind_components(self):        with xr.open_dataset(self.filename) as ds:            self.u = self.de_stagger(ds['U'], axis='west_east_stag')            self.v = self.de_stagger(ds['V'], axis='south_north_stag')            self.w = self.de_stagger(ds['W'], axis='bottom_top_stag')                    return self.u, self.v, self.w        def get_wind_components_at_heights(self, height=100):        with xr.open_dataset(self.filename) as ds:            self.u = self.de_stagger(ds['U'], axis='west_east_stag')            self.v = self.de_stagger(ds['V'], axis='south_north_stag')                        # Calculate geopotential height            height_field = self.compute_geopotential_height(ds)            # Find closest height level to target height            height_idx = self.find_closest_height_level(height_field, height)                        self.u_hgt = self.extract_variable_at_height(self.u, height_idx)            self.v_hgt = self.extract_variable_at_height(self.v, height_idx)                    return self.u_hgt, self.v_hgt                    def get_absolute_vorticity(self, height=500):        """Calculate absolute vorticity at a specified height above ground level (e.g., 500m AGL)."""        with xr.open_dataset(self.filename) as ds:            # Retrieve latitude and longitude coordinates            self.get_lat_lons()            # De-stagger U and V to a consistent grid            U = self.de_stagger(ds['U'], axis='west_east_stag')            V = self.de_stagger(ds['V'], axis='south_north_stag')            # Calculate geopotential height            height_field = self.compute_geopotential_height(ds)            # Find closest height level to target height            height_idx = self.find_closest_height_level(height_field, height)            # Extract U and V at this level            U_at_height = self.extract_variable_at_height(U, height_idx)            V_at_height = self.extract_variable_at_height(V, height_idx)            # Calculate gradients for relative vorticity using numpy's gradient            dx = ds.attrs['DX']            dy = ds.attrs['DY']            dv_dx = np.gradient(V_at_height, dx, axis=-1)            du_dy = np.gradient(U_at_height, dy, axis=-2)            # Calculate Coriolis parameter            f = self.calculate_coriolis_parameter(self.lats)            # Calculate absolute vorticity            zeta = dv_dx - du_dy            absolute_vorticity = (zeta + f)            self.absolute_vorticity = absolute_vorticity * 1000 # Convert to desired units.        return self.absolute_vorticity