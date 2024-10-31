from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords

class GetVariablesWRF:
    def __init__(self, filename):
        self.filename = filename
        self.rainfall_rate_surface = None
        self.absolute_vorticity = None
        self.lats = None
        self.lons = None

    def get_lat_lons(self):
        """ Retrieve latitude and longitude coordinates. """
        with Dataset(self.filename) as wrf_data:
            lats, lons = latlon_coords(getvar(wrf_data, "XLAT"))
            self.lats = to_np(lats)
            self.lons = to_np(lons)
        return self.lats, self.lons

    def get_rainfall_rate(self, timeidx=0):
        """ Calculate rainfall rate from radar reflectivity (dBZ) at a specific time index. """
        with Dataset(self.filename) as wrf_data:
            dbz = getvar(wrf_data, "dbz", timeidx=timeidx)
            rain_rate = (10 ** (to_np(dbz) / 10) / 200) ** 0.625
            self.rainfall_rate_surface = rain_rate[0]  # For surface.
        return self.rainfall_rate_surface

    def get_absolute_vorticity(self, height=500, timeidx=0):
        """ Calculate absolute vorticity at a given height above ground level (e.g., 500m AGL). """
        with Dataset(self.filename) as wrf_data:
            # Calculate absolute vorticity and interpolate to the specified height above ground level.
            abs_vort = getvar(wrf_data, "avo", timeidx=timeidx)
            height_agl = getvar(wrf_data, "height_agl", timeidx=timeidx)
            abs_vort_at_height = interplevel(abs_vort, height_agl, height)
            self.absolute_vorticity = to_np(abs_vort_at_height) * 1e-5
        return self.absolute_vorticity