from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords

class GetVariablesWRF:
    """
    Class for extracting and processing variables from WRF model output files.

    This class provides methods to load and extract key meteorological variables from a WRF output file.
    It uses the netCDF4 library to read the file and the wrf-python package functions (e.g., getvar, interplevel,
    to_np, latlon_coords) to extract variables such as latitude, longitude, radar reflectivity (to compute rainfall rate),
    and absolute vorticity. The class stores the extracted data as NumPy arrays for further processing or plotting.

    Attributes:
      filename (str): Path to the WRF output file.
      rainfall_rate_surface (numpy.ndarray): Array containing the computed surface rainfall rate.
      absolute_vorticity (numpy.ndarray): Array containing the computed absolute vorticity.
      lats (numpy.ndarray): Array of latitude coordinates extracted from the dataset.
      lons (numpy.ndarray): Array of longitude coordinates extracted from the dataset.

    Methods:
      get_lat_lons():
          Opens the WRF file, extracts the "XLAT" variable, and returns the latitude and longitude coordinates
          as NumPy arrays.
      get_rainfall_rate(timeidx=0):
          Reads the radar reflectivity (dbz) variable from the file at a specified time index, computes the rainfall
          rate using a conversion formula, and returns the surface rainfall rate.
      get_absolute_vorticity(height=500, timeidx=0):
          Reads the absolute vorticity ("avo") and height above ground level ("height_agl") from the file,
          interpolates the vorticity to a specified target height (e.g., 500 m AGL), scales the result, and returns it.
    """

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
            self.absolute_vorticity = to_np(abs_vort_at_height) * 0.01
        return self.absolute_vorticity