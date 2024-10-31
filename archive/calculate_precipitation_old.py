import xarray as xr
import numpy as np

class CalculatePrecipitation:
    def __init__(self, current_filename, previous_filename, delta):
        self.current_filename = current_filename
        self.previous_filename = previous_filename
        self.delta = delta  # Time difference in minutes
        self.Rainfall_Rate_surface = None
        self.lats = None
        self.lons = None

    def load_datasets(self):
        self.ds1 = xr.open_dataset(self.current_filename)
        self.ds2 = xr.open_dataset(self.previous_filename)

    def calculate_rainfall_rate(self):
        rainnc1 = self.ds1['RAINNC'] + self.ds1['RAINC']
        rainnc2 = self.ds2['RAINNC'] + self.ds2['RAINC']

        delta_rain = rainnc1 - rainnc2
        delta_time = self.delta / 60.0  # Convert minutes to hours
        self.Rainfall_Rate_surface = delta_rain / delta_time
        self.Rainfall_Rate_surface = self.Rainfall_Rate_surface.where(self.Rainfall_Rate_surface >= 0, 0)

        self.lats = self.ds1['XLAT']
        self.lons = self.ds1['XLONG']

    def run(self):
        self.load_datasets()
        self.calculate_rainfall_rate()
        return self.Rainfall_Rate_surface, self.lats, self.lons