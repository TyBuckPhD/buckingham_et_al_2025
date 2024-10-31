import xarray as xr
import numpy as np

class CalculatePrecipitation:
    def __init__(self, filepath, ivarint=1, iliqskin=1):
        """
        Initialize with filepath and calculation options.
        """
        self.filepath = filepath
        self.ivarint = ivarint
        self.iliqskin = iliqskin
        self.ds = None  # Dataset placeholder
        self.P = None
        self.T = None
        self.qv = None
        self.qr = None
        self.qs = None
        self.qg = None
        self.dbz_surface = None  # Surface reflectivity in dBZ
        self.rain_rate = None    # Surface rain rate in mm/hr
        self.lats = None          # Latitude grid
        self.lons = None          # Longitude grid

        # Load data on initialization
        self.load_data()

    def load_data(self):
        """
        Load data from the WRF file and process necessary variables.
        """
        self.ds = xr.open_dataset(self.filepath)
        self.P = self.ds['P'] + self.ds['PB']  # Full pressure in Pa
        T = self.ds['T'] + 300                 # Temperature in K, assuming T is perturbation potential temperature
        self.T = (self.P / 100000) ** (287.05 / 1004) * T  # Actual temperature in K (assuming dry air)

        self.qv = self.ds['QVAPOR']            # Water vapor mixing ratio
        self.qr = self.ds['QRAIN']             # Rain mixing ratio
        self.qs = self.ds['QSNOW'] if 'QSNOW' in self.ds.variables else xr.zeros_like(self.P)
        self.qg = self.ds['QGRAUP'] if 'QGRAUP' in self.ds.variables else xr.zeros_like(self.P)

        # Load latitude and longitude if available
        if 'XLAT' in self.ds.variables and 'XLONG' in self.ds.variables:
            self.lats = self.ds['XLAT']
            self.lons = self.ds['XLONG']
        else:
            raise ValueError("Latitude (XLAT) and Longitude (XLONG) not found in WRF file.")

    def calculate_reflectivity(self):
        """
        Calculate radar reflectivity (dBZ) from WRF variables at the surface level.
        """
        rho_rain = 1000.0
        rho_snow = 100.0
        rho_graupel = 400.0
        
        N_rain = 8.0e6
        N_snow = 2.0e7
        N_graupel = 4.0e6

        # Adjust intercept parameters if ivarint = 1
        if self.ivarint == 1:
            N_rain *= np.exp(-0.053 * (self.T - 273.15))
            N_snow *= np.exp(-0.1 * (self.T - 273.15))
            N_graupel *= np.exp(-0.07 * (self.T - 273.15))
        
        coef_rain = 720.0 * N_rain / (rho_rain ** 1.75)
        coef_snow = 720.0 * N_snow / (rho_snow ** 1.75)
        coef_graupel = 720.0 * N_graupel / (rho_graupel ** 1.75)

        # Extract only the surface (first vertical level) data for calculation
        reflectivity_rain = coef_rain * (self.qr.isel(bottom_top=0) * self.P.isel(bottom_top=0) / (287.05 * self.T.isel(bottom_top=0))) ** 1.75
        reflectivity_snow = coef_snow * (self.qs.isel(bottom_top=0) * self.P.isel(bottom_top=0) / (287.05 * self.T.isel(bottom_top=0))) ** 1.75
        reflectivity_graupel = coef_graupel * (self.qg.isel(bottom_top=0) * self.P.isel(bottom_top=0) / (287.05 * self.T.isel(bottom_top=0))) ** 1.75

        # Handle scattering if iliqskin = 1
        if self.iliqskin == 1:
            is_above_freezing = self.T.isel(bottom_top=0) > 273.15
            reflectivity_snow = np.where(is_above_freezing, reflectivity_snow * (rho_snow / rho_rain) ** 1.75, reflectivity_snow)
            reflectivity_graupel = np.where(is_above_freezing, reflectivity_graupel * (rho_graupel / rho_rain) ** 1.75, reflectivity_graupel)
        
        # Total surface reflectivity (linear scale) and conversion to dBZ
        total_reflectivity = reflectivity_rain + reflectivity_snow + reflectivity_graupel
        self.dbz_surface = 10.0 * np.log10(np.maximum(total_reflectivity, 1e-12))  # Avoid log(0)

        return self.dbz_surface

    def convert_to_rain_rate(self):
        """
        Convert reflectivity (dBZ) at the surface to rain rate (mm/hr) and return with lat/lon.
        """
        if self.dbz_surface is None:
            raise ValueError("Reflectivity has not been calculated. Run calculate_reflectivity() first.")
        
        # Convert dBZ to Z (linear reflectivity)
        Z = 10 ** (self.dbz_surface / 10.0)
        
        # Empirical conversion from Z to rain rate: R = a * Z^b
        a, b = 200, 0.6
        self.rain_rate = a * Z ** b  # Rain rate in mm/hr
        
        if self.lats is None or self.lons is None:
            raise ValueError("Latitude and Longitude not loaded. Please check the file format.")
        
        return self.rain_rate, self.lats, self.lons

    def run(self):
        """
        Execute the full process to calculate surface rain rate and return with lat/lon.
        """
        self.calculate_reflectivity()
        self.rain_rate, self.lats, self.lons = self.convert_to_rain_rate()
        
        if self.ivarint == 1 or self.iliqskin == 1:
            self.rain_rate = self.rain_rate[0]
            
        return self.rain_rate, self.lats, self.lons
