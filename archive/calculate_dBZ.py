from netCDF4 import Dataset
from wrf import getvar, to_np, latlon_coords
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature

# Filepath to the WRF data
filepath = '/Volumes/Samsung_T5/phd_data/wrfout_d03_2011-11-29_14-00-00.nc'

# Load the WRF data file
wrf_data = Dataset(filepath)

# Extract dBZ for the first timestep
dbz = getvar(wrf_data, "dbz", timeidx=0)  # timeidx=0 for the first timestep

# Get the latitude and longitude points
lats, lons = latlon_coords(dbz)

# Convert dBZ to a NumPy array and calculate precipitation rate
dbz_np = to_np(dbz)[0]
precip_rate = ((10 ** (dbz_np / 10) / 200) ** 0.625)  # Convert dBZ to mm/hr

# Plot the precipitation rate
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=crs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), precip_rate, levels=[0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 32, 64], cmap="viridis", extend="max")
plt.colorbar(ax=ax, label="Precipitation Rate (mm/hr)")

# Add coastlines, borders, and other map features
ax.coastlines('50m', linewidth=0.8)

# Set the title
plt.title("Precipitation Rate (mm/hr) at First Timestep")
plt.show()