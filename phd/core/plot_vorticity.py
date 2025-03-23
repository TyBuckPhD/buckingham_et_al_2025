import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

fname = 'wrfout_d04_2011-11-29_09-10-00.nc'
fpath = '/Volumes/PhDrive/d04_runs/2011-11-29'
file = os.path.join(fpath, fname)

def destagger(data, axis):
    """Destagger a WRF variable along a specified axis."""
    return 0.5 * (data.take(indices=range(data.shape[axis] - 1), axis=axis) +
                  data.take(indices=range(1, data.shape[axis]), axis=axis))

def calculate_absolute_vorticity(file):
    """
    Calculate full absolute vorticity and associated terms for a single WRF file.

    Parameters:
    - wrf_file: Path to the WRF file.

    Returns:
    - absolute_vorticity: xarray.DataArray of absolute vorticity.
    - stretching_term: xarray.DataArray of the stretching term.
    - tilting_term: xarray.DataArray of the tilting term.
    """
    # Load the dataset
    ds = xr.open_dataset(file)

    # Extract and destagger variables
    u = destagger(ds['U'].isel(Time=0).values, axis=2)  # Destagger along the second-to-last dimension
    v = destagger(ds['V'].isel(Time=0).values, axis=1)  # Destagger along the last dimension
    w = destagger(ds['W'].isel(Time=0).values, axis=0)   # Destagger along the first dimension

    # Extract latitudes, grid spacing, and geopotential heights
    lats = ds['XLAT'].isel(Time=0).values
    dx = ds.attrs['DX']
    dy = ds.attrs['DY']
    ph = destagger(ds['PH'].isel(Time=0).values, axis=0)
    phb = destagger(ds['PHB'].isel(Time=0).values, axis=0)
    
    height_agl = (ph + phb) / 9.81
    dz = np.gradient(height_agl, axis=0)  # Compute dz along the vertical axis

    # Calculate gradients
    dv_dx = np.gradient(v, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    dw_dx = np.gradient(w, dx, axis=2)
    dw_dy = np.gradient(w, dy, axis=1)
    du_dz = np.gradient(u, axis=0) / dz
    dv_dz = np.gradient(v, axis=0) / dz
    dw_dz = np.gradient(w, axis=0) / dz

    # Calculate relative vorticity (ζ = dv/dx - du/dy)
    zeta = dv_dx - du_dy

    # Calculate Coriolis parameter (f = 2 * Omega * sin(latitude))
    omega = 7.2921e-5  # Earth's angular velocity (rad/s)
    f = 2 * omega * np.sin(np.radians(lats))

    # Expand Coriolis parameter to match 3D dimensions
    f = np.expand_dims(f, axis=0)  # Add vertical dimension
    f = np.repeat(f, zeta.shape[0], axis=0)  # Repeat for all vertical levels

    # Calculate absolute vorticity
    absolute_vorticity = zeta + f

    # Calculate stretching and tilting terms
    stretching_term = absolute_vorticity * dw_dz
    tilting_term = -(dw_dx * dv_dz - dw_dy * du_dz)
    
    return absolute_vorticity, stretching_term, tilting_term

absolute_vorticity, stretching_term, tilting_term = calculate_absolute_vorticity(file)
plt.pcolor(absolute_vorticity[5])