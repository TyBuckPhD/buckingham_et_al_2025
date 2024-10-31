import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import cartopy.crs as ccrs
from datetime import datetime
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.variables.colorbar_vorticity import ColorbarVorticity
from phd.utils.timer import Timer

@Timer
def plot_vorticity_evolution(directory, start_time, end_time, vorticity_threshold=1e-3):
    """
    Plots the evolution of vorticity over a specified time range.

    Parameters:
        directory (str): The directory containing WRF output files.
        start_time (str): Start time in "HH:MM:SS" format.
        end_time (str): End time in "HH:MM:SS" format.
        vorticity_threshold (float): Threshold for masking vorticity values.
    """
    # Convert start and end times to datetime objects
    start_time_dt = datetime.strptime(start_time, "%H:%M:%S")
    end_time_dt = datetime.strptime(end_time, "%H:%M:%S")
    
    # Load all WRF files in the directory
    files = sorted(glob.glob(f"{directory}/wrfout_d03_*.nc"))
    pattern = r"wrfout_d03_\d{4}-\d{2}-\d{2}_(\d{2}-\d{2}-\d{2})\.nc"

    # Select files based on the time range
    selected_files = []
    for file in files:
        match = re.search(pattern, file)
        if match:
            # Parse the time from the filename
            file_time_str = match.group(1).replace("-", ":")
            file_time_dt = datetime.strptime(file_time_str, "%H:%M:%S")
            if start_time_dt <= file_time_dt <= end_time_dt:
                selected_files.append(file)

    # Check if no files are selected
    if not selected_files:
        raise ValueError(f"No files found in the specified time range {start_time} to {end_time}.")

    # Initialize lists to store data and metadata
    vorticity_data = []
    times = []
    lats = None
    lons = None

    # Calculate absolute vorticity for each file and apply the mask
    for file in selected_files:
        vorticity_calculator = GetVariablesWRF(file)
        vorticity = vorticity_calculator.get_absolute_vorticity()
        
        # Mask values below the threshold
        masked_vorticity = np.where(np.abs(vorticity) > vorticity_threshold, vorticity, np.nan)
        vorticity_data.append(masked_vorticity)
        
        # Record latitude/longitude once and store time
        if lats is None or lons is None:
            lats, lons = vorticity_calculator.get_lat_lons()
        time_str = re.search(r"_(\d{2}-\d{2}-\d{2})\.nc", file).group(1)
        times.append(time_str.replace("-", ":"))
    
    # Stack vorticity data along the time axis
    vorticity_stacked = np.stack(vorticity_data, axis=0)  # Shape: (time, lat, lon)

    # Compute the maximum absolute vorticity over time
    vorticity_collapsed = np.nanmax(np.abs(vorticity_stacked), axis=0)
    vorticity_collapsed = np.nan_to_num(vorticity_collapsed, nan=0.0)
    
    # Initialize the ColorbarVorticity object with the desired vorticity type
    vorticity_cmap = ColorbarVorticity(vorticity_type='Type 1')  # Change to 'Type 2' if needed
    
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Mercator()})
    ax.set_extent([-7, -4.4, 52.8, 54.0], crs=ccrs.PlateCarree())
    
    # Plot the vorticity data using levels and colormap from vorticity_cmap
    cf = ax.contourf(
        lons,
        lats,
        vorticity_collapsed,
        levels=vorticity_cmap.levels,
        cmap=vorticity_cmap.cmap,
        norm=vorticity_cmap.norm,
        extend='both',
        transform=ccrs.PlateCarree()
    )
    
    # Add the customized colorbar using vorticity_cmap
    vorticity_cmap.add_colorbar(cf, ax, label="Maximum Absolute Vorticity (s⁻¹)")
    
    # Add additional map features
    ax.coastlines()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Combined Vorticity Map (> {vorticity_threshold} s⁻¹) from {start_time} to {end_time}")
    
    plt.tight_layout()
    fig_filename = 'figures/figure_03.png'
    plt.savefig(fig_filename, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.show()

# Usage example
if __name__ == "__main__":
    directory = "/Volumes/Samsung_T5/phd_data/2011-11-29"
    start_time = "08:00:00"
    end_time = "12:00:00"
    
    plot_vorticity_evolution(directory, start_time, end_time)