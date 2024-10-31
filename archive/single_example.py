import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from phd.utils.label_and_scale import add_corner_label, add_length_scale

def plot_single_map():
    # Create a figure and an Axes object with the PlateCarree projection
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set the map extent (lon_min, lon_max, lat_min, lat_max)
    extent = [-10, 10, 45, 65]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Define the label text and type
    text_lines = ["(a) Modelled:", "29 Nov 2011"]  # Single line label
    label_type = "Type 1"  # Choose between "Type 1" and "Type 2"
    
    # Add the corner label
    add_corner_label(ax, text_lines, label_type)
    
    # Add the length scale
    add_length_scale(ax)
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_single_map()