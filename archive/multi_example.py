import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import json
from phd.utils.label_and_scale import add_corner_label, add_length_scale

def plot_multiple_maps():
    # Load configurations from JSON file
    with open('inputs/plot_config.json', 'r') as f:
        config = json.load(f)
    
    num_subplots = config["num_subplots"]
    nrows = config["nrows"]
    ncols = config["ncols"]
    labels = config["labels"]
    dates = config["dates"]
    label_types = config["label_types"]
    extents = config["extents"]
    
    # Create the figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Ensure the number of configurations matches the number of subplots
    assert len(labels) == num_subplots, "Number of labels must match num_subplots"
    assert len(dates) == num_subplots, "Number of dates must match num_subplots"
    assert len(label_types) == num_subplots, "Number of label_types must match num_subplots"
    assert len(extents) == num_subplots, "Number of extents must match num_subplots"
    
    for i in range(num_subplots):
        ax = axes[i]
        # Set the map extent
        ax.set_extent(extents[i], crs=ccrs.PlateCarree())
        # Add map features
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        # Prepare text_lines for the corner label
        text_lines = [f"{labels[i]} Modelled:", dates[i]]
        # Add the corner label
        add_corner_label(
            ax=ax,
            text_lines=text_lines,
            label_type=label_types[i]
        )
        # Add the length scale
        add_length_scale(ax)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_multiple_maps()
