import cartopy.crs as ccrs
import numpy as np
import json
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches

def add_length_scale(ax):
    """
    Adds a dynamic scale bar with alternating black and white blocks to the given Axes object.
    Parameters:
    - ax: The Axes object to draw on.
    """
    # Load the scale configuration from config.json
    with open('inputs/length_scale_config.json', 'r') as f:
        scale_config = json.load(f)

    # Extract extent of the map
    extent = ax.get_extent(crs=ccrs.PlateCarree())

    length_km = scale_config.get("length_km", 100)
    units = scale_config.get("units", "km")
    location = tuple(scale_config.get("location", [0.95, 0.05]))
    nblocks = scale_config.get("nblocks", 5)
    color1 = scale_config.get("color1", "black")
    color2 = scale_config.get("color2", "white")
    fontsize = scale_config.get("fontsize", 10)

    # Calculate the length per block without rounding
    length_per_block_km = length_km / nblocks

    # Calculate length in degrees (approximate)
    central_lat = (extent[2] + extent[3]) / 2
    km_per_deg = 111.32 * np.cos(np.radians(central_lat))
    length_per_block_deg = length_per_block_km / km_per_deg
    total_length_deg = length_per_block_deg * nblocks

    # Positioning
    total_lon = extent[1] - extent[0]
    total_lat = extent[3] - extent[2]

    # Starting position in data coordinates
    loc_x = extent[1] - (1 - location[0]) * total_lon - total_length_deg
    loc_y = extent[2] + location[1] * total_lat

    # Draw the scale bar blocks
    for i in range(nblocks):
        block_lon_start = loc_x + i * length_per_block_deg
        block = patches.Rectangle(
            (block_lon_start, loc_y),
            length_per_block_deg,
            0.01 * total_lat,  # Height of the scale bar
            facecolor=color1 if i % 2 == 0 else color2,
            edgecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=999
        )
        ax.add_patch(block)

    # Add labels at each division
    for i in range(nblocks + 1):
        label_km = i * length_per_block_km
        label_lon = loc_x + i * length_per_block_deg
        ax.text(
            label_lon,
            loc_y - 0.02 * total_lat,
            f"{int(label_km)}",
            ha='center',
            va='top',
            fontsize=fontsize,
            weight='bold',
            color='black',
            transform=ccrs.PlateCarree(),
            zorder=999
        )

    # Add the 'km' label at the end of the scale bar
    label_lon_km = loc_x + total_length_deg + 0.01 * total_lon  # Slightly to the right of the scale bar
    ax.text(
        label_lon_km,
        loc_y + 0.005 * total_lat,  # Vertically centered on the scale bar
        units,
        ha='left',
        va='center',
        fontsize=fontsize,
        weight='bold',
        color='black',
        transform=ccrs.PlateCarree(),
        zorder=999
    )

    # Optionally, draw a thin border around the scale bar
    border = patches.Rectangle(
        (loc_x, loc_y),
        total_length_deg,
        0.01 * total_lat,  # Same height as the blocks
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=999
    )
    ax.add_patch(border)

def add_corner_label(ax, text_lines, label_type):
    """
    Adds a corner label to the plot.

    Parameters:
    - ax: The Axes object to draw on.
    - text_lines: List of text lines to display in the label.
    - label_type: String indicating the type ("Type 1" or "Type 2") for color schemes.
    """
    # Common properties
    fontsize = 13
    text_color = "white"
    border_width = 3.0
    borderpad = 0.1
    pad = 0.4

    # Type-specific colors
    if label_type == "Type 1":
        background_color = "lightskyblue"
        border_color = "dodgerblue"
    elif label_type == "Type 2":
        background_color = "tomato"
        border_color = "darkred"
    else:
        raise ValueError("Invalid label_type. Expected 'Type 1' or 'Type 2'.")

    # Extract extent of the map in PlateCarree coordinates
    extent = ax.get_extent(crs=ccrs.PlateCarree())

    # Extract the top-left corner in lon/lat (PlateCarree)
    top_left_lon, top_left_lat = extent[0], extent[3]

    # Transform the lon/lat point to the map projection's data coordinates
    x, y = ax.projection.transform_point(top_left_lon, top_left_lat, ccrs.PlateCarree())

    # Transform the data coordinate to display coordinate
    display_coords = ax.transData.transform((x, y))

    # Transform display coordinate to axes coordinate
    axes_coords = ax.transAxes.inverted().transform(display_coords)

    # Combine the text lines into a single string with line breaks
    text_str = '\n'.join(text_lines)

    # Create the AnchoredText object with 'loc' as a positional argument
    at = AnchoredText(
        text_str,
        'upper left',  # Required positional argument 'loc'
        prop=dict(
            fontsize=fontsize,
            color=text_color,
            weight='bold'
        ),
        frameon=True,
        borderpad=borderpad,  # Padding between the text and the box
        pad=pad,              # Padding between the box and the anchor point
    )

    # Set box alignment to align the top-left corner of the bounding box with the anchor point
    at.box_alignment = (0, 1)  # (horizontal alignment, vertical alignment)

    # Customize the box style and colors
    at.patch.set_facecolor(background_color)
    at.patch.set_edgecolor(border_color)
    at.patch.set_linewidth(border_width)

    # Manually set the position of the AnchoredText using bbox_to_anchor
    at.set_bbox_to_anchor(axes_coords, transform=ax.transAxes)

    # Add the AnchoredText to the axes
    ax.add_artist(at)
