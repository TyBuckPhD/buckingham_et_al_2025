import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import rotate
from pynhhd import nHHD
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.timer import Timer

class HelmholtzDecomposition:
    def __init__(self, config_path):
        # Load configuration from JSON file
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # Extract parameters
        self.directory = config['directory']
        self.start_time = config['start_time']
        self.end_time = config['end_time']
        
        self.angle_start = config['angle_start']
        self.angle_end = config['angle_end']
        
        self.ymin_start = config['ymin_start']
        self.ymax_start = config['ymax_start']
        self.xmin_start = config['xmin_start']
        self.xmax_start = config['xmax_start']
        
        self.ymin_end = config['ymin_end']
        self.ymax_end = config['ymax_end']
        self.xmin_end = config['xmin_end']
        self.xmax_end = config['xmax_end']
        
        # Get the list of selected files
        self.selected_files = self.get_files()
        self.num_files = len(self.selected_files)
        
        # Quiver cropping
        self.crop = 10
        
        if self.num_files == 0:
            raise ValueError("No files found in the specified time range.")
        
        # Generate arrays for angles and bounding box coordinates
        self.generate_parameters()
        self.plot = True  # Set to False to disable plotting

    def get_files(self):
        """
        Retrieves WRF output files within the specified time range.
        """
        # Convert start and end times to datetime objects
        start_time_dt = datetime.strptime(self.start_time, "%H:%M:%S")
        end_time_dt = datetime.strptime(self.end_time, "%H:%M:%S")
        
        # Load all WRF files in the directory
        files = sorted(glob.glob(f"{self.directory}/wrfout_d03_*.nc"))
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
        return selected_files

    def generate_parameters(self):
        """
        Generates arrays for angles and bounding box coordinates using np.linspace().
        """
        num_files = self.num_files
        
        # Generate angles
        self.angles = np.linspace(self.angle_start, self.angle_end, num_files).tolist()
        
        # Generate bounding box coordinates
        self.ymins = np.linspace(self.ymin_start, self.ymin_end, num_files).astype(int).tolist()
        self.ymaxs = np.linspace(self.ymax_start, self.ymax_end, num_files).astype(int).tolist()
        self.xmins = np.linspace(self.xmin_start, self.xmin_end, num_files).astype(int).tolist()
        self.xmaxs = np.linspace(self.xmax_start, self.xmax_end, num_files).astype(int).tolist()

    def get_variables(self, file_path, height=500):
        """
        Extracts x, y coordinates and wind components at a specified height from a WRF output file.
        """
        variables = GetVariablesWRF(file_path)
        lats, lons = variables.get_lat_lons()
        u, v = variables.get_wind_components_at_heights(height=height)
    
        # Convert lat/lon to approximate x/y coordinates in meters
        R = 6371000  # Earth's radius in meters
        lats_rad = np.deg2rad(lats.values)
        lons_rad = np.deg2rad(lons.values)
        mean_lat_rad = np.mean(lats_rad)
        mean_lon_rad = np.mean(lons_rad)
    
        # Approximate x and y (meters)
        x = R * (lons_rad - mean_lon_rad) * np.cos(mean_lat_rad)
        y = R * (lats_rad - mean_lat_rad)  # Correct calculation
    
        return x, y, u.values, v.values

    def plot_wind_field(self, component, xmin, xmax, ymin, ymax, title='Wind Field', is_vector=False):
        """
        Plots the specified wind component with cropping rectangle.

        Parameters:
        - component: np.ndarray
            The wind component to plot. Can be scalar or vector.
        - xmin, xmax, ymin, ymax: int
            Coordinates for the cropping rectangle.
        - title: str
            Title of the plot.
        - is_vector: bool
            Flag indicating if the component is a vector field.
        """
        plt.figure(figsize=(8, 6))
        
        if is_vector:
            # Compute the magnitude of the vector field
            magnitude = np.sqrt(component[..., 0]**2 + component[..., 1]**2)
            plot_data = magnitude
        else:
            plot_data = component
        
        # Plot the scalar field
        plt.pcolor(plot_data, shading='auto', cmap='viridis')
        
        # Define the cropping rectangle
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys, color='black', linewidth=2)
        
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_components(self, x_cropped, y_cropped, non_div, non_rot, harmonic, idx):
        """
        Plots the Helmholtz components as quiver plots.
        """
        # Define cropping step for quiver plots
        crop_step = self.crop

        # Rotational component (non_div)
        plt.figure(figsize=(6, 6))
        plt.quiver(x_cropped[::crop_step, ::crop_step], y_cropped[::crop_step, ::crop_step],
                   non_div[::crop_step, ::crop_step, 0], non_div[::crop_step, ::crop_step, 1],
                   scale=200, pivot='mid')
        plt.title(f'Non-Divergent Component (File {idx+1})')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # Divergent component (non_rot)
        plt.figure(figsize=(6, 6))
        plt.quiver(x_cropped[::crop_step, ::crop_step], y_cropped[::crop_step, ::crop_step],
                   non_rot[::crop_step, ::crop_step, 0], non_rot[::crop_step, ::crop_step, 1],
                   scale=200, pivot='mid')
        plt.title(f'Non-Rotational Component (File {idx+1})')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # Harmonic component
        plt.figure(figsize=(6, 6))
        plt.quiver(x_cropped[::crop_step, ::crop_step], y_cropped[::crop_step, ::crop_step],
                   harmonic[::crop_step, ::crop_step, 0], harmonic[::crop_step, ::crop_step, 1],
                   scale=200, pivot='mid')
        plt.title(f'Harmonic Component (File {idx+1})')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    @Timer
    def run(self):
        non_div_list = []
        non_rot_list = []
        harmonic_list = []
        
        for idx, file_path in enumerate(self.selected_files):
            print(f"Processing file {idx+1}/{self.num_files}: {file_path}")

            # Get variables from the WRF output file
            x, y, u, v = self.get_variables(file_path, height=500)

            angle = self.angles[idx]

            # Grid dimensions
            ny, nx = u.shape

            # Calculate grid spacings from x and y
            # Assuming x and y are 1D arrays; adjust if they are 2D
            if x.ndim == 1 and y.ndim == 1:
                dx = np.mean(np.diff(x))
                dy = np.mean(np.diff(y))
            else:
                # If x and y are 2D, calculate spacings accordingly
                dx = np.mean(np.diff(x, axis=1))
                dy = np.mean(np.diff(y, axis=0))
            spacings = (dy, dx)  # Note the order (y, x)

            # Structure the wind field
            windfield = np.stack((u, v), axis=-1)  # Shape: (num_y, num_x, 2)

            # Handle NaNs in windfield
            if np.isnan(windfield).any():
                print("Warning: NaN values found in windfield. Replacing with zeros.")
                windfield = np.nan_to_num(windfield)

            # Initialize nHHD class
            nhhd = nHHD(grid=(ny, nx), spacings=spacings)

            # Perform the decomposition
            nhhd.decompose(windfield)

            # Extract the components
            non_div = nhhd.r  # Non-divergent component (vector field)
            non_rot = nhhd.d  # Non-rotational component (vector field)
            harmonic = nhhd.h  # Harmonic component (vector field)

            # Rotate the components
            non_div_rot = rotate(non_div, angle=angle, reshape=True, order=1)
            non_rot_rot = rotate(non_rot, angle=angle, reshape=True, order=1)
            harmonic_rot = rotate(harmonic, angle=angle, reshape=True, order=1)
            
            # Rotate the coordinates
            x_rot = rotate(x, angle=angle, reshape=True, order=1)
            y_rot = rotate(y, angle=angle, reshape=True, order=1)

            # Determine the indices for cropping
            ymin = self.ymins[idx]
            ymax = self.ymaxs[idx]
            xmin = self.xmins[idx]
            xmax = self.xmaxs[idx]

            # Ensure indices are within the bounds of the rotated arrays
            ny_rot, nx_rot = non_div_rot.shape[:2]
            ymin = max(0, ymin)
            ymax = min(ny_rot, ymax)
            xmin = max(0, xmin)
            xmax = min(nx_rot, xmax)

            if ymax <= ymin or xmax <= xmin:
                print("Invalid indices for cropping.")
                continue

            # **New Addition: Plot the full rotated irrotational component with cropping rectangle**
            if self.plot:
                self.plot_wind_field(
                    component=non_div_rot,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    title=f'Rotated Non-Divergent Component with Crop Rectangle (File {idx+1})',
                    is_vector=True  # Indicate that this is a vector field
                )

            # Crop the rotated components
            non_div_cropped = non_div_rot[ymin:ymax, xmin:xmax]
            non_rot_cropped = non_rot_rot[ymin:ymax, xmin:xmax]
            harmonic_cropped = harmonic_rot[ymin:ymax, xmin:xmax]
            x_cropped = x_rot[ymin:ymax, xmin:xmax]
            y_cropped = y_rot[ymin:ymax, xmin:xmax]
            
            non_div_list.append(non_div_cropped)
            non_rot_list.append(non_rot_cropped)
            harmonic_list.append(harmonic_cropped)    
            
            # Plot the components
            if self.plot:
                self.plot_components(x_cropped, y_cropped, non_div_cropped, non_rot_cropped, harmonic_cropped, idx)
        
        return non_div_list, non_rot_list, harmonic_list
