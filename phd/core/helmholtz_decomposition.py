import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from scipy.ndimage import rotate
from pynhhd import nHHD
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.timer import Timer

class HelmholtzDecomposition:
    def __init__(self, config_path, event_type, num_domains=10, plot=True):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        self.event_type = event_type
        self.num_domains = num_domains
        
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
        
        self.selected_files = self.get_files()
        self.num_files = len(self.selected_files)
        if self.num_files == 0:
            raise ValueError("No files found in the specified time range.")
        
        self.generate_parameters()
        self.f = 1e-4  # Coriolis parameter
        
        self.plot = plot

    def get_files(self):
        start_time_dt = datetime.strptime(self.start_time, "%H:%M:%S")
        end_time_dt = datetime.strptime(self.end_time, "%H:%M:%S")
        
        files = sorted(glob.glob(f"{self.directory}/wrfout_d03_*.nc"))
        pattern = r"wrfout_d03_\d{4}-\d{2}-\d{2}_(\d{2}-\d{2}-\d{2})\.nc"
        
        selected_files = []
        for file in files:
            match = re.search(pattern, file)
            if match:
                file_time_str = match.group(1).replace("-", ":")
                try:
                    file_time_dt = datetime.strptime(file_time_str, "%H:%M:%S")
                    if start_time_dt <= file_time_dt <= end_time_dt:
                        selected_files.append(file)
                except ValueError:
                    print(f"Warning: Time format mismatch in file {file}. Skipping.")
        return selected_files

    def generate_parameters(self):
        num_files = self.num_files
        
        self.angles = np.linspace(self.angle_start, self.angle_end, num_files).tolist()
        
        self.ymins = np.linspace(self.ymin_start, self.ymin_end, num_files).astype(int).tolist()
        self.ymaxs = np.linspace(self.ymax_start, self.ymax_end, num_files).astype(int).tolist()
        self.xmins = np.linspace(self.xmin_start, self.xmin_end, num_files).astype(int).tolist()
        self.xmaxs = np.linspace(self.xmax_start, self.xmax_end, num_files).astype(int).tolist()

    def get_variables(self, file_path, height=500):
        variables = GetVariablesWRF(file_path)
        lats, lons = variables.get_lat_lons()
        u, v = variables.get_wind_components_at_heights(height=height)
    
        R = 6371000
        lats_rad = np.deg2rad(lats.values)
        lons_rad = np.deg2rad(lons.values)
        mean_lat_rad = np.mean(lats_rad)
        mean_lon_rad = np.mean(lons_rad)
    
        x = R * (lons_rad - mean_lon_rad) * np.cos(mean_lat_rad)
        y = R * (lats_rad - mean_lat_rad)
    
        return x, y, u.values, v.values

    def calculate_vorticity(self, u, v, dx, dy):
        dv_dx = np.gradient(v, axis=1) / dx
        du_dy = np.gradient(u, axis=0) / dy
        vorticity = dv_dx - du_dy
        return vorticity

    def perform_decomposition(self, windfield, dx, dy):
        spacings = (dy, dx)        
        nhhd = nHHD(grid=windfield.shape[:2], spacings=spacings)
        nhhd.decompose(windfield)
        return nhhd.r, nhhd.d, nhhd.h

    def rotate_field(self, field, angle):
        return rotate(field, angle=angle, reshape=True, order=1, mode='nearest')

    def calculate_strain_threshold(self, vorticity_front_mean, vorticity_ambient_mean, f, mu):
        coriolis_component = f / 4
        vorticity_component = ((vorticity_front_mean - vorticity_ambient_mean) / (vorticity_front_mean + vorticity_ambient_mean))
        wavelength_component = np.exp(-2 * mu)
        return coriolis_component * vorticity_component * wavelength_component

    def calculate_frontal_strain(self, v_harmonic, u_harmonic, dy, dx, event_type):
        if event_type == 1:
            dv_dy = np.gradient(v_harmonic, dy, axis=0)
            stretching_deformation = dv_dy.mean()
        if event_type == 2:
            du_dx = np.gradient(u_harmonic, dx, axis=1)
            stretching_deformation = du_dx.mean()           
        return stretching_deformation

    def plot_frontal_domain(self, x_rotated, y_rotated, u_rotated, v_rotated, idx, xmin, xmax, ymin, ymax):
        magnitude = np.sqrt(u_rotated**2 + v_rotated**2)

        # Determine the spatial extent of the entire rotated grid
        x_min_plot = x_rotated.min()
        x_max_plot = x_rotated.max()
        y_min_plot = y_rotated.min()
        y_max_plot = y_rotated.max()

        # Get the number of grid points
        ny, nx = u_rotated.shape

        plt.figure(figsize=(6, 6))
        plt.imshow(
            magnitude,
            extent=(x_min_plot, x_max_plot, y_min_plot, y_max_plot),
            origin='lower',
            cmap='PuOr'
        )

        # Calculate grid spacing
        dx = (x_max_plot - x_min_plot) / nx
        dy = (y_max_plot - y_min_plot) / ny

        # Calculate rectangle position in data coordinates
        rect_x_min = x_min_plot + xmin * dx
        rect_y_min = y_min_plot + ymin * dy
        width = (xmax - xmin) * dx
        height = (ymax - ymin) * dy

        # Create rectangle
        rect = patches.Rectangle(
            (rect_x_min, rect_y_min),
            width,
            height,
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        plt.gca().add_patch(rect)

        plt.xticks([])
        plt.yticks([])
        plt.show()

    @Timer
    def run(self):
        addition = 2
        expansions = [i * addition for i in range(self.num_domains)]

        # Initialize lists to store results for each domain
        strain_threshold_lists = [[] for _ in range(self.num_domains)]
        frontal_strain_lists = [[] for _ in range(self.num_domains)]
        u_harmonic_lists = [[] for _ in range(self.num_domains)]
        v_harmonic_lists = [[] for _ in range(self.num_domains)]

        for idx, file_path in enumerate(self.selected_files):
            print(f"\nProcessing file {idx+1}/{self.num_files}: {file_path}")

            # Get data
            x, y, u, v = self.get_variables(file_path, height=500)
            angle = self.angles[idx]
            
            # Rotate the full field
            u_rotated = self.rotate_field(u, angle)
            v_rotated = self.rotate_field(v, angle)
            x_rotated = self.rotate_field(x, angle)
            y_rotated = self.rotate_field(y, angle)

            ny, nx = u_rotated.shape

            # Original domain boundaries for this time step
            xmin_t = self.xmins[idx]
            xmax_t = self.xmaxs[idx]
            ymin_t = self.ymins[idx]
            ymax_t = self.ymaxs[idx]

            for i in range(self.num_domains):
                expansion = expansions[i]

                # Adjust domain boundaries
                xmin_i = max(0, xmin_t - expansion)
                xmax_i = min(nx, xmax_t + expansion)
                ymin_i = max(0, ymin_t - expansion)
                ymax_i = min(ny, ymax_t + expansion)

                # Ensure indices are integers
                xmin_i = int(xmin_i)
                xmax_i = int(xmax_i)
                ymin_i = int(ymin_i)
                ymax_i = int(ymax_i)

                # Crop the domain
                u_frontal = u_rotated[ymin_i:ymax_i, xmin_i:xmax_i]
                v_frontal = v_rotated[ymin_i:ymax_i, xmin_i:xmax_i]
                x_frontal = x_rotated[ymin_i:ymax_i, xmin_i:xmax_i]
                y_frontal = y_rotated[ymin_i:ymax_i, xmin_i:xmax_i]

                # Calculate approximate horizontal grid resolution for the frontal domain
                dx = np.mean(np.diff(x_frontal, axis=1))
                dy = np.mean(np.diff(y_frontal, axis=0))

                # Perform Helmholtz Decomposition on the frontal domain
                windfield = np.stack((u_frontal, v_frontal), axis=-1)
                _, _, harmonic = self.perform_decomposition(windfield, dx, dy)

                # Separate wind components
                u_harmonic = harmonic[..., 0]
                v_harmonic = harmonic[..., 1]

                # Append harmonic components to lists
                u_harmonic_lists[i].append(u_harmonic)
                v_harmonic_lists[i].append(v_harmonic)

                # Calculate vorticity
                vorticity_frontal = self.calculate_vorticity(u_frontal, v_frontal, dx, dy)
                vorticity_front_mean = np.mean(vorticity_frontal[vorticity_frontal > 0.003])
                vorticity_ambient_mean = np.mean(vorticity_frontal[vorticity_frontal < 0.000])

                # Calculate mu for the strain threshold equation
                vorticity_width = 2.5
                perturbation_wavelength = 17.5
                mu = vorticity_width / perturbation_wavelength
                # mu = 0

                # Calculate theoretical strain threshold
                strain_threshold = self.calculate_strain_threshold(
                    vorticity_front_mean, vorticity_ambient_mean, self.f, mu
                )
                strain_threshold_lists[i].append(strain_threshold)

                print('theoretical strain:', strain_threshold)

                # Calculate harmonic strain
                stretching_deformation = self.calculate_frontal_strain(v_harmonic, u_harmonic, dy, dx, self.event_type)
                frontal_strain_lists[i].append(stretching_deformation)         

                print('frontal strain:', stretching_deformation)

                # Optionally, plot domain location for visual check
                if self.plot:
                    self.plot_frontal_domain(
                        x_rotated, y_rotated, u_rotated, v_rotated, idx, xmin_i, xmax_i, ymin_i, ymax_i
                    )

        return strain_threshold_lists, frontal_strain_lists, self.selected_files
    
