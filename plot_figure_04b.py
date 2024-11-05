import numpy as np
from phd.variables.get_variables_manual import GetVariablesWRF
from phd.utils.timer import Timer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from phd.utils.label_and_scale import add_length_scale  # Ensure this is correctly implemented

class RayleighStabilityAnalyzer:
    def __init__(self,
                 file_path: str,
                 theta_front: float = 22.5,
                 distance: int = 5,
                 levels: list = [100, 500, 1000],
                 thresholds: list = [0, 0.1, 0.5, 1]):
        
        """
        Initializes the RayleighStabilityAnalyzer with specified parameters.
        
        Parameters:
        - file_path (str): Path to the WRF output NetCDF file.
        - theta_front (float): Angle to define the front direction. Default is 22.5 degrees.
        - distance (int): Number of grid points to consider on each side of the center for slicing. Default is 5.
        - levels (list): List of heights (in meters) at which to perform the analysis. Default is [100, 500, 1000].
        - thresholds (list): List of threshold values to apply in stability criteria. Default is [0, 0.1, 0.5, 1].
        """
        
        self.file_path = file_path
        self.theta_front = theta_front
        self.theta_rotate = -theta_front  # Clockwise rotation to align across-front with x-axis
        self.distance = distance
        self.slice_size = 2 * distance + 1
        self.levels = levels  # List of heights in meters
        self.thresholds = thresholds  # List of threshold values
        
        # Initialize a dictionary to store results
        # Structure: {height: {threshold: stability_codes}}
        self.results = {}
        
        # Initialize dictionaries to store absolute vorticity for each height
        self.absolute_vorticity = {}
        
        # Initialize latitude and longitude placeholders
        self.lat = None
        self.lon = None
        
        # Initialize wind components placeholders
        self.u = None
        self.v = None
        
        # Load static data that doesn't depend on height
        self._load_static_data()

    @Timer
    def _load_static_data(self):
        """
        Loads static data such as latitude and longitude from the WRF file.
        """
        variables = GetVariablesWRF(self.file_path)
        self.lat, self.lon = variables.get_lat_lons()

    def _load_dynamic_data(self, height: int):
        """
        Loads dynamic data (wind components and absolute vorticity) for a specific height.
        
        Parameters:
        - height (int): The height (in meters) at which to load the data.
        """
        variables = GetVariablesWRF(self.file_path)
        
        # Load wind components at the specified height
        self.u, self.v = variables.get_wind_components_at_heights(height=height)
        
        # Load absolute vorticity at the specified height
        self.absolute_vorticity[height] = variables.get_absolute_vorticity(height=height)

    def rotate_wind_components(self):
        """
        Rotates the wind components to align the across-front direction with the x-axis.
        """
        theta_rad = np.deg2rad(self.theta_rotate)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        self.u_rot = cos_theta * self.u.values + sin_theta * self.v.values
        self.v_rot = -sin_theta * self.u.values + cos_theta * self.v.values

        # For cross-front analysis, focus on u_rot
        self.wind_across_front = self.u_rot

    def get_wind_slices(self, level: int):
        """
        Extracts row-wise wind slices for the across-front wind component at a specific level.
        
        Parameters:
        - level (int): The height (in meters) at which to extract wind slices.
        
        Returns:
        - np.ndarray: 3D array of wind slices with shape (num_rows, num_cols, slice_size).
        """
        # Extract the specified level
        u_hgt = self.wind_across_front

        # Validate that u_hgt is a 2D array
        if u_hgt.ndim != 2:
            raise ValueError("Wind component at the specified level must be a 2D array.")

        num_rows, num_cols = u_hgt.shape
        slice_size = self.slice_size

        # Pad u_hgt with NaNs to handle edge slices
        # For row-wise slices, pad columns
        u_padded = np.pad(
            u_hgt,
            pad_width=((0, 0), (self.distance, self.distance)),
            mode='constant',
            constant_values=np.nan
        )

        # Initialize empty list to store slices
        u_slices_list = []

        # Extract row-wise slices
        for i in range(num_rows):
            for j in range(num_cols):
                slice_start = j
                slice_end = j + slice_size
                u_slice = u_padded[i, slice_start: slice_end]
                u_slices_list.append(u_slice)

        # Convert list to NumPy array and reshape
        slices_across_front = np.array(u_slices_list).reshape(num_rows, num_cols, slice_size)
        return slices_across_front

    def compute_double_gradient(self, slices_across_front: np.ndarray):
        """
        Computes the second gradient (double gradient) of the wind slices.
        
        Parameters:
        - slices_across_front (np.ndarray): 3D array of wind slices.
        
        Returns:
        - np.ndarray: 3D array of double gradients.
        """
        # Compute the first gradient
        first_gradient = np.gradient(slices_across_front, axis=2)

        # Compute the second gradient based on the first gradient
        second_gradient = np.gradient(first_gradient, axis=2)

        # Initialize the double_gradient array with the second gradient
        double_gradient = second_gradient.copy()

        # Create a mask where any NaN exists in the input slices
        nan_mask = np.isnan(slices_across_front).any(axis=2)  # Shape: (num_rows, num_cols)

        # Set entire slices to NaN where nan_mask is True
        double_gradient[nan_mask, :] = np.nan

        return double_gradient

    def apply_rayleigh_criterion(self, double_gradient: np.ndarray, threshold: float):
        """
        Applies Rayleigh's Stability Criterion based on the double gradients with a threshold.
        
        Parameters:
        - double_gradient (np.ndarray): The computed double gradient array.
        - threshold (float): The threshold value for applying the criterion.
        
        Returns:
        - np.ndarray: Boolean array where Rayleigh's criterion is satisfied.
        """
        centre = self.slice_size // 2
        idx_left, idx_right = centre - 1, centre + 1

        # Extract the values at the adjacent indices
        left_values = double_gradient[:, :, idx_left]
        right_values = double_gradient[:, :, idx_right]

        # Rayleigh's criterion: sign change with magnitude above threshold
        condition_sign_change = (left_values * right_values) < 0
        condition_magnitude = ((left_values < -threshold) & (right_values > threshold)) | \
                              ((left_values > threshold) & (right_values < -threshold))

        # Combine both conditions
        rayleigh_across_front = condition_sign_change & condition_magnitude

        return rayleigh_across_front

    def apply_fjortoft_criterion(self, double_gradient: np.ndarray, slices_across_front: np.ndarray, threshold: float):
        """
        Applies Fjortoft's Stability Criterion based on the double gradients with a threshold.
        
        Parameters:
        - double_gradient (np.ndarray): The computed double gradient array.
        - slices_across_front (np.ndarray): The wind slices across front.
        - threshold (float): The threshold value for applying the criterion.
        
        Returns:
        - np.ndarray: Boolean array where Fjortoft's criterion is satisfied.
        """        
        centre = self.slice_size // 2
        idx_left, idx_right = centre - 1, centre + 1

        # Extract wind speed at the inflection point (central index)
        V_inflection = slices_across_front[:, :, centre]

        # Compute (V(y) - V_inflection) for each point in the slice
        V_diff = slices_across_front - V_inflection[:, :, np.newaxis]

        # Compute Fjortoft's value: double_gradient * (V(y) - V_inflection)
        fjortoft = double_gradient * V_diff

        # Extract Fjortoft values at adjacent indices
        fjortoft_left = fjortoft[:, :, idx_left]
        fjortoft_right = fjortoft[:, :, idx_right]

        # Fjortoft's criterion: both fjortoft_left and fjortoft_right must be less than -threshold
        fjortoft_satisfied = (fjortoft_left < -threshold) & (fjortoft_right < -threshold)

        return fjortoft_satisfied

    def classify_stability(self, rayleigh: np.ndarray, fjortoft: np.ndarray):
        """
        Classifies stability based on Rayleigh and Fjortoft criteria.
        
        Parameters:
        - rayleigh (np.ndarray): Boolean array where Rayleigh's criterion is satisfied.
        - fjortoft (np.ndarray): Boolean array where Fjortoft's criterion is satisfied.
        
        Returns:
        - np.ndarray: Integer array with stability codes.
                    0 - Stable
                    1 - Rayleigh's Criterion Satisfied
                    2 - Fjortoft's Criterion Satisfied
        """
        # Initialize stability codes with 0 (Stable)
        stability_codes = np.zeros(rayleigh.shape, dtype=int)

        # Assign 1 where only Rayleigh's Criterion is satisfied
        rayleigh_only = rayleigh & ~fjortoft
        stability_codes[rayleigh_only] = 1

        # Assign 2 where Fjortoft's Criterion is satisfied (includes cases where both are satisfied)
        stability_codes[fjortoft] = 2

        return stability_codes

    def analyze(self):
        """
        Executes the full stability analysis workflow for all combinations of heights and thresholds.
        """
        for height in self.levels:
            print(f"Analyzing height: {height}m")
            self._load_dynamic_data(height)
            self.rotate_wind_components()
            slices_across_front = self.get_wind_slices(level=height)
            double_gradient = self.compute_double_gradient(slices_across_front)
            
            self.results[height] = {}
            
            for threshold in self.thresholds:
                print(f"  Applying threshold: {threshold}")
                rayleigh = self.apply_rayleigh_criterion(double_gradient, threshold)
                fjortoft = self.apply_fjortoft_criterion(double_gradient, slices_across_front, threshold)
                stability_codes = self.classify_stability(rayleigh, fjortoft)
                self.results[height][threshold] = stability_codes

    @Timer
    def plot_stability(self):
        """
        Plots the stability distribution across the geographical grid using a grid of subplots.
        Rows correspond to different heights, and columns correspond to different thresholds.
        Subplots are closer together by adjusting spacing parameters.
        """
        if not self.results:
            raise ValueError("Stability analysis not yet performed. Call the 'analyze()' method first.")
        
        num_rows = len(self.levels)
        num_cols = len(self.thresholds)
        
        # Create a figure with specified size and adjusted spacing
        fig, axes = plt.subplots(
            num_rows, num_cols, 
            figsize=(10, 15),  # Adjust figsize as needed
            subplot_kw={'projection': ccrs.Mercator()},
            gridspec_kw={'wspace': 0.0, 'hspace': 0.03}
        )
        
        # Define colormap and normalization
        cmap = ListedColormap(['white', 'lightblue', 'purple'])  # Adjust colors as desired
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Ensure axes is a 2D array even if num_rows or num_cols is 1
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif num_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        
        for row_idx, height in enumerate(self.levels):
            for col_idx, threshold in enumerate(self.thresholds):
                ax = axes[row_idx, col_idx]
                
                stability_codes = self.results[height][threshold]
                
                # Plot using pcolormesh on the axes with Mercator projection
                ax.pcolormesh(
                    self.lon, self.lat, stability_codes, 
                    cmap=cmap, norm=norm, 
                    transform=ccrs.PlateCarree(), 
                    zorder=0
                )
                
                # Plot absolute vorticity contours
                if height in self.absolute_vorticity:
                    ax.contour(
                        self.lon, self.lat, self.absolute_vorticity[height], 
                        levels=[3], colors='black', linewidths=1, 
                        transform=ccrs.PlateCarree(), zorder=1
                    )
                
                # Set plot extent to the range of your data (adjust as needed)
                ax.set_extent([-7.4, -5.8, 52.9, 54.9], crs=ccrs.PlateCarree())
                
                # Optionally, remove axis ticks for a cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Only add the length scale to the top right plot. Too messy otherwise.
                if row_idx == 0 and col_idx == 3:
                    add_length_scale(ax)
        
        plt.savefig('figures/figure_04b.png', dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    @Timer
    def run(self):
        """
        Runs the full analysis and plots the stability distribution.
        """
        self.analyze()
        self.plot_stability()

# Example Usage
if __name__ == "__main__":
    file_path = "/Volumes/Samsung_T5/phd_data/2011-11-29/wrfout_d03_2011-11-29_09-00-00.nc"
    heights = [100, 500, 1000]  # Heights in meters
    thresholds = [0, 0.1, 0.5, 1]  # Threshold values
    
    rsa = RayleighStabilityAnalyzer(file_path, levels=heights, thresholds=thresholds)
    rsa.run()
