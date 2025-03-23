import numpy as np
from buckingham_et_al_2025.core.trajectory_analysis import ParticleTrajectoryAnalysis

if __name__ == "__main__":
    """
    Script for analyzing particle trajectories using ParticleTrajectoryAnalysis for 
    the vortex environment of a Type 2 event.

    This script performs the following steps:
      1. Loads configuration parameters from a JSON file to set up the analysis.
      2. Defines seed location ranges (x0_range, y0_range, z0_range), a vorticity threshold, and a buffer.
      3. Creates an instance of ParticleTrajectoryAnalysis with the specified parameters.
      4. Plots the initial seed locations and applies an initial filter (without labels).
      5. Interactively validates and filters the seed locations.
      6. Computes particle trajectories based on the validated seeds.
      7. Re-plots the seed locations with labels after displaying 3D trajectory plots.
      8. Selects trajectories of interest.
      9. Plots the selected trajectories and saves the resulting figure.
     10. Plots time series of vorticity components (absolute vorticity, stretching, and tilt)
         for the selected trajectories and saves the corresponding figure.

    Configuration Parameters:
      - config_path: Path to the JSON configuration file for Type 2 high-resolution analysis.
      - x0_range: Range for initial x-coordinate seed values.
      - y0_range: Range for initial y-coordinate seed values.
      - z0_range: Range for initial z-coordinate seed values (typically a single level).
      - vorticity_threshold: Threshold value for filtering seeds based on vorticity.
      - buffer: Buffer value to adjust seed selection criteria.
      - levels: Array of levels (e.g., strain thresholds) used in the analysis.

    Output:
      - Figure files saved as:
          'figures/figure_08_type1_trajectories.png'
          'figures/figure_10_type1_vorticity_budget.png'
    """
    
    # Define configuration parameters
    config_path = 'inputs/type2_high_resolution_config.json' # Config file (edit as needed for pre/post vortexgenesis)
    x0_range = [825, 875]     # Range for initial x-coordinate seed values
    y0_range = [400, 450]     # Range for initial y-coordinate seed values
    z0_range = [5]            # Range for initial z-coordinate seed values (typically one level)
    vorticity_threshold = 15  # Vorticity threshold for filtering seeds
    buffer = 10               # Buffer value for seed selection
    levels = np.arange(-30, 30 + 3, 3) # Array of vorticity levels for plotting

    # Create instance of ParticleTrajectoryAnalysis
    pta = ParticleTrajectoryAnalysis(
        config_path=config_path,
        x0_range=x0_range,
        y0_range=y0_range,
        z0_range=z0_range,
        vorticity_threshold=vorticity_threshold,
        buffer=buffer,
        levels=levels
    )

    # Step 1: Plot initial seed locations and filter seeds (without labels)
    pta.plot_seed_locations_and_filter(labels=False)

    # Step 2: Validate and filter seeds interactively
    pta.validate_and_filter_seeds()

    # Step 3: Compute particle trajectories based on the validated seeds
    pta.compute_trajectories()

    # Step 4: After generating 3D plots, plot seed locations again with labels for clarity
    pta.plot_seed_locations_and_filter(labels=True)

    # Step 5: Select trajectories of interest from the computed trajectories
    pta.select_trajectories()

    # Step 6: Plot the selected trajectories and save the figure
    pta.plot_selected_trajectories(filename='figures/figure_13_type2_trajectories.png')

    # Step 7: Plot time series of vorticity components (absolute, stretching, tilt) for the selected trajectories
    avo, stretch, tilt = pta.plot_vorticity_components(filename='figures/figure_15_type2_vorticity_budget.png')
