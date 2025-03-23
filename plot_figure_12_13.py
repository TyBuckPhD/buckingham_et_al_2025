import numpy as np
from phd.core.trajectory_analysis import ParticleTrajectoryAnalysis

if __name__ == "__main__":
    config_path = 'inputs/type2_high_resolution_config.json'
    x0_range = [825, 875]
    y0_range = [400, 450]
    z0_range = [5]
    vorticity_threshold = 15
    buffer = 10
    levels = np.arange(-30, 30 + 3, 3) 

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

    # Plot seed locations and filter seeds
    pta.plot_seed_locations_and_filter(labels=False)

    # Validate and filter seeds interactively
    pta.validate_and_filter_seeds()

    # Compute trajectories
    pta.compute_trajectories()

    # After the 3D plots, plot seed locations again with labels
    pta.plot_seed_locations_and_filter(labels=True)

    # Select trajectories of interest
    pta.select_trajectories()
    
    pta.plot_selected_trajectories(filename='figures/figure_12_type2_trajectories.png')

    # Plot time series of vorticity components for selected trajectories
    avo, stretch, tilt = pta.plot_vorticity_components(filename='figures/figure_13_type2_vorticity_budget.png')
