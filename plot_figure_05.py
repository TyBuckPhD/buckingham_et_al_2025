import re
import matplotlib.pyplot as plt
import numpy as np
from phd.core.helmholtz_decomposition import HelmholtzDecomposition

class CalculateFrontalStrains:
    def __init__(self, config_path, output_fig_path, event_type):
        self.config_path = config_path
        self.output_fig_path = output_fig_path
        self.event_type = event_type
        self.hhd = HelmholtzDecomposition(config_path=self.config_path, event_type=self.event_type)
        self.strain_threshold_list = []
        self.frontal_strain_list = []
        self.files = []
        self.times = []
        self.colors = ['blue', 'green', 'red', 'purple', 'orange',
                       'brown', 'pink', 'gray', 'olive', 'cyan']
        self.num_time_steps = 0
        self.x_indices = None

    def run_decomposition(self):
        # Run the Helmholtz decomposition and store results
        self.strain_threshold_list, self.frontal_strain_list, self.files = self.hhd.run()
        self.num_time_steps = len(self.files)
        self.x_indices = np.arange(self.num_time_steps)

    def extract_times(self):
        # Extract times from file names for x-axis labels
        self.times = []
        for file in self.files:
            match = re.search(
                r'wrfout_d\d{2}_\d{4}-\d{2}-\d{2}_(\d{2})-(\d{2})-\d{2}\.nc',
                file
            )
            if match:
                hour = match.group(1)
                minute = match.group(2)
                self.times.append(f"{hour}:{minute}")
            else:
                self.times.append('Unknown')

    def plot_data(self):
        plt.figure(figsize=(8, 6))

        # Set y-axis and x-axis limits
        # plt.ylim(1.2e-5, 3.0e-5)
        plt.xlim(0, self.num_time_steps - 1)

        # Plot all strains for each domain
        for i in range(len(self.strain_threshold_list)):
            color = self.colors[i % len(self.colors)]

            strain_threshold = self.strain_threshold_list[i]
            frontal_strain = self.frontal_strain_list[i]

            # Ensure data lengths match
            if (len(frontal_strain) != self.num_time_steps or
                    len(strain_threshold) != self.num_time_steps):
                print(f"Warning: Data length mismatch in domain {i+1}")
                continue

            plt.plot(
                self.x_indices,
                frontal_strain,
                color=color,
                linewidth=1.0,
                linestyle='-'
            )
            plt.plot(
                self.x_indices,
                strain_threshold,
                color=color,
                linestyle='--'
            )

        # Compute the average frontal strain over domains
        frontal_strain_array = np.array(self.frontal_strain_list)
        frontal_strain_avg = np.mean(frontal_strain_array, axis=0)

        # Plot the average frontal strain as a thicker black line
        plt.plot(
            self.x_indices,
            frontal_strain_avg,
            color='black',
            linestyle='-',
            linewidth=3.0,
            label='Average Frontal Strain'
        )

        # Identify positions where minutes are '00' or '30' (every half hour)
        label_positions = []
        label_times = []
        for idx, time in enumerate(self.times):
            if time != 'Unknown':
                hour, minute = time.split(':')
                if minute in ('00', '30'):
                    label_positions.append(idx)
                    label_times.append(f"{hour}:{minute}")

        # Create labels for all positions, empty where we don't want labels
        all_labels = [''] * self.num_time_steps
        for idx, label in zip(label_positions, label_times):
            all_labels[idx] = label

        # Set x-axis labels
        plt.xticks(self.x_indices, all_labels, rotation=45)

        # Set axis labels
        plt.xlabel('Time (UTC)')
        plt.ylabel('Stretching Deformation (s$^{-1}$)')

        # Adjust layout
        plt.tight_layout()

    def save_figure(self):
        # Save the figure to a file
        plt.savefig(
            self.output_fig_path,
            dpi=200,
            bbox_inches='tight',
            pad_inches=0.05
        )
        plt.show()

    def run(self):
        self.run_decomposition()
        self.extract_times()
        self.plot_data()
        self.save_figure()

if __name__ == "__main__":
    config_path = 'inputs/type1_front_config.json'
    output_fig_path = 'figures/figure_05_type1.png'
    event_type = 2

    cfs = CalculateFrontalStrains(config_path, output_fig_path, event_type)
    cfs.run()