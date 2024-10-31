from phd.ukmo_composites.ukmo_composite_file_converter import UKMOCompositeProcessor

# Filepath to the data file
filepath = '/Volumes/Samsung_T5/phd_data/ukmo_composite_data/ukmo_20111129/metoffice-c-band-rain-radar_uk_201111291300_1km-composite.dat'

# Initialize the processor and plot with a custom extent
processor = UKMOCompositeProcessor(filepath)
processor.plot_radar_composite(extent=[2.0, -8.0, 49.0, 56.0], output_file="composite.png")