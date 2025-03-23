import os
import xarray as xr
import pandas as pd
from tqdm import tqdm
from phd.variables.get_variables_manual import GetVariablesWRF

class WRFPreprocessor:
    """
    Class for preprocessing WRF model output files.

    This class handles the loading, processing, and saving of WRF datasets based on a provided JSON 
    configuration file. It leverages xarray for efficient data handling and Dask for parallel computation 
    via chunking. The preprocessing includes checking for the presence of a 'Time' dimension in each dataset,
    adding a dummy time coordinate if necessary, and saving the processed files with compression.

    Attributes:
      config_path (str): Path to the JSON configuration file containing WRF settings.
      output_dir (str): Directory where the processed output files will be saved.
      chunks (str or dict): Chunk sizes for Dask operations when opening datasets.
      dummy_time (pandas.DatetimeIndex): Dummy timestamp to assign if a dataset is missing a 'Time' dimension.
      gvw (GetVariablesWRF): Instance to load WRF variables and configuration.
      config (dict): Configuration parameters loaded from the JSON file.
      files (list): List of WRF output files retrieved based on the configuration.

    Methods:
      open_single_dataset(filename):
          Opens a single WRF dataset using xarray with the specified chunking.
      open_multiple_datasets(files, concat_dim, combine, parallel):
          Opens multiple WRF datasets and concatenates them along the 'Time' dimension.
      add_time_dimension(ds):
          Checks and adds a 'Time' dimension to a dataset if it is missing, using the dummy timestamp.
      close_dataset(ds):
          Closes an open xarray Dataset.
      preprocess_and_save(input_file):
          Processes a single WRF file by adding a time dimension (if needed), compressing, and saving the result.
      run():
          Iterates over all WRF files obtained from the configuration, preprocessing and saving each one.
    """
    
    def __init__(self, config_path, output_dir='outputs', chunks='auto', dummy_time='2000-01-01T00:00:00'):
        self.config_path = config_path
        self.output_dir = output_dir
        self.chunks = chunks
        self.dummy_time = pd.to_datetime([dummy_time])

        # Initialize the GetVariablesWRF class with the configuration path
        self.gvw = GetVariablesWRF(config_path=self.config_path)

        # Load the configuration
        self.config = self.gvw.get_config()

        # Retrieve the list of WRF output files based on the configuration
        self.files = self.gvw.get_files()

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def open_single_dataset(self, filename):
        try:
            ds = xr.open_dataset(filename, chunks=self.chunks)
            return ds
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        except Exception as e:
            raise RuntimeError(f"Error opening dataset {filename}: {e}")

    def open_multiple_datasets(self, files, concat_dim='Time', combine='nested', parallel=True):
        if not files:
            raise ValueError("No files provided to open_multiple_datasets.")

        try:
            ds = xr.open_mfdataset(
                files,
                combine=combine,
                concat_dim=concat_dim,
                parallel=parallel
            )
            return ds
        except FileNotFoundError as e:
            raise FileNotFoundError(f"One or more files not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error opening multiple datasets: {e}")

    def add_time_dimension(self, ds):
        if 'Time' not in ds.dims:
            # Add a new 'Time' dimension
            ds = ds.expand_dims('Time')

            # Assign a dummy Time coordinate
            ds['Time'] = ('Time', self.dummy_time)

            print("Added 'Time' dimension with a dummy timestamp to the dataset.")
        else:
            print("'Time' dimension already exists in the dataset.")
        return ds

    def close_dataset(self, ds):
        if ds is not None:
            ds.close()
            print("Dataset closed.")

    def preprocess_and_save(self, input_file):
        try:
            # Open the dataset
            ds = self.open_single_dataset(input_file)

            # Add 'Time' dimension if missing
            ds = self.add_time_dimension(ds)

            # Define the output file path (e.g., add '_processed' suffix)
            base_filename = os.path.basename(input_file)
            name, ext = os.path.splitext(base_filename)
            output_file = os.path.join(self.output_dir, f"{name}{ext}")

            # Save the modified dataset with compression
            encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
            ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
            print(f"Processed and saved: {output_file}")

            # Close the dataset
            self.close_dataset(ds)

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

    def run(self):
        if not self.files:
            print("No files found to process.")
            return

        print(f"Found {len(self.files)} files to process.")

        # Iterate over each file with a progress bar
        for file in tqdm(self.files, desc="Processing WRF files"):
            self.preprocess_and_save(input_file=file)

# Example usage
if __name__ == "__main__":
    # Define the path to your JSON configuration file
    config_path = 'inputs/type1_front_config.json'

    # Instantiate the WRFPreprocessor class
    preprocessor = WRFPreprocessor(config_path=config_path, output_dir='outputs')

    # Run the preprocessing
    preprocessor.run()
