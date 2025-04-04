## Buckingham et al. (2025)
This repository contains the code and data processing scripts for the paper 
"Two Archetypes of Tornadic Quasilinear Convective Systems in the United Kingdom: 
Relevance of Horizontal Shearing Instability" by Buckingham et al. (2025). 
It includes scripts for:

- Processing WRF and UKMO rainfall composite files
- Performing the Helmholtz Decomposition on a limited domain
- Backwards trajectory analysis from gridded data

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Contact](#contact)

## Installation

```bash
conda env create -f environment.yml
conda activate your_env_name
```
Please be aware the current environment.yml file is bloated and can be trimmed.

## Usage
Unfortunately, the provided scripts are currently hard-coded to specific WRF output (which was far too large to be added into the repository); therefore, the provided scripts are expected to be used as guidelines for replication rather than as a flexible package. Development of such a package would require implementing an automated cold front detection algorithm for high-resolution gridded data - an exceedingly challenging task. 

However, Figure 4 is standalone, and can be run via the terminal using

```python
Python3 scripts/plot_figure_04.py
```

or simply executing it within your chosen IDE.

## Repository Structure
The repository is structured as so

```plaintext
buckingham_et_al_2025/       # Main package with core modules
├── core/                    # Core data processing functions
├── ukmo_composites/         # UKMO precipitation mosaic processing functions
├── utils/                   # Utility functions (e.g., plotting helpers, timer)
└── variables/               # Modules for variable definitions and colorbars
figures/                     # Output directory for generated figures (png files)
scripts/                     # Stand-alone scripts for generating figures
inputs/                      # JSON files for input configs (hard-coded to specific WRF output)
README.md                    # This file
environment.yml              # Conda environment file for reproducibility (currently bloated)
```

## Data Sources
- The Met Office 5-min, 1-km grid-spacing precipitation-rate mosaics were accessed via CEDA (https://data.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km). An account is required.
- WRF namelists are available upon request.

## Contact
- The lead author of Buckingham et al. (2025) is available at tyjamesbuckingham@gmail.com

