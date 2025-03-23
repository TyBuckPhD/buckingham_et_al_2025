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

## Installation

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Usage
Unfortunately, the provided scripts are currently hard-coded to specific WRF output (which was far too large to be added into the repository); therefore, the provided scripts are expected to be used as guidelines for replication rather than as a flexible package. Development of such a package would require implementing an automated cold front detection algorithm for high-resolution gridded data - which is an exceedingly challenging task. 

However, Figure 4 is standalone, and can be run via the terminal using either

```python
Python3 scripts/plot_figure_04.py
```

or simply executing it within your chosen IDE>

## Repository Structure
```plaintext
buckingham_et_al_2025/       # Main package with core modules
├── core/                    # Core data processing functions
├── utils/                   # Utility functions (e.g., plotting helpers, timer)
└── variables/               # Modules for variable definitions and colorbars
ukmo_composites/             # Modules for processing UKMO composite files
scripts/                     # Stand-alone scripts for generating figures
figures/                     # Output directory for generated figures (png files)
environment.yml              # Conda environment file for reproducibility (currently bloated)
README.md                    # This file
```
