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
- [Repository Structure](#project-structure)

## Installation

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Usage

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
