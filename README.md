# Tsunami Surrogates
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15337936.svg)](https://doi.org/10.5281/zenodo.15337936)

Machine learning surrogates for approximating tsunami wave height time series nearshore and maximum inundation depth onshore for the Japan Tohoku region.
Related article available as preprint - Advancing nearshore and onshore tsunami hazard approximation with machine learning surrogates available at https://doi.org/10.5194/nhess-2024-72

## Contents

### **geoclaw** (2D Nonlinear Shallow Water Equations tsunami runs)
- `_input/`
- `_output/`
- `_tsunami/`
<img src="/paper/_plots/model_region_geoclaw.png" alt="Model Region" height="400">

### **rupture** (earthquake rupture and displacement modeling)
- `_inputs/` (input source parameters for DOE and historic events)
- `dtopo_his/` (dtopo files for historic events and plotting)
- `dtopo_sift/` (for type B)
- `dtopo_slab/` (for type A)
<img src="/paper/_plots/displacement_fault_fuji.png" alt="Displacement Ex" height="400">

### **surrogates** (machine learning models)
- `MAP/` (onshore surrogate for maximum inundation depth prediction)
- `TS/` (nearshore surrogate for time series prediction)
<img src="/paper/_plots/VEDArch.png" alt="VED" height="400">

### **paper** (Jupyter notebooks for analysis, plots, and results)
- `_plots/`
- `_results/`
- `_stats/`
<img src="/paper/_plots/scatter_TS__riku_6042.png" alt="Plots" height="400">

## Usage
Following are the YAML files with information on the Python packages and requirements to run:
- GeoClaw 2DNLSE simulation: [`/geoclaw/geoclaw.yml`](/geoclaw/geoclaw.yml)
- Machine learning: [`/surrogates/pytorch.yml`](/surrogates/pytorch.yml)
- PyGMT plotting: [`/paper/pygmt.yml`](/paper/pygmt.yml)

Each directory contains a more detailed README.md.

Some large input files for the geoclaw simulation and the post-processed inputs for machine learning need to be downloaded from https://doi.org/10.5281/zenodo.10817116
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10817116.svg)](https://doi.org/10.5281/zenodo.10817116)

## Useful References and Projects

- [https://github.com/rjleveque/MLSJdF2021](https://github.com/rjleveque/MLSJdF2021.git) - A project using VAE for tsunami forecasting problem, developed in Python/Pytorch. Liu, C.M., Rim, D., Baraldi, R. et al. Comparison of Machine Learning Approaches for Tsunami Forecasting from Sparse Observations. Pure Appl. Geophys. 178, 5129–5153 (2021).DOI: [10.1007/s00024-021-02841-9](https://doi.org/10.1007/s00024-021-02841-9)

- [Tsunami Inundation Emulator](https://github.com/norwegian-geotechnical-institute/tsunami-inundation-emulator.git) - A project for tsunami inundation depth prediction using machine learning, developed in Julia/Flex. Erlend Briseid Storrøsten, Naveen Ragu Ramalingam, Stefano Lorito, Manuela Volpe, Carlos Sánchez-Linares, Finn Løvholt, Steven J Gibbons, Machine Learning Emulation of High Resolution Inundation Maps, Geophysical Journal International, 2024;ggae151.  DOI:  [https://doi.org/10.1093/gji/ggae151](https://doi.org/10.1093/gji/ggae151)
