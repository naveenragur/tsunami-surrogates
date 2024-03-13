# Tsunami Surrogates

Machine learning surrogates for approximating tsunami wave height time series nearshore and maximum inundation depth onshore for the Japan Tohoku region.

## Contents
- **geoclaw** (2D Nonlinear Shallow Water Equations tsunami runs)
  - _input
  - _output
  - _tsunami
<img src="/paper/_plots/model_region_geoclaw.png" alt="Model Region" height="400">

- **rupture** (earthquake rupture and displacement modeling)
  - _inputs (input source parameters for DOE and historic events)
  - dtopo_his (dtopo files for historic events and plotting)
  - dtopo_sift (for type B)
  - dtopo_slab (for type A)
<img src="/paper/_plots/displacement_fault_fuji.png" alt="Displacement Ex" height="400">

- **surrogates**
  - MAP (onshore surrogate for maximum inundation depth prediction)
  - TS (nearshore surrogate for time series prediction)
<img src="/paper/_plots/VEDArch.png" alt="VED" height="400">

- **paper** (analysis, plots, and results)
  - _plots
  - _results
  - _stats
<img src="/paper/_plots/scatter_TS__riku_6042.png" alt="Plots" height="400">

## Usage

Following are the YAML files with information on the Python packages and requirements to run:
- GeoClaw 2DNLSE simulation: [`/geoclaw/geoclaw.yml`](/geoclaw/geoclaw.yml)
- Machine learning: [`/surrogates/pytorch.yml`](/surrogates/pytorch.yml)
- PyGMT plotting: [`/paper/pygmt.yml`](/paper/pygmt.yml)

Each directory contains a more detailed README.md.

## Useful References and Projects

- "Comparison of Machine Learning Approaches for Tsunami Forecasting from Sparse Observations" by C.M. Liu, D. Rim, R. Baraldi, and R.J. LeVeque, Pure and Applied Geophysics, 2021. DOI: [10.1007/s00024-021-02841-9](https://doi.org/10.1007/s00024-021-02841-9)


