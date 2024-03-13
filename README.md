# tsunami-surrogates
Machine learning surrogates for approximating tsunami wave height time series nearshore and max inundation depth onshore for Japan Tohoku region

# Contents
- geoclaw(2D Nonlinear Shallow Water Equations tsunami runs)
  - _input
  - _output
  - _tsunami

- paper(analysis, plots and results)
  - _plots
  - _results
  - _stats

- rupture(earthquake rupture and displacement modelling)
  - _inputs(input source parameters)
    - historic(for historic)
  - dtopo_his(dtopo file and plotting)
    - dtopopng
  - dtopo_sift(for type B)
    - dtopofil
    - dtopopng
  - dtopo_slab(for type A)
    - dtopopng

- surrogates
  - MAP(onshore surrogate for max inundation depth prediction)
    - 1GaugeRiku
    - 3GaugeIshi
    - 3GaugeShio
  - TS(nearshore surrogate for time series prediction)
    - _ishi_5675
    - _riku_6042
    - _shio_5440

# Usage
Following are the yml files with info on the python packages and requirements to run:
- GeoClaw 2DNLSE simulation - /geoclaw/geoclaw.yml
- Machine learning - /surrogates/HPC.yml
- Pygmt plotting - /paper/pygmt.yml

README.md in each directory contains more details

## Useful References and projects
Comparison of Machine Learning Approaches for Tsunami Forecasting from Sparse Observations,
by C.M. Liu, D. Rim, R. Baraldi, and R.J. LeVeque, Pure and Applied Geophysics, 2021
DOI 10.1007/s00024-021-02841-9



