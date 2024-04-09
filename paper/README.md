# Analysis for the paper
This directory contains scripts and resources for preparing results, analysis and figures for the paper.

## Contents and Folder Structure
- `README.md`: This README file.
- `_results/`: Directory for saving resultsfrom analysis.
- `_stats/`: Directory for storing statistics generated during the geoclaw simulations.
- `_plots/`: Directory for storing generated plots.
- `pygmt.yml`: Configuration file for Python packages for pygmt plots

## Overview of notebooks
The folder includes various notebooks and tools for:
- `plotModel_JPData.ipynb`: for plotting data related to the rupture modelling(displacement,fault location etc.)
- `plotResults_JPGeoclawModel.ipynb`: for plotting GeoClaw model region.
- `plotResults_JPModel.ipynb`: for plotting results from the JP model for 2011 Tohoku event.
- `plotResultsML_TS.ipynb`:  for plotting results from the nearshore time sereis(TS) surrogate.
- `plotResultsML_MAP_historic.ipynb`: for plotting historic results from the ML MAP model.
- `plotResultsML_MAP.ipynb`:for plotting results from the onshore (MAP) surrogate.
- `plotResultsML_stats.ipynb`: for plotting results from the ML model.

## Usage
### Setting up the environment
The repository uses a conda environment to manage the python packages, for plotting pygmt was used and can be installed using the following command.

```bash
conda env create -f pygmt.yml
```
This will create a conda environment named with all the required packages.

### Running the scripts
The notebooks can be run using jupyter notebook or jupyter lab. The notebooks are self-explanatory and provide the results discussed in the article.