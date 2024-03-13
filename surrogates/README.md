# ML surrogates(variational encoder decoder) 
This repository contains scripts and resources for building and testing the variational encoder decoder model 

## Overview
The repository includes various scripts and tools for:
- 2 surrogates for tsunami prediction nearshore(time series) and onshore(maximum inundation depth).
- each surrogate is tested for the 3 test sites in Japan - Rikuzentakata(Riku), Ishinomaki(Ishi) and SendaI(Shio)
- 

## Folder Structure
Introductions to this and requisits.
- `README.md`: This README file.

input and output folders
- `MAP`: Contains source parameter info for events - historic scenarios, slab(typeA) and sift(typeB).
- `TS`: Contains dtopo files(.tt3), plots for each historic event(dtopopng) and event info file

## Usage
Certain heavy input ,output and result plot files (.npy,.csv and .png) are not included in the repository. These files are available on request and will be available at a download link soon.

### Setting up the environment
The repository uses a conda environment to manage the python packages.

```bash
conda env create -f pytorch.yml
```
This will create a conda environment named with all the required packages.

### Files and Example to Run Nearshore Surrogate for Rikuzentakata

**Working Directory:** `/surrogates/TS/_riku_6042`

#### Directories:
- `_data/`: Contains input data files used in the model training and testing.
- `_gaugeplots/`: Contains plots related to gauge data.
- `_output/`: Likely contains output files generated during the execution of scripts.
- `_plots/`: Contains various plots generated during the analysis of the ML training and testing.

#### Scripts:
- `aeplot.py`: Scripts for plotting data related to the autoencoder model.
- `process_gaugedata.py`: Scripts for processing gauge data as input for training stored in npy format.
- `vae.py`: Implementation of the variational autoencoder model.
- `vae_train.py`: Script for training the model.
- `vae_test.py`: Script for testing the model.
- `vae_test_historic.py`: Script for testing the model for historic events.
- `vae_plot.py`: Scripts for plotting results from the variational autoencoder.

#### Job Scripts:
- `run.sbatch`, `run.sh`: Scripts for running training and test jobs in one go.

### Files and Example to Run Onshore Surrogate for Rikuzentakata

**Working Directory:** `/surrogates/MAP/1GaugeRiku`

#### Directories:
- `_data/`: Contains data files used in the model.
- `_gaugeplots/`: Contains plots related to gauge data.
- `_output/`: Contains output files generated during the execution of scripts.
- `_plots/`: Contains various plots generated during the analysis.

#### Scripts:
- `process_fgmax.py`: Script for processing maximum depth data.
- `process_gaugedata.py`: Script for processing gauge data as input for training.
- `vae_plot.py`: Script for plotting results from the model.
- `vae.py`: Implementation of the model.
- `vae_train.py`: Script for training the model.
- `vae_test.py`: Script for testing the model.
- `vae_test_historic.py`: Script for testing the model for historic events.
