# Tohoku Tsunami Modeling Repository
This directory contains scripts and resources for conducting geoclaw simulation for the Tohoku region in Japan.

## Overview
The repository includes various scripts and tools for:
- Modified python modules used in GEOCLAW
- GEOCLAW input folder
- Scripts to prepare inputs and run simulations
- Post-processing scripts

## Contents and Folder Structure
Introductions to this and requisits.
- `README.md`: This README file.
- `geoclaw.yml`: Configuration file for python packages managed in a conda environment

Modified GEOCLAW modules, make files and executables
- `dtopotools.py`: Modified GeoClaw module.
- `plotclaw.py`: Modified GeoClaw module.
- `runclaw.py`: Python script for running GeoClaw simulations.
- `Makefile`: Makefile for the project.
- `Makefile.common`: Common Makefile settings.
- `xgeoclaw`: Executable for GeoClaw.

GEOCLAW input and output folders
- `_input/`: Directory for geoclaw model input files.
- `_tsunami/`: Empty directory which usually contains dtopo files(.tt3) which will run as tsunami forcing in a batch of simulations.
- `_output/`: Empty directory where output files are written for each simulation ie the gauge observation and the grid observations.

Preprocessing inputs for GEOCLAW model, setup model
- `process_dat.py`: Script for processing topo data from JP cabinet project.
- `process_dat_deform.py`: Script for processing deformed datafrom JP cabinet project.
- `process_fg_max.py`: Script for processing maximum depth data.
- `makegaugepts.py`: Python script for generating gauge points for recording observations.

GEOCLAW run and plot settings, batch run and post processing scripts
- `setplot.py`: Python script for setting up plots.
- `setrun.py`: Python script for setting up GeoClaw runs.
- `startbatch_run.py`: Python script for starting batch runs.
- `startbatch_postprocess.py`: Script for starting batch post-processing.
- `startbatch_postprocess_synthetic.py`: Script for starting batch post-processing of synthetic data.

## Usage

### Setting up the environment
The repository uses a conda environment to manage the python packages.

```bash
conda env create -f geoclaw.yml
```
This will create a conda environment named `geoclaw` with all the required packages.

### GeoClaw code and modules
See the [clawpack installation guide](https://www.clawpack.org/installing.html) for more information on installing clawpack.

### Running the simulations
The simulations can be run using the `startbatch_run.py` script.


