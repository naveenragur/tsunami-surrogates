# EQ rupture modelling and Geoclaw input processing folder
This repository contains scripts and resources for generating geoclaw dtopo files used as input for tsunami simulation for historic and DOE events for the the Tohoku region in Japan.

## Overview
The repository includes various scripts and tools for:
- inputs to prepare eq rupture for historic events
- inputs to prepare eq rupture for DOE type A events
- inputs to prepare eq rupture for DOE type B events
- dtop folders containing .tt3 file to force geoclaw model

## Folder Structure
Introductions to this and requisits.
- `README.md`: This README file.

Modified GEOCLAW modules, make files and executables
- `dtopotools.py`: Modified GeoClaw module.

input and output folders
- `_input/`: Contains source parameter info for events - historic scenarios, slab(typeA) and sift(typeB).
- `dtopo_his`: Contains dtopo files(.tt3), plots for each historic event(dtopopng) and event info file
- `dtopo_sift`: Empty directory where .tt3 files are written for each type B event, contains plots for each event(dtopopng) and overall summary info files
- `dtopo_slab`: Empty directory where .tt3 files are written for each type A event, contains plots for each event(dtopopng) and overall summary info files

## Usage

### Setting up the environment
The repository uses a conda environment to manage the python packages.

```bash
conda env create -f geoclaw.yml
```
This will create a conda environment named `geoclaw` with all the required packages.

### GeoClaw code and dtopo modules
See the [clawpack installation guide](https://www.clawpack.org/installing.html) for more information on installing clawpack.

### Generating dtopo files for tsunami simulations
The simulations can be run using the python scripts
makedtopo9Mwsift.py
maketopodtopo_hist.py
maketopodtopo_hist2increaseext.py
maketopodtopo_SLAB.py
maketopodtopoSanrikuCentroid.py


