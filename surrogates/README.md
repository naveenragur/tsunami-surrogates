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

### Setting up the environment
The repository uses a conda environment to manage the python packages.

```bash
conda env create -f pytorch.yml
```
This will create a conda environment named with all the required packages.

### Running the scripts