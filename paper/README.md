# Analysis for the paper
This repository contains scripts and resources for preparing results, analysis and figures for the paper.

## Overview
The folder includes various scripts and tools for:

- 

## Folder Structure
Introductions to this and requisits.
- `README.md`: This README file.

input and output folders
- `MAP`: Contains source parameter info for events - historic scenarios, slab(typeA) and sift(typeB).
- `TS`: Contains dtopo files(.tt3), plots for each historic event(dtopopng) and event info file

## Usage

### Setting up the environment
The repository uses a conda environment to manage the python packages, for plotting pygmt was used and can be installed using the following command.

```bash
conda env create -f pygmt.yml
```
This will create a conda environment named with all the required packages.

### Running the scripts