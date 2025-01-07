# PinkRigs Data Tools

A Python package for querying, formatting, and processing data from the PinkRigs dataset. This repository provides easy-to-use tools to interact with the PinkRigs data, handle ONE folders, and format the data for further analysis. PinkRigs data follows the ONE nomenclature (Open Neurophysiology Environment, for more details go [here](https://int-brain-lab.github.io/iblenv/one_docs/one_reference.html)). 

# Installation 

### with Git
- git clone this repository 
- create a new environment: `conda create -n PinkRigs_data_analysis_project python=3.10`
- conda activate new environment `conda activate PinkRigs_data_analysis_project`
- navigate to where your repository is saved in terminal `cd Documents/Github/PinkRigs-pytools`
- install the repository in your environment `pip install .`

### with pip
Currently you can install the query functions as a site-package with pip. For this run: 
`pip install pinkrigs_tools @ git+https://github.com/takacsflora/PinkRigs-pytools.git@main`

## To update after changes to the repository
After pulling the repository run `pip install .`

# To query the PinkRigs data
- follow the tutorial found [here](https://github.com/takacsflora/PinkRigs-pytools/blob/main/tests/load_datasets_tutorial.ipynb). 

# Common issues
This code is currently for CortexLab members and collaborators only and requires server access. Please make sure that you mapped all of our active servers. This code is only compatible with pandas 1.3> at the moment. Please get it touch if you have ideas on how to refactor for newer version of pandas.

