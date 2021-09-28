# More Alike Than Different - Quantifying Deviations of Brain Structure and Function in Major Depressive Disorder across Neuroimaging Modalities

This repo contains all scripts necessary to the analyses of our currently submitted manuscript on differences between healthy and depressed 
individuals in multiple neuroimaging modalities.

## Dependencies
To start of, create a Python 3.7 (or higher) environment (e.g. using Anaconda). You can install all necessary packages 
via `pip install -r requirements.txt`. Please note that photonai_graph is currently still under development and has
not been officially released by our group, yet. If you want to run the graph metric analyses, feel free to send us a short
email and we will be able to grant you access to the code of PHOTONAI Graph. 

## Install
To download the code of this repository, run the following command within the terminal:

`git clone --recurse-submodules https://github.com/wwu-mmll/more-alike-than-different-paper2021.git`

## Code structure
There are two important folders within this repo. First, all Python scripts are organized in the `pipeline` folder. 
There, you will find the `pipeline.py` script that includes the foundation to all analyses. The analysis pipeline
runs a consecutively number of steps (e.g. preprocessing, statistics, plots). Second, all pipeline (therefore analysis)
definitions are defined in the `analyses/configs` folder. These are .yaml files that define the data preprocessing, 
data loading, statistical analyses and visualization of every individual analysis. 

To run an analysis, you can run `analyses/run_analyses.py`. In this script, `main_pipeline` is called with a specific 
.yaml file that defines an analysis. Results are saved to the folder specified in the .yaml file.