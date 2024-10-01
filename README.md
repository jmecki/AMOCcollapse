# AMOCcollapse

Scripts used for analysis and creating figures for Drijfhout et al. (Atlantic overturning collapses in global warming projections after 2100)

There are two setups required for producing all the figures. For each of the two setups follows a separate description.

1.  The first setup was used on [`SurfSara`](https://www.surf.nl/en/services/surf-research-cloud). Here we produced the analysis of the raw CMIP6 data (AMOC-strength, mixed-layer depth, temperature etc.).
2.  The second setup was used on [`Jasmin`](https://jasmin.ac.uk/). This setup provides the methods for analyzing the EN4 data, as well as computing Overturning heat transport and the AMOC-strength (used for `CanESM5`).

## Data

The main data used in this study can be downloaded from the ESGF data nodes, this can be done through the website (https://aims2.llnl.gov/search/cmip6), the python notebook provided (downloadCMIP/downloadCMIP.ipynb) and by your favorite means. Mesh mask data and basin data are provided as gzipped files in the data directory, they must be unzipped prior to doin computations.

EN4 data as well as RAPID data used can be downloaded from their websites https://www.metoffice.gov.uk/hadobs/en4/ and https://rapid.ac.uk/, respectively.

# 1. [`SurfSara`](https://www.surf.nl/en/services/surf-research-cloud) analyses

## Setup

For these analyses, one has to install `cdo` and `cartopy`. On top of these packages we install the custom build [`optim_esm_tools`](https://github.com/JoranAngevaare/optim_esm_tools) and this package for the final plotting:

```
git clone https://github.com/jmecki/AMOCcollapse
# The -e ensures that any local edits also affect the code as desired
pip install -e AMOCcollapse
```

If you encounter issues getting `cdo`, `cartopy`, `optim_esm_tools` and everything else working in your environment, there is a step by step guide that allows you to retrieve a set of software versions that should work, you can follow this step by step installation in the [`optim_esm_base`](https://github.com/JoranAngevaare/optim_esm_base/blob/master/README.md#partial-installation-on-linux-without-synda) repository (follow the "Partial installation (on linux) without `synda`").

The file `amoc_collapse_scripts/path_setup.py` needs to be updated with the specific paths where one has downloaded the appropriate data.

## AMOC strength

The notebook `notebooks/AMOC_strength.ipynb` combines the information from all the readily available CMIP6 stream-functions (CMIP6 variables `msftyz` and `msftmz`).
A requirement is that all the data is pre-downloaded and follows a conventional `ESGF` folder structure (e.g. such as provided by the [`synda`](https://espri-mod.github.io/synda/index.html) tool).

## AMOC deep mixing regions

`notebooks/AMOC_deep_mixing_regions.ipynb` investigates the extended data of the 9 models that show a collapse in the ssp585 scenario, specifically by further analyzing the mixed layer depth (CMIP6 variable `mlotst`). This notebook requires the variables of interest to be downloaded and merged in time. It also requires the `notebooks/Surface_fresh_water_convergence_calculation.ipynb` notebook to be computed for the surface fresh water convergence calculation.

## EN4 figure

This notebook (`notebooks/EN4_figure.ipynb`) only plots the output of the "EN4 mixed layer depth" as discussed below.

# 2. [`Jasmin`](https://jasmin.ac.uk/) analyses

## Setup

The analysis was done using python scripts and notebooks. The python environment provided (`environment.yml`) contains the python packages needed and maybe a few extra. Once installed to compute the ocean heat transports (others are also computed at the same time as running the script). In order to make use of these scripts `transportsComp/CMIPinformation.py` needs to have the directories and search scripts altered to match the directory setup you are using. The EN4 scripts and downloadCMIP will also require some edits to match your directory structure.

## EN4 mixed layer depth:

The python notebooks to compute the EN4 mixed layer depth are found in the EN4_mld directory.

## OHT and other transports:

The main script to do this is found in the transportsCMIP directory and can be run as follows from the commandline (where <> means you have to specify the setup you want to run). Important do not forget to update CMIPinformation to your directory structure, otherwise these scripts won't run!!!

`python compute_meridional_transports.py <model> <experiment> <ensemble member> <grid type, typically gn> <outfile> <runfile>`

Note that the runfile and outfile need to be different, the data will be moved from the runfile to the outfile when the computation has completed.
