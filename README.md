# AMOCcollapse
Scripts used for analysis and creating figures for Drijfhout et al. (Atlantic overturning collapses in global warming projections after 2100)

# Setup
The analysis was done using python scripts and notebooks.  The python environment provided (environment.yml) contains the python packages needed and maybe a few extra.  Once installed to compute the ocean heat transports (others are also computed at the same time as running the script).  In order to make use of these scripts transportsComp/CMIPinformation.py needs to have the directories and search scripts altered to match the directory setup you are using.  The EN4 scripts and downloadCMIP will also require some edits to match your directory structure.

# Data
The main data used in this study can be downloaded from the ESGF data nodes, this can be done through the website (https://aims2.llnl.gov/search/cmip6), the python notebook provided (downloadCMIP/downloadCMIP.ipynb) and by your favorite means.  Mesh mask data and basin data are provided as gzipped files in the data directory, they must be unzipped prior to doin computations. 

EN4 data as well as RAPID data used can be downloaded from their websites https://www.metoffice.gov.uk/hadobs/en4/ and https://rapid.ac.uk/, respectively.

# EN4 mixed layer depth:
The python notebooks to compute the EN4 mixed layer depth are found in the EN4_mld directory.

# OHT and other transports:
The main script to do this is found in the transportsCMIP directory and can be run as follows from the commandline (where <> means you have to specify the setup you want to run).  Important do not forget to updata CMIPinformation to your directory structure, otherwise these scripts won't run!!!

python compute_meridional_transports.py <model> <experiment> <ensemble member> <grid type, typically gn> <outfile> <runfile> 

Note that the runfile and outfile need to be different, the data will be moved from the runfile to the outfile when the computation has completed.
