"""Set of hardcoded paths."""
###
# AMOC_strength - files
###
# There is one file that was merged by hand to both include the historical and the scenario data. It was obtained using the transportsComp/CMIPinformation.py file.
# The data on ESGF does only extend until 2180, so using the abovementioned script, we had to extend the data until 2300.
MANUALLY_PATCHED_SCENARIO_AND_HISTORICAL_FILE = "/data/volume_2/2024_06_17_msft_manual_stage/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r1i1p1f1/Omon/msftmz/gn/v20240820/merged.nc"

STEAMFUNCTION_BASE = '/data/volume_2/2024_06_17_msft_manual_stage/data'
# Folder structure is
# {STEAMFUNCTION_BASE}/CMIP6/ScenarioMIP/CAS/CAS-ESM2-0/ssp126/r1i1p1f1/Omon/msftmz/gn/v20201230
# Therefore, the index in path.split(os.sep) is as follows:
SCENARIO_INDEX = 9
VERSION_INDEX = -1
VARIANT_LABEL_INDEX = 10
SOURCE_ID_INDEX = 8
VARIABLE_ID_INDEX = 12
GRID_ID_INDEX = -2

AMOC_STRENGTH_WORKING_DIR = '/data/volume_2/tipping_figures/2024_02_27_msft_v12'

# Computed with transportsComp/CMIPinformation.py
OHT_TRANSPORT_CMIP6 = '/data/volume_2/tipping_figures/2024_05_07_amoc/data/transports_from_jenny/Atlantic_OHT_H_None_26.5N_SybrenPaper.nc'

RAPID_OHT = '/data/volume_2/2024_08_12_rapid_data/mocha_mht_data_ERA5_v2020.nc'
RAPID_AMOC = '/data/volume_2/2024_08_12_rapid_data/moc_vertical.nc'


###
# General CMIP6 data
###
GENERAL_CMIP6_BASE = '/data/volume_2/synda/year_data_merged/CMIP6'

###
# Mixed layer depth
###
MIXED_LAYER_BASE = "/data/volume_2/tipping_figures/2024_05_07_amoc/"
FRESHWATER_TRANSPORT_BASE = "/data/volume_2/manual_download"


###
# EN4 data
###
EN4_MLD_DATA_SET = "/data/volume_2/tipping_figures/2024_05_07_amoc/data/figure_4_from_jenny/EN4_mld_figure.nc"
EN4_MLD_TIME_SERIES = "/data/volume_2/tipping_figures/2024_05_07_amoc/data/figure_4_from_jenny/EN4_mld_timeseries.nc"
EN4_SSS_TIME_SERIES = "/data/volume_2/tipping_figures/2024_05_07_amoc/data/figure_4_from_jenny/EN4_SSS_timeseries.nc"
