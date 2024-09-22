##############################################################################
#
# CMIP5 and CMIP6 functions and dictionaries to extract useful information:
#
# #############################################################################

import glob

from netCDF4 import Dataset

cmip6dir  = '/badc/cmip6/data/CMIP6/'  # Where masks and basins are stored
cmip5dir  = '/badc/cmip5/data/cmip5/output1/'
cmip6down = '/gws/nopw/j04/canari/users/jmecking001/CMIP/downloaded/CMIP6/'  # any downloaded CMIP data
cmipsave  = '/home/users/jmecking001/jpython/AMOCcollapse/data/'  # Where computed data is stored as well as masks and basin files

# Make list of all models available:
def listModels(cmip,MIP='*',inst='*',):
    models = []
    if cmip == '6':
        search_str = (cmip6dir + MIP + '/' + inst + '/*')
        tmp = glob.glob(search_str)
        for tt in tmp:
            models.append(tt.split('/')[-1])
            models = list(set(models))
    elif cmip == '5':
        search_str = (cmip5dir + inst + '/*')
        tmp = glob.glob(search_str)
        for tt in tmp:
            models.append(tt.split('/')[-1])
            models = list(set(models)) 

    models.sort()
    return models

# Make list of all available ensemble members:
def listENS(cmip,model,EXP='*',MIP='*',inst='*',):
    ens = []
    if cmip == '6':
        search_str = (cmip6dir + MIP + '/' + inst + '/' + model + '/' + EXP + '/r*')
        tmp = glob.glob(search_str)
        for tt in tmp:
            ens.append(tt.split('/')[-1])
            ens = list(set(ens))
        search_str = (cmip6down + model + '/' + EXP + '/r*')
        tmp = glob.glob(search_str)
        for tt in tmp:
            ens.append(tt.split('/')[-1])
            ens = list(set(ens))
    elif cmip == '5':
        search_str = (cmip5dir + inst + '/' + model + '/' + EXP + '/*/*/*/r*')
        tmp = glob.glob(search_str)
        for tt in tmp:
            ens.append(tt.split('/')[-1])
            ens = list(set(ens)) 

    ens.sort()
    return ens

# Make list of files available (assume they are .nc files):
def listFiles(cmip,model,ENS,EXP,var,MIP='*',inst='*',vtype='*',gtype='gn',version='v*'):
    if cmip == '6':
        # Change the search string for models that have files downloaded:
        # Check downloaded directory first:
        search_str = (cmip6down + model + '/' + EXP + '/' + ENS + '/' + version + '/' + var + '*.nc')
        files = glob.glob(search_str)
        if len(files) == 0:
            search_str = (cmip6dir + MIP + '/' + inst + '/' + model + '/' + EXP + '/' + ENS + '/' + vtype + '/' + var + '/' + gtype + '/' + version + '/*.nc')
            files = glob.glob(search_str)
    elif cmip == '5':
        search_str = (cmip5dir + inst + '/' + model + '/' + EXP + '/*/*/' + vtype + '/' + ENS + '/' + version + '/' + var + '/*.nc')
        files = glob.glob(search_str)
    else:
        print('please specify CMIP phase, currently 5 and 6 are coded')
        files = []

    # Find unique files if latest or a specific version is not specified:
    if ((len(files) != 0 ) & (version == 'v*')):
        files = list(reversed(files))
        singlefiles = []
        infiles     = []
        nl = len(files)
        for ll in range(0,nl):
            fname = files[ll].split('/')[-1]
            if not fname in singlefiles:
                singlefiles.append(fname)
                infiles.append(files[ll])
        files = infiles

    files.sort()
    
    return files

# Determine dimensions using a given file and variable:
def getDims(infile,var):
    ncid = Dataset(infile,'r')
    dims = ncid.variables[var].get_dims()
    ncid.close
    
    return dims

# CMIP5 ocean dictionary (grid, haloEW, flipNS, reg, extraT):
cmip5ocean = {
    'ACCESS1-0'       :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'ACCESS1-3'       :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'BNU-ESM'         :{'grid':'Btl',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'CCSM4'           :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-BGC'       :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-CAM5'      :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-CAM5-1-FV2':{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-FASTCHEM'  :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-WACCM'     :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CFSv2-2011'      :{'grid':'A',       'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'CMCC-CESM'       :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-CM'         :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-CMS'        :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CNRM-CM5'        :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CNRM-CM5-2'      :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CSIRO-Mk3-6-0'   :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'CanAM4'          :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'CanCM4'          :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'CanESM2'         :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'EC-EARTH'        :{'grid':'Ctr',     'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'FGOALS-g2'       :{'grid':'A',       'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'FGOALS-gl'       :{'grid':'Bbl',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'FGOALS-s2'       :{'grid':'Bbl',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'FIO-ESM'         :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GEOS-5'          :{'grid':'A',       'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'GFDL-CM2p1'      :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-CM3'        :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-ESM2G'      :{'grid':'Ctr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-ESM2M'      :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-HIRAM-C180' :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'GFDL-HIRAM-C360' :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'GISS-E2-H'       :{'grid':'Bbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-H-CC'    :{'grid':'Bbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-R'       :{'grid':'Cbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-R-CC'    :{'grid':'Cbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'HadCM3'          :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':True},
    'HadGEM2-A'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'HadGEM2-AO'      :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':True},
    'HadGEM2-CC'      :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':True},
    'HadGEM2-ES'     :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':True},
    'IPSL-CM5A-LR'    :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'IPSL-CM5A-MR'    :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'IPSL-CM5B-LR'    :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'MIROC-ESM'       :{'grid':'Btl', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MIROC-ESM-CHEM'  :{'grid':'Btl', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MIROC4h'         :{'grid':'Bbl', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'MIROC5'          :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'MPI-ESM-LR'      :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MPI-ESM-MR'      :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MPI-ESM-P'       :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MRI-AGCM3-2H'    :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MRI-AGCM3-2S'    :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MRI-CGCM3'       :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'MRI-ESM1'        :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'NICAM-09'        :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'NorESM1-M'       :{'grid':'Cbl',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'NorESM1-ME'      :{'grid':'Cbl',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'bcc-csm1-1'      :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'bcc-csm1-1-m'    :{'grid':'Btr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'inmcm4'          :{'grid':'Ctr',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
}

# CMIP6 ocean dictionary (grid, haloEW, flipNS, reg, extraT):
cmip6ocean = {
    'ACCESS-CM2'       :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'ACCESS-ESM1-5'    :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'ACCESS-OM2'       :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'ACCESS-OM2-025'   :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'AWI-CM-1-1-HR'    :{'grid':'unknown',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'AWI-CM-1-1-LR'    :{'grid':'unknown',     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'AWI-CM-1-1-MR'    :{'grid':'Unstructured','haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'AWI-ESM-1-1-LR'   :{'grid':'Unstructured','haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'BCC-CSM2-HR'      :{'grid':'Btl',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'BCC-CSM2-MR'      :{'grid':'Btl',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'BCC-ESM1'         :{'grid':'Btl',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CAMS-CSM1-0'      :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CAS-ESM2-0'       :{'grid':'Bbl',         'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'CESM1-CAM5-SE-HR' :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM1-CAM5-SE-LR' :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM2'            :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM2-FV2'        :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM2-WACCM'      :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CESM2-WACCM-FV2'  :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CIESM'            :{'grid':'Btr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-CM2-HR4'     :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-CM2-SR5'     :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-CM2-VHR4'    :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CMCC-ESM2'        :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CNRM-CM6-1'       :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CNRM-CM6-1-HR'    :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CNRM-ESM2-1'      :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'CanESM5'          :{'grid':'Ctr',         'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'CanESM5-CanOE'    :{'grid':'unknown',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'E3SM-1-0'         :{'grid':'A-gr'   ,     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'E3SM-1-1'         :{'grid':'A-gr'   ,     'haloEW':[0, 0], 'flipNS':False, 'reg':True,  'extraT':False},
    'E3SM-1-1-ECA'     :{'grid':'A-gr',        'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'EC-Earth3'        :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3-AerChem':{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3-CC'     :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3-LR'     :{'grid':'unknown',     'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3-Veg'    :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3-Veg-LR' :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3P'       :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3P-HR'    :{'grid':'Ctr',         'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'EC-Earth3P-VHR'   :{'grid':'unknown',     'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'ECMWF-IFS-HR'     :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'ECMWF-IFS-LR'     :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'ECMWF-IFS-MR'     :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'FGOALS-f3-H'      :{'grid':'Bbl', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'FGOALS-f3-L'      :{'grid':'Bbl', 'haloEW':[0, 0], 'flipNS':True, 'reg':False, 'extraT':False},
    'FGOALS-g3'        :{'grid':'Bbl', 'haloEW':[0, 0], 'flipNS':True, 'reg':False, 'extraT':False},
    'FIO-ESM-2-0'      :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-AM4'         :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GFDL-CM4'         :{'grid':'Cbl', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-CM4C192'     :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GFDL-ESM2M'       :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-ESM4'        :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GFDL-OM4p5B'      :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'GISS-E2-1-G'      :{'grid':'Cbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-1-G-CC'   :{'grid':'Cbr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-1-H'      :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-2-G'      :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'GISS-E2-2-H'      :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'HadGEM3-GC31-HH'  :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'HadGEM3-GC31-HM'  :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'HadGEM3-GC31-LL'  :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'HadGEM3-GC31-LM'  :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'HadGEM3-GC31-MH'  :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'HadGEM3-GC31-MM'  :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'HiRAM-SIT-HR'     :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'HiRAM-SIT-LR'     :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'ICON-ESM-LR'      :{'grid':'Unstructured', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'IITM-ESM'         :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'INM-CM4-8'        :{'grid':'A-gr1', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'INM-CM5-0'        :{'grid':'A-gr1', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'INM-CM5-H'        :{'grid':'A-gr1', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'IPSL-CM5A2-INCA'  :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'IPSL-CM6A-ATM-HR' :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'IPSL-CM6A-LR'     :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'IPSL-CM6A-LR-INCA':{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'KACE-1-0-G'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'KIOST-ESM'        :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MCM-UA-1-0'       :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MIROC-ES2H'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MIROC-ES2L'       :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'MIROC6'           :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'MPI-ESM-1-2-HAM'  :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MPI-ESM1-2-HR'    :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MPI-ESM1-2-LR'    :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MPI-ESM1-2-XR'    :{'grid':'Cbr', 'haloEW':[1, 1], 'flipNS':True, 'reg':False, 'extraT':False},
    'MRI-AGCM3-2-H'    :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MRI-AGCM3-2-S'    :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'MRI-ESM2-0'       :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':True},
    'NESM3'            :{'grid':'Ctr', 'haloEW':[1, 1], 'flipNS':False, 'reg':False, 'extraT':False},
    'NICAM16-7S'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'NICAM16-8S'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'NICAM16-9S'       :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'NorCPM1'          :{'grid':'Cbl-gr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'NorESM1-F'        :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'NorESM2-LM'       :{'grid':'Cbl-gr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'NorESM2-MM'       :{'grid':'Cbl-gr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'SAM0-UNICON'      :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'TaiESM1'          :{'grid':'Btr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'TaiESM1-TIMCOM'   :{'grid':'A', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'UKESM1-0-LL'      :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'UKESM1-ice-LL'    :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'UKESM1-1-LL'      :{'grid':'Ctr', 'haloEW':[0, 0], 'flipNS':False, 'reg':False, 'extraT':False},
    'E3SM-2-0'         :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
    'EC-Earth3-HR'     :{'grid':'unknown', 'haloEW':[0, 0], 'flipNS':False, 'reg':True, 'extraT':False},
}